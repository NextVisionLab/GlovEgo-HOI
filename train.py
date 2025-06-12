# import some common libraries
import argparse
import numpy as np
import random
import os
import json
import torch
from collections import OrderedDict
import logging
import debugpy

# import some common detectron2 utilities
from detectron2.config import get_cfg 
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog,  build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances 
from detectron2.data.ehoi_dataset_mapper_v1 import *
from detectron2.evaluation import COCOEvaluator, EHOIEvaluator, inference_on_dataset
from detectron2.utils.converters import *
from detectron2.modeling.meta_arch import MMEhoiNetv1
from detectron2.utils.logger import setup_logger
from detectron2.engine import default_writers
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage
import detectron2.utils.comm as comm
from detectron2.config import get_cfg, CfgNode as CN

parser = argparse.ArgumentParser(description='EHOI Training Script')
parser.add_argument('--train_json', dest='train_json', help='train json path', type=str, required = True)
parser.add_argument('--weights_path', dest='weights', help='weights path', type=str, default="detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl")
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')

parser.add_argument('--test_json', dest='test_json', nargs='*', help='test json paths', type=str)
parser.add_argument('--test_dataset_names', dest='test_dataset_names', nargs='*', help='test dataset names', type=str)

parser.add_argument('--no_predict_mask', dest='predict_mask', action='store_false', default=True)
parser.add_argument('--mask_gt', action='store_true', default=False)
parser.add_argument('--no_depth_module', dest='depth_module', action='store_false', default=True)

parser.add_argument('--contact_state_modality', default="mask+rgb+depth+keypoints+fusion", help="contact state modality", type=str, 
                    choices=[
                        "rgb", 
                        "cnn_rgb", 
                        "depth", 
                        "mask", 
                        "rgb+depth", 
                        "mask+rgb", 
                        "mask+depth", 
                        "mask+rgb+depth", 
                        "mask+rgb+depth+fusion", 
                        "mask+rgb+fusion", 
                        "rgb+depth+fusion", 
                        "rgb+fusion",
                        "keypoints",
                        "keypoints+fusion", 
                        "mask+rgb+depth+keypoints+fusion"
                    ])
parser.add_argument('--contact_state_cnn_input_size', default="128", help="input size for the CNN contact state classification module", type=int)

parser.add_argument('--cuda_device', default=0, help='CUDA device id', type=int)
parser.add_argument('--base_lr', default=0.001, help='base learning rate.', type=float)
parser.add_argument('--ims_per_batch', default=4, help='ims per batch', type=int)
parser.add_argument('--solver_steps', default=[40000, 60000], help='solver_steps', nargs='+', type=int)
parser.add_argument('--max_iter', default=80000, help='max_iter', type=int)
parser.add_argument('--checkpoint_period', default=5000, help='checkpoint_period', type=int)
parser.add_argument('--eval_period', default=5000, help='eval_period', type=int)
parser.add_argument('--warmup_iters', default=1000, help='warmup_iters', type=int)
parser.add_argument('--debug', action='store_true', default=False)

def parse_args():
    args = parser.parse_args()
    if type(args.test_json) != type(args.test_dataset_names):
        assert False, "len of test_json and test_dataset_names must be the same"
    if args.test_json == None:
        args.test_json = []
        args.test_dataset_names = []
    if len(args.test_json) != len(args.test_dataset_names): 
        assert False, "len of test_json and test_dataset_names must be the same"
    
    args.use_keypoints = "keypoints" in args.contact_state_modality
    args.keypoint_early_fusion = "keypoints" in args.contact_state_modality and "fusion" in args.contact_state_modality
    args.num_keypoints = 21
    args.keypoint_loss_weight = 1.0
    
    return args

def setup_keypoint_metadata(dataset_name, use_keypoints=True):
    if use_keypoints:
        from detectron2.data import MetadataCatalog
        
        keypoint_names = [
            "wrist",
            "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip", 
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        ]
        
        keypoint_connection_rules = [
            ("wrist", "thumb_mcp", (255, 0, 0)),
            ("thumb_mcp", "thumb_pip", (255, 0, 0)),
            ("thumb_pip", "thumb_dip", (255, 0, 0)), 
            ("thumb_dip", "thumb_tip", (255, 0, 0)),
            ("wrist", "index_mcp", (0, 255, 0)),
            ("index_mcp", "index_pip", (0, 255, 0)),
            ("index_pip", "index_dip", (0, 255, 0)),
            ("index_dip", "index_tip", (0, 255, 0)),
            ("wrist", "middle_mcp", (0, 0, 255)),
            ("middle_mcp", "middle_pip", (0, 0, 255)),
            ("middle_pip", "middle_dip", (0, 0, 255)),
            ("middle_dip", "middle_tip", (0, 0, 255)),
            ("wrist", "ring_mcp", (255, 255, 0)),
            ("ring_mcp", "ring_pip", (255, 255, 0)),
            ("ring_pip", "ring_dip", (255, 255, 0)),
            ("ring_dip", "ring_tip", (255, 255, 0)),
            ("wrist", "pinky_mcp", (255, 0, 255)),
            ("pinky_mcp", "pinky_pip", (255, 0, 255)), 
            ("pinky_pip", "pinky_dip", (255, 0, 255)),
            ("pinky_dip", "pinky_tip", (255, 0, 255)),
        ]
        
        MetadataCatalog.get(dataset_name).set(
            keypoint_names=keypoint_names,
            keypoint_flip_map=[],
            keypoint_connection_rules=keypoint_connection_rules
        )

def load_cfg(args, num_classes):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.merge_from_file("./configs/Custom/custom.yaml")

    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = tuple(args.test_dataset_names)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    cfg.ADDITIONAL_MODULES.USE_MASK_GT = args.mask_gt
    cfg.ADDITIONAL_MODULES.USE_MASK = True if "mask" in args.contact_state_modality else args.predict_mask
    cfg.ADDITIONAL_MODULES.DEPTH_MODULE.USE_DEPTH_MODULE = True if "depth" in args.contact_state_modality else args.depth_module
    cfg.ADDITIONAL_MODULES.CONTACT_STATE_MODALITY = args.contact_state_modality
    cfg.ADDITIONAL_MODULES.CONTACT_STATE_CNN_INPUT_SIZE = args.contact_state_cnn_input_size
    
    cfg.ADDITIONAL_MODULES.USE_KEYPOINTS = args.use_keypoints
    cfg.ADDITIONAL_MODULES.USE_KEYPOINT_EARLY_FUSION = args.keypoint_early_fusion
    cfg.ADDITIONAL_MODULES.NORMALIZE_KEYPOINT_COORDS = True
    
    cfg.MODEL.KEYPOINT_ON = args.use_keypoints
    
    if cfg.ADDITIONAL_MODULES.USE_KEYPOINTS:
        cfg.MODEL.ROI_KEYPOINT_HEAD = CN()
        cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = args.num_keypoints
        cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = args.keypoint_loss_weight
        cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = (512, 512, 512, 512)
        cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True
        cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
        cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
        cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = "ROIAlignV2"
        cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1
        cfg.MODEL.ROI_KEYPOINT_HEAD.POSITIVE_FRACTION = 0.25
        cfg.MODEL.ROI_KEYPOINT_HEAD.BATCH_SIZE_PER_IMAGE = 64
        
        cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
        cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.STEPS = tuple(args.solver_steps)
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.SOLVER.WARMUP_ITERS = args.warmup_iters

    cfg.MODEL.WEIGHTS = args.weights
    cfg.OUTPUT_DIR = "./output_dir/last_training/"
    
    if args.debug:
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.TEST.EVAL_PERIOD = 100
        cfg.OUTPUT_DIR = "./output_dir/debug_training/"
    
    cfg.freeze()
    setup_logger(output = cfg.OUTPUT_DIR)

    with open(os.path.join(cfg.OUTPUT_DIR, "cfg.yaml"), "w") as f:
        f.write(cfg.dump())

    return cfg

def log_epoch_summary(losses_accumulated, iteration, max_iter, optimizer, logger):
    """Log epoch summary with all losses in Detectron2 format"""
    if not losses_accumulated:
        return
    
    # Calculate averages
    avg_losses = {}
    for loss_name, loss_values in losses_accumulated.items():
        if loss_values:
            avg_losses[loss_name] = sum(loss_values) / len(loss_values)
    
    # Calculate ETA
    eta_seconds = 0  # Would need actual timing for real ETA
    eta_str = f"{eta_seconds//3600}:{(eta_seconds%3600)//60:02d}:{eta_seconds%60:02d}"
    
    # Get current learning rate
    current_lr = optimizer.param_groups[0]["lr"]
    
    # Get max memory
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() // (1024**2)  # Convert to MB
    else:
        max_mem = 0
    
    # Format losses with proper precision
    loss_items = []
    
    # Required losses in specific order
    loss_order = [
        "total_loss", "loss_cls", "loss_box_reg", "loss_rpn_cls", "loss_rpn_loc",
        "loss_classification_contact_state", "loss_classification_hand_lr", 
        "loss_regression_dxdymagn", "loss_depth", "loss_mask", "loss_keypoint"
    ]
    
    for loss_name in loss_order:
        if loss_name in avg_losses:
            value = avg_losses[loss_name]
            if loss_name == "total_loss":
                loss_items.append(f"{loss_name}: {value:.1f}")
            elif value >= 0.001:
                loss_items.append(f"{loss_name}: {value:.5f}")
            else:
                loss_items.append(f"{loss_name}: {value:.3e}")
    
    # Build the log message
    log_message = f"eta: {eta_str}  iter: {iteration}  " + "  ".join(loss_items)
    log_message += f"  lr: {current_lr:.0e}  max_mem: {max_mem}M"
    
    logger.info(log_message)

def do_test(cfg, model, converter, mapper, data):
    """Run evaluation on test dataset"""
    logger = logging.getLogger("detectron2")
    
    results = {}
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        evaluator = EHOIEvaluator(cfg, dataset_name, converter)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        
        if comm.is_main_process():
            logger.info(f"Evaluation results for {dataset_name}:")
            for task, metrics in results_i.items():
                logger.info(f"  {task}:")
                for metric_name, value in metrics.items():
                    logger.info(f"    {metric_name}: {value}")
    
    return results

if __name__ == "__main__":
    args = parse_args()
    
    # Clean startup log
    print("=" * 60)
    print("EHOI TRAINING PIPELINE")
    print("=" * 60)
    print(f"Contact State Modality: {args.contact_state_modality}")
    print(f"Keypoints: {'ENABLED' if args.use_keypoints else 'DISABLED'}")
    if args.use_keypoints:
        print(f"Early Fusion: {'ENABLED' if args.keypoint_early_fusion else 'DISABLED'}")
    print(f"Max Iterations: {args.max_iter}")
    print(f"Batch Size: {args.ims_per_batch}")
    print(f"Learning Rate: {args.base_lr}")
    print("=" * 60)

    if args.debug:
        debugpy.listen(56763)
        print("Debug mode enabled, waiting for client...")
        debugpy.wait_for_client()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_images_path = os.path.join(args.train_json[:[x for x, v in enumerate(args.train_json) if v == '/'][-2]], "images/")
    register_coco_instances("dataset_train", {}, args.train_json, train_images_path)
    
    setup_keypoint_metadata("dataset_train", args.use_keypoints)
    
    dataset_train_metadata = MetadataCatalog.get("dataset_train")
    dataset_dict_train = DatasetCatalog.get("dataset_train")
    num_classes = len(dataset_train_metadata.as_dict()["thing_dataset_id_to_contiguous_id"])
    
    with open(dataset_train_metadata.json_file) as json_file: 
        data_anns_train_sup = json.load(json_file)
    
    for json_, name_ in zip(args.test_json, args.test_dataset_names):
        images_path = os.path.join(json_[:[x for x, v in enumerate(json_) if v == '/'][-2]], "images/")
        register_coco_instances(name_, {}, json_, images_path)
        setup_keypoint_metadata(name_, args.use_keypoints)
        dataset_test_dicts = DatasetCatalog.get(name_)
        test_metadata = MetadataCatalog.get(name_)
        test_metadata.set(coco_gt_hands = test_metadata.json_file.replace(".json", "_hands.json"))

    cfg = load_cfg(args, num_classes=num_classes)

    mapper = EhoiDatasetMapperDepthv1(cfg, data_anns_sup = data_anns_train_sup)
    mapper_test = EhoiDatasetMapperDepthv1
    if len(args.test_json): 
        converter = MMEhoiNetConverterv1(cfg, test_metadata)
    
    model = MMEhoiNetv1(cfg, dataset_train_metadata)

    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR)
    _ = checkpointer.load(cfg.MODEL.WEIGHTS)

    device = "cuda:" + str(args.cuda_device)
    model.to(device)
    model.train()
    
    logger = logging.getLogger("detectron2")
    logger.info(f"Model loaded on device: {device}")
    if cfg.ADDITIONAL_MODULES.USE_KEYPOINTS:
        logger.info("Keypoint support: ENABLED")

    base_parameters = [param for name, param in model.named_parameters() if 'depth_module' not in name]
    depth_parameters = [param for name, param in model.named_parameters() if 'depth_module' in name]
    optimizer = torch.optim.SGD([
            {'params': base_parameters},
            {'params': depth_parameters, "lr": float(cfg.ADDITIONAL_MODULES.DEPTH_MODULE.LR)}],
            lr = cfg["SOLVER"].BASE_LR, 
            momentum = cfg["SOLVER"]["MOMENTUM"], 
            weight_decay = cfg["SOLVER"]["WEIGHT_DECAY"])
    scheduler = build_lr_scheduler(cfg, optimizer)
    
    start_iter = 1
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    data_loader = build_detection_train_loader(cfg, mapper = mapper)

    logger.info(f"Starting training from iteration {start_iter}")
    logger.info(f"Total iterations: {max_iter}")

    losses_accumulated = {}
    
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            
            # Accumulate losses for epoch summary
            for loss_name, loss_value in loss_dict_reduced.items():
                if loss_name not in losses_accumulated:
                    losses_accumulated[loss_name] = []
                losses_accumulated[loss_name].append(loss_value)
            
            if "total_loss" not in losses_accumulated:
                losses_accumulated["total_loss"] = []
            losses_accumulated["total_loss"].append(losses_reduced)
            
            if comm.is_main_process(): 
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            
            # Log every 100 iterations with key metrics only
            if iteration % 100 == 0:
                logger.info(f"iter: {iteration:5d} | "
                           f"total_loss: {losses_reduced:.4f} | "
                           f"lr: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Evaluation
            if len(args.test_json) and cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
                log_epoch_summary(losses_accumulated, iteration, max_iter, optimizer, logger)
                losses_accumulated = {}  # Reset for next epoch
                
                logger.info("Starting evaluation...")
                results_val = do_test(cfg, model, converter = converter, mapper= mapper_test, data = data_anns_train_sup)
                logger.info("Evaluation completed")

            # Write and checkpoint
            if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
    
    # Final epoch summary
    log_epoch_summary(losses_accumulated, max_iter, max_iter, optimizer, logger)
    logger.info("Training completed successfully!")