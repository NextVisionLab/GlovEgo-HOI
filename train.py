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
import wandb

# import some common detectron2 utilities
from detectron2.config import get_cfg, CfgNode as CN
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, build_detection_train_loader
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

##### ArgumentParser
parser = argparse.ArgumentParser(description='EHOI Training Script with Wandb')
parser.add_argument('--train_json', dest='train_json', help='train json path', type=str, required=True)
parser.add_argument('--weights_path', dest='weights', help='weights path', type=str, default="detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl")
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')

parser.add_argument('--test_json', dest='test_json', nargs='*', help='test json paths', type=str)
parser.add_argument('--test_dataset_names', dest='test_dataset_names', nargs='*', help='test dataset names', type=str)

parser.add_argument('--no_predict_mask', dest='predict_mask', action='store_false', default=True)
parser.add_argument('--mask_gt', action='store_true', default=False)
parser.add_argument('--no_depth_module', dest='depth_module', action='store_false', default=True)

parser.add_argument('--contact_state_modality', default="mask+rgb+depth+fusion", help="contact state modality", type=str, 
                    choices=["rgb", "cnn_rgb", "depth", "mask", "rgb+depth", "mask+rgb", "mask+depth", 
                            "mask+rgb+depth", "mask+rgb+depth+fusion", "mask+rgb+fusion", "rgb+depth+fusion", "rgb+fusion"])

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

# Wandb arguments
parser.add_argument('--wandb_project', default="ehoi-hand-object-interaction", help='Wandb project name', type=str)
parser.add_argument('--wandb_run_name', default=None, help='Wandb run name', type=str)
parser.add_argument('--no_wandb', action='store_true', default=False, help='Disable wandb logging')

def parse_args():
    """Parse command line arguments and set derived parameters"""
    args = parser.parse_args()
    
    # Validate test arguments
    if args.test_json is None:
        args.test_json = []
        args.test_dataset_names = []
    
    if len(args.test_json) != len(args.test_dataset_names): 
        raise ValueError("test_json and test_dataset_names must have the same length")
    
    return args

def setup_keypoint_metadata(dataset_name):
    """Configure keypoint metadata for dataset with 21 hand keypoints"""
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

def get_evaluators(cfg, dataset_name, output_folder, converter):
    """Get both COCO and EHOI evaluators"""
    coco_evaluator = COCOEvaluator(dataset_name, output_dir=output_folder, tasks=("bbox",))
    ehoi_evaluator = EHOIEvaluator(cfg, dataset_name, converter)
    return [coco_evaluator, ehoi_evaluator]

def do_test(cfg, model, *, converter, mapper, data):
    """Run evaluation"""
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper(cfg, data, is_train=False))
        evaluators = get_evaluators(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name), converter)
        results_i = inference_on_dataset(model, data_loader, evaluators)
        results[dataset_name] = results_i
    
    if len(results) == 1:
        results = list(results.values())[0]
    
    return results

def load_cfg(args, num_classes):
    """Load and configure training configuration"""
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.merge_from_file("./configs/Custom/custom.yaml")

    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = tuple(args.test_dataset_names)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # Additional modules configuration
    cfg.ADDITIONAL_MODULES.USE_MASK_GT = args.mask_gt
    cfg.ADDITIONAL_MODULES.USE_MASK = True if "mask" in args.contact_state_modality else args.predict_mask
    cfg.ADDITIONAL_MODULES.DEPTH_MODULE.USE_DEPTH_MODULE = True if "depth" in args.contact_state_modality else args.depth_module
    cfg.ADDITIONAL_MODULES.CONTACT_STATE_MODALITY = args.contact_state_modality
    cfg.ADDITIONAL_MODULES.CONTACT_STATE_CNN_INPUT_SIZE = args.contact_state_cnn_input_size
    cfg.ADDITIONAL_MODULES.NORMALIZE_KEYPOINT_COORDS = True
    
    # Always enable keypoints
    cfg.MODEL.KEYPOINT_ON = True
    
    # ROI Keypoint Head configuration
    cfg.MODEL.ROI_KEYPOINT_HEAD = CN()
    cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 21
    cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = (512, 512, 512, 512)
    cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True
    cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
    cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = "ROIAlignV2"
    cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 0
    cfg.MODEL.ROI_KEYPOINT_HEAD.POSITIVE_FRACTION = 0.25
    cfg.MODEL.ROI_KEYPOINT_HEAD.BATCH_SIZE_PER_IMAGE = 64
    
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    
    # Solver configuration
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.STEPS = tuple(args.solver_steps)
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.SOLVER.WARMUP_ITERS = args.warmup_iters

    cfg.MODEL.WEIGHTS = args.weights
    cfg.OUTPUT_DIR = "./output_dir/debug_training/" if args.debug else "./output_dir/last_training/"
    
    if args.debug:
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.TEST.EVAL_PERIOD = 100
    
    cfg.freeze()
    setup_logger(output=cfg.OUTPUT_DIR)

    with open(os.path.join(cfg.OUTPUT_DIR, "cfg.yaml"), "w") as f:
        f.write(cfg.dump())

    return cfg

def setup_wandb(args, cfg, num_classes):
    """Initialize Wandb tracking with robust error handling"""
    if args.no_wandb:
        return None
    
    try:
        # Try to initialize with simple configuration
        import os
        
        # Set offline mode if there are permission issues
        if 'WANDB_MODE' not in os.environ:
            os.environ['WANDB_MODE'] = 'offline'
        
        print("Initializing Wandb in offline mode...")
        
        run = wandb.init(
            project="ehoi-training",  # Simple project name
            name=args.wandb_run_name or "ehoi_run",
            config={
                "learning_rate": args.base_lr,
                "batch_size": args.ims_per_batch,
                "max_iterations": args.max_iter,
                "contact_state_modality": args.contact_state_modality,
                "num_classes": num_classes,
            },
            tags=["detectron2", "keypoints"],
            mode="offline"  # Force offline mode
        )
        
        print("✅ Wandb initialized successfully in offline mode")
        print("Run 'wandb sync wandb/latest-run' after training to upload logs")
        return run
        
    except Exception as e:
        print(f"❌ Wandb initialization failed: {e}")
        print("Continuing training without Wandb...")
        return None
def log_losses_to_wandb(loss_dict_reduced, iteration, lr):
    """Log losses and metrics to wandb"""
    if wandb.run is None:
        return
    
    # Prepare logging dictionary
    log_dict = {"iteration": iteration, "learning_rate": lr}
    
    # Add all losses
    for loss_name, loss_value in loss_dict_reduced.items():
        log_dict[f"losses/{loss_name}"] = loss_value
    
    # Calculate total loss
    total_loss = sum(loss for loss in loss_dict_reduced.values())
    log_dict["losses/total_loss"] = total_loss
    
    # Log to wandb
    wandb.log(log_dict, step=iteration)

def log_evaluation_to_wandb(results, iteration):
    """Log evaluation results to wandb"""
    if wandb.run is None or not results:
        return
    
    log_dict = {"iteration": iteration}
    
    # Process evaluation results
    for dataset_name, dataset_results in results.items():
        if isinstance(dataset_results, dict):
            for metric_category, metrics in dataset_results.items():
                if isinstance(metrics, dict):
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            log_dict[f"eval/{dataset_name}/{metric_category}/{metric_name}"] = metric_value
                elif isinstance(metrics, (int, float)):
                    log_dict[f"eval/{dataset_name}/{metric_category}"] = metrics
    
    # Log to wandb
    wandb.log(log_dict, step=iteration)

if __name__ == "__main__":
    args = parse_args()
    print("=" * 50)
    print("EHOI TRAINING WITH WANDB")
    print("=" * 50)
    print(args)

    if args.debug:
        debugpy.listen(56763)
        print("Debug mode enabled, waiting for client...")
        debugpy.wait_for_client()

    # Set seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Train register
    train_images_path = os.path.join(args.train_json[:[x for x, v in enumerate(args.train_json) if v == '/'][-2]], "images/")
    register_coco_instances("dataset_train", {}, args.train_json, train_images_path)
    setup_keypoint_metadata("dataset_train")
    
    dataset_train_metadata = MetadataCatalog.get("dataset_train")
    dataset_dict_train = DatasetCatalog.get("dataset_train")
    
    # Load the JSON data FIRST to calculate num_classes correctly
    with open(dataset_train_metadata.json_file) as json_file: 
        data_anns_train_sup = json.load(json_file)
    
    # Calculate num_classes from JSON data
    if 'categories' in data_anns_train_sup:
        num_classes = len(data_anns_train_sup['categories'])
    else:
        category_ids = set()
        for ann in data_anns_train_sup.get('annotations', []):
            category_ids.add(ann['category_id'])
        num_classes = len(category_ids)
    
    print(f"Number of classes detected: {num_classes}")
    
    # Test register
    for json_, name_ in zip(args.test_json, args.test_dataset_names):
        images_path = os.path.join(json_[:[x for x, v in enumerate(json_) if v == '/'][-2]], "images/")
        print(json_, name_, images_path)
        register_coco_instances(name_, {}, json_, images_path)
        setup_keypoint_metadata(name_)
        dataset_test_dicts = DatasetCatalog.get(name_)
        test_metadata = MetadataCatalog.get(name_)
        test_metadata.set(coco_gt_hands=test_metadata.json_file.replace(".json", "_hands.json"))

    # Load config
    cfg = load_cfg(args, num_classes=num_classes)
    
    # Setup Wandb
    wandb_run = setup_wandb(args, cfg, num_classes)

    # Init model
    mapper = EhoiDatasetMapperDepthv1(cfg, data_anns_sup=data_anns_train_sup)
    mapper_test = EhoiDatasetMapperDepthv1
    if len(args.test_json):
        converter = MMEhoiNetConverterv1(cfg, test_metadata)
    model = MMEhoiNetv1(cfg, dataset_train_metadata)

    # Watch model with wandb
    if wandb_run is not None:
        wandb.watch(model, log="all", log_freq=1000)

    # Load weights
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR)
    _ = checkpointer.load(cfg.MODEL.WEIGHTS)

    # Model to device
    device = "cuda:" + str(args.cuda_device)
    model.to(device)
    model.train()
    
    logger = logging.getLogger("detectron2")
    logger.info(f"Model loaded on device: {device}")

    # Optimizer and scheduler init
    base_parameters = [param for name, param in model.named_parameters() if 'depth_module' not in name]
    depth_parameters = [param for name, param in model.named_parameters() if 'depth_module' in name]
    optimizer = torch.optim.SGD([
        {'params': base_parameters},
        {'params': depth_parameters, "lr": float(cfg.ADDITIONAL_MODULES.DEPTH_MODULE.LR)}
    ], lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = build_lr_scheduler(cfg, optimizer)
    
    # Training parameters
    start_iter = 1
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    data_loader = build_detection_train_loader(cfg, mapper=mapper)

    logger.info("Starting training from iteration {}".format(start_iter))

    # Training loop
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]["lr"]
            storage.put_scalar("lr", current_lr, smoothing_hint=False)
            
            # Log to Wandb
            if comm.is_main_process():
                log_losses_to_wandb(loss_dict_reduced, iteration, current_lr)
            
            # Evaluation
            if len(args.test_json) and cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
                logger.info("Starting evaluation...")
                results_val = do_test(cfg, model, converter=converter, mapper=mapper_test, data=data_anns_train_sup)
                
                # Log evaluation to Wandb
                if comm.is_main_process():
                    log_evaluation_to_wandb(results_val, iteration)
                
                logger.info("Evaluation completed")

            # Write and checkpoint
            if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
    
    logger.info("Training completed successfully!")
    
    # Finish wandb run
    if wandb_run is not None:
        wandb.finish()