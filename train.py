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

##### ArgumentParser
parser = argparse.ArgumentParser(description='Train script')
parser.add_argument('--train_json', dest='train_json', help='train json path', type=str, required = True)
parser.add_argument('--weights_path', dest='weights', help='weights path', type=str, default="detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl")
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')

parser.add_argument('--test_json', dest='test_json', nargs='*', help='test json paths', type=str)
parser.add_argument('--test_dataset_names', dest='test_dataset_names', nargs='*', help='test dataset names', type=str)

parser.add_argument('--no_predict_mask', dest='predict_mask', action='store_false', default=True)
parser.add_argument('--mask_gt', action='store_true', default=False)
parser.add_argument('--no_depth_module', dest='depth_module', action='store_false', default=True)

# UPDATED: Extended choices to include all keypoint combinations
parser.add_argument('--contact_state_modality', default="mask+rgb+depth+fusion", help="contact state modality", type=str, 
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
                        # Keypoint-only modalities
                        "keypoints",
                        "keypoints+fusion",
                        # Keypoint + RGB combinations
                        "rgb+keypoints",
                        "rgb+keypoints+fusion",
                        # Keypoint + Depth combinations
                        "depth+keypoints",
                        "depth+keypoints+fusion",
                        # Keypoint + Mask combinations
                        "mask+keypoints",
                        "mask+keypoints+fusion",
                        # Keypoint + RGB + Depth combinations
                        "rgb+depth+keypoints",
                        "rgb+depth+keypoints+fusion",
                        # Keypoint + Mask + RGB combinations
                        "mask+rgb+keypoints",
                        "mask+rgb+keypoints+fusion",
                        # Keypoint + Mask + Depth combinations
                        "mask+depth+keypoints",
                        "mask+depth+keypoints+fusion",
                        # Full multimodal combinations
                        "mask+rgb+depth+keypoints",
                        "mask+rgb+depth+keypoints+fusion"
                    ])
parser.add_argument('--contact_state_cnn_input_size', default="128", help="input size for the CNN contact state classification module", type=int)

# Keypoint related arguments
parser.add_argument('--use_keypoints', action='store_true', default=False, help='Enable keypoint detection and Early Fusion')
parser.add_argument('--keypoint_early_fusion', action='store_true', default=False, help='Use Early Fusion with keypoints for contact state')
parser.add_argument('--num_keypoints', default=21, help='Number of keypoints (default: 21 for hands)', type=int)
parser.add_argument('--keypoint_loss_weight', default=1.0, help='Weight for keypoint loss', type=float)
parser.add_argument('--normalize_keypoint_coords', action='store_true', default=True, help='Normalize keypoint coordinates')

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
    
    # Auto-enable keypoints if specified in modality
    if "keypoints" in args.contact_state_modality:
        if not args.use_keypoints:
            print("INFO: keypoints detected in contact_state_modality. Automatically enabling --use_keypoints")
            args.use_keypoints = True
        if not args.keypoint_early_fusion:
            print("INFO: keypoints detected in contact_state_modality. Automatically enabling --keypoint_early_fusion")
            args.keypoint_early_fusion = True
    
    # Validation: keypoint_early_fusion requires use_keypoints
    if args.keypoint_early_fusion and not args.use_keypoints:
        print("WARNING: --keypoint_early_fusion enabled but --use_keypoints is False. Automatically enabling --use_keypoints")
        args.use_keypoints = True
    
    if args.use_keypoints:
        print("=== KEYPOINT CONFIGURATION ===")
        print(f"Use Keypoints: {args.use_keypoints}")
        print(f"Early Fusion: {args.keypoint_early_fusion}")
        print(f"Contact State Modality: {args.contact_state_modality}")
        print(f"Number of Keypoints: {args.num_keypoints}")
        print(f"Keypoint Loss Weight: {args.keypoint_loss_weight}")
        print("==============================")
    
    return args

def get_evaluators(cfg, dataset_name, output_folder, converter):
    cocoEvaluator = COCOEvaluator(dataset_name, output_dir=output_folder, tasks = ("bbox",)) 
    return [cocoEvaluator, EHOIEvaluator(cfg, dataset_name, converter)]

def do_test(cfg, model, * ,converter, mapper, data):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper = mapper(cfg, data, is_train = False))
        evaluators = get_evaluators(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name), converter)
        results_i = inference_on_dataset(model, data_loader, evaluators)
        results[dataset_name] = results_i
    if len(results) == 1: results = list(results.values())[0]
    return results

def load_cfg(args, num_classes):
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.merge_from_file("./configs/Custom/custom.yaml")

    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = tuple(args.test_dataset_names)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    # CONFIGURAZIONI ESISTENTI
    cfg.ADDITIONAL_MODULES.USE_MASK_GT = args.mask_gt
    cfg.ADDITIONAL_MODULES.USE_MASK = True if "mask" in args.contact_state_modality else args.predict_mask
    cfg.ADDITIONAL_MODULES.DEPTH_MODULE.USE_DEPTH_MODULE = True if "depth" in args.contact_state_modality else args.depth_module
    cfg.ADDITIONAL_MODULES.CONTACT_STATE_MODALITY = args.contact_state_modality
    cfg.ADDITIONAL_MODULES.CONTACT_STATE_CNN_INPUT_SIZE = args.contact_state_cnn_input_size
    
    # NUOVE CONFIGURAZIONI PER KEYPOINTS
    cfg.ADDITIONAL_MODULES.USE_KEYPOINTS = args.use_keypoints or "keypoints" in args.contact_state_modality
    cfg.ADDITIONAL_MODULES.USE_KEYPOINT_EARLY_FUSION = args.keypoint_early_fusion or "keypoints" in args.contact_state_modality
    cfg.ADDITIONAL_MODULES.NORMALIZE_KEYPOINT_COORDS = args.normalize_keypoint_coords
    
    # ROI KEYPOINT HEAD CONFIGURATION
    if cfg.ADDITIONAL_MODULES.USE_KEYPOINTS:
        # Abilita keypoint head se non già presente
        cfg.MODEL.ROI_KEYPOINT_HEAD = CN()
        cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = args.num_keypoints
        cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = args.keypoint_loss_weight
        cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = (512, 512, 512, 512)
        cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True
        
        # Aggiungi keypoints ai ROI HEADS
        cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
        if "keypoint" not in cfg.MODEL.ROI_HEADS.NAME:
            cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"  # Supporta keypoints
    
    # SOLVER CONFIGURATIONS
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.STEPS = tuple(args.solver_steps)
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.SOLVER.WARMUP_ITERS = args.warmup_iters

    cfg.MODEL.WEIGHTS = args.weights
    cfg.OUTPUT_DIR = "./output_dir/last_training/"
    
    # LOGGING E DEBUG
    if args.debug:
        cfg.SOLVER.IMS_PER_BATCH = 1  # Batch size ridotto per debug
        cfg.TEST.EVAL_PERIOD = 100    # Eval più frequente per debug
        cfg.OUTPUT_DIR = "./output_dir/debug_training/"
    
    cfg.freeze()
    setup_logger(output = cfg.OUTPUT_DIR)

    with open(os.path.join(cfg.OUTPUT_DIR, "cfg.yaml"), "w") as f:
        f.write(cfg.dump())
    
    # LOG DELLA CONFIGURAZIONE KEYPOINTS
    logger = logging.getLogger("detectron2")
    if cfg.ADDITIONAL_MODULES.USE_KEYPOINTS:
        logger.info("=== KEYPOINT CONFIGURATION ===")
        logger.info(f"Use Keypoints: {cfg.ADDITIONAL_MODULES.USE_KEYPOINTS}")
        logger.info(f"Early Fusion: {cfg.ADDITIONAL_MODULES.USE_KEYPOINT_EARLY_FUSION}")
        logger.info(f"Num Keypoints: {cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS}")
        logger.info(f"Contact State Modality: {cfg.ADDITIONAL_MODULES.CONTACT_STATE_MODALITY}")
        logger.info("==============================")

    return cfg

if __name__ == "__main__":
    args = parse_args()
    print("=== TRAINING ARGUMENTS ===")
    print(f"Use Keypoints: {args.use_keypoints}")
    print(f"Keypoint Early Fusion: {args.keypoint_early_fusion}")
    print(f"Contact State Modality: {args.contact_state_modality}")
    print(f"Number of Keypoints: {args.num_keypoints}")
    print("==========================")

    if args.debug:
        debugpy.listen(56763)
        print("Debug...")
        debugpy.wait_for_client()

    ###SET SEED
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ###TRAIN REGISTER
    train_images_path = os.path.join(args.train_json[:[x for x, v in enumerate(args.train_json) if v == '/'][-2]], "images/")
    register_coco_instances("dataset_train", {}, args.train_json, train_images_path)
    dataset_train_metadata = MetadataCatalog.get("dataset_train")
    dataset_dict_train = DatasetCatalog.get("dataset_train")
    num_classes = len(dataset_train_metadata.as_dict()["thing_dataset_id_to_contiguous_id"])
    with open(dataset_train_metadata.json_file) as json_file: 
        data_anns_train_sup = json.load(json_file)
    
    ###TEST REGISTER
    for json_, name_ in zip(args.test_json, args.test_dataset_names):
        images_path = os.path.join(json_[:[x for x, v in enumerate(json_) if v == '/'][-2]], "images/")
        print(json_, name_, images_path)
        register_coco_instances(name_, {}, json_, images_path)
        dataset_test_dicts = DatasetCatalog.get(name_)
        test_metadata = MetadataCatalog.get(name_)
        test_metadata.set(coco_gt_hands = test_metadata.json_file.replace(".json", "_hands.json"))

    ###LOAD CFG 
    cfg = load_cfg(args, num_classes=num_classes)

    ###INIT MODEL 
    mapper = EhoiDatasetMapperDepthv1(cfg, data_anns_sup = data_anns_train_sup)
    mapper_test = EhoiDatasetMapperDepthv1
    if len(args.test_json): 
        converter = MMEhoiNetConverterv1(cfg, test_metadata)
    
    model = MMEhoiNetv1(cfg, dataset_train_metadata)

    ###LOAD WEIGHTS
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR)
    _ = checkpointer.load(cfg.MODEL.WEIGHTS)

    ###MODEL TO DEVICE
    device = "cuda:" + str(args.cuda_device)
    model.to(device)
    model.train()
    print("Modello caricato:", model.device)
    
    # LOG 
    logger = logging.getLogger("detectron2")
    if cfg.ADDITIONAL_MODULES.USE_KEYPOINTS:
        logger.info("Model initialized with KEYPOINT support")
        if hasattr(model, 'keypoint_feature_extractor'):
            logger.info("Keypoint feature extractor: LOADED")
        if cfg.ADDITIONAL_MODULES.USE_KEYPOINT_EARLY_FUSION:
            logger.info("Early Fusion for Contact State: ENABLED")

    ###OPTIMIZER AND SCHEDULER INIT
    base_parameters = [param for name, param in model.named_parameters() if 'depth_module' not in name]
    depth_parameters = [param for name, param in model.named_parameters() if 'depth_module' in name]
    optimizer = torch.optim.SGD([
            {'params': base_parameters},
            {'params': depth_parameters, "lr": float(cfg.ADDITIONAL_MODULES.DEPTH_MODULE.LR)}],
            lr = cfg["SOLVER"].BASE_LR, 
            momentum = cfg["SOLVER"]["MOMENTUM"], 
            weight_decay = cfg["SOLVER"]["WEIGHT_DECAY"])
    scheduler = build_lr_scheduler(cfg, optimizer)
    
    ###PARAMS
    start_iter = 1
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    data_loader = build_detection_train_loader(cfg, mapper = mapper)

    logger = logging.getLogger("detectron2")
    logger.info("Starting training from iteration {}".format(start_iter))

    ###TRAIN LOOP
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            
            loss_dict = model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process(): storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            
            if len(args.test_json) and cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
                results_val = do_test(cfg, model, converter = converter, mapper= mapper_test, data = data_anns_train_sup)

            if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)