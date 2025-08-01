import argparse
import numpy as np
import random
import os
import json
import torch
from collections import OrderedDict
import logging
import wandb 

# import some common detectron2 utilities
from detectron2.config import get_cfg, CfgNode
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog,  build_detection_test_loader, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data.ehoi_dataset_mapper_v1 import *
from detectron2.evaluation import COCOEvaluator, EHOIEvaluator, inference_on_dataset
from detectron2.utils.converters import *
from detectron2.modeling.meta_arch import MMEhoiNetv1
from detectron2.utils.logger import setup_logger
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage
import detectron2.utils.comm as comm

# --- Argument Parser Setup ---
parser = argparse.ArgumentParser(description='EHOI Training Script')

# --- Dataset and Weights Arguments ---
parser.add_argument('--train_json', dest='train_json', help='Path to the training COCO-style JSON file', type=str, required = True)
parser.add_argument('--weights', dest='weights', help='Path to initial weights (local file or Detectron2 model zoo URL)', type=str, default="detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl")
parser.add_argument('--seed', type=int, default=42, metavar='S', help='Random seed for reproducibility (default: 42)')
parser.add_argument('--test_json', dest='test_json', nargs='*', help='List of paths to test/validation JSON files')
parser.add_argument('--test_dataset_names', dest='test_dataset_names', nargs='*', help='List of names for the test/validation datasets')

# --- Model Configuration Arguments ---
parser.add_argument('--no_predict_mask', dest='predict_mask', action='store_false', default=True, help="Disables the mask prediction branch if not specified in modality")
parser.add_argument('--mask_gt', action='store_true', default=False, help="Use ground truth masks to train the mask head")
parser.add_argument('--no_depth_module', dest='depth_module', action='store_false', default=True, help="Disables the depth prediction module if not specified in modality")

parser.add_argument(
    '--contact_state_modality', 
    default="mask+rgb+depth+keypoints+fusion",  
    help="Defines the input modalities for the contact state prediction head.", 
    type=str, 
    choices=[
        "rgb", "cnn_rgb", "depth", "mask", "rgb+depth", "mask+rgb", "mask+depth", 
        "mask+rgb+depth", "mask+rgb+depth+fusion", "mask+rgb+fusion", 
        "rgb+depth+fusion", "rgb+fusion",
        "keypoints", "rgb+keypoints", "mask+keypoints", "depth+keypoints",
        "mask+rgb+keypoints", "mask+depth+keypoints", "rgb+depth+keypoints",
        "mask+rgb+depth+keypoints", "mask+rgb+depth+keypoints+fusion"
    ]
)
parser.add_argument('--contact_state_cnn_input_size', default="128", help="Input patch size (HxW) for the contact state CNN", type=int)

# --- Solver and Hardware Arguments ---
parser.add_argument('--cuda_device', default=0, help='CUDA device ID to use', type=int)
parser.add_argument('--base_lr', default=0.001, help='Base learning rate', type=float)
parser.add_argument('--ims_per_batch', default=4, help='Images per batch across all GPUs', type=int)
parser.add_argument('--solver_steps', default=[40000, 60000], help='Iterations at which to decay learning rate', nargs='+', type=int)
parser.add_argument('--max_iter', default=80000, help='Total number of training iterations', type=int)
parser.add_argument('--checkpoint_period', default=5000, help='Save a checkpoint every this number of iterations', type=int)
parser.add_argument('--eval_period', default=5000, help='Run evaluation every this number of iterations', type=int)
parser.add_argument('--warmup_iters', default=1000, help='Number of warm-up iterations', type=int)

# --- Wandb Integration Arguments ---
parser.add_argument('--wandb_project', type=str, default="ehoi-project", help='Weights & Biases project name.')
parser.add_argument('--wandb_run_name', type=str, default=None, help='A specific name for the Wandb run.')
parser.add_argument('--no_wandb', action='store_true', default=False, help='Disable all Weights & Biases logging.')


def parse_args():
    """Parses command-line arguments and performs basic validation."""
    args = parser.parse_args()
    if args.test_json is None: args.test_json = []
    if args.test_dataset_names is None: args.test_dataset_names = []
    if len(args.test_json) != len(args.test_dataset_names): 
        raise ValueError("The number of test_json files must match the number of test_dataset_names.")
    return args

def get_evaluators(cfg, dataset_name, output_folder, converter):
    """Builds a list of evaluators for a given dataset."""
    cocoEvaluator = COCOEvaluator(dataset_name, output_dir=output_folder, tasks=("bbox",)) 
    return [cocoEvaluator, EHOIEvaluator(cfg, dataset_name, converter)]

def do_test(cfg, model, *, converter, mapper):
    """Runs evaluation on all configured test datasets."""
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        test_mapper = mapper(cfg, is_train=False) 
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=test_mapper)
        evaluators = get_evaluators(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name), converter)
        results_i = inference_on_dataset(model, data_loader, evaluators)
        results[dataset_name] = results_i
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def load_cfg(args, num_classes):
    cfg = get_cfg()
    cfg.set_new_allowed(True)  

    cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.merge_from_file("./configs/Custom/custom.yaml")

    # --- Dataset and Model settings ---
    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = tuple(args.test_dataset_names)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.WEIGHTS = args.weights

    # --- EHOI-Specific Overrides from command line ---
    cfg.ADDITIONAL_MODULES.USE_MASK_GT = args.mask_gt
    cfg.ADDITIONAL_MODULES.CONTACT_STATE_MODALITY = args.contact_state_modality
    cfg.ADDITIONAL_MODULES.CONTACT_STATE_CNN_INPUT_SIZE = args.contact_state_cnn_input_size
    
    cfg.ADDITIONAL_MODULES.USE_MASK = 'mask' in args.contact_state_modality or args.predict_mask
    cfg.ADDITIONAL_MODULES.DEPTH_MODULE.USE_DEPTH_MODULE = 'depth' in args.contact_state_modality or args.depth_module

    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.STEPS = tuple(args.solver_steps)
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.SOLVER.WARMUP_ITERS = args.warmup_iters
    cfg.TEST.EVAL_PERIOD = args.eval_period

    # --- Output Directory ---
    cfg.OUTPUT_DIR = "./output_dir/last_training/"
    
    cfg.freeze()
        
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(cfg.OUTPUT_DIR, "cfg.yaml"), "w") as f:
        f.write(cfg.dump())

    return cfg

def setup_wandb(args, cfg):
    """Initializes a new Weights & Biases run for experiment tracking."""
    if args.no_wandb:
        logging.getLogger("detectron2").info("Wandb logging is disabled by user.")
        return None
    try:
        run_name = args.wandb_run_name or f"ehoi_{args.contact_state_modality}"
        run = wandb.init(project=args.wandb_project, name=run_name, config=cfg)
        logging.getLogger("detectron2").info(f"Wandb run '{run.name}' initialized.")
        return run
    except ImportError:
        logging.getLogger("detectron2").warning("Wandb not found. To enable logging, run: pip install wandb")
        return None
    except Exception as e:
        logging.getLogger("detectron2").error(f"Failed to initialize Wandb: {e}. Continuing without logging.")
        return None

if __name__ == "__main__":
    args = parse_args()

    # --- Setup ---
    output_dir = "./output_dir/last_training/"
    os.makedirs(output_dir, exist_ok=True)
    setup_logger(output=output_dir)
    logger = logging.getLogger("detectron2") 
    logger.info("Parsed Arguments: %s", args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- Dataset Registration and Metadata Population ---
    train_images_path = os.path.dirname(os.path.dirname(args.train_json)) + "/images/"
    register_coco_instances("dataset_train", {}, args.train_json, train_images_path)
    load_coco_json(args.train_json, train_images_path, "dataset_train")
    dataset_train_metadata = MetadataCatalog.get("dataset_train")
    num_classes = len(dataset_train_metadata.thing_classes)
    
    with open(dataset_train_metadata.json_file) as json_file: 
        data_anns_train_sup = json.load(json_file)
    
    for json_, name_ in zip(args.test_json, args.test_dataset_names):
        images_path = os.path.dirname(os.path.dirname(json_)) + "/images/"
        register_coco_instances(name_, {}, json_, images_path)
        test_metadata = MetadataCatalog.get(name_)
        test_metadata.set(thing_classes=dataset_train_metadata.thing_classes)
        test_metadata.set(coco_gt_hands=test_metadata.json_file.replace(".json", "_hands.json"))

    # --- Configuration and Model Initialization ---
    cfg = load_cfg(args, num_classes=num_classes)
    wandb_run = setup_wandb(args, cfg)
    
    mapper = EhoiDatasetMapperDepthv1(cfg, data_anns_sup=data_anns_train_sup, is_train=True)
    mapper_test = EhoiDatasetMapperDepthv1
    converter = None
    if len(args.test_json):
        converter_metadata = MetadataCatalog.get(args.test_dataset_names[0])
        converter = MMEhoiNetConverterv1(cfg, converter_metadata)
    
    model = MMEhoiNetv1(cfg, dataset_train_metadata)

    # --- Checkpointing and Device Placement ---
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR)
    _ = checkpointer.load(cfg.MODEL.WEIGHTS)
    
    device = "cuda:" + str(args.cuda_device)
    model.to(device)
    model.train()
    logger.info(f"Model loaded on device: {device}")

    # --- Optimizer and Scheduler ---
    base_parameters = [param for name, param in model.named_parameters() if 'depth_module' not in name]
    depth_parameters = [param for name, param in model.named_parameters() if 'depth_module' in name]
    optimizer = torch.optim.SGD([
            {'params': base_parameters},
            {'params': depth_parameters, "lr": float(cfg.ADDITIONAL_MODULES.DEPTH_MODULE.LR)}],
            lr=cfg.SOLVER.BASE_LR, 
            momentum=cfg.SOLVER.MOMENTUM, 
            weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    scheduler = build_lr_scheduler(cfg, optimizer)
    
    # --- Training Setup ---
    start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER
    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)

    logger.info("Starting training from iteration {}".format(start_iter))

    # --- Main Training Loop ---
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            
            loss_dict = model(data)

            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            # Reduce losses from all GPUs for logging
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            # --- Optimization Step ---
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()
            
            # --- Logging to Console and Wandb ---
            if comm.is_main_process():
                if wandb_run:
                    wandb.log({
                        "total_loss": losses_reduced, 
                        "lr": storage.latest()["lr"][0], 
                        **loss_dict_reduced
                    }, step=iteration)

                if (iteration + 1) % 20 == 0 or iteration == max_iter - 1:
                    loss_str = "  ".join([f"{k}: {v:.4f}" for k, v in loss_dict_reduced.items()])
                    logger.info(
                        f"iter: {iteration+1:05d}  "
                        f"lr: {storage.latest()['lr'][0]:.6f}  "
                        f"time: {storage.latest().get('time', [0])[0]:.4f}s  "
                        f"data_time: {storage.latest().get('data_time', [0])[0]:.4f}s  "
                        f"{loss_str}" 
                    )

            # --- Evaluation ---
            if len(args.test_json) and cfg.TEST.EVAL_PERIOD > 0 and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0:
                results_val = do_test(cfg, model, converter=converter, mapper=mapper_test)
                if wandb_run and comm.is_main_process():
                    main_metric = results_val.get("bbox", {}).get("AP", 0)
                    wandb.log({"evaluation/AP50": main_metric}, step=iteration)
            
            periodic_checkpointer.step(iteration)

    logger.info("Training finished.")
    if wandb_run:
        wandb_run.finish()