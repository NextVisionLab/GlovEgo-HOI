import argparse
import numpy as np
import cv2
import random
import os
import copy
import json
import torch
from collections import OrderedDict
import logging

from detectron2.config import get_cfg, CfgNode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data.ehoi_dataset_mapper_v1 import EhoiDatasetMapperDepthv1
from detectron2.evaluation import COCOEvaluator, EHOIEvaluator, inference_on_dataset
from detectron2.modeling.meta_arch.mm_ehoi_net_v1 import MMEhoiNetv1
from detectron2.utils.converters import MMEhoiNetConverterv1
from detectron2.utils.logger import setup_logger

def get_evaluators(cfg, dataset_name, output_folder, converter):
    tasks = ["bbox"]
    kpt_sigmas = ()
    if cfg.MODEL.KEYPOINT_ON:
        tasks.append("keypoints")
        if hasattr(cfg.TEST, "KEYPOINT_OKS_SIGMAS") and cfg.TEST.KEYPOINT_OKS_SIGMAS:
            kpt_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
            
    coco_evaluator = COCOEvaluator(
        dataset_name, tasks=tuple(tasks), output_dir=output_folder, kpt_oks_sigmas=kpt_sigmas
    )
    ehoi_evaluator = EHOIEvaluator(cfg, dataset_name, converter)
    return [coco_evaluator, ehoi_evaluator]

def do_test(cfg, model, *, converter):
    results = OrderedDict()
    dataset_name = cfg.DATASETS.TEST[0]
    mapper = EhoiDatasetMapperDepthv1(cfg, is_train=False)
    data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    
    evaluators = get_evaluators(
        cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name), converter
    )
    results_i = inference_on_dataset(model, data_loader, evaluators)
    results[dataset_name] = results_i
        
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def setup_cfg(args):
    cfg_path = os.path.join(args.weights.rsplit("/", 1)[0], "cfg.yaml")
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(cfg_path)
    
    cfg.defrost()
    cfg.DATASETS.TEST = ("ehoi_test_dataset",)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.OUTPUT_DIR = args.weights.rsplit("/", 1)[0]
    cfg.freeze()
    
    return cfg

def main(args):
    cfg = setup_cfg(args)
    setup_logger(output=cfg.OUTPUT_DIR)
    
    # --- The Final, Correct, and Cleanest Fix ---

    # 1. Register the test dataset
    test_dataset_name = cfg.DATASETS.TEST[0]
    register_coco_instances(test_dataset_name, {}, args.dataset_json, args.dataset_images)
    
    # 2. Force-load the dataset to populate its metadata in the global catalog
    DatasetCatalog.get(test_dataset_name)
    
    # 3. Get the now-populated metadata object
    metadata = MetadataCatalog.get(test_dataset_name)
    
    # 4. Add ALL necessary custom fields to THIS single metadata object
    KEYPOINT_NAMES = [
        "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip", "index_dip", "index_tip",
        "middle_mcp", "middle_pip", "middle_dip", "middle_tip", "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
    ]
    metadata.set(keypoint_names=KEYPOINT_NAMES, keypoint_flip_map=[])
    metadata.set(coco_gt_hands=args.dataset_json.replace(".json", "_hands.json"))
    
    # 5. Initialize ALL components with this single, complete metadata object
    model = MMEhoiNetv1(cfg, metadata) 
    converter = MMEhoiNetConverterv1(cfg, metadata)
    
    logging.getLogger("detectron2").info(f"Loading weights from {cfg.MODEL.WEIGHTS}")
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    
    device = f"cuda:{args.cuda_device}"
    model.to(device)
    logging.getLogger("detectron2").info(f"Model loaded successfully on: {model.device}")
    
    results = do_test(cfg, model, converter=converter)
    
    if "ehoi" in results:
        print("\n--- EHOI Evaluation Results (AP Hand | mAP Objects | mAP Target | AP Side | AP State | mAP HOI | mAP All) ---")
        res_ehoi = results["ehoi"]
        for key in ["AP Hand", "mAP Objects", "mAP Target Objects", "AP Hand + Side", "AP Hand + State", "mAP Hand + Target Objects", "mAP All"]:
            value = res_ehoi.get(key, 'N/A')
            try:
                print(f"{float(value):.2f}\t", end="")
            except (ValueError, TypeError):
                print(f"{value}\t", end="")
        print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--dataset_json', dest='dataset_json', help='reference dataset json path', type=str, required=True)
    parser.add_argument('--dataset_images', dest='dataset_images', help='reference dataset images path', type=str, required=True)
    parser.add_argument('--weights_path', dest='weights', help='weights path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
    parser.add_argument('--cuda_device', default=0, help='CUDA device id', type=int)
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    main(args)