import argparse
import numpy as np
import cv2
import random
import os
import copy
import json
import torch
from collections import OrderedDict
from detectron2.config import get_cfg, CfgNode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import EhoiDatasetMapperDepthv1
from detectron2.evaluation import *
from detectron2.modeling.meta_arch.mm_ehoi_net_v1 import MMEhoiNetv1
from detectron2.utils.converters import *
from detectron2.utils.logger import setup_logger

parser = argparse.ArgumentParser(description='Inference script')
parser.add_argument('--dataset_json', dest='dataset_json', help='reference dataset json path', type=str, required=True)
parser.add_argument('--dataset_images', dest='dataset_images', help='reference dataset images path', type=str, required=True)
parser.add_argument('--weights_path', dest='weights', help='weights path', type=str, required=True)
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
parser.add_argument('--cuda_device', default=0, help='CUDA device id', type=int)
parser.add_argument('--use_keypoints', action='store_true', help='enable keypoint evaluation')
parser.add_argument('--contact_state_modality', type=str, default="mask+rgb+depth", help='contact state modality')
args = parser.parse_args()

def get_evaluators(cfg, dataset_name, output_folder, converter):
    tasks = ["bbox"]
    if args.use_keypoints:
        tasks.append("keypoints")
    cocoEvaluator = COCOEvaluator(dataset_name, output_dir=output_folder, tasks=tasks)
    return [cocoEvaluator, EHOIEvaluator(cfg, dataset_name, converter)]

def do_test(cfg, model, *, converter, mapper, data):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper(cfg, data, is_train=False))
        evaluators = get_evaluators(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name), converter)
        results_i = inference_on_dataset(model, data_loader, evaluators)
        results[dataset_name] = results_i
    if len(results) == 1: 
        results = list(results.values())[0]
    return results

if __name__ == "__main__":
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    register_coco_instances("test", {}, args.dataset_json, args.dataset_images)
    dataset_test_dicts = DatasetCatalog.get("test")
    test_metadata = MetadataCatalog.get("test")
    test_metadata.set(coco_gt_hands=test_metadata.json_file.replace(".json", "_hands.json"))
    
    with open(test_metadata.json_file) as json_file:
        data_anns_test_sup = json.load(json_file)
    
    cfg_dir = os.path.join(args.weights.split("model_")[0], "cfg.yaml")
    cfg = CfgNode(CfgNode.load_yaml_with_base(cfg_dir))
    cfg.set_new_allowed(True)
    cfg.DATASETS.TEST = ("test",)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.OUTPUT_DIR = args.weights[:args.weights.rfind("/") + 1]
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    if args.use_keypoints and hasattr(cfg.MODEL, 'ROI_KEYPOINT_HEAD'):
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = len(test_metadata.get("keypoint_names", []))
    
    if hasattr(cfg, 'CONTACT_STATE_MODALITY'):
        cfg.CONTACT_STATE_MODALITY = args.contact_state_modality
    
    cfg.freeze()
    setup_logger(output=cfg.OUTPUT_DIR)
    
    if "EHOINET_VERSION" not in cfg or cfg.EHOINET_VERSION == 1:
        mapper_test = EhoiDatasetMapperDepthv1
        converter = MMEhoiNetConverterv1(cfg, test_metadata)
        model = MMEhoiNetv1(cfg, test_metadata)
    
    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR)
    _ = checkpointer.load(cfg.MODEL.WEIGHTS)
    device = "cuda:" + str(args.cuda_device)
    model.to(device)
    print("Modello caricato:", model.device)
    
    results = do_test(cfg, model, converter=converter, mapper=mapper_test, data=data_anns_test_sup)
    print(results)
    
    metrics_output = []
    if "ehoi" in results:
        ehoi_results = results["ehoi"]
        metrics = ["AP Hand", "mAP Objects", "mAP Target Objects", "AP Hand + Side", "AP Hand + State", "mAP Hand + Target Objects", "mAP All"]
        for metric in metrics:
            metrics_output.append(str(ehoi_results.get(metric, 0.0)))
    
    if "bbox" in results:
        bbox_results = results["bbox"]
        metrics_output.extend([str(bbox_results.get("AP", 0.0)), str(bbox_results.get("AP50", 0.0)), str(bbox_results.get("AP75", 0.0))])
    
    if args.use_keypoints and "keypoints" in results:
        keypoint_results = results["keypoints"]
        metrics_output.extend([str(keypoint_results.get("AP", 0.0)), str(keypoint_results.get("AP50", 0.0)), str(keypoint_results.get("AP75", 0.0))])
    
    print('\t'.join(metrics_output))