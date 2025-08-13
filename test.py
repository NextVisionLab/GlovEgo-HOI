# import some common libraries
import argparse
import numpy as np
import random
import os
import logging
import json
import torch
from collections import OrderedDict
from telegram_notifier import notify

# import some common detectron2 utilities
from detectron2.config import get_cfg, CfgNode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetCatalog,  build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import EhoiDatasetMapperDepthv1
from detectron2.evaluation import *
from detectron2.modeling.meta_arch.mm_ehoi_net_v1 import MMEhoiNetv1
from detectron2.utils.converters import *
from detectron2.utils.logger import setup_logger

##### ArgumentParser
parser = argparse.ArgumentParser(description='Inference script')
parser.add_argument('--dataset_json', dest='dataset_json', help='reference dataset json path', type=str, required = True)
parser.add_argument('--dataset_images', dest='dataset_images', help='reference dataset images path', type=str, required = True)
parser.add_argument('--weights_path', dest='weights', help='weights path', type=str, required = True)
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
parser.add_argument('--cuda_device', default=0, help='CUDA device id', type=int)

args = parser.parse_args()

def print_row(metric_name, value):
    print(f"| {metric_name:<28} | {value:>9.2f} |")

def get_evaluators(cfg, dataset_name, output_folder, converter):
    cocoEvaluator = COCOEvaluator(dataset_name, output_dir=output_folder, tasks = ("bbox",)) 
    return [cocoEvaluator, EHOIEvaluator(cfg, dataset_name, converter)]

def do_test(cfg, model, *, converter, mapper, data):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper = mapper(cfg, data, is_train = False))
        evaluators = get_evaluators(cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name), converter)
        results_i = inference_on_dataset(model, data_loader, evaluators)
        results[dataset_name] = results_i
    if len(results) == 1: results = list(results.values())[0]
    return results

header = f"| {'Metric':<28} | {'Score (%)':>10} |"
separator = "+" + "-"*30 + "+" + "-"*12 + "+"

@notify("Test Keypoints")
def main():
    ###SET SEED
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ###REGISTER COCO INSTANCES
    register_coco_instances("test", {}, args.dataset_json, args.dataset_images)
    dataset_test_dicts = DatasetCatalog.get("test")
    test_metadata = MetadataCatalog.get("test")
    test_metadata.set(coco_gt_hands = test_metadata.json_file.replace(".json", "_hands.json"))
    with open(test_metadata.json_file) as json_file: 
        data_anns_test_sup = json.load(json_file)

    ###LOAD CFG
    cfg_dir = os.path.join(args.weights.split("model_")[0], "cfg.yaml")
    cfg = CfgNode(CfgNode.load_yaml_with_base(cfg_dir))
    cfg.set_new_allowed(True)
    cfg.DATASETS.TEST = ("test",)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.OUTPUT_DIR = args.weights[:args.weights.rfind("/") + 1]
    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)

    log_file_path = os.path.join(cfg.OUTPUT_DIR, "log_test.txt")
    setup_logger(output=log_file_path) 
    logger = logging.getLogger("detectron2")
    
    cfg.freeze()

    ###INIT MODEL
    if "EHOINET_VERSION" not in cfg or cfg.EHOINET_VERSION == 1:
        mapper_test = EhoiDatasetMapperDepthv1
        converter = MMEhoiNetConverterv1(cfg, test_metadata)
        model = MMEhoiNetv1(cfg, test_metadata)

    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR)
    _ = checkpointer.load(cfg.MODEL.WEIGHTS)
    device = "cuda:" + str(args.cuda_device)
    model.to(device)
    print("Modello caricato:", model.device)
    
    results = do_test(cfg, model, converter = converter, mapper= mapper_test, data = data_anns_test_sup)

    ###OUTPUT
    bbox_results = results.get('bbox', {})
    ehoi_results = results.get('ehoi', {})
    
    logger.info("\n\n")
    logger.info(" EVALUATION RESULTS ".center(80, "="))
    
    logger.info("\n\n--- [ Task: Standard Object Detection (COCOEvaluator) ] ---")
    for key, value in bbox_results.items():
        logger.info(f"{key:<35} : {value:.2f}")
    
    logger.info("\n\n--- [ Task: Egocentric HOI Evaluation (EHOIEvaluator) ] ---")
    for key, value in ehoi_results.items():
        logger.info(f"{key:<35} : {value:.2f}")
    
    ap_hand = bbox_results.get('AP-hand', 0.0)
    map_objects = bbox_results.get('AP', 0.0)
    map_target_objects = ehoi_results.get('mAP Target Objects', 0.0)
    ap_hand_side = ehoi_results.get('AP Hand + Side', 0.0)
    ap_hand_state = ehoi_results.get('AP Hand + State', 0.0)
    map_all_hoi = ehoi_results.get('mAP All', 0.0)
    
    summary_string = "\n"
    summary_string += "="*80 + "\n"
    summary_string += "MMEhoiNetv1 PERFORMANCE SUMMARY\n"
    summary_string += "="*80 + "\n"
    summary_string += separator + "\n"
    summary_string += header + "\n"
    summary_string += separator + "\n"
    summary_string += f"| {'Object Detection (mAP)':<28} | {map_objects:>9.2f} |\n"
    summary_string += f"| {'Hand Detection (AP)':<28} | {ap_hand:>9.2f} |\n"
    summary_string += f"| {'Target Object (mAP)':<28} | {map_target_objects:>9.2f} |\n"
    summary_string += f"| {'Hand Side (AP)':<28} | {ap_hand_side:>9.2f} |\n"
    summary_string += f"| {'Contact State (AP)':<28} | {ap_hand_state:>9.2f} |\n"
    summary_string += f"| {f'Overall HOI (mAP All)':<28} | {map_all_hoi:>9.2f} |\n"
    summary_string += separator
    
    logger.info(summary_string)

if __name__ == "__main__":
    main()