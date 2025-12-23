# Common libraries
import argparse
import numpy as np
import cv2
import random
import os
import torch
import warnings
import logging
import time # Aggiunto per misurazioni temporali
import json
from tqdm import tqdm

# Detectron2 utilities
from detectron2.config import get_cfg, CfgNode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, Metadata
from detectron2.utils.logger import setup_logger

# Project-specific modules
from detectron2.modeling.meta_arch.mm_ehoi_net_v1 import MMEhoiNetv1
from detectron2.utils.custom_visualizer import EhoiVisualizerv1
from detectron2.utils.converters import MMEhoiNetConverterv1
from detectron2.data import SimpleMapper

# --- Argument Parser  ---
parser = argparse.ArgumentParser(description='Inference script with Benchmarking')
parser.add_argument('--dataset', dest='ref_dataset_json', help='reference json', default='./data/ref_enigma.json', type=str)
parser.add_argument('-w', '--weights_path', dest='weights', help='weights path', type=str, required=True)
parser.add_argument('--cfg_path', dest='cfg_path', help='cfg .yaml path', type=str)
parser.add_argument('--nms', dest='nms', help='nms', default=0.3, type=float)
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
parser.add_argument('--cuda_device', default=0, help='CUDA device id', type=int)
parser.add_argument('--images_path', type=str, help='directory/file to load images')
parser.add_argument('--video_path', type=str, help='video to process')
parser.add_argument('--save_dir', type=str, help='directory to save results', required=True)
parser.add_argument('--skip_the_first_frames', type=int, help='skip the first n frames of the video.', default=0)
parser.add_argument('--duration_of_record_sec', type=int, help='time (seconds) of the video to process', default=10000000)
parser.add_argument('--hide_depth', action='store_true', default=False)
parser.add_argument('--hide_ehois', action='store_true', default=False)
parser.add_argument('--hide_bbs', action='store_true', default=False)
parser.add_argument('--hide_masks', action='store_true', default=False)
parser.add_argument('--hide_kpts', action='store_true', default=False)
parser.add_argument('--save_masks', action='store_true', help='save masks of the image', default=False)
parser.add_argument('--save_depth_map', action='store_true', help='save depth of the image', default=False)
parser.add_argument('--thresh', help='thresh of the score', default=0.5, type=float)
# Nuovi argomenti per il benchmark
parser.add_argument('--benchmark', action='store_true', help='Run latency benchmark', default=True)
parser.add_argument('--warmup', type=int, default=10, help='Number of warmup iterations')

args = parser.parse_args()

def print_header(text: str):
    bar = "=" * 80
    print(f"\n{bar}")
    print(f"\033[1;34m{text.center(80)}\033[0m")
    print(bar)

def get_thing_classes_from_json(json_path: str):
    if not os.path.exists(json_path): return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if "categories" in data and isinstance(data["categories"], list):
            categories = sorted(data["categories"], key=lambda x: x["id"])
            return [cat["name"] for cat in categories]
    except: return None
    return None

def setup_cfg() -> CfgNode:
    cfg_path = args.cfg_path or os.path.join(os.path.dirname(args.weights), "cfg.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found at '{cfg_path}'")
    
    cfg = CfgNode(CfgNode.load_yaml_with_base(cfg_path))
    cfg.set_new_allowed(True)
    cfg.defrost() 
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms
    cfg.OUTPUT_DIR = args.save_dir
    cfg.freeze()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def process_images(model: torch.nn.Module, mapper, visualizer, device):
    if os.path.isdir(args.images_path):
        image_paths = [os.path.join(args.images_path, f) for f in sorted(os.listdir(args.images_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_paths = [args.images_path]
        
    if not image_paths:
        print("\033[1;31mError: No valid image files found.\033[0m")
        return

    # --- BENCHMARK WARMUP ---
    if args.benchmark and "cuda" in device:
        print(f"Warming up GPU with {args.warmup} iterations...")
        warmup_input = mapper(image_paths[0])
        for _ in range(args.warmup):
            _ = model([warmup_input])
        torch.cuda.synchronize()

    save_dir_images = os.path.join(args.save_dir, "images_processed")
    os.makedirs(save_dir_images, exist_ok=True)
    
    latencies = []

    for image_path in tqdm(image_paths, desc="Processing Images", ncols=100):
        model_input = mapper(image_path)
        
        # --- INFERENCE & LATENCY MEASUREMENT ---
        if "cuda" in device: torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        predictions = model([model_input])[0]
        
        if "cuda" in device: torch.cuda.synchronize()
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)
        # ----------------------------------------

        # Salvataggio risultati (come da script originale)
        original_image = cv2.imread(image_path)
        if original_image is None: continue
        base_fname = os.path.splitext(os.path.basename(image_path))[0]

        # Logica Depth/Masks/Visualizer (Ometta qui per brevità, resta uguale a quella fornita)
        vis_output = visualizer.draw_results(original_image, predictions)
        cv2.imwrite(os.path.join(save_dir_images, f"{base_fname}_annotated.png"), vis_output)

    # --- REPORT FINALE ---
    if args.benchmark and latencies:
        avg_ms = np.mean(latencies) * 1000
        std_ms = np.std(latencies) * 1000
        fps = 1 / np.mean(latencies)
        print_header("BENCHMARK RESULTS")
        print(f"Average Inference Time: {avg_ms:.2f} ms (+/- {std_ms:.2f} ms)")
        print(f"Throughput: {fps:.2f} FPS")
        print(f"Device: {device}")
        print("=" * 80)

def main():
    print_header("INITIALIZING GLOVEGO-NET")
    device = "cuda:" + str(args.cuda_device) if not args.no_cuda and torch.cuda.is_available() else "cpu"
    
    cfg = setup_cfg()
    thing_classes = get_thing_classes_from_json(args.ref_dataset_json) or cfg.DATASETS.THING_CLASSES
    
    metadata = MetadataCatalog.get("__temp_metadata")
    metadata.set(thing_classes=thing_classes)

    model = MMEhoiNetv1(cfg, metadata)
    model.to(device)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    
    converter = MMEhoiNetConverterv1(cfg, metadata)
    visualizer = EhoiVisualizerv1(cfg, metadata, converter, cuda_device=args.cuda_device)
    mapper = SimpleMapper(cfg)
    
    with torch.no_grad():
        if args.images_path:
            process_images(model, mapper, visualizer, device)
            
    MetadataCatalog.remove("__temp_metadata")

if __name__ == "__main__":
    main()