# Common libraries
import argparse
import numpy as np
import cv2
import random
import os
import torch
import warnings
import logging
import sys
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
parser = argparse.ArgumentParser(description='Inference script')
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
parser.add_argument('--save_masks', action='store_true', help='save masks of the image (only supported for images)', default=False)
parser.add_argument('--save_depth_map', action='store_true', help='save depth of the image (only supported for images)', default=False)
parser.add_argument('--thresh', help='thresh of the score', default=0.5, type=float)

args = parser.parse_args()


def print_header(text: str):
    bar = "=" * 80
    print(f"\n{bar}")
    print(f"\033[1;34m{text.center(80)}\033[0m")
    print(bar)

def get_thing_classes_from_json(json_path: str):
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if "categories" in data and isinstance(data["categories"], list):
            categories = sorted(data["categories"], key=lambda x: x["id"])
            return [cat["name"] for cat in categories]
    except (json.JSONDecodeError, KeyError):
        return None
    return None

def setup_cfg() -> CfgNode:
    cfg_path = args.cfg_path or os.path.join(os.path.dirname(args.weights), "cfg.yaml")
    
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found. Expected at '{cfg_path}' or specify with --cfg_path.")
    
    cfg = CfgNode(CfgNode.load_yaml_with_base(cfg_path))
    cfg.set_new_allowed(True)
    cfg.defrost() 

    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms
    cfg.OUTPUT_DIR = args.save_dir
    #cfg.UTILS.VISUALIZER.DRAW_EHOI = not args.hide_ehois
    #cfg.UTILS.VISUALIZER.DRAW_MASK = not args.hide_masks
    #cfg.UTILS.VISUALIZER.DRAW_OBJS = not args.hide_bbs
    #cfg.UTILS.VISUALIZER.DRAW_DEPTH = not args.hide_depth
    #cfg.UTILS.VISUALIZER.DRAW_KEYPOINTS = not args.hide_kpts
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    
    return cfg

def process_images(model: torch.nn.Module, mapper, visualizer):
    if os.path.isdir(args.images_path):
        image_paths = [os.path.join(args.images_path, f) for f in sorted(os.listdir(args.images_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_paths = [args.images_path]
        
    if not image_paths or not any(os.path.exists(p) for p in image_paths):
        print(f"\033[1;31mError: No valid image files found in '{args.images_path}'.\033[0m")
        return
        
    save_dir_images = os.path.join(args.save_dir, "images_processed")
    os.makedirs(save_dir_images, exist_ok=True)
    
    for image_path in tqdm(image_paths, desc="Processing Images", ncols=100, bar_format="{l_bar}{bar:40}{r_bar}{bar:-40b}"):
        model_input = mapper(image_path)
        predictions = model([model_input])[0]
        original_image = cv2.imread(image_path)
        if original_image is None: 
            print(f"Warning: Could not read image {image_path}, skipping.")
            continue

        base_fname = os.path.splitext(os.path.basename(image_path))[0]

        if args.save_depth_map and "depth_map" in predictions:
            try:
                depth_map_tensor = predictions["depth_map"].squeeze().detach().cpu().numpy()
                depth_normalized = cv2.normalize(depth_map_tensor, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                
                depth_fname = f"{base_fname}_depth.png"
                cv2.imwrite(os.path.join(save_dir_images, depth_fname), depth_colormap)
            except Exception as e:
                print(f"Error saving depth map: {e}")

        if args.save_masks and "instances" in predictions:
            try:
                instances = predictions["instances"].to("cpu")
                confident_instances = visualizer._converter.generate_confident_instances(instances)
                
                if confident_instances.has("pred_masks"):
                    mask_image = np.zeros_like(original_image, dtype=np.uint8)
                    masks = visualizer._mask_postprocess(confident_instances, original_image.shape[:2]) 
                    
                    if masks is not None:
                        for idx, mask in enumerate(masks):
                            color = visualizer._colors_classes[confident_instances[idx].pred_classes.item() % len(visualizer._colors_classes)]
                            mask_image[mask] = tuple(int(c) for c in color) 
                        
                        mask_fname = f"{base_fname}_masks.png"
                        cv2.imwrite(os.path.join(save_dir_images, mask_fname), mask_image)
            except Exception as e:
                print(f"Error saving masks: {e}")
        
        vis_output = visualizer.draw_results(original_image, predictions)
        output_fname = f"{base_fname}.png" 
        cv2.imwrite(os.path.join(save_dir_images, output_fname), vis_output)
    print(f"\n\033[1;32mSuccess!\033[0m Processed images saved to: {save_dir_images}")

def process_video(model: torch.nn.Module, mapper, visualizer):
    video_capture = cv2.VideoCapture(args.video_path)
    if not video_capture.isOpened():
        print(f"\033[1;31mError: Could not open video file '{args.video_path}'.\033[0m")
        return

    width, height, fps = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    save_dir_videos = os.path.join(args.save_dir, "videos_processed")
    os.makedirs(save_dir_videos, exist_ok=True)
    output_fname = os.path.basename(args.video_path)
    video_writer = cv2.VideoWriter(os.path.join(save_dir_videos, f"processed_{output_fname}"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    max_frames = min(total_frames, (args.duration_of_record_sec * fps) + args.skip_the_first_frames)
    with tqdm(total=max_frames, desc="Processing Video", ncols=100, bar_format="{l_bar}{bar:40}{r_bar}{bar:-40b}") as pbar:
        for frame_count in range(int(max_frames)):
            has_frame, frame = video_capture.read()
            if not has_frame: break
            pbar.update(1)
            if frame_count < args.skip_the_first_frames: continue
            model_input = mapper(frame)
            predictions = model([model_input])[0]
            vis_output = visualizer.draw_results(frame, predictions)
            video_writer.write(vis_output)
    video_capture.release()
    video_writer.release()
    print(f"\n\033[1;32mSuccess!\033[0m Processed video saved to: {os.path.join(save_dir_videos, f'processed_{output_fname}')}")


def main():
    print_header("INITIALIZING MODEL AND CONFIGURATION")
    warnings.filterwarnings("ignore", category=UserWarning, module='torch.functional')
    setup_logger(name="detectron2").setLevel(logging.WARNING)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:" + str(args.cuda_device) if not args.no_cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model configuration...")
    cfg = setup_cfg()
    
    # --- METADATA HANDLING ---
    thing_classes = None
    
    thing_classes = get_thing_classes_from_json(args.ref_dataset_json)
    if thing_classes:
        print(f"Loaded {len(thing_classes)} class names from: {args.ref_dataset_json}")
        
    if thing_classes is None:
        if hasattr(cfg.DATASETS, "THING_CLASSES") and cfg.DATASETS.THING_CLASSES:
            print("Loading class names from config file (DATASETS.THING_CLASSES)...")
            thing_classes = cfg.DATASETS.THING_CLASSES
    
    metadata = MetadataCatalog.get("__temp_inference_metadata")
    metadata.set(thing_classes=thing_classes)

    cfg.defrost()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
    cfg.freeze()

    print(f"Model configured for {len(metadata.thing_classes)} classes.")

    print("Building MMEhoiNetv1 model...")
    converter = MMEhoiNetConverterv1(cfg, metadata)
    model = MMEhoiNetv1(cfg, metadata)
    model.to(device)

    print(f"Loading weights from: {cfg.MODEL.WEIGHTS}")
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    print("\033[1;32mModel loaded successfully!\033[0m")

    visualizer = EhoiVisualizerv1(cfg, metadata, converter, cuda_device=args.cuda_device)
    mapper = SimpleMapper(cfg)
    
    with torch.no_grad():
        if args.images_path:
            print_header("PROCESSING IMAGES")
            process_images(model, mapper, visualizer)
        elif args.video_path:
            print_header("PROCESSING VIDEO")
            process_video(model, mapper, visualizer)
        else:
             print("\033[1;33mWarning: No input specified. Please provide --images_path or --video_path.\033[0m")

    MetadataCatalog.remove("__temp_inference_metadata")

if __name__ == "__main__":
    main()