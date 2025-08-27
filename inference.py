# Common libraries
import argparse
import numpy as np
import cv2
import random
import os
import torch
import sys
from tqdm import tqdm

# Detectron2 utilities
from detectron2.config import get_cfg, CfgNode
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data.detection_utils import read_image

# Project-specific modules
from detectron2.modeling.meta_arch.mm_ehoi_net_v1 import MMEhoiNetv1
from detectron2.utils.custom_visualizer import EhoiVisualizerv1
from detectron2.utils.converters import MMEhoiNetConverterv1
from detectron2.data import SimpleMapper

# Command-line arguments
parser = argparse.ArgumentParser(description='Inference script')
parser.add_argument('--dataset', dest='ref_dataset_json', help='reference json', default='./data/ref_enigma.json', type=str)
parser.add_argument('-w', '--weights_path', dest='weights', help='weights path', type=str, required = True)
parser.add_argument('--cfg_path', dest='cfg_path', help='cfg .yaml path', type=str)
parser.add_argument('--nms', dest='nms', help='nms', default = 0.3, type=float)
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
parser.add_argument('--cuda_device', default=0, help='CUDA device id', type=int)
parser.add_argument('--images_path', type=str, help='directory/file to load images')
parser.add_argument('--video_path', type=str, help='video to process')
parser.add_argument('--save_dir', type=str, help='directory to save results', default = "./output_dir_kpts_gloves/inference/")
parser.add_argument('--skip_the_fist_frames', type=int, help='skip the first n frames of the video.', default = 0)
parser.add_argument('--duration_of_record_sec', type=int, help='time (seconds) of the video to process', default = 10000000)
parser.add_argument('--hide_depth', action='store_true', default=False)
parser.add_argument('--hide_ehois', action='store_true', default=False)
parser.add_argument('--hide_bbs', action='store_true', default=False)
parser.add_argument('--hide_masks', action='store_true', default=False)
parser.add_argument('--hide_kpts', action='store_true', default=False)
parser.add_argument('--hide_skeleton', action='store_true', default=False)
parser.add_argument('--save_masks', action='store_true', help='save masks of the image (only supported for images)', default=False)
parser.add_argument('--save_depth_map', action='store_true', help='save depth of the image (only supported for images)', default=False)
parser.add_argument('--thresh', help='thresh of the score', default=0.5, type = float)

args = parser.parse_args()


def format_times(times_dict):
    """Formats the inference time dictionary for printing."""
    if not times_dict:
        return ""
    return "\n".join([f"\t{k}: {v} ms" for k, v in times_dict.items()])

def setup_cfg(args, metadata):
    """Loads model configuration from the YAML file associated with the weights."""
    cfg_path = args.cfg_path or os.path.join(os.path.dirname(args.weights), "cfg.yaml")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found. Expected at '{cfg_path}' or specify with --cfg_path.")
    
    cfg = CfgNode(CfgNode.load_yaml_with_base(cfg_path))
    cfg.set_new_allowed(True)

    # Override settings for inference
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = args.nms
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
    cfg.OUTPUT_DIR = args.save_dir
    
    # Custom module settings
    cfg.ADDITIONAL_MODULES.NMS_THRESH = args.nms
    cfg.UTILS.VISUALIZER.THRESH_OBJS = args.thresh
    cfg.UTILS.VISUALIZER.DRAW_EHOI = not args.hide_ehois
    cfg.UTILS.VISUALIZER.DRAW_MASK = not args.hide_masks
    cfg.UTILS.VISUALIZER.DRAW_OBJS = not args.hide_bbs
    cfg.UTILS.VISUALIZER.DRAW_DEPTH = not args.hide_depth
    cfg.UTILS.VISUALIZER.DRAW_KEYPOINTS = not args.hide_kpts
    cfg.UTILS.VISUALIZER.DRAW_SKELETON = args.hide_skeleton

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    return cfg, metadata

def process_images(args, model, mapper, visualizer):
    """Processes a single image or a directory of images."""
    image_paths = []
    if os.path.isdir(args.images_path):
        image_paths = [os.path.join(args.images_path, f) for f in sorted(os.listdir(args.images_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        image_paths.append(args.images_path)

    save_dir_images = os.path.join(args.save_dir, "images_processed")
    os.makedirs(save_dir_images, exist_ok=True)

    for image_path in tqdm(image_paths, desc="Processing Images"):
        model_input = mapper(image_path)
        predictions = model([model_input])[0]
        
        original_image = cv2.imread(image_path)
        vis_output = visualizer.draw_results(original_image, predictions)
        
        output_fname = os.path.basename(image_path)
        cv2.imwrite(os.path.join(save_dir_images, output_fname), vis_output)
    print(f'Inferred images saved in {save_dir_images}.')

def process_video(args, model, mapper, visualizer):
    """Processes a video file frame by frame."""
    video_capture = cv2.VideoCapture(args.video_path)
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    save_dir_videos = os.path.join(args.save_dir, "videos_processed")
    os.makedirs(save_dir_videos, exist_ok=True)
    output_fname = os.path.basename(args.video_path)
    video_writer = cv2.VideoWriter(os.path.join(save_dir_videos, f"processed_{output_fname}"), 
                                   cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    max_frames = args.duration_of_record_sec * fps + args.skip_the_fist_frames
    
    with tqdm(total=min(total_frames, max_frames), desc="Processing Video") as pbar:
        while True:
            has_frame, frame = video_capture.read()
            if not has_frame or frame_count >= max_frames:
                break
            
            frame_count += 1
            if frame_count <= args.skip_the_fist_frames:
                pbar.update(1)
                continue
            
            model_input = mapper(frame)
            predictions = model([model_input])[0]
            
            vis_output = visualizer.draw_results(frame, predictions)
            video_writer.write(vis_output)
            pbar.update(1)
            
    video_capture.release()
    video_writer.release()

def main():
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load metadata
    register_coco_instances("inference_ref", {}, args.ref_dataset_json, os.path.dirname(args.ref_dataset_json))
    load_coco_json(args.ref_dataset_json, os.path.dirname(args.ref_dataset_json), "inference_ref")
    metadata = MetadataCatalog.get("inference_ref")

    # Setup configuration and model
    cfg, metadata = setup_cfg(args, metadata)
    converter = MMEhoiNetConverterv1(cfg, metadata)
    model = MMEhoiNetv1(cfg, metadata)

    # Load weights
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    device = "cuda:" + str(args.cuda_device) if not args.no_cuda and torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print("Model loaded:", model.device)

    # Initialize visualizer and mapper
    visualizer = EhoiVisualizerv1(cfg, metadata, converter, cuda_device=args.cuda_device)
    mapper = SimpleMapper(cfg)

    with torch.no_grad():
        if args.images_path:
            process_images(args, model, mapper, visualizer)
        elif args.video_path:
            process_video(args, model, mapper, visualizer)

    print("\nDone.")

if __name__ == "__main__":
    main()