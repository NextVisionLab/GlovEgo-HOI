#  Copyright (c) [Your Name/Company]. All rights reserved.
#  Licensed under the MIT License.

import argparse
import logging
import os
import random
import cv2
import torch
import numpy as np
import json
from pathlib import Path

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

# Project-specific imports
from detectron2.modeling.meta_arch import MMEhoiNetv1


# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Define the hand skeleton for visualization.
HAND_SKELETON = [
    [0, 1], [1, 2], [2, 3], [3, 4],      # Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],      # Index Finger
    [0, 9], [9, 10], [10, 11], [11, 12], # Middle Finger
    [0, 13], [13, 14], [14, 15], [15, 16],# Ring Finger
    [0, 17], [17, 18], [18, 19], [19, 20] # Pinky Finger
]

def draw_hand_keypoints(image, keypoints_tensor, confidence_threshold=0.1, color=(0, 255, 0), line_thickness=2, circle_radius=4):
    """Draws keypoints and their connections (skeleton) on an image for a single hand."""
    keypoints = keypoints_tensor.cpu().numpy()
    for p1_idx, p2_idx in HAND_SKELETON:
        if p1_idx < len(keypoints) and p2_idx < len(keypoints):
            p1, p2 = keypoints[p1_idx], keypoints[p2_idx]
            if p1[2] > confidence_threshold and p2[2] > confidence_threshold:
                cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, line_thickness)
    for x, y, score in keypoints:
        if score > confidence_threshold:
            cv2.circle(image, (int(x), int(y)), circle_radius, color, -1)

def create_comparison_visualization(image_dict, prediction, metadata):
    """Creates a side-by-side image comparing ground truth annotations and model predictions."""
    img = cv2.imread(image_dict["file_name"])
    if img is None:
        logger.error(f"Failed to read image: {image_dict['file_name']}")
        return None

    gt_canvas, pred_canvas = img.copy(), img.copy()

    gt_visualizer = Visualizer(gt_canvas, metadata, scale=1.0)
    gt_vis = gt_visualizer.draw_dataset_dict(image_dict)
    gt_image = gt_vis.get_image()
    cv2.putText(gt_image, "Ground Truth", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    pred_visualizer = Visualizer(pred_canvas, metadata, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
    instances = prediction["instances"].to("cpu")
    pred_vis = pred_visualizer.draw_instance_predictions(instances)
    pred_image = pred_vis.get_image()

    additional_outputs = prediction.get("additional_outputs")
    hand_instances = instances[instances.pred_classes == metadata.hand_id]

    if additional_outputs and len(hand_instances) > 0 and additional_outputs.has("sides"):
        for i in range(len(additional_outputs.boxes)):
            box = additional_outputs.boxes[i].numpy().astype(int)
            side = "Right" if additional_outputs.sides[i].item() == 1 else "Left"
            contact = "CONTACT" if additional_outputs.contact_states[i].item() == 1 else "NO-CONTACT"
            score = additional_outputs.scores[i].item()
            label = f"{side} Hand, {contact} ({score:.2f})"
            color = (0, 255, 0) if contact == "CONTACT" else (0, 0, 255)
            cv2.rectangle(pred_image, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(pred_image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    if hand_instances.has("pred_keypoints"):
        for kps in hand_instances.pred_keypoints:
             draw_hand_keypoints(pred_image, kps, color=(255, 0, 0))

    cv2.putText(pred_image, "Prediction", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    return np.concatenate((gt_image, pred_image), axis=1)

def find_latest_training_dir(base_dir="./output_dir"):
    """Finds the most recently modified training sub-directory."""
    all_subdirs = [d for d in Path(base_dir).iterdir() if d.is_dir()]
    if not all_subdirs: return None
    return max(all_subdirs, key=lambda d: d.stat().st_mtime)

def find_best_weights_file(training_dir: Path):
    """Finds the best weights file: model_final.pth or the latest checkpoint."""
    final_model = training_dir / "model_final.pth"
    if final_model.exists():
        logger.info(f"Found final model: {final_model}")
        return final_model
    checkpoints = sorted(training_dir.glob("model_*.pth"))
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        logger.info(f"Found latest checkpoint: {latest_checkpoint}")
        return latest_checkpoint
    return None

def main(args):
    """Main function to run the visualization script."""
    
    # --- 1. Auto-detect paths from the latest training run ---
    training_dir = find_latest_training_dir()
    if not training_dir:
        logger.error("Could not find any training directory in './output_dir/'.")
        return
    logger.info(f"Using artifacts from latest training directory: {training_dir}")
    
    config_file = training_dir / "cfg.yaml"
    weights_file = find_best_weights_file(training_dir)

    if not config_file.exists() :
        logger.error(f"Could not find 'cfg.yaml'")
        return

    if not weights_file:
        logger.error(f"Could not find model weights in {training_dir}")
        return

    # --- 2. Load Configuration ---
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.merge_from_file("./configs/Custom/custom.yaml")
    cfg.merge_from_file(config_file)
    
    cfg.MODEL.WEIGHTS = str(weights_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf_threshold
    
    # --- 3. Determine and Register Dataset ---
    val_json_path = args.input_json
    if not val_json_path:
        # Assume a standard project structure if no json is provided
        default_path = Path("data/egoism-hoi-dataset/annotations/val_coco.json")
        if default_path.exists():
            val_json_path = str(default_path)
            logger.info(f"Auto-detected validation JSON: {val_json_path}")
        else:
            logger.error("Validation JSON not found at default path. Please specify it with --input-json.")
            return

    dataset_name = "ehoi_visualizer_dataset"
    if dataset_name in DatasetCatalog.list(): DatasetCatalog.remove(dataset_name)
    
    image_root = str(Path(val_json_path).parent.parent / "images")
    register_coco_instances(dataset_name, {}, val_json_path, image_root)
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    metadata = MetadataCatalog.get(dataset_name)

    try:
        hand_class_name = "hand" if "hand" in metadata.thing_classes else "mano"
        metadata.hand_id = metadata.thing_classes.index(hand_class_name)
    except (AttributeError, ValueError):
        logger.error("Could not find 'hand' or 'mano' in dataset categories.")
        return

    if not hasattr(metadata, "keypoint_names"):
         metadata.keypoint_names = [
            "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip", "index_dip", 
            "index_tip", "middle_mcp", "middle_pip", "middle_dip", "middle_tip", "ring_mcp", "ring_pip", 
            "ring_dip", "ring_tip", "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        ]
         metadata.keypoint_connection_rules = HAND_SKELETON

    # --- 4. Build Model and Visualize ---
    predictor = DefaultPredictor(cfg)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    num_to_visualize = min(args.num_samples, len(dataset_dicts))
    logger.info(f"Selecting {num_to_visualize} random samples for visualization...")
    random_samples = random.sample(dataset_dicts, num_to_visualize)
    
    for i, image_dict in enumerate(random_samples):
        file_name = os.path.basename(image_dict["file_name"])
        logger.info(f"Processing sample {i+1}/{num_to_visualize}: {file_name}")
        
        with torch.no_grad():
            prediction = predictor.model([image_dict])[0]

        comparison_image = create_comparison_visualization(image_dict, prediction, metadata)
        if comparison_image is not None:
            output_path = os.path.join(output_dir, f"comparison_{file_name}")
            cv2.imwrite(output_path, comparison_image)

    logger.info(f"Visualizations successfully saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize model predictions against ground truth for the EHOI project."
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of random samples to visualize. (Default: 10)"
    )
    parser.add_argument(
        "--input-json", type=str, default=None, help="Path to the COCO JSON annotation file. If not set, it will try a default path."
    )
    parser.add_argument(
        "--output-dir", default="./visualization_output", help="Directory to save the visualization images."
    )
    parser.add_argument(
        "--conf-threshold", type=float, default=0.5, help="Confidence threshold for displaying predictions."
    )
    
    args = parser.parse_args()
    main(args)