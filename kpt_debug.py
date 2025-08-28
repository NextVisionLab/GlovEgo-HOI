import argparse
import json
import os

import cv2
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data.detection_utils import read_image
from detectron2.modeling.meta_arch.mm_ehoi_net_v1 import MMEhoiNetv1
from detectron2.structures import Instances
from detectron2.utils.converters import MMEhoiNetConverterv1
from detectron2.data import SimpleMapper

# Command-line arguments
parser = argparse.ArgumentParser(description='Keypoint Debug Script')
parser.add_argument('--dataset', dest='ref_dataset_json', help='reference json', default='./data/train_coco.json', type=str)
parser.add_argument('-w', '--weights_path', dest='weights', help='weights path', type=str, required = True)
parser.add_argument('--cfg_path', dest='cfg_path', help='cfg .yaml path', type=str)
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--cuda_device', default=0, help='CUDA device id', type=int)
parser.add_argument('--image_path', type=str, help='Path to the image to debug', default = "./data/test_images/camera_33.png")
args = parser.parse_args()

def setup_cfg(args, metadata):
    """Loads model configuration from the YAML file associated with the weights."""
    cfg_path = args.cfg_path or os.path.join(os.path.dirname(args.weights), "cfg.yaml")

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found. Expected at '{cfg_path}' or specify with --cfg_path.")
    
    cfg = CfgNode(CfgNode.load_yaml_with_base(cfg.yaml))
    cfg.set_new_allowed(True)

    # Override settings for inference
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a reasonable threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
    # Custom module settings
    cfg.ADDITIONAL_MODULES.NMS_THRESH = 0.3
    cfg.UTILS.VISUALIZER.THRESH_OBJS = 0.5

    cfg.freeze()
    return cfg, metadata

def load_ground_truth_keypoints(json_path, image_path):
    """
    Loads ground truth keypoints from the COCO JSON annotation file for a specific image.

    Returns:
        list: A list of dictionaries, where each dictionary contains the keypoints and
              hand side for a single hand annotation.
              Returns an empty list if no hand annotations with keypoints are found.
    """
    image_name = os.path.basename(image_path)

    with open(json_path, 'r') as f:
        data = json.load(f)

    image_id = None
    for image_info in data['images']:
        if image_info['file_name'] == image_name:
            image_id = image_info['id']
            break

    if image_id is None:
        print(f"Warning: Image {image_name} not found in annotations.")
        return []

    hand_annotations = []
    for annotation in data['annotations']:
        if (annotation['image_id'] == image_id and
                annotation['category_id'] == 19 and
                annotation['num_keypoints'] > 0):
            hand_annotations.append({
                'keypoints': np.array(annotation['keypoints']).reshape(-1, 3),
                'hand_side': annotation['hand_side'],
                'bbox': annotation['bbox']  # Store bounding box for matching
            })

    return hand_annotations

def main():
    # Load metadata
    register_coco_instances("kpt_debug", {}, args.ref_dataset_json, os.path.dirname(args.ref_dataset_json))
    load_coco_json(args.ref_dataset_json, os.path.dirname(args.ref_dataset_json), "kpt_debug")
    metadata = MetadataCatalog.get("kpt_debug")

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

    # Load image
    image = read_image(args.image_path, format="BGR")
    height, width = image.shape[:2]

    # Prepare input
    mapper = SimpleMapper(cfg)
    model_input = mapper(args.image_path)

    # Inference
    with torch.no_grad():
        outputs = model([model_input])[0]
        predictions = outputs["instances"].to("cpu")
        additional_outputs = outputs["additional_outputs"].to("cpu")

        # Find hand instances
        hand_indices = (predictions.pred_classes == metadata.thing_classes.index("hand")).nonzero(as_tuple=True)[0]
        num_hands = len(hand_indices)
        if num_hands == 0:
            print("No hands detected in the image.")
            return

        # Load ground truth keypoints
        gt_hands = load_ground_truth_keypoints(args.ref_dataset_json, args.image_path)
        num_gt_hands = len(gt_hands)

        print(f"Number of hands detected: {num_hands}")
        print(f"Number of ground truth hands: {num_gt_hands}")
        print(f"predictions.pred_classes: {predictions.pred_classes}")
        print(f"hand_indices: {hand_indices}")
        print(f"len(predictions.pred_boxes): {len(predictions.pred_boxes)}")

        # ***NUOVE STAMPE DI DEBUG***
        print(f"Type of predictions.pred_boxes: {type(predictions.pred_boxes)}")
        if isinstance(predictions.pred_boxes, torch.Tensor):
            print(f"Shape of predictions.pred_boxes: {predictions.pred_boxes.shape}")
        else:
            print(f"predictions.pred_boxes is not a torch.Tensor")


        # Sort predicted hands based on bounding box x coordinate (left to right)
        predicted_hands = []

        # Check if pred_boxes is not empty AND hand_indices is not empty
        if len(predictions.pred_boxes) > 0 and len(hand_indices) > 0:
            for i, hand_index in enumerate(hand_indices):
                try:
                    # ***PROTEZIONE TOTALE: Catturiamo *qualsiasi* errore***
                    bbox = predictions.pred_boxes[hand_index].tensor.cpu().numpy()[0]  # Extract bounding box
                    predicted_hands.append((hand_index, bbox[0]))  # Store index and x coordinate
                except Exception as e:
                    print(f"Warning: Error accessing bounding box for hand with index {hand_index}. Skipping. Error: {e}")
                    continue

            predicted_hands.sort(key=lambda x: x[1])  # Sort based on x coordinate
        else:
            print("Warning: No bounding boxes or hand indices found. Skipping hand processing.")
            return

        # Sort ground truth hands based on bounding box x coordinate (left to right)
        gt_hands.sort(key=lambda x: x['bbox'][0])

        # Iterate through sorted detected hands
        for i, (hand_index, _) in enumerate(predicted_hands):
            print(f"\n--- Hand {i+1} ---")

            # Hand side (if available)
            if "sides" in predictions.get_fields():
                hand_side = predictions.sides[hand_indices[hand_index]].item()
                side_str = "Left" if hand_side == 0 else "Right" if hand_side == 1 else "Unknown"
                print(f"  Predicted hand side: {side_str}")

            # Predicted keypoints
            if additional_outputs.has("pred_keypoints"):
                keypoints_pred = additional_outputs.get("pred_keypoints")[hand_index].cpu().numpy()  # Access keypoints based on original index
                print("  Predicted Keypoints (x, y, score):")
                for kp_idx, (x, y, score) in enumerate(keypoints_pred):
                    print(f"    Keypoint {kp_idx}: ({x:.2f}, {y:.2f}, {score:.2f})")
            else:
                print("  No keypoints predicted.")

            # Try to match with a GT annotation (simple matching by index)
            if i < num_gt_hands:
                gt_keypoints = gt_hands[i]['keypoints']
                gt_hand_side = gt_hands[i]['hand_side']

                # Correct the hand side strings
                gt_side_str = "Left" if gt_hand_side == 0 else "Right" if gt_hand_side == 1 else "Unknown"
                print(f"  Ground Truth hand side: {gt_side_str}")
                print("  Ground Truth Keypoints (x, y, visibility):")
                for kp_idx, (x, y, visibility) in enumerate(gt_keypoints):
                    print(f"    Keypoint {kp_idx}: ({x:.2f}, {y:.2f}, {visibility})")
            else:
                print("  No matching ground truth annotation.")

if __name__ == "__main__":
    main()