#!/usr/bin/env python3

import os
import json
import cv2
import torch
import numpy as np
import argparse

from detectron2.config import get_cfg
from detectron2.data.datasets import load_coco_json
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger

from detectron2.modeling.meta_arch.mm_ehoi_net_v1 import MMEhoiNetv1

# --- Funzione di Disegno per le Predizioni ---

def draw_predictions(image, annotations, metadata):
    result = image.copy()
    class_names = metadata.thing_classes
    hand_class_id = class_names.index('hand') if 'hand' in class_names else -1
    
    # Colori casuali ma consistenti per le classi
    category_colors = {i: tuple(np.random.randint(80, 255, 3).tolist()) for i in range(len(class_names))}
    
    hand_anns = [ann for ann in annotations if ann.get('category_id') == hand_class_id]
    other_anns = [ann for ann in annotations if ann.get('category_id') != hand_class_id]
    
    # Trova l'oggetto in contatto per evidenziarlo
    contacting_object_bbox = None
    contacting_hand = next((h for h in hand_anns if h.get('contact_state') == 1), None)
    if contacting_hand and other_anns:
        hand_box = contacting_hand['bbox']
        hand_center = np.array([hand_box[0] + hand_box[2]/2, hand_box[1] + hand_box[3]/2])
        obj_centers = {i: np.array([o['bbox'][0] + o['bbox'][2]/2, o['bbox'][1] + o['bbox'][3]/2]) for i, o in enumerate(other_anns)}
        if obj_centers:
            distances = {i: np.linalg.norm(hand_center - center) for i, center in obj_centers.items()}
            closest_obj_idx = min(distances, key=distances.get)
            contacting_object_bbox = other_anns[closest_obj_idx]['bbox']
            
    # Disegna oggetti
    for ann in other_anns:
        x, y, w, h = [int(c) for c in ann['bbox']]
        cat_id = ann.get('category_id')
        is_contacted = contacting_object_bbox is not None and ann['bbox'] == contacting_object_bbox
        
        color = (0, 255, 255) if is_contacted else category_colors.get(cat_id, (200, 200, 200))
        thickness = 3 if is_contacted else 2
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        label = f"{class_names[cat_id]}|{ann.get('score', 0):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result, (x, y - th - 10), (x + tw + 5, y), color, -1)
        cv2.putText(result, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # Disegna mani
    for ann in hand_anns:
        x, y, w, h = [int(c) for c in ann['bbox']]
        contact = ann.get('contact_state', 0) == 1
        
        color = (0, 255, 0) if contact else (0, 0, 255)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        side = "R" if ann.get('hand_side') == 1 else "L"
        glove = "Glove" if ann.get('gloves') == 1 else "No Glove"
        score = f"|{ann.get('score', 1.0):.2f}"
        contact_text = "|Contact" if contact else ""
        label = f"{side} Hand ({glove}){contact_text}{score}"
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(result, (x, y - th - 10), (x + tw + 5, y), color, -1)
        cv2.putText(result, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        kpts_data = ann.get('keypoints')
        if kpts_data is not None and len(kpts_data) > 0:
            kpts = np.array(kpts_data).reshape(-1, 3)
            for kx, ky, v in kpts:
                if v > 0.5: cv2.circle(result, (int(kx), int(ky)), 4, (255, 0, 255), -1)

        if contact and ann.get('dxdymagn_hand') is not None:
            dx, dy, magn = ann['dxdymagn_hand']
            if magn > 0.01:
                hand_center = (x + w // 2, y + h // 2)
                arrow_length = magn * 500
                end_point = (int(hand_center[0] + dx * arrow_length), int(hand_center[1] + dy * arrow_length))
                cv2.arrowedLine(result, hand_center, end_point, (255, 0, 0), 2, tipLength=0.3)
    return result

class EHOIPredictionVisualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def setup_model(self, config_file, weights_path, metadata):
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        cfg.merge_from_file(config_file)
        cfg.defrost()
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(metadata.thing_classes)
        cfg.MODEL.META_ARCHITECTURE = "MMEhoiNetv1"
        cfg.MODEL.KEYPOINT_ON = True
        cfg.freeze()
        
        model = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)(cfg, metadata)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(weights_path)
        model.eval()
        self.hand_class_id = metadata.thing_classes.index('hand') if 'hand' in metadata.thing_classes else -1
        return model, cfg

    def create_visualization(self, model, data_dict, metadata):
        image = cv2.imread(data_dict['file_name'])
        if image is None: return None

        with torch.no_grad():
            predictions = model([data_dict])[0]
        
        pred_anns = self._convert_pred_to_anns(predictions["instances"], predictions.get("additional_outputs"))
        pred_image_vis = draw_predictions(image.copy(), pred_anns, metadata)
        
        return pred_image_vis

    def _convert_pred_to_anns(self, instances, additional_outputs):
        if not instances: return []
        annotations = []
        pred_boxes, pred_scores, pred_classes = instances.pred_boxes.tensor.cpu(), instances.scores.cpu(), instances.pred_classes.cpu()
        pred_keypoints = instances.pred_keypoints.cpu() if instances.has("pred_keypoints") else None

        non_hand_indices = torch.where(pred_classes != self.hand_class_id)[0]
        for i in non_hand_indices:
            x1, y1, x2, y2 = pred_boxes[i]
            annotations.append({'id': i.item(), 'category_id': pred_classes[i].item(), 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': pred_scores[i].item()})

        if additional_outputs and len(additional_outputs) > 0:
            hand_indices_in_main_preds = torch.where(pred_classes == self.hand_class_id)[0]
            for i in range(len(additional_outputs)):
                x1, y1, x2, y2 = additional_outputs.boxes[i]
                kpts = pred_keypoints[hand_indices_in_main_preds[i]] if pred_keypoints is not None and len(hand_indices_in_main_preds) > i else []
                annotations.append({
                    'category_id': self.hand_class_id, 'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': additional_outputs.scores[i].item(), 'hand_side': additional_outputs.sides[i].item(),
                    'gloves': additional_outputs.gloves[i].item(), 'contact_state': additional_outputs.contact_states[i].item(),
                    'keypoints': kpts, 'dxdymagn_hand': additional_outputs.dxdymagn_hand[i].cpu().numpy()
                })
        return annotations

def main():
    parser = argparse.ArgumentParser(description='EHOI Prediction Visualization')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--weights', required=True, help='Path to model weights')
    parser.add_argument('--val_json', required=True, help='Validation COCO JSON file')
    parser.add_argument('--output_dir', default='./predictions_vis', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()
    
    setup_logger()
    
    val_json_path = args.val_json
    images_root = os.path.join(os.path.dirname(os.path.dirname(val_json_path)), "images")
    val_dataset_name = "vis_dataset_preds"
    
    DatasetCatalog.register(val_dataset_name, lambda: load_coco_json(val_json_path, images_root, dataset_name=val_dataset_name))
    
    with open(val_json_path, 'r') as f: coco_data = json.load(f)
    categories = sorted(coco_data.get('categories', []), key=lambda x: x['id'])
    thing_classes = [cat['name'] for cat in categories]
    metadata = MetadataCatalog.get(val_dataset_name); metadata.set(thing_classes=thing_classes)
    
    visualizer = EHOIPredictionVisualizer(args.output_dir)
    print("Loading model and configuration...")
    model, cfg = visualizer.setup_model(args.config, args.weights, metadata)
    cfg.defrost(); cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold; cfg.freeze()
    model.to("cuda")
    
    dataset_dicts = DatasetCatalog.get(val_dataset_name)
    
    print(f"Creating {args.num_samples} visualizations...")
    for i in range(min(args.num_samples, len(dataset_dicts))):
        data_dict = dataset_dicts[i]
        
        # Il mapper di D2 non è necessario, passiamo il dizionario con il percorso del file
        # che il modello stesso caricherà.
        vis_image = visualizer.create_visualization(model, [data_dict], metadata)
        
        if vis_image is not None:
            save_path = os.path.join(args.output_dir, os.path.basename(data_dict['file_name']))
            cv2.imwrite(save_path, vis_image)
            print(f"Saved: {save_path}")
    
    print(f"\nVisualization complete! Check {args.output_dir} for results.")

if __name__ == "__main__":
    main()