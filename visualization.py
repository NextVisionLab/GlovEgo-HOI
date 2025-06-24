#!/usr/bin/env python3

import os
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import argparse

from detectron2.config import get_cfg, CfgNode as CN
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.ehoi_dataset_mapper_v1 import EhoiDatasetMapperDepthv1
from detectron2.modeling.meta_arch import MMEhoiNetv1
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger

class EHOIVisualizer:
    
    def __init__(self, output_dir="./visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.colors = {
            'gt_hand': (0, 255, 0),
            'pred_hand': (0, 0, 255),
            'gt_keypoints': (255, 255, 0),
            'pred_keypoints': (255, 0, 255),
            'gt_contact': (255, 0, 0),
            'pred_contact': (255, 165, 0),
            'gt_mask': (0, 255, 255),
            'pred_mask': (255, 255, 255),
        }
    
    def setup_model(self, weights_path, val_json):
        with open(val_json, 'r') as f:
            coco_data = json.load(f)
        
        val_images_path = os.path.dirname(val_json) + "/../images/"
        register_coco_instances("vis_dataset", {}, val_json, val_images_path)
        
        keypoint_names = [
            "wrist", "thumb_mcp", "thumb_pip", "thumb_dip", "thumb_tip",
            "index_mcp", "index_pip", "index_dip", "index_tip", 
            "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
            "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
            "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"
        ]
        
        metadata = MetadataCatalog.get("vis_dataset")
        if 'categories' in coco_data:
            categories = sorted(coco_data['categories'], key=lambda x: x['id'])
            thing_classes = [cat['name'] for cat in categories]
            metadata.set(
                thing_classes=thing_classes,
                keypoint_names=keypoint_names,
                keypoint_flip_map=[],
                keypoint_connection_rules=[]
            )
        
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        cfg.merge_from_file("./configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        cfg.merge_from_file("./configs/Custom/custom.yaml")
        
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes) if 'categories' in coco_data else 20
        cfg.ADDITIONAL_MODULES.USE_MASK_GT = True
        cfg.ADDITIONAL_MODULES.USE_MASK = True
        cfg.ADDITIONAL_MODULES.DEPTH_MODULE.USE_DEPTH_MODULE = True
        cfg.ADDITIONAL_MODULES.CONTACT_STATE_MODALITY = "mask+rgb+depth+fusion"
        cfg.ADDITIONAL_MODULES.CONTACT_STATE_CNN_INPUT_SIZE = 128
        cfg.ADDITIONAL_MODULES.NORMALIZE_KEYPOINT_COORDS = True
        
        cfg.MODEL.KEYPOINT_ON = True
        cfg.MODEL.ROI_KEYPOINT_HEAD = CN()
        cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 21
        cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0
        cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = (512, 512, 512, 512)
        cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True
        cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
        cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
        cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = "ROIAlignV2"
        cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
        cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
        
        cfg.MODEL.WEIGHTS = weights_path
        cfg.freeze()
        
        model = MMEhoiNetv1(cfg, metadata)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(weights_path)
        model.eval()
        
        self.hand_class_id = self._find_hand_class_id(thing_classes)
        print(f"Hand class ID: {self.hand_class_id}")
        
        self.coco_data = coco_data
        
        return model, cfg, metadata, coco_data
    
    def _find_hand_class_id(self, thing_classes):
        for i, class_name in enumerate(thing_classes):
            if 'hand' in class_name.lower():
                return i
        return len(thing_classes) - 1
    
    def _get_gt_annotations_from_coco(self, image_id):
        if not hasattr(self, 'coco_data') or 'annotations' not in self.coco_data:
            return []
        
        gt_annotations = []
        for ann in self.coco_data['annotations']:
            if ann['image_id'] == image_id:
                gt_annotations.append(ann)
        
        return gt_annotations
    
    def visualize_sample(self, model, batch, save_path):
        image_dict = batch[0]
        image = image_dict["image"].permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        image_id = image_dict.get("image_id", None)
        gt_inst = batch[0]["instances"] if "instances" in batch[0] else None
        
        gt_annotations_coco = []
        if image_id is not None:
            gt_annotations_coco = self._get_gt_annotations_from_coco(image_id.item() if torch.is_tensor(image_id) else image_id)
        
        with torch.no_grad():
            predictions = model(batch)
        
        pred_instances = predictions[0]["instances"]
        additional_outputs = predictions[0].get("additional_outputs", None)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        ax1.imshow(image)
        ax1.set_title('Ground Truth', fontsize=16, fontweight='bold')
        ax1.axis('off')
        
        gt_drawn = False
        if gt_inst is not None and len(gt_inst) > 0:
            try:
                self._draw_annotations(ax1, gt_inst, None, is_gt=True)
                gt_drawn = True
            except Exception as e:
                pass
        
        if not gt_drawn and gt_annotations_coco:
            try:
                self._draw_coco_annotations(ax1, gt_annotations_coco, image.shape[:2])
                gt_drawn = True
            except Exception as e:
                pass
        
        if not gt_drawn:
            ax1.text(0.5, 0.5, 'No GT annotations found', 
                    transform=ax1.transAxes, ha='center', va='center',
                    fontsize=14, color='red', weight='bold')
        
        ax2.imshow(image)
        ax2.set_title('Predictions', fontsize=16, fontweight='bold')
        ax2.axis('off')
        self._draw_annotations(ax2, pred_instances, additional_outputs, is_gt=False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _draw_coco_annotations(self, ax, annotations, image_shape):
        for ann in annotations:
            if ann.get('category_id', -1) != self.hand_class_id:
                continue
                
            bbox = ann.get('bbox', [])  
            if len(bbox) < 4:
                continue
                
            x, y, w, h = bbox
            
            color_mpl = tuple(c/255.0 for c in self.colors['gt_hand'])
            rect = patches.Rectangle((x, y), w, h, 
                                   linewidth=3, edgecolor=color_mpl, facecolor='none')
            ax.add_patch(rect)
            
            if 'keypoints' in ann and ann['keypoints']:
                kpts = np.array(ann['keypoints'])
                if len(kpts) >= 63:
                    kpts = kpts.reshape(-1, 3)
                    color_kpts = tuple(c/255.0 for c in self.colors['gt_keypoints'])
                    
                    for i, (kx, ky, vis) in enumerate(kpts):
                        if vis > 0:
                            circle = Circle((kx, ky), radius=5, color=color_kpts, alpha=0.9, linewidth=2)
                            ax.add_patch(circle)
            
            if ann.get('contact_state', 0) == 1:
                if 'dx' in ann and 'dy' in ann and 'magnitude' in ann:
                    dx, dy, magnitude = ann['dx'], ann['dy'], ann['magnitude']
                    center_x = x + w/2
                    center_y = y + h/2
                    
                    color_contact = tuple(c/255.0 for c in self.colors['gt_contact'])
                    scale = 150
                    ax.arrow(center_x, center_y, dx*scale, dy*scale, 
                            head_width=20, head_length=20, fc=color_contact, ec=color_contact, 
                            alpha=0.9, linewidth=4)
    
    def _draw_annotations(self, ax, instances, additional_outputs, is_gt=True):
        try:
            if is_gt:
                boxes = getattr(instances, 'gt_boxes', None)
                classes = getattr(instances, 'gt_classes', None)
                keypoints = getattr(instances, 'gt_keypoints', None)
                contact_states = getattr(instances, 'gt_contact_states', None)
                vectors = getattr(instances, 'gt_dxdymagn_hands', None)
                color_box = self.colors['gt_hand']
                color_kpts = self.colors['gt_keypoints']
                color_contact = self.colors['gt_contact']
            else:
                boxes = getattr(instances, 'pred_boxes', None)
                classes = getattr(instances, 'pred_classes', None)
                keypoints = getattr(instances, 'pred_keypoints', None)
                contact_states = None
                vectors = None
                
                if additional_outputs:
                    contact_states = getattr(additional_outputs, 'contact_states', None)
                    vectors = getattr(additional_outputs, 'dxdymagn_hand', None)
                
                color_box = self.colors['pred_hand']
                color_kpts = self.colors['pred_keypoints']
                color_contact = self.colors['pred_contact']
            
            if boxes is None or classes is None:
                return
            
            if torch.is_tensor(classes):
                classes_np = classes.cpu().numpy()
            else:
                classes_np = np.array(classes)
            
            hand_indices = np.where(classes_np == self.hand_class_id)[0]
            
            if len(hand_indices) == 0:
                return
            
            if hasattr(boxes, 'tensor'):
                boxes_np = boxes.tensor.cpu().numpy()
            else:
                boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else np.array(boxes)
            
            color_box_mpl = tuple(c/255.0 for c in color_box)
            for i in hand_indices:
                x1, y1, x2, y2 = boxes_np[i]
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=3, edgecolor=color_box_mpl, facecolor='none')
                ax.add_patch(rect)
            
            if keypoints is not None:
                color_kpts_mpl = tuple(c/255.0 for c in color_kpts)
                
                if torch.is_tensor(keypoints):
                    keypoints_np = keypoints.cpu().numpy()
                elif hasattr(keypoints, 'tensor'):
                    keypoints_np = keypoints.tensor.cpu().numpy()
                else:
                    keypoints_np = np.array(keypoints)
                
                for i in hand_indices:
                    if i < len(keypoints_np):
                        kpts_np = keypoints_np[i]
                        
                        if len(kpts_np.shape) == 1 and len(kpts_np) == 63:
                            kpts_np = kpts_np.reshape(-1, 3)
                        elif len(kpts_np.shape) == 2 and kpts_np.shape[1] == 3:
                            pass
                        else:
                            continue
                        
                        for x, y, vis in kpts_np:
                            if vis > 0.5:
                                circle = Circle((x, y), radius=5, color=color_kpts_mpl, alpha=0.9, linewidth=2)
                                ax.add_patch(circle)
            
            if contact_states is not None and vectors is not None:
                if torch.is_tensor(contact_states):
                    contact_states_np = contact_states.cpu().numpy()
                else:
                    contact_states_np = np.array(contact_states)
                
                if torch.is_tensor(vectors):
                    vectors_np = vectors.cpu().numpy()
                elif hasattr(vectors, 'cpu'):
                    vectors_np = vectors.cpu().numpy()
                else:
                    vectors_np = np.array(vectors)
                
                color_contact_mpl = tuple(c/255.0 for c in color_contact)
                
                for i in hand_indices:
                    if i < len(contact_states_np) and i < len(vectors_np):
                        if contact_states_np[i] == 1:
                            x1, y1, x2, y2 = boxes_np[i]
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            
                            if len(vectors_np[i]) >= 3:
                                dx, dy, magnitude = vectors_np[i][:3]
                                
                                scale = 150
                                ax.arrow(center_x, center_y, dx*scale, dy*scale, 
                                        head_width=20, head_length=20, fc=color_contact_mpl, ec=color_contact_mpl, 
                                        alpha=0.9, linewidth=4)
            
        except Exception as e:
            pass


def main():
    parser = argparse.ArgumentParser(description='EHOI Visualization')
    parser.add_argument('--weights', required=True, help='Path to model weights')
    parser.add_argument('--val_json', required=True, help='Validation JSON file')
    parser.add_argument('--output_dir', default='./visualizations', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index for samples')
    
    args = parser.parse_args()
    
    setup_logger()
    torch.cuda.set_device(args.device)
    
    visualizer = EHOIVisualizer(args.output_dir)
    
    print("Loading model...")
    model, cfg, metadata, coco_data = visualizer.setup_model(args.weights, args.val_json)
    model = model.to(args.device)
    
    mapper = EhoiDatasetMapperDepthv1(cfg, data_anns_sup=coco_data, is_train=True)
    data_loader = build_detection_test_loader(cfg, "vis_dataset", mapper=mapper)
    
    print(f"Creating {args.num_samples} visualizations starting from index {args.start_idx}...")
    
    processed = 0
    for idx, batch in enumerate(data_loader):
        if idx < args.start_idx:
            continue
        if processed >= args.num_samples:
            break
                
        save_path = os.path.join(args.output_dir, f"sample_{idx:03d}.png")
        visualizer.visualize_sample(model, batch, save_path)
        
        print(f"Saved: {save_path}")
        processed += 1
    
    print(f"\nVisualization complete! Check {args.output_dir} for results.")


if __name__ == "__main__":
    main()