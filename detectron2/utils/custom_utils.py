import cv2
import torch
from torch import nn
import copy
import kornia

import detectron2
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.structures import PolygonMasks, BitMasks
from detectron2.structures import ROIMasks, Instances
from typing import Dict, List, Optional, Tuple
from detectron2.layers.mask_ops import paste_masks_in_image
import logging
import numpy as np


logger = logging.getLogger(__name__)

def deep_merge(_dict1: dict, _dict2: dict) -> dict:
    dict1, dict2 = copy.deepcopy(_dict1), copy.deepcopy(_dict2)
    def _val(v1, v2):
        if isinstance(v1, dict) and isinstance(v2, dict):
            return deep_merge(v1, v2)
        elif isinstance(v1, list) and isinstance(v2, list):
            return v1 + v2
        return v2 or v1
    return {k: _val(dict1.get(k), dict2.get(k)) for k in dict1.keys() | dict2.keys()}

def same_type_conv(dict_1, dict_2):
    for k in dict_1:
        dict_1[k] = type(dict_2[k])(dict_1[k])
    if isinstance(dict_1[k], dict): same_type_conv(dict_1[k], dict_2[k])

def RMSELoss(x, y):
    eps = 1e-6
    loss = torch.sqrt(nn.functional.mse_loss(x, y) + eps)
    return loss

def expand_box(bb_tensor, max_width, max_height, ratio = 0.3):
    for bb in bb_tensor:
        x0, y0, x1, y1 = bb
        width, height = x1 - x0, y1 - y0
        bb[0] = (x0 - width * ratio) if (x0 - width * ratio) > 0 else 0
        bb[1] = (y0 - height * ratio) if (y0 - height * ratio) > 0 else 0
        bb[2] = x1 + (width * ratio) if x1 + (width * ratio) < max_width else max_width
        bb[3] = y1 + (height * ratio) if y1 + (height * ratio) < max_height else max_height
    return bb_tensor

def get_iou(a, b, type="xyxy", epsilon=1e-5):
    if type == "xyxy":
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    elif type == "xywh":
        x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2] + a[0], b[2] + b[0]), min(a[3] + a[1], b[3] + b[1])
    width, height = (x2 - x1), (y2 - y1)
    if (width<0) or (height <0): return 0.0
    area_overlap = width * height
    if type == "xyxy":
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
    elif type == "xywh":
        area_a = a[2] * a[3]
        area_b = b[2] * b[3]

    area_combined = area_a + area_b - area_overlap
    iou = area_overlap / (area_combined+epsilon)
    return iou

def calculate_center(bb, xyxy = True):
    if xyxy: return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]
    else: return [bb[0] + (bb[2]/2), bb[1] + (bb[3]/2)]

def select_hands_proposals(proposals: List[Instances], hand_label: int) -> Tuple[List[Instances], List[torch.Tensor]]:
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    hands_proposals = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes == hand_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        hands_proposals.append(proposals_per_image[fg_idxs])
    return hands_proposals

#### INPUT ARRAY OF INSTANCES [B, I] OUTPUT TENSOR [B, N, OUTPUT_SIZE_H, OUTPUT_SIZE_W]
def extract_masks_and_resize(instances_list, output_size, class_id):
    output_masks = []
    
    is_training = len(instances_list) > 0 and instances_list[0].has("gt_classes")

    for instances in instances_list:
        if is_training:
            hand_instances = instances[instances.gt_classes == class_id]
            mask_field_name = "gt_masks"
        else:
            hand_instances = instances[instances.pred_classes == class_id]
            mask_field_name = "pred_masks"

        if len(hand_instances) == 0 or not hand_instances.has(mask_field_name):
            continue

        instance_masks = getattr(hand_instances, mask_field_name)

        if isinstance(instance_masks, PolygonMasks):
            image_h, image_w = instances.image_size
            rasterized_masks = []
            for polygons_per_instance in instance_masks.polygons:
                mask = np.zeros((image_h, image_w), dtype=np.uint8)

                valid_polygons = [p for p in polygons_per_instance if isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] == 2]

                if not valid_polygons:
                    continue

                polygons_for_cv2 = [p.reshape(-1, 1, 2).astype(np.int32) for p in valid_polygons]
                
                cv2.fillPoly(mask, polygons_for_cv2, 1)
                rasterized_masks.append(torch.from_numpy(mask))
            
            if not rasterized_masks:
                continue
            
            device = instances.gt_boxes.device if instances.has("gt_boxes") else instances.pred_boxes.device
            masks_tensor = torch.stack(rasterized_masks).to(device)

        elif isinstance(instance_masks, BitMasks):
            masks_tensor = instance_masks.tensor
        elif isinstance(instance_masks, torch.Tensor):
            if instance_masks.dim() == 4 and instance_masks.shape[1] == 1:
                masks_tensor = instance_masks.squeeze(1)
            else:
                masks_tensor = instance_masks
        else:
            logger.warning(f"Unknown mask type: {type(instance_masks)}")
            continue

        if masks_tensor.numel() == 0:
            continue
            
        masks_tensor = masks_tensor.float()

        if masks_tensor.dim() == 3:
             masks_tensor = masks_tensor.unsqueeze(1)
        
        res_masks = kornia.geometry.transform.resize(masks_tensor, output_size, align_corners=False).squeeze(1)
        output_masks.append(res_masks)

    return output_masks