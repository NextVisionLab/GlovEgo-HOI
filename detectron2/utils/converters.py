import numpy as np
import torch
from torchvision.ops.boxes import nms
from pycocotools.coco import COCO
import math
from abc import abstractmethod
import cv2
import pprint

from detectron2.structures import BoxMode
from detectron2.structures.instances import Instances
from detectron2.structures.masks import ROIMasks
from detectron2.utils.custom_utils import calculate_center, get_iou


def get_hoi_categories(categories, is_object=False):
    if not is_object: 
        return categories
    object_categories = [
        cat for cat in categories if 'hand' not in cat['name'].lower() and 'mano' not in cat['name'].lower()
    ]
    return object_categories

class Converter:
    def __init__(self, cfg, metadata) -> None:
        self._cfg = cfg
        self._metadata = metadata
        self._thing_classes = self._metadata.as_dict()["thing_classes"]
        self._id_hand = self._thing_classes.index("hand") if "hand" in self._thing_classes else self._thing_classes.index("mano")
        self._thresh_objs = cfg.ADDITIONAL_MODULES.THRESH_OBJS
        self._nms_thresh = cfg.ADDITIONAL_MODULES.NMS_THRESH

    def convert_instances_to_coco(self, instances, img_id, convert_boxes_xywh_abs=False):
        boxes = instances.pred_boxes.tensor.detach().clone()
        if convert_boxes_xywh_abs:
            boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()
        
        has_keypoints = instances.has("pred_keypoints")
        if has_keypoints:
            keypoints = instances.pred_keypoints.tolist()

        results = []
        for k in range(len(instances)):
            result = {
                "image_id": img_id,
                "category_id": classes[k],
                "bbox": boxes[k],
                "score": scores[k],
            }
            
            if has_keypoints:
                # Convert lists from [[x1,y1,v1], [x2,y2,v2],..] to [x1,y1,v1,x2,y2,v2,...]
                flat_keypoints = [coord for kp in keypoints[k] for coord in kp]
                result["keypoints"] = flat_keypoints
            
            results.append(result)
        return results
        
    def convert_coco_to_coco_target_object(self, coco_hands, coco_gt_all):
        tmp_dateset = {}
        tmp_dateset["images"] = coco_hands.dataset["images"]
        tmp_dateset["categories"] = get_hoi_categories(coco_hands.dataset["categories"], is_object=True)
        tmp_dateset["annotations"] = []

        all_annotations_by_id = {ann['id']: ann for ann in coco_gt_all.dataset['annotations']}
        
        for hand_ann in coco_hands.anns.values():
            if hand_ann.get("contact_state") == 1:
                object_id = hand_ann.get("id_obj")
                if object_id is not None and object_id in all_annotations_by_id:
                    object_ann = all_annotations_by_id[object_id]
                    new_target_ann = {
                        "id": hand_ann["id"],  
                        "image_id": hand_ann["image_id"],
                        "category_id": object_ann["category_id"], 
                        "area": object_ann.get("area", 0.0), 
                        "bbox": hand_ann["bbox_obj"],       
                        "iscrowd": 0
                    }
                    tmp_dateset["annotations"].append(new_target_ann)
        return tmp_dateset

    @abstractmethod
    def generate_predictions(self, image_id, confident_instances, instances_hand, **kwargs):
        pass
    
    def generate_confident_instances(self, instances):
        confident_mask = instances.scores >= self._thresh_objs
        confident_instances = instances[confident_mask]
        if len(confident_instances) == 0: return confident_instances
        keep = nms(confident_instances.pred_boxes.tensor, confident_instances.scores.float(), self._nms_thresh)
        final_instances = confident_instances[keep]
        return final_instances

class MMEhoiNetConverterv1(Converter):
    def __init__(self, cfg, metadata) -> None:
        super().__init__(cfg, metadata)
        self._diag = math.sqrt((math.pow(int(cfg.UTILS.TARGET_SHAPE_W), 2) + math.pow(int(cfg.UTILS.TARGET_SHAPE_H), 2)))
        self._scale_factor = cfg.ADDITIONAL_MODULES.ASSOCIATION_VECTOR_SCALE_FACTOR
        self.HAND_ID_FOR_EHOI_EVALUATOR = 1 

    def generate_predictions(self, image_id, confident_instances, **kwargs):
        results = []
        results_target = []
        
        obj_indices = (confident_instances.pred_classes != self._id_hand)
        hand_indices = (confident_instances.pred_classes == self._id_hand)
        
        instances_objs = confident_instances[obj_indices]
        instances_hands = confident_instances[hand_indices]
        
        object_boxes_xyxy = instances_objs.pred_boxes.tensor.cpu()
        object_categories = instances_objs.pred_classes.cpu().tolist()
        object_scores = instances_objs.scores.cpu().tolist()
        
        hand_boxes_tensor = instances_hands.pred_boxes.tensor.cpu()
        dxdymagn_tensor = instances_hands.get("dxdymagn_hand").cpu()
        contact_states_tensor = instances_hands.get("contact_states").cpu()
        scores_tensor = instances_hands.scores.cpu()
        sides_tensor = instances_hands.get("sides").cpu()
        
        has_gloves = instances_hands.has("gloves")
        if has_gloves:
            gloves_tensor = instances_hands.get("gloves").cpu()
        
        for i in range(len(instances_hands)):
            bbox_hand_xyxy = hand_boxes_tensor[i].tolist()
            bbox_hand_xywh = BoxMode.convert(np.array(bbox_hand_xyxy).reshape(1,4), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS).flatten().tolist()
            
            dxdymag_v = dxdymagn_tensor[i].numpy()
            contact_state = int(contact_states_tensor[i].item())

            element = {
                "id": len(results),
                "image_id": image_id, 
                "category_id": self._metadata.thing_classes.index("hand"), 
                "bbox": bbox_hand_xywh, 
                "score": float(scores_tensor[i].item()), 
                "hand_side": int(sides_tensor[i].item()), 
                "contact_state": contact_state, 
                "bbox_obj": [], 
                "category_id_obj": -1, 
                "dx": float(dxdymag_v[0]),
                "dy": float(dxdymag_v[1]),
                "magnitude": float(dxdymag_v[2])
            }

            if has_gloves:
                element["gloves"] = int(gloves_tensor[i].item())

            if contact_state == 1 and len(instances_objs) > 0:
                hand_cc = np.array(calculate_center(bbox_hand_xyxy))
                magn_in_pixels = dxdymag_v[2] * self._diag 
                target_point = np.array([
                    (hand_cc[0] + dxdymag_v[0] * magn_in_pixels), 
                    (hand_cc[1] + dxdymag_v[1] * magn_in_pixels)
                ])

                all_obj_centers = [calculate_center(box) for box in object_boxes_xyxy]
                distances = np.sum((all_obj_centers - target_point)**2, axis=1)
                idx_closest_obj = np.argmin(distances)

                matched_obj_box_xyxy = object_boxes_xyxy[idx_closest_obj].tolist()
                matched_obj_box_xywh = BoxMode.convert(np.array(matched_obj_box_xyxy).reshape(1,4), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS).flatten().tolist()

                element["bbox_obj"] = matched_obj_box_xywh
                element["category_id_obj"] = int(object_categories[idx_closest_obj])
                element["score_obj"] = float(object_scores[idx_closest_obj])
                
                results_target.append({
                    "id": len(results_target), "image_id": image_id, "category_id": element["category_id_obj"], 
                    "bbox": element["bbox_obj"], "score": element["score_obj"]
                })

            results.append(element)

        return results, results_target

class MMEhoiNetConverterv2(Converter):
    def __init__(self, cfg, metadata) -> None:
        super().__init__(cfg, metadata)

    def generate_predictions(self, image_id, confident_instances, instances_hand, **kwargs):
        start_id = kwargs.get("start_id", 0)
        results = []
        results_target = []
            
        for idx_hand in range(len(instances_hand)):
            instance_hand = instances_hand[idx_hand]
            bbox_hand = instance_hand.pred_boxes.tensor.cpu().numpy()[0]
            x0_hand, y0_hand, x1_hand, y1_hand = bbox_hand
            width_hand, heigth_hand = x1_hand - x0_hand, y1_hand - y0_hand
            interaction_box = instance_hand.pred_interaction_boxes.numpy()[0]
            x0_ib, y0_ib, x1_ib, y1_ib = interaction_box
            w_ib, h_ib = x1_ib - x0_ib, y1_ib - y0_ib
            target_object_box = instance_hand.pred_target_object_boxes.numpy()[0]
            x0_to, y0_to, x1_to, y1_to = target_object_box
            w_to, h_to = x1_to - x0_to, y1_to - y0_to
            target_object_cls = instance_hand.pred_target_object_cls.item()
            contact_state = instance_hand.contact_states.item()
            score = instance_hand.scores.item()
            side = instance_hand.sides.item()
            pred_mask_hand = instance_hand.pred_masks_hand
            pred_mask_target_object = instance_hand.pred_masks_target_object

            element = {
                "id": start_id + idx_hand,
                "image_id": image_id, 
                "category_id": self._id_hand, 
                "bbox": [x0_hand, y0_hand, width_hand, heigth_hand], 
                "score": score, 
                "hand_side": side, 
                "contact_state": contact_state, 
                "bbox_obj": [x0_to, y0_to, w_to, h_to], 
                "category_id_obj": target_object_cls, 
                "interaction_box": [x0_ib, y0_ib, w_ib, h_ib],
                "mask_hand": pred_mask_hand,
                "mask_target_object": pred_mask_target_object
            }

            if contact_state: 
                results_target.append({"image_id": image_id, "category_id": element["category_id_obj"], "bbox": element["bbox_obj"], "score": 100})
            results.append(element)

        return results, results_target

class MMEhoiNetConverterv1DepthMask(Converter):
    def __init__(self, cfg, metadata) -> None:
        super().__init__(cfg, metadata)
        self._diag = math.sqrt((math.pow(int(cfg.UTILS.TARGET_SHAPE_W), 2) + math.pow(int(cfg.UTILS.TARGET_SHAPE_H), 2)))
        self._scale_factor = cfg.ADDITIONAL_MODULES.ASSOCIATION_VECTOR_SCALE_FACTOR

    def _mask_postprocess(self, results: Instances, size: tuple, mask_threshold: float = 0.5):
        results = Instances(size, **results.get_fields())
        if results.has("pred_masks"):
            if isinstance(results.pred_masks, ROIMasks): roi_masks = results.pred_masks
            else: roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
            results.pred_masks = roi_masks.to_bitmasks(results.pred_boxes, size[0], size[1], mask_threshold).tensor
        return results.pred_masks

    def resize_depth(self, depth):
        depth = torch.squeeze(depth[:, :, :]).to("cpu")
        depth = cv2.resize(depth.numpy(), (self._cfg.UTILS.TARGET_SHAPE_W, self._cfg.UTILS.TARGET_SHAPE_H), interpolation=cv2.INTER_CUBIC)
        if not np.isfinite(depth).all():
            depth=np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            print("WARNING: Non-finite depth values present")
        depth_min = depth.min()
        depth_max = depth.max()
        max_val = (2**(8*1))-1
        depth = max_val * (depth - depth_min) / (depth_max - depth_min) if depth_max - depth_min > np.finfo("float").eps else np.zeros(depth.shape, dtype=depth.dtype)
        return depth

    def mean_pxl_depth(self, depth, mask):
        locs = np.where(mask)
        pixels = depth[locs]
        mean = np.mean(pixels)
        return mean

    def match_object(self, obj_dets, hand_bb, hand_dxdymag):
        object_cc_list = np.array([calculate_center(bbox) for bbox in obj_dets]) # object center list
        magn =  hand_dxdymag[2] / self._scale_factor
        hand_cc = np.array(calculate_center(hand_bb)) # hand center points
        point_cc = np.array([(hand_cc[0] + hand_dxdymag[0] * magn * self._diag), (hand_cc[1] + hand_dxdymag[1] * magn * self._diag)])
        dist = np.sum((object_cc_list - point_cc)**2,axis=1)
        dist_min = np.argmin(dist) # find the nearest 
        return dist_min

    def generate_predictions(self, image_id, confident_instances, instances_hand, **kwargs):
        start_id = kwargs.get("start_id", 0)
        results = []
        results_target = []
        depth_map = self.resize_depth(kwargs.get("depth_map"))
        objs = self.convert_instances_to_coco(confident_instances[confident_instances.pred_classes != self._id_hand], image_id)
        instances_hand = self.generate_confident_instances(instances_hand)
        masks_hand = self._mask_postprocess(confident_instances[confident_instances.pred_classes == self._id_hand], confident_instances.image_size)
        
        if len(instances_hand):
            masks_objs = self._mask_postprocess(confident_instances[confident_instances.pred_classes != self._id_hand], confident_instances.image_size)
            mean_objs_pxls = [self.mean_pxl_depth(depth_map, mask_obj) for mask_obj in masks_objs]

        for idx_hand in range(len(instances_hand)):
            instance_hand = instances_hand[idx_hand]
            bbox_hand = instance_hand.pred_boxes.tensor.cpu().numpy()[0]
            x0_hand, y0_hand, x1_hand, y1_hand = bbox_hand
            width_hand, heigth_hand = x1_hand - x0_hand, y1_hand - y0_hand
            dxdymag_v = instance_hand.dxdymagn_hand.numpy()[0]
            contact_state = instance_hand.contact_states.item()
            score = instance_hand.scores.item()
            side = instance_hand.sides.item()

            mean_pxl_hand = self.mean_pxl_depth(depth_map, masks_hand[idx_hand])
            best_match_idx, best_mean = -1, 256
            for idx_obj, mean_pxl_obj in enumerate(mean_objs_pxls):
                if np.abs(mean_pxl_hand - mean_pxl_obj) < best_mean:
                    best_mean = np.abs(mean_pxl_hand - mean_pxl_obj)
                    best_match_idx = idx_obj

            contact_state = 1 if best_mean < 20 else contact_state
            contact_state = 0 if best_mean > 100 else contact_state

            element = {
                "id": start_id + idx_hand,
                "image_id": image_id, 
                "category_id": self._id_hand, 
                "bbox": [x0_hand, y0_hand, width_hand, heigth_hand], 
                "score": score, 
                "hand_side": side, 
                "contact_state": contact_state, 
                "bbox_obj": [], 
                "category_id_obj": -1, 
                "dx":dxdymag_v[0],
                "dy": dxdymag_v[1],
                "magnitude": dxdymag_v[2] / self._scale_factor * self._diag
            }

            if contact_state and best_match_idx != -1:
                x0, y0, x1, y1 = objs[best_match_idx]["bbox"]
                width, heigth = x1 - x0, y1 - y0                
                element["bbox_obj"] = [x0, y0, width, heigth]
                element["category_id_obj"] = int(objs[best_match_idx]["category_id"])
                element["score_obj"] = objs[best_match_idx]["score"]
                results_target.append({"image_id": image_id, "category_id": element["category_id_obj"], "bbox": element["bbox_obj"], "score": element["score_obj"]})
            
            results.append(element)

        return results, results_target