from abc import abstractmethod
import numpy as np
import cv2
import torch
import math
from detectron2.structures.instances import Instances
from detectron2.structures.masks import ROIMasks
from detectron2.utils.converters import Converter

class BaseEhoiVisualizer:
    def __init__(self, cfg, metadata, converter: Converter, **kwargs):
        self._thing_classes = metadata.as_dict()["thing_classes"]
        self._id_hand = self._thing_classes.index("hand") if "hand" in self._thing_classes else self._thing_classes.index("mano")
        self.cfg = cfg
        self.class_names = {i: name for i, name in enumerate(metadata.thing_classes)}
        self._input_size = (cfg.UTILS.TARGET_SHAPE_W, cfg.UTILS.TARGET_SHAPE_H)
        self._diag = math.sqrt(self._input_size[0]**2 + self._input_size[1]**2)
        self._converter = converter
        self._draw_ehoi = self.cfg.UTILS.VISUALIZER.DRAW_EHOI
        self._draw_masks = self.cfg.UTILS.VISUALIZER.DRAW_MASK
        self._draw_objs = self.cfg.UTILS.VISUALIZER.DRAW_OBJS
        self._draw_depth = self.cfg.UTILS.VISUALIZER.DRAW_DEPTH and cfg.ADDITIONAL_MODULES.DEPTH_MODULE.USE_DEPTH_MODULE
        self._predict_gloves = self.cfg.ADDITIONAL_MODULES.get("PREDICT_GLOVES", True)
        self.create_colors()

    def create_colors(self):
        colors = np.array([(190, 209, 18), (108, 233, 247), (255, 188, 73), (221, 149, 42), (191, 80, 61), (144, 183, 3), (14, 160, 41), (75, 229, 96), (78, 80, 183), (35, 33, 150), (103, 252, 103), (38, 116, 193), (72, 52, 153), (51, 198, 154), (191, 70, 22), (160, 14, 29), (150, 242, 101), (214, 17, 30), (11, 229, 142), (190, 234, 32)], np.uint8)
        self._colors_classes = {k: v for k, v in enumerate(colors)}

    @abstractmethod
    def _draw_masks_f(self, *args, **kwargs):
        pass

    @abstractmethod
    def _mask_postprocess(self, results: Instances, size: tuple, mask_threshold: float = 0.5, *args, **kwargs):
        pass

    def _draw_ehoi_f(self, image, predictions):
        hand_instances = predictions[predictions.pred_classes == self._id_hand]
        if not len(hand_instances): return image
        
        predictions_hands_coco, _ = self._converter.generate_predictions("", hand_instances)
        
        annotations_active_objs = [x for x in predictions_hands_coco if x.get("contact_state") == 1 and x.get("category_id_obj", -1) != -1]
        for element in annotations_active_objs:
            if 'bbox_obj' in element and element['bbox_obj']:
                x,y,w,h = np.array(element['bbox_obj'], int)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        for i in range(len(hand_instances)):
            instance = hand_instances[i]
            x1, y1, x2, y2 = instance.pred_boxes.tensor[0].int().numpy()
            
            score = instance.scores.item()
            hand_state = instance.contact_states.item() if instance.has("contact_states") else -1
            color = (0, 255, 0) if hand_state == 1 else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            
            label_parts = ["Hand"]
            if instance.has("sides"):
                side = instance.sides.item()
                if side != -1: label_parts[0] = f'{"Right" if side == 1 else "Left"} Hand'
            if self._predict_gloves and instance.has("gloves"):
                gloves = instance.gloves.item()
                if gloves != -1: label_parts.append(f'({"Glove" if gloves == 1 else "No Glove"})')
            label_parts.append(f'{score:.1%}')
            label_text = " ".join(label_parts)
            
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - th - 5), (x1 + tw + 5, y1), (255,255,255), -1)
            cv2.putText(image, label_text, (x1 + 5, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            if i < len(predictions_hands_coco):
                coco_hand = predictions_hands_coco[i]
                if coco_hand.get("contact_state") == 1 and all(k in coco_hand for k in ['dx', 'dy', 'magnitude']):
                    dx, dy, magn = coco_hand['dx'], coco_hand['dy'], coco_hand['magnitude']
                    magn_in_pixels = magn * self._diag
                    hand_center = np.array([(x1 + x2) // 2, (y1 + y2) // 2])
                    target_point = np.array([hand_center[0] + dx * magn_in_pixels, hand_center[1] + dy * magn_in_pixels])
                    cv2.arrowedLine(image, tuple(hand_center.astype(int)), tuple(target_point.astype(int)), (255, 0, 0), 3, tipLength=0.03)
                    cv2.circle(image, tuple(hand_center.astype(int)), 5, (0, 0, 255), -1)
        return image

    def _draw_objs_f(self, image, predictions):
        predictions_obj = predictions[predictions.pred_classes != self._id_hand]
        if not len(predictions_obj): return image
        
        predictions_objs_coco = self._converter.convert_instances_to_coco(predictions_obj, "", convert_boxes_xywh_abs=True)
        for element in predictions_objs_coco:
            x,y,w,h = np.array(element['bbox'], int)
            label = f"{self.class_names[element['category_id']]} {element['score']:.1%}"
            cv2.rectangle(image, (x, y), (x+w, y+h), (128,128,128), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x, y - th - 5), (x + tw + 5, y), (255,255,255), -1) 
            cv2.putText(image, label, (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return image

    def _draw_depth_f(self, image, outputs, **kwargs):
        if "depth_map" not in outputs: return image
        depth = cv2.resize(outputs["depth_map"].detach().cpu().numpy().transpose(1, 2, 0), self._input_size)
        depth = np.array(depth).astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        return np.concatenate((image, depth), axis=0)

    def draw_results(self, image_, outputs, **kwargs):
        image = cv2.resize(image_, self._input_size)
        predictions = outputs["instances"].to("cpu")

        if len(predictions) == 0:
            if self._draw_depth: return self._draw_depth_f(image, outputs, **kwargs), predictions
            return image, predictions

        additional_outputs = outputs.get("additional_outputs")
        hand_indices = (predictions.pred_classes == self._id_hand).nonzero(as_tuple=True)[0]
        
        if len(hand_indices) > 0 and additional_outputs and len(additional_outputs) > 0:
            if len(hand_indices) == len(additional_outputs):
                device = predictions.pred_boxes.tensor.device
                num_predictions = len(predictions)
                for key in ["dxdymagn_hand", "contact_states", "sides", "gloves"]:
                    if not additional_outputs.has(key) or (key == "gloves" and not self._predict_gloves): continue
                    data = additional_outputs.get(key)
                    if key == "dxdymagn_hand":
                        full_tensor = torch.zeros((num_predictions, 3), device=device, dtype=data.dtype)
                    else:
                        data = data.squeeze()
                        if data.dim() == 0: data = data.unsqueeze(0)
                        full_tensor = torch.full((num_predictions,), -1, device=device, dtype=data.dtype)
                    full_tensor[hand_indices] = data
                    predictions.set(key, full_tensor)
        
        confident_instances = self._converter.generate_confident_instances(predictions)

        if len(confident_instances) == 0:
            if self._draw_depth: image = self._draw_depth_f(image, outputs, **kwargs)
            return image, confident_instances

        if self._draw_masks and confident_instances.has("pred_masks"): 
            image = self._draw_masks_f(image, confident_instances, **kwargs)
        if self._draw_ehoi: 
            image = self._draw_ehoi_f(image, confident_instances)
        if self._draw_objs: 
            image = self._draw_objs_f(image, confident_instances)
        if self._draw_depth:
            image = self._draw_depth_f(image, outputs, **kwargs)

        return image, confident_instances

class EhoiVisualizerv1(BaseEhoiVisualizer):
    def __init__(self, cfg, metadata, converter: Converter, **kwargs):
        super().__init__(cfg, metadata, converter, **kwargs)
        self._draw_keypoints = self.cfg.UTILS.VISUALIZER.get("DRAW_KEYPOINTS", False)
        
    def _mask_postprocess(self, results: Instances, size: tuple, mask_threshold: float = 0.5):
        if not results.has("pred_masks"): return None
        results = Instances(size, **results.get_fields())
        roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        return roi_masks.to_bitmasks(results.pred_boxes, size[0], size[1], mask_threshold).tensor

    def _draw_masks_f(self, image, predictions, **kwargs):
        masks = self._mask_postprocess(predictions, predictions.image_size)
        if masks is None: return image
        overlay = image.copy()
        for idx, mask in enumerate(masks):
            color = self._colors_classes[predictions[idx].pred_classes.item() % len(self._colors_classes)]
            overlay[mask] = color
        return cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

    def _draw_keypoints_f(self, image, predictions):
        hand_instances = predictions[predictions.pred_classes == self._id_hand]
        if not hand_instances.has("pred_keypoints"): return image

        for i in range(len(hand_instances)):
            hand_instance = hand_instances[i]
            keypoints = hand_instance.pred_keypoints[0].cpu().numpy()
            
            if self._draw_keypoints:
                for kp in keypoints:
                    if kp[2] > 0:
                        cv2.circle(image, (int(kp[0]), int(kp[1])), 3, (0, 0, 200), -1, cv2.LINE_AA)
        return image

    def draw_results(self, image_, outputs, **kwargs):
        predictions = outputs["instances"].to("cpu")
        additional_outputs = outputs.get("additional_outputs")

        hand_indices = (predictions.pred_classes == self._id_hand).nonzero(as_tuple=True)[0]
        if len(hand_indices) > 0 and additional_outputs and len(additional_outputs) > 0 and additional_outputs.has("pred_keypoints"):
             if len(additional_outputs) == len(hand_indices):
                kp_data = additional_outputs.get("pred_keypoints").to("cpu")
                kp_full = torch.zeros((len(predictions), kp_data.shape[1], 3), device=kp_data.device, dtype=kp_data.dtype)
                kp_full[hand_indices] = kp_data
                predictions.set("pred_keypoints", kp_full)
        
        outputs["instances"] = predictions
        
        image, confident_instances = super().draw_results(image_, outputs, **kwargs)
        
        if len(confident_instances) > 0:
            if self._draw_keypoints and confident_instances.has("pred_keypoints"):
                image = self._draw_keypoints_f(image, confident_instances)
        
        return image