import numpy as np
import cv2
from detectron2.structures import Instances
from detectron2.utils.converters import Converter

def draw_text_with_outline(img, text, pos, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, text_color=(255, 255, 255), thickness=2):
    x, y = pos
    cv2.putText(img, text, (x, y), font_face, font_scale, (0, 0, 0), thickness * 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font_face, font_scale, text_color, thickness, cv2.LINE_AA)
    return img

class BaseEhoiVisualizer:
    def __init__(self, cfg, metadata, converter: Converter, **kwargs):
        self._thing_classes = metadata.as_dict()["thing_classes"]
        self._id_hand = next(i for i, name in enumerate(self._thing_classes) if name in ["hand", "mano"])
        self.cfg = cfg
        self.class_names = metadata.thing_classes
        self._converter = converter
        self._draw_keypoints = self.cfg.UTILS.VISUALIZER.DRAW_KEYPOINTS
        
        self.colors = {
            "hand_contact": (0, 255, 0),
            "hand_no_contact": (0, 0, 255),
            "obj_contact": (0, 255, 0),
            "obj_no_contact": (255, 100, 0),
            "interaction_line": (0, 0, 255),
            "keypoints": (255, 255, 0),
            "vector": (255, 150, 50)
        }
        self.create_class_colors(metadata)

    def create_class_colors(self, metadata):
        self.class_colors = {}
        num_classes = len(self.class_names)
        for i in range(num_classes):
            hue = int(180 * i / num_classes)
            self.class_colors[i] = tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 200, 220]]]), cv2.COLOR_HSV2BGR)[0][0]))

    def _manual_mask_paste(self, mask, box, image_h, image_w):
        x1, y1, x2, y2 = box.astype(int)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0: return np.zeros((image_h, image_w), dtype=bool)
        mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        full_res_mask = np.zeros((image_h, image_w), dtype=bool)
        full_res_mask[y1:y2, x1:x2] = mask_resized > 0.5
        return full_res_mask

    def _draw_all_predictions(self, image, instances, predictions_hands, predictions_objs):
        overlay = image.copy()
        image_h, image_w = image.shape[:2]
        
        contacting_obj_ids = {p["id_obj"] for p in predictions_hands if p.get("contact_state") == 1 and p.get("id_obj")}

        if instances.has("pred_masks"):
            masks_small = instances.pred_masks.cpu().numpy()
            boxes_abs = instances.pred_boxes.tensor.cpu().numpy()
            class_ids = instances.pred_classes.cpu().numpy()
            for i in range(len(instances)):
                mask_full = self._manual_mask_paste(np.squeeze(masks_small[i]), boxes_abs[i], image_h, image_w)
                color = self.class_colors.get(class_ids[i], (255, 255, 255))
                overlay[mask_full] = color
        
        image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
        
        obj_id_map = {p.get('id'): p for p in predictions_objs}

        for p_obj in predictions_objs:
            x, y, w, h = [int(c) for c in p_obj['bbox']]
            is_in_contact = any(p_hand.get("id_obj") == p_obj.get("id") for p_hand in predictions_hands if p_hand.get("contact_state") == 1)
            color = self.colors["obj_contact"] if is_in_contact else self.colors["obj_no_contact"]
            label = f"{self.class_names[p_obj['category_id']]} {p_obj['score']:.1%}"
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            draw_text_with_outline(image, label, (x, y - 10), text_color=color)

        for p_hand in predictions_hands:
            x, y, w, h = [int(c) for c in p_hand['bbox']]
            is_contact = p_hand.get('contact_state') == 1
            color = self.colors["hand_contact"] if is_contact else self.colors["hand_no_contact"]
            
            side_text = "R Hand" if p_hand.get('hand_side', 1) == 1 else "L Hand"
            label = f"{side_text} {p_hand['score']:.1%}"
            if is_contact:
                label += " CONTACT"
            
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            draw_text_with_outline(image, label, (x, y - 10), text_color=color)

            if is_contact and "bbox_obj" in p_hand and p_hand["bbox_obj"]:
                obj_box = p_hand['bbox_obj']
                hand_center = (x + w // 2, y + h // 2)
                obj_center = (int(obj_box[0] + obj_box[2] / 2), int(obj_box[1] + obj_box[3] / 2))
                cv2.arrowedLine(image, hand_center, obj_center, self.colors["interaction_line"], 2, tipLength=0.3)
        
        if self._draw_keypoints and instances.has("pred_keypoints"):
            keypoints_all = instances.pred_keypoints.cpu().numpy()
            class_ids = instances.pred_classes.cpu().numpy()
            for i in range(len(instances)):
                if class_ids[i] == self._id_hand:
                    for kx, ky, v in keypoints_all[i]:
                        if v > 0.2:
                            cv2.circle(image, (int(kx), int(ky)), 3, self.colors["keypoints"], -1, cv2.LINE_AA)
        
        return image

    def draw_results(self, image_, outputs, **kwargs):
        image = image_.copy()
        
        predictions = outputs["instances"].to("cpu")
        additional_outputs = outputs.get("additional_outputs")
        if additional_outputs:
             additional_outputs = additional_outputs.to("cpu")

        confident_instances = self._converter.generate_confident_instances(predictions)
        if len(confident_instances) == 0:
            return image
        
        kwargs_for_converter = {
            "depth_map": outputs.get("depth_map"),
            **kwargs 
        }

        instances_hand_confident = confident_instances[confident_instances.pred_classes == self._id_hand]
        
        predictions_hands, _ = self._converter.generate_predictions(
            "", confident_instances, instances_hand_confident, **kwargs_for_converter
        )
        
        instances_objs_confident = confident_instances[confident_instances.pred_classes != self._id_hand]
        predictions_objs = self._converter.convert_instances_to_coco(instances_objs_confident, "", True)
        
        return self._draw_all_predictions(image, confident_instances, predictions_hands, predictions_objs)

class EhoiVisualizerv1(BaseEhoiVisualizer):
    def __init__(self, cfg, metadata, converter: Converter, **kwargs):
        super().__init__(cfg, metadata, converter, **kwargs)

    def _draw_all_predictions(self, image, instances, predictions_hands, predictions_objs):
        image = super()._draw_all_predictions(image, instances, predictions_hands, predictions_objs)
        
        for p_hand in predictions_hands:
            if all(k in p_hand for k in ['dx', 'dy', 'magnitude']):
                x, y, w, h = [int(c) for c in p_hand['bbox']]
                dx, dy, magn = p_hand['dx'], p_hand['dy'], p_hand['magnitude']
                
                hand_center = np.array([x + w / 2, y + h / 2])
                end_point = hand_center + np.array([dx, dy]) * magn
                cv2.arrowedLine(image, tuple(hand_center.astype(int)), tuple(end_point.astype(int)), self.colors["vector"], 2, tipLength=0.4)
        
        return image