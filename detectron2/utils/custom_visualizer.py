from abc import abstractmethod
import numpy as np
import cv2
import copy
import torch
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.structures.masks import ROIMasks

from detectron2.utils.converters import Converter

def normalizeData(data):
    """
    Normalizes a NumPy array to the range [0, 1].
    Returns the original data if the max and min are the same to avoid division by zero.
    """
    if np.max(data) - np.min(data) == 0:
        return data
    return (data - np.min(data)) / (np.max(data) - np.min(data))

class BaseEhoiVisualizer:
    """
    An abstract base class for visualizing predictions from an EHOI (Extended Hand-Object Interaction) model.
    It defines the core structure and methods for drawing various components like bounding boxes,
    masks, and depth maps.
    """
    def __init__(self, cfg, metadata, converter: Converter, **kwargs):
        """
        Initializes the visualizer.

        Args:
            cfg (CfgNode): The configuration object.
            metadata (Metadata): The dataset metadata, containing class names and other info.
            converter (Converter): A utility to convert model outputs to a desired format (e.g., COCO).
        """
        self._thing_classes = metadata.as_dict()["thing_classes"]
        self._id_hand = self._thing_classes.index("hand") if "hand" in self._thing_classes else self._thing_classes.index("mano")
        self.cfg = cfg
        self.class_names = {i: name for i, name in enumerate(metadata.thing_classes)}
        self._input_size = (cfg.UTILS.TARGET_SHAPE_W, cfg.UTILS.TARGET_SHAPE_H)

        self._converter = converter
        self._draw_ehoi = self.cfg.UTILS.VISUALIZER.DRAW_EHOI
        self._draw_masks = self.cfg.UTILS.VISUALIZER.DRAW_MASK
        self._draw_objs = self.cfg.UTILS.VISUALIZER.DRAW_OBJS
        self._draw_depth = self.cfg.UTILS.VISUALIZER.DRAW_DEPTH and cfg.ADDITIONAL_MODULES.DEPTH_MODULE.USE_DEPTH_MODULE
        
        self.create_colors()

    def create_colors(self):
        """Creates a dictionary mapping class IDs to specific RGB colors for visualization."""
        colors = np.array([(190, 209, 18), (108, 233, 247), (255, 188, 73), (221, 149, 42), (191, 80, 61), (144, 183, 3), (14, 160, 41), (75, 229, 96), (78, 80, 183), (35, 33, 150), (103, 252, 103), (38, 116, 193), (72, 52, 153), (51, 198, 154), (191, 70, 22), (160, 14, 29), (150, 242, 101), (214, 17, 30), (11, 229, 142), (190, 234, 32)], np.uint8 )
        self._colors_classes = {k: v for k, v in enumerate(colors)}
            
    @abstractmethod
    def _draw_masks_f(self, *args, **kwargs):
        """Abstract method for drawing instance masks. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _mask_postprocess(self, results: Instances, size: tuple, mask_threshold: float = 0.5, *args, **kwargs):
        """Abstract method for post-processing masks (e.g., converting to bitmasks)."""
        pass

    def _draw_ehoi_f(self, image, predictions):
        """Draws hand-object interaction visualizations (boxes, labels, contact lines)."""
        predictions_hands, _ = self._converter.generate_predictions("", predictions)
        if not len(predictions_hands): 
            return image 
        
        annotations_active_objs = [x for x in copy.deepcopy(predictions_hands) if x["contact_state"] and x["category_id_obj"] != -1]
        for element in annotations_active_objs:
            x,y,w,h = np.array(element['bbox_obj'], int)
            cv2.rectangle(image, (x, y), ((x+w), (y+h)), (0, 255, 0), 2)

        for element in predictions_hands:
            x,y,w,h = np.array(element['bbox'], int)
            hand_side = element["hand_side"]
            hand_state = element["contact_state"]

            color = (0, 255, 0) if hand_state == 1 else (0, 0, 255)
            cv2.rectangle(image, (x, y), ((x+w), (y+h)), color, 2)
            cv2.rectangle(image, (x, y), ((x+w), y+15), (255,255,255), -1)

            side_text = f'Right Hand {round(element["score"] * 100, 2)} %' if hand_side == 1 else f'Left Hand {round(element["score"] * 100, 2)} %'
            cv2.putText(image, side_text, (x + 5, y + 11), 1,  1, (0, 0, 0), 1, cv2.LINE_AA)         

            if hand_state and element["category_id_obj"] != -1:
                obj_box = np.array(element['bbox_obj'], int)
                hand_cc = (x + w//2, y + h//2)
                point_cc = (obj_box[0] + obj_box[2]//2, obj_box[1] + obj_box[3]//2)
                cv2.line(image, hand_cc, point_cc, (0, 255, 0), 4)
                cv2.circle(image, hand_cc, 4, (0, 0, 255), -1)
                cv2.circle(image, point_cc, 4, (0, 255, 0), -1)
        return image

    def _draw_objs_f(self, image, predictions):
        """Draws bounding boxes and labels for all detected objects (non-hands)."""
        predictions_obj = predictions[predictions.pred_classes != self._id_hand]
        predictions_objs = self._converter.convert_instances_to_coco(predictions_obj, "", convert_boxes_xywh_abs=True)
        for element in predictions_objs:
            x,y,w,h = np.array(element['bbox'], int)
            class_name = self.class_names[element['category_id']] + " " + str(round(element["score"] * 100, 2)) + " %"
            cv2.rectangle(image, (x, y), ((x+w), (y+h)), (128,128,128), 1)
            cv2.rectangle(image, (x, y), ((x+w), y+15), (255,255,255), -1) 
            cv2.putText(image, class_name, (x + 5, y + 11), 1,  1, (0, 0, 0), 1, cv2.LINE_AA)
        return image

    def _draw_depth_f(self, image, outputs, **kwargs):
        """Draws the predicted depth map and concatenates it below the main image."""
        depth = cv2.resize(outputs["depth_map"].detach().to("cpu").numpy().transpose(1, 2, 0), self._input_size)
        depth = np.array(depth).astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        if "save_depth_map" in kwargs.keys() and kwargs["save_depth_map"]: 
            cv2.imwrite(kwargs["save_depth_map_path"], depth)
        image =  np.concatenate((image, depth), axis=0)
        return image

    def draw_results(self, image_, outputs, **kwargs):
        """
        Main entry point for drawing all visualizations.
        It orchestrates the alignment of data and calls the specific drawing functions.
        """
        image = cv2.resize(image_, self._input_size)
        predictions = outputs["instances"].to("cpu")
        additional_outputs = outputs["additional_outputs"].to("cpu")
        
        num_predictions = len(predictions)
        hand_indices = (predictions.pred_classes == self._id_hand).nonzero(as_tuple=True)[0]
        
        if len(hand_indices) > 0:
            num_hands_in_head = len(additional_outputs) if hasattr(additional_outputs, "__len__") else 0
            if num_hands_in_head > 0:
                 assert len(hand_indices) == num_hands_in_head, \
                    f"Mismatch: Found {len(hand_indices)} hands in predictions, but EHOI head produced {num_hands_in_head} outputs."
        
        device = predictions.pred_boxes.tensor.device
        
        if additional_outputs.has("dxdymagn_hand"):
            dxdymagn_data = additional_outputs.get("dxdymagn_hand")
            dxdymagn_full = torch.zeros((num_predictions, 3), device=device, dtype=dxdymagn_data.dtype)
            dxdymagn_full[hand_indices] = dxdymagn_data
            predictions.set("dxdymagn_hand", dxdymagn_full)

        if additional_outputs.has("contact_states"):
            contact_data = additional_outputs.get("contact_states").squeeze()
            contact_full = torch.full((num_predictions,), -1, device=device, dtype=contact_data.dtype)
            contact_full[hand_indices] = contact_data
            predictions.set("contact_states", contact_full)

        if additional_outputs.has("sides"):
            sides_data = additional_outputs.get("sides").squeeze()
            sides_full = torch.full((num_predictions,), -1, device=device, dtype=sides_data.dtype)
            sides_full[hand_indices] = sides_data
            predictions.set("sides", sides_full)
        
        confident_instances = self._converter.generate_confident_instances(predictions)

        if self._draw_masks and confident_instances.has("pred_masks"): 
            image = self._draw_masks_f(image, confident_instances, **kwargs)
        if self._draw_ehoi: 
            image = self._draw_ehoi_f(image, confident_instances)
        if self._draw_objs: 
            image = self._draw_objs_f(image, confident_instances)
        if self._draw_depth:
            image = self._draw_depth_f(image, outputs, **kwargs)

        return image

class EhoiVisualizerv1(BaseEhoiVisualizer):
    def __init__(self, cfg, metadata, converter: Converter, **kwargs):
        super().__init__(cfg, metadata, converter, **kwargs)
        self.metadata = metadata
        
        self._draw_keypoints = self.cfg.UTILS.VISUALIZER.get("DRAW_KEYPOINTS", False)
        self._draw_skeleton = self.cfg.UTILS.VISUALIZER.get("DRAW_SKELETON", False)

    def _mask_postprocess(self, results: Instances, size: tuple, mask_threshold: float = 0.5):
        results = Instances(size, **results.get_fields())
        if results.has("pred_masks"):
            if isinstance(results.pred_masks, ROIMasks): roi_masks = results.pred_masks
            else: roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
            results.pred_masks = roi_masks.to_bitmasks(results.pred_boxes, size[0], size[1], mask_threshold).tensor
        return results.pred_masks

    def _draw_masks_f(self, image, predictions, save_masks = False, save_masks_path = "./masks.png", **kwargs):
        masks = self._mask_postprocess(predictions, predictions.image_size)
        masked_img = np.zeros(image.shape)
        for idx, mask in enumerate(masks):
            id_class = predictions[idx].pred_classes.item()
            masked_img = np.where(mask[...,None], self._colors_classes[id_class], masked_img)
        if save_masks: cv2.imwrite(save_masks_path, masked_img)
        masked_img = np.where(masked_img != 0, masked_img, image)
        image = cv2.addWeighted(image, 0.4, np.asarray(masked_img, np.uint8), 0.6, 0)
        return image

    def _draw_keypoints_f(self, image, predictions):
        """
        Draws keypoints and a colored skeleton for hand instances.
        """
        hand_instances = predictions[predictions.pred_classes == self._id_hand]
        if not len(hand_instances):
            return image

        # --- THIS IS THE CORRECTED AND COMPLETE SKELETON DEFINITION ---
        skeleton_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index Finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle Finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring Finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky Finger
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm Connections
            (5, 9), (9, 13), (13, 17)
        ]

        color_map = {
            'thumb': (0, 0, 255),      # Red
            'index': (0, 255, 0),      # Green
            'middle': (255, 0, 0),     # Blue
            'ring': (0, 255, 255),     # Yellow
            'pinky': (255, 0, 255),    # Magenta
            'wrist': (255, 255, 255),  # White
            'palm': (255, 128, 0)      # Light Blue (for palm connections)
        }

        keypoint_to_part = {0: 'wrist'}
        for i in range(1, 5): keypoint_to_part[i] = 'thumb'
        for i in range(5, 9): keypoint_to_part[i] = 'index'
        for i in range(9, 13): keypoint_to_part[i] = 'middle'
        for i in range(13, 17): keypoint_to_part[i] = 'ring'
        for i in range(17, 21): keypoint_to_part[i] = 'pinky'

        for i in range(len(hand_instances)):
            keypoints = hand_instances.pred_keypoints[i].cpu().numpy()
            num_kps = len(keypoints)

            if self._draw_skeleton:
                for p1_idx, p2_idx in skeleton_connections:
                    if p1_idx >= num_kps or p2_idx >= num_kps:
                        continue
                    
                    # Determine color: if connecting two fingers, use a palm color
                    is_palm_connection = p1_idx in [5, 9, 13] and p2_idx in [9, 13, 17]
                    part_name = 'palm' if is_palm_connection else keypoint_to_part.get(p2_idx, 'wrist')
                    color = color_map.get(part_name, (255, 255, 255))
                    
                    visibility1 = keypoints[p1_idx, 2]
                    visibility2 = keypoints[p2_idx, 2]
                    
                    if visibility1 > 0 and visibility2 > 0:
                        pt1 = (int(keypoints[p1_idx, 0]), int(keypoints[p1_idx, 1]))
                        pt2 = (int(keypoints[p2_idx, 0]), int(keypoints[p2_idx, 1]))
                        cv2.line(image, pt1, pt2, color, 2, cv2.LINE_AA)

            if self._draw_keypoints:
                for kp_idx, (x, y, visibility) in enumerate(keypoints):
                    if visibility > 0:
                        part_name = keypoint_to_part.get(kp_idx, 'wrist')
                        color = color_map.get(part_name, (255, 255, 255))
                        cv2.circle(image, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
        return image

    def _draw_vector(self, image, annotations_hands):
        for element in annotations_hands:
            x,y,w,h = np.array(element['bbox'], int)
            dx, dy, magn = float(element['dx']), float(element['dy']), float(element['magnitude'])
            hand_cc = np.array([x + w//2, y + h//2])
            point_cc = np.array([(hand_cc[0] + dx * magn), (hand_cc[1] + dy * magn)])
            cv2.line(image, tuple(hand_cc.astype(int)), tuple(point_cc.astype(int)), (255, 0, 0), 4)
            cv2.circle(image, tuple(hand_cc.astype(int)), 4, (255, 0, 255), -1)
        return image
    
    def _draw_ehoi_f(self, image, predictions):
        predictions_hands, _ = self._converter.generate_predictions("", predictions)
        if not len(predictions_hands): return image
        image = self._draw_vector(image, predictions_hands)
        return super()._draw_ehoi_f(image, predictions)

    def draw_results(self, image_, outputs, **kwargs):
        """
        Overrides the base method to control the full drawing pipeline.
        This ensures all data is aligned before any drawing occurs.
        """
        image = cv2.resize(image_, self._input_size)
        predictions = outputs["instances"].to("cpu")
        additional_outputs = outputs["additional_outputs"].to("cpu")
        
        num_predictions = len(predictions)
        hand_indices = (predictions.pred_classes == self._id_hand).nonzero(as_tuple=True)[0]
        
        if len(hand_indices) > 0:
            num_hands_in_head = len(additional_outputs) if hasattr(additional_outputs, "__len__") else 0
            if num_hands_in_head > 0:
                 assert len(hand_indices) == num_hands_in_head, \
                    f"Mismatch: Found {len(hand_indices)} hands in predictions, but EHOI head produced {num_hands_in_head} outputs."

        device = predictions.pred_boxes.tensor.device
        
        if additional_outputs.has("dxdymagn_hand"):
            dxdymagn_data = additional_outputs.get("dxdymagn_hand")
            dxdymagn_full = torch.zeros((num_predictions, 3), device=device, dtype=dxdymagn_data.dtype)
            dxdymagn_full[hand_indices] = dxdymagn_data
            predictions.set("dxdymagn_hand", dxdymagn_full)

        if additional_outputs.has("contact_states"):
            contact_data = additional_outputs.get("contact_states").squeeze()
            contact_full = torch.full((num_predictions,), -1, device=device, dtype=contact_data.dtype)
            contact_full[hand_indices] = contact_data
            predictions.set("contact_states", contact_full)

        if additional_outputs.has("sides"):
            sides_data = additional_outputs.get("sides").squeeze()
            sides_full = torch.full((num_predictions,), -1, device=device, dtype=sides_data.dtype)
            sides_full[hand_indices] = sides_data
            predictions.set("sides", sides_full)
            
        if additional_outputs.has("pred_keypoints"):
            kp_data = additional_outputs.get("pred_keypoints")
            kp_full = torch.zeros((num_predictions, kp_data.shape[1], 3), device=device, dtype=kp_data.dtype)
            if len(hand_indices) > 0:
                kp_full[hand_indices] = kp_data
            predictions.set("pred_keypoints", kp_full)

        confident_instances = self._converter.generate_confident_instances(predictions)
        
        if self._draw_masks and confident_instances.has("pred_masks"): 
            image = self._draw_masks_f(image, confident_instances, **kwargs)
        if self._draw_ehoi: 
            image = self._draw_ehoi_f(image, confident_instances)
        if self._draw_objs: 
            image = self._draw_objs_f(image, confident_instances)
        
        if (self._draw_keypoints or self._draw_skeleton) and confident_instances.has("pred_keypoints"):
            image = self._draw_keypoints_f(image, confident_instances)
        
        if self._draw_depth:
            image = self._draw_depth_f(image, outputs, **kwargs)

        return image