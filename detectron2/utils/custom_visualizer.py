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
        
        # --- MODIFICA CHIAVE: Leggiamo il flag per i guanti dalla config ---
        self._predict_gloves = self.cfg.ADDITIONAL_MODULES.get("PREDICT_GLOVES", True)
        # -----------------------------------------------------------------
        
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
        hand_instances = predictions[predictions.pred_classes == self._id_hand]
        predictions_hands_coco, _ = self._converter.generate_predictions("", hand_instances)
        
        if not len(hand_instances): 
            return image 
        
        annotations_active_objs = [x for x in copy.deepcopy(predictions_hands_coco) if x["contact_state"] and x["category_id_obj"] != -1]
        for element in annotations_active_objs:
            x,y,w,h = np.array(element['bbox_obj'], int)
            cv2.rectangle(image, (x, y), ((x+w), (y+h)), (0, 255, 0), 2)

        for i in range(len(hand_instances)):
            instance = hand_instances[i]
            
            x, y, x2, y2 = instance.pred_boxes.tensor[0].int().numpy()
            w, h = x2 - x, y2 - y
            
            score = instance.scores.item()
            hand_state = instance.contact_states.item() if instance.has("contact_states") else -1

            color = (0, 255, 0) if hand_state == 1 else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(image, (x, y), (x+w, y+15), (255,255,255), -1)

            # --- MODIFICA CHIAVE: Costruzione dinamica e controllata della label ---
            label_parts = ["Hand"] # Partiamo con "Hand"
            
            # Aggiungiamo il lato solo se presente e valido
            if instance.has("sides"):
                hand_side = instance.sides.item()
                if hand_side != -1: # Controlla che non sia il valore di default
                    side_text = "Right" if hand_side == 1 else "Left"
                    label_parts[0] = f'{side_text} Hand' # Sostituisce "Hand" con "Right/Left Hand"

            # Aggiungiamo i guanti solo se il modello li predice (flag in cfg) E se il valore è valido
            if self._predict_gloves and instance.has("gloves"):
                has_gloves = instance.gloves.item()
                if has_gloves != -1: # Controlla che non sia il valore di default
                    glove_text = "Glove" if has_gloves == 1 else "No Glove"
                    label_parts.append(f"({glove_text})")
            
            label_parts.append(f'{score:.1%}')
            label_text = " ".join(label_parts)
            # ----------------------------------------------------------------------
            
            cv2.putText(image, label_text, (x + 5, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
     
            if i < len(predictions_hands_coco):
                coco_hand = predictions_hands_coco[i]
                if coco_hand["contact_state"] and coco_hand["category_id_obj"] != -1:
                    obj_box = np.array(coco_hand['bbox_obj'], int)
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
        
        if len(hand_indices) > 0 and len(additional_outputs) > 0:
            assert len(hand_indices) == len(additional_outputs), \
                f"Mismatch: Found {len(hand_indices)} hands in predictions, but EHOI head produced {len(additional_outputs)} outputs."
        
        device = predictions.pred_boxes.tensor.device
        
        if additional_outputs.has("dxdymagn_hand"):
            dxdymagn_data = additional_outputs.get("dxdymagn_hand")
            dxdymagn_full = torch.zeros((num_predictions, 3), device=device, dtype=dxdymagn_data.dtype)
            dxdymagn_full[hand_indices] = dxdymagn_data
            predictions.set("dxdymagn_hand", dxdymagn_full)

        if additional_outputs.has("contact_states"):
            contact_data = additional_outputs.get("contact_states").squeeze()
            if contact_data.dim() == 0: contact_data = contact_data.unsqueeze(0)
            contact_full = torch.full((num_predictions,), -1, device=device, dtype=contact_data.dtype)
            contact_full[hand_indices] = contact_data
            predictions.set("contact_states", contact_full)

        if additional_outputs.has("sides"):
            sides_data = additional_outputs.get("sides").squeeze()
            if sides_data.dim() == 0: sides_data = sides_data.unsqueeze(0)
            sides_full = torch.full((num_predictions,), -1, device=device, dtype=sides_data.dtype)
            sides_full[hand_indices] = sides_data
            predictions.set("sides", sides_full)
        
        # Allineiamo i guanti solo se il modello li predice (controllato dal flag)
        if self._predict_gloves and additional_outputs.has("gloves"):
            gloves_data = additional_outputs.get("gloves").squeeze()
            if gloves_data.dim() == 0:
                gloves_data = gloves_data.unsqueeze(0)
            
            gloves_full = torch.full((num_predictions,), -1, device=device, dtype=gloves_data.dtype)
            gloves_full[hand_indices] = gloves_data
            predictions.set("gloves", gloves_full)
        
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

        self.skeleton_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        
        self.skeleton_colors = [
            (255, 0, 0), (255, 60, 0), (255, 120, 0), (255, 180, 0), 
            (0, 255, 0), (60, 255, 0), (120, 255, 0), (180, 255, 0), 
            (0, 0, 255), (0, 60, 255), (0, 120, 255), (0, 180, 255), 
            (255, 0, 255), (255, 0, 180), (255, 0, 120), (255, 0, 60), 
            (0, 255, 255), (0, 180, 255), (0, 120, 255), (0, 60, 255), 
            (255, 255, 255), (255, 255, 255), (255, 255, 255) 
        ]

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
        hand_instances = predictions[predictions.pred_classes == self._id_hand]

        if not hand_instances.has("pred_keypoints") or len(hand_instances) == 0:
            return image

        for i in range(len(hand_instances)):
            hand_instance = hand_instances[i] 

            if not hand_instance.has("pred_keypoints"):
                continue

            keypoints = hand_instance.pred_keypoints[0].cpu().numpy()
            
            if self._draw_skeleton:
                for j, connection in enumerate(self.skeleton_connections):
                    p1_idx, p2_idx = connection
                    
                    if keypoints[p1_idx, 2] > 0 and keypoints[p2_idx, 2] > 0:
                        p1 = tuple(keypoints[p1_idx, :2].astype(int))
                        p2 = tuple(keypoints[p2_idx, :2].astype(int))
                        color = self.skeleton_colors[j % len(self.skeleton_colors)]
                        cv2.line(image, p1, p2, tuple(int(c) for c in color), 2, cv2.LINE_AA)
            
            if self._draw_keypoints:
                for kp_idx in range(keypoints.shape[0]):
                    if keypoints[kp_idx, 2] > 0:
                        x, y = int(keypoints[kp_idx, 0]), int(keypoints[kp_idx, 1])
                        cv2.circle(image, (x, y), 3, (0, 0, 200), -1, cv2.LINE_AA)
        
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
        
        if len(hand_indices) > 0 and len(additional_outputs) > 0:
            assert len(hand_indices) == len(additional_outputs), \
                f"Mismatch: Found {len(hand_indices)} hands in predictions, but EHOI head produced {len(additional_outputs)} outputs."
        
        device = predictions.pred_boxes.tensor.device
        
        if additional_outputs.has("dxdymagn_hand"):
            dxdymagn_data = additional_outputs.get("dxdymagn_hand")
            dxdymagn_full = torch.zeros((num_predictions, 3), device=device, dtype=dxdymagn_data.dtype)
            dxdymagn_full[hand_indices] = dxdymagn_data
            predictions.set("dxdymagn_hand", dxdymagn_full)

        if additional_outputs.has("contact_states"):
            contact_data = additional_outputs.get("contact_states").squeeze()
            if contact_data.dim() == 0: contact_data = contact_data.unsqueeze(0)
            contact_full = torch.full((num_predictions,), -1, device=device, dtype=contact_data.dtype)
            contact_full[hand_indices] = contact_data
            predictions.set("contact_states", contact_full)

        if additional_outputs.has("sides"):
            sides_data = additional_outputs.get("sides").squeeze()
            if sides_data.dim() == 0: sides_data = sides_data.unsqueeze(0)
            sides_full = torch.full((num_predictions,), -1, device=device, dtype=sides_data.dtype)
            sides_full[hand_indices] = sides_data
            predictions.set("sides", sides_full)

        # Allineiamo i guanti solo se il modello li predice (controllato dal flag)
        if self._predict_gloves and additional_outputs.has("gloves"):
            gloves_data = additional_outputs.get("gloves").squeeze()
            if gloves_data.dim() == 0:
                gloves_data = gloves_data.unsqueeze(0)
            
            gloves_full = torch.full((num_predictions,), -1, device=device, dtype=gloves_data.dtype)
            gloves_full[hand_indices] = gloves_data
            predictions.set("gloves", gloves_full)
            
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
    
        should_draw_kpts = (self._draw_keypoints or self._draw_skeleton)
        if should_draw_kpts and confident_instances.has("pred_keypoints"):
            image = self._draw_keypoints_f(image, confident_instances)
        
        if self._draw_depth and "depth_map" in outputs:
            image = self._draw_depth_f(image, outputs, **kwargs)
        
        return image