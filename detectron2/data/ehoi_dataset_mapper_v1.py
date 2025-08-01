import copy
import os
import numpy as np
import torch
import math
import cv2
import logging
import re

from . import detection_utils as utils
from . import transforms as T
from . import BaseEhoiDatasetMapper

from torchvision.transforms import Compose
from detectron2.modeling.meta_arch.MiDaS.midas.transforms import Resize, PrepareForNet
from detectron2.modeling.meta_arch.MiDaS import utils as midas_utils

logger = logging.getLogger(__name__)

class EhoiDatasetMapperv1(BaseEhoiDatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 format and maps it into
    a format suitable for training the EHOI model.

    This class handles the core logic of loading images, processing annotations,
    applying augmentations, and converting data into tensors.
    """
    def __init__(self, cfg, data_anns_sup=None, is_train=True, **kwargs):
        super().__init__(cfg, data_anns_sup, is_train, **kwargs)

    def __call__(self, dataset_dict):
        """
        Processes a single data sample from the dataset.

        Args:
            dataset_dict (dict): A dict in Detectron2 dataset format.

        Returns:
            dict: The transformed dict with image and instances tensors.
                  Returns None if the image file is corrupt or cannot be read.
        """
        # Route to the appropriate method based on the training mode.
        if not self.is_train:
            return self.inference(dataset_dict)

        # Create a deep copy to avoid modifying the original dataset dict.
        dataset_dict = copy.deepcopy(dataset_dict)
        
        # Section 1: Image Loading and Validation
        # ------------------------------------------
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        if image is None:
            logger.warning(f"Could not read image: {dataset_dict['file_name']}. Skipping this sample.")
            return None
        
        image_shape = image.shape[:2]  # (H, W)
        # Pre-calculate image diagonal for normalization of the association vector magnitude.
        diag = math.sqrt(image_shape[0]**2 + image_shape[1]**2)

        # Section 2: Annotation Processing
        # ----------------------------------
        # Merges supplementary annotations and prepares them for the model.
        annotations_sup = {ann["id"]: ann for ann in self._data_anns_sup['annotations'] if ann['image_id'] == dataset_dict['image_id']}
        
        processed_annotations = []
        for ann in dataset_dict["annotations"]:
            sup_ann = annotations_sup.get(ann["id"])
            if not sup_ann:
                continue

            # Filter out instances without segmentation if masks are required.
            if self._masks_gt and (not ann.get("segmentation") or len(ann["segmentation"]) == 0):
                continue
            
            tmp_ann = ann.copy()
            
            # Conditionally remove segmentation data if not used.
            if not self._masks_gt:
                tmp_ann.pop("segmentation", None)

            # Standardize keypoints format if they are enabled.
            if self._keypoints_gt:
                keypoints = tmp_ann.get("keypoints", [])
                if keypoints:
                    keypoints_np = np.array(keypoints).flatten()
                    tmp_ann["keypoints"] = keypoints_np.tolist()
                else:
                    # Pad with zeros if no keypoints are annotated.
                    tmp_ann["keypoints"] = [0.0] * (self._num_keypoints * 3)
                    tmp_ann["num_keypoints"] = 0
            elif "keypoints" in tmp_ann:
                tmp_ann.pop("keypoints")

            # Enrich annotation with custom EHOI-specific labels.
            tmp_ann["hand_side"] = sup_ann.get("hand_side", 0)
            tmp_ann["contact_state"] = sup_ann.get("contact_state", 0)
            tmp_ann["gloves"] = sup_ann.get("gloves", 0) 
            
            # Process the association vector (dx, dy, magnitude) for hands in contact.
            if tmp_ann["contact_state"] == 1:
                tmp_ann["dx"] = float(sup_ann.get("dx", 0))
                tmp_ann["dy"] = float(sup_ann.get("dy", 0))
                # Normalize magnitude by image diagonal to make it scale-invariant.
                tmp_ann["magnitude"] = (float(sup_ann.get("magnitude", 0)) / diag) * self._scale_factor
            else:
                # Set to zero for non-contact states.
                tmp_ann["dx"], tmp_ann["dy"], tmp_ann["magnitude"] = 0.0, 0.0, 0.0
                
            processed_annotations.append(tmp_ann)
                
        dataset_dict["annotations"] = processed_annotations
        
        # Section 3: Data Augmentation
        # ------------------------------
        # Apply a sequence of color-based augmentations.
        transform_list = [
            T.RandomContrast(self._cfg.AUG.RANDOM_CONTRAST_MIN, self._cfg.AUG.RANDOM_CONTRAST_MAX), 
            T.RandomBrightness(self._cfg.AUG.RANDOM_BRIGHTNESS_MIN, self._cfg.AUG.RANDOM_BRIGHTNESS_MAX), 
            T.RandomLighting(scale=self._cfg.AUG.RANDOM_LIGHTING_SCALE),
        ]
        image, transforms = T.apply_transform_gens(transform_list, image)
        
        # Section 4: Tensor Conversion
        # ------------------------------
        # Convert the augmented image to a PyTorch tensor in CHW format.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        # Apply the same geometric transforms to all annotations (bboxes, masks, keypoints).
        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.get("annotations", [])
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        
        # Create custom ground-truth tensors and attach them to the Instances object.
        if len(annos) > 0:
            instances.gt_ids = torch.tensor([x["id"] for x in annos], dtype=torch.int64)
            instances.gt_contact_states = torch.tensor([x["contact_state"] for x in annos], dtype=torch.int64)
            instances.gt_sides = torch.tensor([x["hand_side"] for x in annos], dtype=torch.int64)
            instances.gt_gloves = torch.tensor([x["gloves"] for x in annos], dtype=torch.int64)
            instances.gt_dxdymagn_hands = torch.tensor([[x["dx"], x["dy"], x["magnitude"]] for x in annos], dtype=torch.float32)
        
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

class EhoiDatasetMapperDepthv1(EhoiDatasetMapperv1):
    """
    Extends the base EHOI mapper to additionally handle depth data.
    
    This class is responsible for loading ground-truth depth maps during training
    and preparing images for the MiDaS depth estimation module.
    """
    def __init__(self, cfg, data_anns_sup=None, is_train=True, _gt=True, **kwargs):
        super().__init__(cfg, data_anns_sup, is_train, **kwargs)
        self.net_w = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.NET_W
        self.net_h = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.NET_H
        self.resize_mode = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.RESIZE_MODE
        self._gt = _gt # Flag to control loading of ground-truth depth maps.
        
        # Pre-defined transformation pipeline for MiDaS input.
        self.transform = Compose([
            Resize(
                self.net_w, self.net_h, resize_target=True, keep_aspect_ratio=True,
                ensure_multiple_of=32, resize_method=self.resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            PrepareForNet()
        ])
        
    def __call__(self, dataset_dict):
        """
        Processes a data sample, adding depth information.
        """
        if not self.is_train: 
            return self.inference(dataset_dict)
            
        # First, process the sample using the parent class logic.
        element = super().__call__(dataset_dict)
        if element is None:
            return None
            
        # Prepare the image specifically for the depth estimation module.
        # MiDaS uses its own reading and transformation utilities.
        img = midas_utils.read_image(dataset_dict["file_name"])
        if img is None: # Add a safety check here as well
            logger.warning(f"[Depth] Could not read image for depth module: {dataset_dict['file_name']}. Skipping sample.")
            return None
        element["image_for_depth_module"] = self.transform({"image": img})["image"]
        
        # Load ground-truth depth map if required for training.
        if self._gt: 
            file_name = dataset_dict["file_name"]
            base_name = os.path.basename(file_name)
            
            # Heuristic to extract a unique image ID from various filename formats.
            match = re.search(r'\d+', base_name)
            image_id = match.group(0) if match else None
            
            if image_id:
                # Construct the path to the corresponding depth map.
                depth_path = f"./data/egoism-hoi-dataset/depth_maps/map_{image_id}.png"
                
                if os.path.exists(depth_path):
                    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                    if depth is not None:
                        depth_resized = cv2.resize(depth, (self.net_w, self.net_h), interpolation=cv2.INTER_NEAREST)
                        # Invert depth map (convention where 255 is far, 0 is near).
                        element["depth_gt"] = np.subtract(255, depth_resized.astype(np.float32))

        return element
    
    def inference(self, dataset_dict):
        """
        Prepares a dataset dict for inference, including depth module input.
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        if image is None:
            # During inference, we cannot return None as it would break the batch.
            # Raising an error is more informative. The dataset should be clean.
            raise FileNotFoundError(f"[Inference] Could not read image: {dataset_dict['file_name']}")

        # Apply standard inference-time transformations (typically just resizing).
        transform_list = [
            T.ResizeShortestEdge(
                [self._cfg.INPUT.MIN_SIZE_TEST, self._cfg.INPUT.MIN_SIZE_TEST], 
                self._cfg.INPUT.MAX_SIZE_TEST
            )
        ]
        image, _ = T.apply_transform_gens(transform_list, image)
        
        # Prepare standard model input tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        # The model's post-processing might need original height and width.
        dataset_dict["height"] = image.shape[0]
        dataset_dict["width"] = image.shape[1]

        # Prepare input for the depth module, using the original, unresized image for best quality.
        img_for_depth = utils.read_image(dataset_dict["file_name"], format="RGB") # MiDaS expects RGB
        if img_for_depth is not None:
             dataset_dict["image_for_depth_module"] = self.transform({"image": img_for_depth})["image"]
        else:
             # If reading fails, create a black image placeholder to prevent a crash.
             logger.warning(f"[Inference] Could not read image for depth module: {dataset_dict['file_name']}. Using a placeholder.")
             placeholder = torch.zeros((3, self.net_h, self.net_w))
             dataset_dict["image_for_depth_module"] = placeholder

        return dataset_dict