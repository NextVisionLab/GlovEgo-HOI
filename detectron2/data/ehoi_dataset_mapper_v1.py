import copy
import os
import numpy as np
import torch
import math
import cv2

from . import detection_utils as utils
from . import transforms as T
from . import BaseEhoiDatasetMapper

from torchvision.transforms import Compose
from detectron2.modeling.meta_arch.MiDaS.midas.transforms import Resize, PrepareForNet
from detectron2.modeling.meta_arch.MiDaS import utils as midas_utils

class EhoiDatasetMapperv1(BaseEhoiDatasetMapper):
    def __init__(self, cfg, data_anns_sup=None, is_train=True, **kwargs):
        super().__init__(cfg, data_anns_sup, is_train, **kwargs)

    def __call__(self, dataset_dict):
        if not self.is_train:
            return self.inference(dataset_dict)

        dataset_dict = copy.deepcopy(dataset_dict)
        
        # 1. Load Image
        image = cv2.imread(dataset_dict["file_name"])
        if image is None:
            print(f"ERROR: Could not read image: {dataset_dict['file_name']}. Skipping sample.")
            return None
        
        image_shape = image.shape[:2]  # (H, W)
        diag = math.sqrt(image_shape[0]**2 + image_shape[1]**2)

        # 2. Process Annotations
        annotations_sup = {ann["id"]: ann for ann in self._data_anns_sup['annotations'] if ann['image_id'] == dataset_dict['image_id']}
        
        processed_annotations = []
        for ann in dataset_dict["annotations"]:
            sup_ann = annotations_sup.get(ann["id"])
            if not sup_ann:
                continue

            if self._masks_gt and (not ann.get("segmentation") or len(ann["segmentation"]) == 0):
                continue
            
            tmp_ann = ann.copy()
            if not self._masks_gt:
                tmp_ann.pop("segmentation", None)

            if self._keypoints_gt:
                keypoints = tmp_ann.get("keypoints", [])
                if keypoints:
                    keypoints_np = np.array(keypoints)
                    if keypoints_np.ndim == 2:
                        keypoints_np = keypoints_np.flatten()
                    tmp_ann["keypoints"] = keypoints_np.tolist()
                else:
                    tmp_ann["keypoints"] = [0.0] * (self._num_keypoints * 3)
                    tmp_ann["num_keypoints"] = 0
            elif "keypoints" in tmp_ann:
                tmp_ann.pop("keypoints")

            tmp_ann["hand_side"] = sup_ann.get("hand_side", 0)
            tmp_ann["contact_state"] = sup_ann.get("contact_state", 0)
            tmp_ann["gloves"] = sup_ann.get("gloves", 0) 
            
            if tmp_ann["contact_state"] == 1:
                tmp_ann["dx"] = float(sup_ann.get("dx", 0))
                tmp_ann["dy"] = float(sup_ann.get("dy", 0))
                tmp_ann["magnitude"] = (float(sup_ann.get("magnitude", 0)) / diag) * self._scale_factor
            else:
                tmp_ann["dx"], tmp_ann["dy"], tmp_ann["magnitude"] = 0.0, 0.0, 0.0
                
            processed_annotations.append(tmp_ann)
                
        dataset_dict["annotations"] = processed_annotations
        
        # 3. Apply Data Augmentation
        transform_list = [
            T.RandomContrast(self._cfg.AUG.RANDOM_CONTRAST_MIN, self._cfg.AUG.RANDOM_CONTRAST_MAX), 
            T.RandomBrightness(self._cfg.AUG.RANDOM_BRIGHTNESS_MIN, self._cfg.AUG.RANDOM_BRIGHTNESS_MAX), 
            T.RandomLighting(scale=self._cfg.AUG.RANDOM_LIGHTING_SCALE),
        ]
        image, transforms = T.apply_transform_gens(transform_list, image)
        
        # 4. Prepare Final Tensors for the Model
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        # Apply transforms to annotations
        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.get("annotations", [])
        ]
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        
        # Create GT tensors
        if len(annos) > 0:
            instances.gt_ids = torch.tensor([x["id"] for x in annos], dtype=torch.int64)
            instances.gt_contact_states = torch.tensor([x["contact_state"] for x in annos], dtype=torch.int64)
            instances.gt_sides = torch.tensor([x["hand_side"] for x in annos], dtype=torch.int64)
            instances.gt_gloves = torch.tensor([x["gloves"] for x in annos], dtype=torch.int64)
            instances.gt_dxdymagn_hands = torch.tensor([[x["dx"], x["dy"], x["magnitude"]] for x in annos], dtype=torch.float32)
        
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict

class EhoiDatasetMapperDepthv1(EhoiDatasetMapperv1):
    def __init__(self, cfg, data_anns_sup=None, is_train=True, _gt=True, **kwargs):
        super().__init__(cfg, data_anns_sup, is_train, **kwargs)
        self.net_w = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.NET_W
        self.net_h = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.NET_H
        self.resize_mode = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.RESIZE_MODE
        self._gt = _gt
        
        self.transform = Compose([
            Resize(
                self.net_w,
                self.net_h,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=self.resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            PrepareForNet()
        ])
        
        self.transform_depth = Compose([
            Resize(
                self.net_w,
                self.net_h,
                resize_target=True,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=self.resize_mode,
                image_interpolation_method=cv2.INTER_NEAREST
            )
        ])

    def __call__(self, dataset_dict):
        if not self.is_train: 
            return self.inference(dataset_dict)
            
        element = super().__call__(dataset_dict)
        if element is None:  # Safety check
            return None
            
        img = midas_utils.read_image(dataset_dict["file_name"])
        element["image_for_depth_module"] = self.transform({"image": img})["image"]
        
        if self._gt: 
            file_name = dataset_dict["file_name"]
            base_name = os.path.basename(file_name)
            
            import re
            if base_name.startswith('rgb_'):
                image_id = base_name.replace('rgb_', '').replace('.png', '')
            elif base_name.startswith('camera_'):
                image_id = base_name.replace('camera_', '').replace('.png', '')
            else:
                numbers = re.findall(r'\d+', base_name)
                image_id = numbers[0] if numbers else None
            
            if image_id:
                depth_path = f"./data/egoism-hoi-dataset/depth_maps/map_{image_id}.png"
                
                if os.path.exists(depth_path):
                    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                    if depth is not None:
                        depth_resized = cv2.resize(depth, (self.net_w, self.net_h), interpolation=cv2.INTER_NEAREST)
                        depth_final = depth_resized.astype(np.float32)
                        element["depth_gt"] = np.subtract(255, depth_final)
        return element

    def inference(self, dataset_dict):
        element = super().inference(dataset_dict)
        img = midas_utils.read_image(dataset_dict["file_name"])
        img_input = self.transform({"image": img})["image"]
        element["image_for_depth_module"] = img_input
        return element