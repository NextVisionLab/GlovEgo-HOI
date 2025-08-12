import copy
import os
import numpy as np
import torch
import math
import cv2

from . import detection_utils as utils
from . import transforms as T
from . import BaseEhoiDatasetMapper
from detectron2.structures import Keypoints, Instances

from torchvision.transforms import Compose
from detectron2.modeling.meta_arch.MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet
from detectron2.modeling.meta_arch.MiDaS import utils as midas_utils

class EhoiDatasetMapperv1(BaseEhoiDatasetMapper):
    def __init__(self, cfg, data_anns_sup=None, is_train=True, **kwargs):       
        super().__init__(cfg, data_anns_sup, is_train, **kwargs)
        self._keypoint_hflip_indices = None
        if is_train and cfg.MODEL.KEYPOINT_ON:
            self._keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        self._num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        self._keypoints_gt = cfg.ADDITIONAL_MODULES.KEYPOINTS_GT

    def __call__(self, dataset_dict):
        if not self.is_train:
            return self.inference(dataset_dict)

        dataset_dict = copy.deepcopy(dataset_dict)
        image = cv2.imread(dataset_dict["file_name"])
        if image is None:
            return None
        
        annotations_sup = [x for x in self._data_anns_sup['annotations'] if x['image_id'] == dataset_dict['image_id']]
        sup_ann_dict = {s_ann["id"]: s_ann for s_ann in annotations_sup}
        
        for i, ann in enumerate(dataset_dict["annotations"]):
            ann_id = ann.get("id")            
            tmp_sup_ann = sup_ann_dict.get(ann_id)
            
            if tmp_sup_ann:
                contact_state_raw = tmp_sup_ann.get("contact_state", "MISSING")
                magnitude_raw = tmp_sup_ann.get("magnitude", "MISSING")
                ann["contact_state"] = contact_state_raw if contact_state_raw != "MISSING" else -1
                ann["magnitude"] = float(magnitude_raw) if ann["contact_state"] == 1 else 0.0
            else:
                ann["contact_state"] = -1
                ann["magnitude"] = 0.0

        diag = math.sqrt(image.shape[0]**2 + image.shape[1]**2)
        
        for ann in dataset_dict["annotations"]:
            tmp_sup_ann = sup_ann_dict.get(ann["id"])
            if tmp_sup_ann:
                ann["hand_side"] = tmp_sup_ann.get("hand_side", -1)
                ann["contact_state"] = tmp_sup_ann.get("contact_state", -1)
                if ann["contact_state"] == 1:
                    ann["dx"] = float(tmp_sup_ann.get("dx", 0.0))
                    ann["dy"] = float(tmp_sup_ann.get("dy", 0.0))
                    ann["magnitude"] = float(tmp_sup_ann.get("magnitude", 0.0))
                else:
                    ann["dx"], ann["dy"], ann["magnitude"] = 0.0, 0.0, 0.0
        
        transform_list = [
            T.RandomContrast(self._cfg.AUG.RANDOM_CONTRAST_MIN, self._cfg.AUG.RANDOM_CONTRAST_MAX),
            T.RandomBrightness(self._cfg.AUG.RANDOM_BRIGHTNESS_MIN, self._cfg.AUG.RANDOM_BRIGHTNESS_MAX),
            T.RandomLighting(scale=self._cfg.AUG.RANDOM_LIGHTING_SCALE),
        ]
        
        image, transforms = T.apply_transform_gens(transform_list, image)
        image_shape = image.shape[:2]
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self._keypoint_hflip_indices
            )
            for obj in dataset_dict["annotations"]
            if obj.get("iscrowd", 0) == 0
        ]
        
        use_masks = self._masks_gt
        mask_format = self._cfg.INPUT.MASK_FORMAT if use_masks else "none"

        annos_for_instances = []
        for ann in annos:
            ann_copy = copy.deepcopy(ann)
            if "keypoints" in ann_copy:
                ann_copy.pop("keypoints")

            if not use_masks:
                if "segmentation" in ann_copy:
                    ann_copy.pop("segmentation")
            else:
                if "segmentation" not in ann_copy:
                    ann_copy["segmentation"] = []
            
            annos_for_instances.append(ann_copy)
        
        instances = utils.annotations_to_instances(annos_for_instances, image_shape, mask_format=mask_format)

        keypoints_list = []
        for obj in annos:
            kps_flat = obj.get("keypoints", [])
            if len(kps_flat) > 0:
                keypoints_list.append(np.array(kps_flat).reshape(self._num_keypoints, 3))
            else:
                keypoints_list.append(np.zeros((self._num_keypoints, 3), dtype=np.float32))

        keypoints_array = np.array(keypoints_list, dtype=np.float32)
        instances.gt_keypoints = Keypoints(torch.as_tensor(keypoints_array))
        
        ids = [x.get("id", -1) for x in annos]
        contact_states = [x.get("contact_state", -1) for x in annos]
        sides = [x.get("hand_side", -1) for x in annos]
        dxdymagn_hands = [[x.get("dx", 0.0), x.get("dy", 0.0), x.get("magnitude", 0.0)] for x in annos]

        instances.set("gt_id", torch.tensor(ids, dtype=torch.int64))
        instances.set("gt_contact_states", torch.tensor(contact_states, dtype=torch.int64))
        instances.set("gt_sides", torch.tensor(sides, dtype=torch.int64))
        instances.set("gt_dxdymagn_hands", torch.tensor(dxdymagn_hands, dtype=torch.float32))

        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        if not len(dataset_dict["instances"]):
             return None

        return dataset_dict

class EhoiDatasetMapperDepthv1(EhoiDatasetMapperv1):
	def __init__(self, cfg, data_anns_sup = None, is_train = True, _gt = True,  **kwargs):
		super().__init__(cfg, data_anns_sup, is_train, **kwargs)
		self.net_w = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.NET_W
		self.net_h = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.NET_H
		self.resize_mode = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.RESIZE_MODE
		self._gt = _gt
		#self.normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
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
		        #self.normalization,
		        PrepareForNet()])
		self.transform_depth = Compose([
			Resize(
				self.net_w,
				self.net_h,
				resize_target=True,
				keep_aspect_ratio=True,
				ensure_multiple_of=32,
				resize_method=self.resize_mode,
				image_interpolation_method=cv2.INTER_NEAREST)])

	def __call__(self, dataset_dict):
		if not self.is_train: return self.inference(dataset_dict)
		element = super().__call__(dataset_dict)
		img = midas_utils.read_image(dataset_dict["file_name"])
		element["image_for_depth_module"] = self.transform({"image": img})["image"]
		if self._gt: 
			depth_path = dataset_dict["file_name"].replace("images", "depth_maps").replace("camera", "map").replace("jpg", "png")
			if os.path.exists(depth_path):
				depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE) if self._gt else []
				depth = np.ascontiguousarray((self.transform_depth({"image": depth})["image"].astype(np.float32)))
				element["depth_gt"] = np.subtract(255, depth) 
		return element

	def inference(self, dataset_dict):
		element = super().inference(dataset_dict)
		img = midas_utils.read_image(dataset_dict["file_name"])
		img_input = self.transform({"image": img})["image"]
		element["image_for_depth_module"] = img_input
		return element