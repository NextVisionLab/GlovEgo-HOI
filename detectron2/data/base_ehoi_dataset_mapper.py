import copy
import numpy as np
import torch
import cv2
from abc import abstractmethod
from . import transforms as T

class BaseEhoiDatasetMapper:
    def __init__(self, cfg, data_anns_sup=None, is_train=True, **kwargs):
        self._data_anns_sup = data_anns_sup
        self._cfg = cfg
        
        # Handle both config object and dict format
        if hasattr(cfg, 'ADDITIONAL_MODULES'):
            # Standard config object
            self._scale_factor = cfg.ADDITIONAL_MODULES.ASSOCIATION_VECTOR_SCALE_FACTOR
            self._masks_gt = cfg.ADDITIONAL_MODULES.USE_MASK_GT
            self._keypoints_gt = (
                cfg.ADDITIONAL_MODULES.get('USE_KEYPOINTS', False) or 
                cfg.MODEL.get('KEYPOINT_ON', False) or
                cfg.ADDITIONAL_MODULES.get('KEYPOINTS_GT', False)
            )
            if self._keypoints_gt:
                self._num_keypoints = (
                    cfg.MODEL.get('ROI_KEYPOINT_HEAD', {}).get('NUM_KEYPOINTS', None) or
                    cfg.ADDITIONAL_MODULES.get('NUM_KEYPOINTS', 21)
                )
            else:
                self._num_keypoints = 21
        else:
            # Fallback for dict-like config
            additional_modules = cfg.get('ADDITIONAL_MODULES', {})
            model_cfg = cfg.get('MODEL', {})
            
            self._scale_factor = additional_modules.get('ASSOCIATION_VECTOR_SCALE_FACTOR', 1.0)
            self._masks_gt = additional_modules.get('USE_MASK_GT', False)
            self._keypoints_gt = (
                additional_modules.get('USE_KEYPOINTS', False) or 
                model_cfg.get('KEYPOINT_ON', False) or
                additional_modules.get('KEYPOINTS_GT', False)
            )
            self._num_keypoints = (
                model_cfg.get('ROI_KEYPOINT_HEAD', {}).get('NUM_KEYPOINTS', 21) or
                additional_modules.get('NUM_KEYPOINTS', 21)
            )
            
        # Force keypoints if contact state modality includes keypoints
        if hasattr(cfg, 'ADDITIONAL_MODULES'):
            contact_state_modality = cfg.ADDITIONAL_MODULES.get('CONTACT_STATE_MODALITY', '')
        else:
            contact_state_modality = cfg.get('ADDITIONAL_MODULES', {}).get('CONTACT_STATE_MODALITY', '')
            
        if 'keypoints' in contact_state_modality and not self._keypoints_gt:
            self._keypoints_gt = True
            
        self.is_train = is_train

    @abstractmethod
    def __call__(self, dataset_dict):
        pass

    def inference(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = cv2.imread(dataset_dict["file_name"])
        
        # Data augmentation
        transform_list = []
        image, transforms = T.apply_transform_gens(transform_list, image)
        dataset_dict["height"] = image.shape[0]
        dataset_dict["width"] = image.shape[1]
        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy())
        dataset_dict["image"] = image_t
        return dataset_dict