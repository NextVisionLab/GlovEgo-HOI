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
from detectron2.modeling.meta_arch.MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet
from detectron2.modeling.meta_arch.MiDaS import utils as midas_utils

class EhoiDatasetMapperv1(BaseEhoiDatasetMapper):
    def __init__(self, cfg, data_anns_sup=None, is_train=True, **kwargs):
        super().__init__(cfg, data_anns_sup, is_train, **kwargs)

    def __call__(self, dataset_dict):
        if not self.is_train:  
            return self.inference(dataset_dict)

        dataset_dict = copy.deepcopy(dataset_dict)
        base_path = "./data/egoism-hoi-dataset/images"
        image_name = os.path.basename(dataset_dict["file_name"])
        image = cv2.imread(os.path.join(base_path, image_name))
        
        new_anns = []
        annotations_sup = [x for x in self._data_anns_sup['annotations'] if x['image_id'] == dataset_dict['image_id']]
        diag = math.sqrt((math.pow(image.shape[0], 2) + math.pow(image.shape[1], 2)))
        
        for ann in dataset_dict["annotations"]:
            tmp_ann = ann.copy()

            if self._masks_gt:
                if not "segmentation" in ann.keys() or len(ann["segmentation"])==0: 
                    continue
            elif "segmentation" in tmp_ann.keys(): 
                tmp_ann.pop("segmentation")

            if self._keypoints_gt:
                if len(tmp_ann["keypoints"]) != 0:
                    # Converti keypoints esistenti in formato corretto
                    keypoints = np.array(tmp_ann["keypoints"])
                    if keypoints.ndim == 2:
                        # Se è 2D [[x,y,v], [x,y,v], ...] -> 1D [x,y,v,x,y,v,...]
                        keypoints = keypoints.flatten()
                    tmp_ann["keypoints"] = keypoints.tolist()
                else:
                    # Crea keypoints vuoti in formato 1D: [0,0,0,0,0,0,...]
                    tmp_ann["keypoints"] = [0.0] * (self._num_keypoints * 3)
                    tmp_ann["num_keypoints"] = 0
            elif "keypoints" in tmp_ann.keys():
                tmp_ann.pop("keypoints")

            # Trova l'annotazione di supervisione corrispondente
            tmp_sup_ann_list = [_ann for _ann in annotations_sup if _ann["id"] == ann["id"]]
            if tmp_sup_ann_list:
                tmp_sup_ann = tmp_sup_ann_list[0]
                tmp_ann["hand_side"] = tmp_sup_ann["hand_side"] if "hand_side" in tmp_sup_ann.keys() and tmp_sup_ann["hand_side"] in [0, 1] else 0
                tmp_ann["contact_state"] = tmp_sup_ann["contact_state"] if "contact_state" in tmp_sup_ann.keys() and tmp_sup_ann["contact_state"] in [0, 1] else 0
                tmp_ann["dx"] = float(tmp_sup_ann["dx"]) if "dx" in tmp_sup_ann.keys() and tmp_ann["contact_state"] else 0
                tmp_ann["dy"] = float(tmp_sup_ann["dy"]) if "dy" in tmp_sup_ann.keys() and tmp_ann["contact_state"] else 0
                tmp_ann["magnitude"] = (float(tmp_sup_ann["magnitude"]) / diag) * self._scale_factor if "magnitude" in tmp_sup_ann.keys() and tmp_ann["contact_state"] else 0
            else:
                # Valori di default se non trova l'annotazione di supervisione
                tmp_ann["hand_side"] = 0
                tmp_ann["contact_state"] = 0
                tmp_ann["dx"] = 0
                tmp_ann["dy"] = 0
                tmp_ann["magnitude"] = 0
                
            new_anns.append(tmp_ann)
            
        dataset_dict["annotations"] = new_anns
        
        ####DATA AUGMENTATION
        transform_list = [
            T.RandomContrast(self._cfg.AUG.RANDOM_CONTRAST_MIN, self._cfg.AUG.RANDOM_CONTRAST_MAX), 
            T.RandomBrightness(self._cfg.AUG.RANDOM_BRIGHTNESS_MIN, self._cfg.AUG.RANDOM_BRIGHTNESS_MAX), 
            T.RandomLighting(scale=self._cfg.AUG.RANDOM_LIGHTING_SCALE),
        ]
        
        image, transforms = T.apply_transform_gens(transform_list, image)
        dataset_dict["height"] = image.shape[0]
        dataset_dict["width"] = image.shape[1]    
        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy())
        dataset_dict["image"] = image_t

        for annotation in dataset_dict["annotations"]:
            utils.transform_instance_annotations(annotation, transforms, image_t.shape[1:])    
        
        try: 
            instances = utils.annotations_to_instances(dataset_dict["annotations"], image_t.shape[1:])
            ids = [x["id"] for x in dataset_dict["annotations"]]
            contact_states = [x["contact_state"] for x in dataset_dict["annotations"]]
            sides = [x["hand_side"] for x in dataset_dict["annotations"]]
            dxdymagn_hands = [[x["dx"], x["dy"], x["magnitude"]] for x in dataset_dict["annotations"]]

            instances.set("gt_id", torch.tensor(ids))
            instances.set("gt_contact_states", torch.tensor(contact_states))
            instances.set("gt_sides", torch.tensor(sides))
            instances.set("gt_dxdymagn_hands", torch.tensor(dxdymagn_hands))
                
        except Exception as e: 
            print(f"Error in mapper: {e}, file: {dataset_dict['file_name']}")
            import traceback
            traceback.print_exc()

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