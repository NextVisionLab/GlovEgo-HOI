import contextlib
import torch
import os
import numpy as np
import copy
import logging

from _pycocotools.coco import COCO
from _pycocotools.cocoeval import COCOeval
from _pycocotools.custom_handside_cocoeval import CustomHandSideCOCOeval
from _pycocotools.custom_handstate_cocoeval import CustomHandContactStateCOCOeval
from _pycocotools.custom_hand_target_object_w_classification import CustomHandTargetCOCOeval
from _pycocotools.custom_hand_all_cocoeval_w_classification import CustomHandAllCOCOeval

from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.data import MetadataCatalog
from detectron2.utils.custom_utils import get_iou
from detectron2.utils.file_io import PathManager

from .evaluator import DatasetEvaluator
from torchvision.ops.boxes import nms

class EHOIEvaluator(DatasetEvaluator):
    def __init__(self, cfg, dataset_name, converter):
        self._cfg = cfg
        self._output_dir = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        self._logger = logging.getLogger(__name__)
        
        self._metadata = MetadataCatalog.get(dataset_name)
        self._thing_classes = self._metadata.as_dict()["thing_classes"]
        self._id_hand = self._thing_classes.index("hand") if "hand" in self._thing_classes else self._thing_classes.index("mano")
        self._class_names_objs = [cls for cls in self._metadata.as_dict()["thing_classes"] if cls not in ["hand", "mano"]]

        self._converter = converter 
        self._coco_gt_all = COCO(PathManager.get_local_path(self._metadata.json_file))
        self._coco_gt = COCO(PathManager.get_local_path(self._metadata.coco_gt_hands))
        self._enrich_hand_gt_with_object_category()
        
        coco_gt_targets_dict = self._converter.convert_coco_to_coco_target_object(self._coco_gt, self._coco_gt_all)
        self._coco_gt_targets = COCO()
        if coco_gt_targets_dict['annotations']:
            self._coco_gt_targets.dataset = coco_gt_targets_dict
            self._coco_gt_targets.createIndex()

    def _enrich_hand_gt_with_object_category(self):
        all_anns_by_id = {ann['id']: ann for ann in self._coco_gt_all.dataset['annotations']}
        
        for ann_id, ann in self._coco_gt.anns.items():
            ann['category_id_obj'] = -1 
            
            if ann.get('contact_state') == 1 and 'id_obj' in ann and ann['id_obj'] != -1:
                obj_id = ann['id_obj']
                if obj_id in all_anns_by_id:
                    ann['category_id_obj'] = all_anns_by_id[obj_id]['category_id']
        
    def reset(self):
        self._predictions = []
        self._predictions_all = []
        self._predictions_targets = []
        self._prediction_counter = 0
        self._prediction_target_counter = 0 

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]

            if "instances" not in output:
                continue
            
            instances = output["instances"].to(torch.device("cpu"))
            additional_outputs = output.get("additional_outputs")
            instances_with_hand_data = None
            
            if additional_outputs is not None and additional_outputs.has("pred_boxes"):
                instances_with_hand_data = additional_outputs
            
            elif instances.has("sides") and instances.has("contact_states"):
                hand_indices = (instances.pred_classes == self._id_hand)
                instances_with_hand_data = instances[hand_indices]

            if instances_with_hand_data is not None and len(instances_with_hand_data) > 0:
                if instances_with_hand_data.has("boxes") and not instances_with_hand_data.has("pred_boxes"):
                    instances_with_hand_data.pred_boxes = instances_with_hand_data.boxes

                confident_instances = self._converter.generate_confident_instances(instances)
                predictions, predictions_target = self._converter.generate_predictions(
                    image_id, 
                    confident_instances, 
                    **output 
                )
                self._predictions.extend(predictions)
                self._predictions_targets.extend(predictions_target)
                self._prediction_counter += len(predictions)
                self._prediction_target_counter += len(predictions_target)

            confident_instances_all = self._converter._nms(instances)
            self._predictions_all.extend(self._converter.convert_instances_to_coco(confident_instances_all, image_id, convert_boxes_xywh_abs = True))

    def evaluate(self):
        valid_hand_gt_img_ids = set(self._coco_gt.getImgIds())
        valid_target_gt_img_ids = set(self._coco_gt_targets.getImgIds())
        
        filtered_hoi_predictions = [p for p in self._predictions if p['image_id'] in valid_hand_gt_img_ids]
        filtered_target_predictions = [p for p in self._predictions_targets if p['image_id'] in valid_target_gt_img_ids]

        if len(filtered_hoi_predictions) == 0:
            self._logger.warning("No HOI predictions found after confidence filtering. Skipping EHOI evaluation.")
            return {"ehoi": {}}
        
        cocoPreds = self._coco_gt.loadRes(filtered_hoi_predictions)
        coco_dt_all = self._coco_gt_all.loadRes(self._predictions_all)
        
        cocoPreds_target = None
        if filtered_target_predictions and self._coco_gt_targets.dataset.get('annotations'):
            cocoPreds_target = self._coco_gt_targets.loadRes(filtered_target_predictions)

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            with PathManager.open(os.path.join(self._output_dir, "ehoi_predictions.pth"), "wb") as f:
                torch.save(self._predictions, f)
            with PathManager.open(os.path.join(self._output_dir, "ehoi_predictions_all.pth"), "wb") as f:
                torch.save(self._predictions_all, f)
            with PathManager.open(os.path.join(self._output_dir, "ehoi_predictions_targets.pth"), "wb") as f:
                torch.save(self._predictions_targets, f)

        annType = 'bbox'
        coco_results = {}
        
        cat_name_to_id = {cat['name']: cat['id'] for cat in self._coco_gt_all.cats.values()}
        hand_id = -1
        for cat in self._coco_gt_all.dataset['categories']:
            if cat['name'] in ['hand', 'mano']:
                hand_id = cat['id']
                break

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            ##### HAND
            eval_hand = COCOeval(self._coco_gt_all, coco_dt_all, annType)
            eval_hand.params.iouThrs = np.array([0.5]); eval_hand.params.catIds = [hand_id]
            eval_hand.evaluate(); eval_hand.accumulate(); eval_hand.summarize()
            coco_results["AP Hand"] = round(eval_hand.stats[0] * 100, 2)

            ##### OBJECTS
            eval_obj = COCOeval(self._coco_gt_all, coco_dt_all, annType)
            eval_obj.params.iouThrs = np.array([0.5]); eval_obj.params.catIds = [c['id'] for c in self._coco_gt_all.cats.values() if c['id'] != hand_id]
            eval_obj.evaluate(); eval_obj.accumulate(); eval_obj.summarize()
            coco_results["mAP Objects"] = round(eval_obj.stats[0] * 100, 2)
            
            ##### TARGET OBJECTS
            if cocoPreds_target:
                eval_target = COCOeval(self._coco_gt_targets, cocoPreds_target, annType)
                eval_target.params.iouThrs = np.array([0.5])
                eval_target.evaluate(); eval_target.accumulate(); eval_target.summarize()
                coco_results["mAP Target Objects"] = round(eval_target.stats[0] * 100, 2) if eval_target.stats[0] > -1 else 0.0
            else:
                coco_results["mAP Target Objects"] = -1

            ##### HAND + SIDE
            eval_side = CustomHandSideCOCOeval(self._coco_gt, cocoPreds, annType)
            eval_side.params.iouThrs = np.array([0.5]); eval_side.evaluate(); eval_side.accumulate(); eval_side.summarize()
            coco_results["AP Hand + Side"] = round(eval_side.stats[0] * 100, 2)

            ##### HAND + CONTACT_STATE
            eval_state = CustomHandContactStateCOCOeval(self._coco_gt, cocoPreds, annType)
            eval_state.params.iouThrs = np.array([0.5]); eval_state.evaluate(); eval_state.accumulate(); eval_state.summarize()
            coco_results["AP Hand + State"] = round(eval_state.stats[0] * 100, 2)
            
            ##### mAP HAND + CONTACT_STATE
            tmp_results = {}
            for class_name in self._class_names_objs:
                original_cat_id = cat_name_to_id.get(class_name, -1)
                gt_filtred = copy.deepcopy(self._coco_gt)
                pred_filtred = copy.deepcopy(cocoPreds)
                gt_filtred.dataset["annotations"] = [ann for ann in self._coco_gt.dataset["annotations"] if ann.get("category_id_obj") == original_cat_id]
                gt_filtred.createIndex()
                pred_filtred.dataset["annotations"] = [ann for ann in cocoPreds.dataset["annotations"] if ann.get("category_id_obj") == original_cat_id]
                pred_filtred.createIndex()
                eval_contact_cls = CustomHandContactStateCOCOeval(gt_filtred, pred_filtred, annType)
                eval_contact_cls.params.iouThrs = np.array([0.5]); eval_contact_cls.evaluate(); eval_contact_cls.accumulate(); eval_contact_cls.summarize()
                tmp_results[class_name] = round(eval_contact_cls.stats[0] * 100, 2) if eval_contact_cls.stats[0] > -1 else 0.0
            coco_results["mAP Hand + State"] = round(np.array(list(tmp_results.values())).mean(), 2)

            tmp_results = {}
            for class_name in self._class_names_objs:
                original_cat_id = cat_name_to_id.get(class_name, -1)
                
                gt_filtred = copy.deepcopy(self._coco_gt)
                pred_filtred = copy.deepcopy(cocoPreds)
                
                gt_anns = [ann for ann in self._coco_gt.dataset["annotations"] if ann.get("category_id_obj") == original_cat_id]
                pred_anns = [ann for ann in cocoPreds.dataset["annotations"] if ann.get("category_id_obj") == original_cat_id]
                
                gt_filtred.dataset["annotations"] = gt_anns
                pred_filtred.dataset["annotations"] = pred_anns
                gt_filtred.createIndex()
                pred_filtred.createIndex()

                cocoEval = CustomHandTargetCOCOeval(gt_filtred, pred_filtred, annType)
                cocoEval.params.iouThrs = np.array([0.5]); cocoEval.evaluate(); cocoEval.accumulate(); cocoEval.summarize()
                score = 0.0
                if cocoEval.stats[0] > -1: score = round(cocoEval.stats[0] * 100, 2)
                if len(gt_filtred.anns) == 0: score = 100.0 if len(pred_filtred.anns) == 0 else 0.0
                tmp_results[class_name] = score

            coco_results["mAP Hand + Target Objects"] = round(np.array(list(tmp_results.values())).mean(), 2)
            
            ##### HAND + ALL
            all_class_results = {}
            for class_name in self._class_names_objs:
                original_cat_id = cat_name_to_id.get(class_name, -1)
                gt_filtred = copy.deepcopy(self._coco_gt)
                pred_filtred = copy.deepcopy(cocoPreds)
                gt_filtred.dataset["annotations"] = [ann for ann in self._coco_gt.dataset["annotations"] if ann.get("category_id_obj") == original_cat_id]
                gt_filtred.createIndex()
                pred_filtred.dataset["annotations"] = [ann for ann in cocoPreds.dataset["annotations"] if ann.get("category_id_obj") == original_cat_id]
                pred_filtred.createIndex()
                eval_all_cls = CustomHandAllCOCOeval(gt_filtred, pred_filtred, annType)
                eval_all_cls.params.iouThrs = np.array([0.5]); eval_all_cls.evaluate(); eval_all_cls.accumulate(); eval_all_cls.summarize()
                score = 0.0
                if eval_all_cls.stats[0] > -1: score = round(eval_all_cls.stats[0] * 100, 2)
                if len(gt_filtred.anns) == 0: score = 100.0 if len(pred_filtred.anns) == 0 else 0.0
                all_class_results[class_name] = score
            coco_results.update(all_class_results)
            
            ### AP HAND + ALL
            eval_all = CustomHandAllCOCOeval(self._coco_gt, cocoPreds, annType, agn = True)
            eval_all.params.iouThrs = np.array([0.5]); eval_all.evaluate(); eval_all.accumulate(); eval_all.summarize()
            coco_results["AP All"] = round(eval_all.stats[0] * 100, 2)
        
        list_results = np.array([coco_results.get(class_name, 0) for class_name in self._class_names_objs])
        coco_results["mAP All"] = round(list_results.mean(), 2)

        return {"ehoi" : coco_results}