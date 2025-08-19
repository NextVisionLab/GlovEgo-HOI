import logging
import torch
from typing import Dict, List, Optional
import numpy as np
import time
import kornia.augmentation as K
import kornia
import cv2
import os

from detectron2.structures import Instances
from detectron2.structures.boxes import Boxes
from detectron2.utils.custom_utils import expand_box, extract_masks_and_resize
from .rcnn import GeneralizedRCNN
from .additional_modules import *
from .ehoi_net import EhoiNet
from .build import META_ARCH_REGISTRY
from ..roi_heads import select_foreground_proposals
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference_in_training, mask_rcnn_loss
from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_loss, keypoint_rcnn_inference

__all__ = ["MMEhoiNetv1"]
logger = logging.getLogger(__name__)
@META_ARCH_REGISTRY.register()
class MMEhoiNetv1(EhoiNet):
    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)

        self._expand_hand_box_ratio = cfg.ADDITIONAL_MODULES.EXPAND_HAND_BOX_RATIO

        self.association_vector_regressor = AssociationVectorRegressor(cfg)

        input_size_cnn_contact_state = cfg.ADDITIONAL_MODULES.CONTACT_STATE_CNN_INPUT_SIZE if "CONTACT_STATE_CNN_INPUT_SIZE" in cfg.ADDITIONAL_MODULES else 128
        self._roi_align_cnn_contact_state = torchvision.ops.RoIAlign((input_size_cnn_contact_state, input_size_cnn_contact_state), 1, -1)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training: return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs] if "instances" in batched_inputs[0] else None
        
        features = self.backbone(images.tensor)
        self._last_extracted_features["rgb"] = features

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        proposals_match, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        
        self._prepare_gt_labels(proposals_match)

        if self._use_depth_module: 
            self._last_extracted_features["depth"], self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
            if "depth_gt" in batched_inputs[0]:
                gt_depth_maps =  torch.tensor(np.array([e["depth_gt"] for e in batched_inputs])).to(self.device)
                loss_depth_estimation = self._loss_depth_f(self._depth_maps_predicted, gt_depth_maps)
            else:
                loss_depth_estimation = torch.tensor([0], dtype=torch.float32).to(self.device)

        mask_losses = {}
        keypoint_losses = {}

        if self._predict_mask or self._predict_keypoints:
            proposals_fg, _ = select_foreground_proposals(proposals_match, self._num_classes)
        
            if self._predict_mask:
                features_mask = self._mask_pooler([features[f] for f in self._mask_in_features], [x.proposal_boxes for x in proposals_fg])
                pred_mask_logits = self._mask_rcnn_head.layers(features_mask)
                if self._mask_gt:
                    mask_losses = {"loss_mask": mask_rcnn_loss(pred_mask_logits, proposals_fg) * self._mask_rcnn_head.loss_weight}
                proposals_match = mask_rcnn_inference_in_training(pred_mask_logits, proposals_fg)
            
            if self._predict_keypoints:
                proposals_hands_kpts = [p[p.gt_classes == self._id_hand] for p in proposals_fg]
                
                features_kpts = self._keypoint_pooler(
                    [features[f] for f in self._keypoint_in_features], 
                    [x.proposal_boxes for x in proposals_hands_kpts]
                )
                
                if features_kpts.shape[0] > 0:
                    keypoint_losses = self._keypoint_head(features_kpts, proposals_hands_kpts)
                    
                    weight = self.cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
                    for k in keypoint_losses.keys():
                        keypoint_losses[k] *= weight
                else:
                    keypoint_losses = {"loss_keypoint": torch.tensor(0.0, device=features['p2'].device)}

        self._prepare_hands_features(batched_inputs, proposals_match)
        self._last_proposals_match = proposals_match

        _, loss_classification_hand_lr = self.classification_hand_lr(self._c_hands_features, self._c_gt_hands_lr) 

        if self._contact_state_modality == "rgb":   
            _, loss_classification_contact_state = self.classification_contact_state(self._c_hands_features_padded, self._c_gt_hands_contact_state)
        elif "fusion" in self._contact_state_modality:
            _, loss_classification_contact_state = self.classification_contact_state(self._c_hands_features_cnn, self._c_hands_features_padded, self._c_gt_hands_contact_state)
        else:
            _, loss_classification_contact_state = self.classification_contact_state(self._c_hands_features_cnn, self._c_gt_hands_contact_state)

        indexes_contact = [i for i, x in enumerate(self._c_gt_hands_contact_state) if x == 1]
        _, loss_regression_vector = self.association_vector_regressor(self._c_hands_features_padded[indexes_contact], np.array(self._c_gt_hands_dxdymagnitude)[indexes_contact])            

        total_loss = {}
        total_loss.update(detector_losses)
        total_loss.update(proposal_losses)
        if isinstance(loss_classification_contact_state, dict): total_loss.update(loss_classification_contact_state)
        else: total_loss['loss_classification_contact_state'] =  loss_classification_contact_state
        total_loss['loss_classification_hand_lr'] =  loss_classification_hand_lr
        total_loss['loss_regression_dxdymagn'] =  loss_regression_vector
        if self._use_depth_module: total_loss['loss_depth'] = loss_depth_estimation
        if self._predict_mask and self._mask_gt: total_loss.update(mask_losses)
        if self._predict_keypoints: total_loss.update(keypoint_losses)

        return total_loss

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True):
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        self._last_inference_times ={k: 0 for k, v in self._last_inference_times.items()} 
        
        start_time = time.time()
        features = self.backbone(images.tensor)
        self._last_extracted_features["rgb"] = features
        self._last_inference_times["backbone"] = time.time() - start_time

        tmp_time = time.time()
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)
        self._last_inference_times["roi_heads"] = time.time() - tmp_time

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        instances = results[0]["instances"]
        results[0]['additional_outputs'] = Instances(instances.image_size)

        if self._use_depth_module: 
            tmp_time = time.time()
            self._last_extracted_features["depth"], self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
            self._last_inference_times["depth.extract_features_maps"] = time.time() - tmp_time

        if self._predict_mask:
            tmp_time = time.time()   
            features_mask = self._mask_pooler([features[f] for f in self._mask_in_features], [instances.pred_boxes])
            instances = self._mask_rcnn_head(features_mask, [instances])[0]
            self._last_inference_times["mask_rcnn_head"] = time.time() - tmp_time
        
        if self._predict_keypoints:
            tmp_time = time.time()
            features_kpts = self._keypoint_pooler(
                [features[f] for f in self._keypoint_in_features], 
                [instances.pred_boxes]
            )

            if features_kpts.shape[0] > 0:
                updated_instances_list = self._keypoint_head(features_kpts, [instances])
                instances = updated_instances_list[0]
                
            self._last_inference_times["keypoint_head"] = time.time() - tmp_time

        tmp_time = time.time()
        instances_hands = instances[instances.pred_classes == self._id_hand]
        self._prepare_hands_features_inference(batched_inputs, instances_hands)
        self._last_inference_times["data.prep.additional_modules"] = time.time() - tmp_time

        tmp_time = time.time()
        output_classification_side = torch.round(torch.sigmoid(self.classification_hand_lr(self._c_hands_features))).int()
        self._last_inference_times["classification_hand_lr"] = time.time() - tmp_time
        
        tmp_time = time.time()
        output_dxdymagn = self.association_vector_regressor(self._c_hands_features_padded)
        self._last_inference_times["association_vector_regressor"] = time.time() - tmp_time

        tmp_time = time.time()   
        if self._contact_state_modality == "rgb":
            output_classification_contact = torch.round(torch.sigmoid(self.classification_contact_state(self._c_hands_features_padded))).int()
        elif "fusion" in self._contact_state_modality:
            self.scores_contact = self.classification_contact_state(self._c_hands_features_cnn, self._c_hands_features_padded)
            output_classification_contact = torch.round(self.scores_contact).int()
        else:
            self.scores_contact = torch.sigmoid(self.classification_contact_state(self._c_hands_features_cnn))
            output_classification_contact = torch.round(self.scores_contact).int()
        self._last_inference_times["classification_contact_state"] = time.time() - tmp_time
        
        instances_hands = instances[instances.pred_classes == self._id_hand]
        
        if len(instances_hands) == len(output_classification_side):
            num_instances = len(instances)
            device = output_classification_side.device
            sides_full = torch.full((num_instances,), -1, dtype=torch.int32, device=device)
            contact_states_full = torch.full((num_instances,), -1, dtype=torch.int32, device=device)
            dxdymagn_hand_full = torch.full((num_instances, 3), -1.0, dtype=torch.float32, device=device)
            hand_mask = (instances.pred_classes == self._id_hand)
            sides_full[hand_mask] = output_classification_side.squeeze()
            contact_states_full[hand_mask] = output_classification_contact.squeeze()
            dxdymagn_hand_full[hand_mask] = output_dxdymagn
            results[0]["instances"].set("sides", sides_full)
            results[0]["instances"].set("contact_states", contact_states_full)
            results[0]["instances"].set("dxdymagn_hand", dxdymagn_hand_full)

        else:
            logger.warning(
                f"Mismatch between number of detected hands ({len(instances_hands)}) "
                f"and number of side/contact predictions ({len(output_classification_side)}). "
                "Skipping attachment of additional outputs for visualization."
            )

        if len(instances_hands) > 0:
            additional_outputs = Instances(instances_hands.image_size)
            additional_outputs.set("pred_boxes", instances_hands.pred_boxes)
            additional_outputs.set("scores", instances_hands.scores)
            additional_outputs.set("pred_classes", instances_hands.pred_classes)
            if instances_hands.has("sides"):
                additional_outputs.set("sides", instances_hands.sides)
                additional_outputs.set("contact_states", instances_hands.contact_states)
                additional_outputs.set("dxdymagn_hand", instances_hands.dxdymagn_hand)
            if instances_hands.has("pred_keypoints"):
                additional_outputs.set("pred_keypoints", instances_hands.pred_keypoints)
            
            results[0]["additional_outputs"] = additional_outputs
        
        if self._use_depth_module: results[0]['depth_map'] = self._depth_maps_predicted

        _total = round(sum(self._last_inference_times.values()) * 1000, 2)
        self._last_inference_times = {k: round(v * 1000, 2) for k, v in self._last_inference_times.items()}
        self._last_inference_times["total"] = _total
        self._last_instances_hands = instances_hands
        return results

    def _prepare_gt_labels(self, proposals_match):
        self._c_gt_hands_lr, self._c_gt_hands_contact_state, self._c_gt_hands_dxdymagnitude = [], [], []
        for batch_proposal in proposals_match:
            batch_proposal_hands = batch_proposal[batch_proposal.gt_classes == self._id_hand]
            for idx_proposal in range(len(batch_proposal_hands)): 
                self._c_gt_hands_lr.append(batch_proposal_hands[idx_proposal].gt_sides.item())
                self._c_gt_hands_contact_state.append(batch_proposal_hands[idx_proposal].gt_contact_states.item())
                self._c_gt_hands_dxdymagnitude.append(batch_proposal_hands[idx_proposal].gt_dxdymagn_hands.detach().cpu().numpy()[0])

    def _prepare_hands_features(self, batched_inputs, proposals_match):
        image_width, image_height = batched_inputs[0]['width'], batched_inputs[0]['height']
        boxes, boxes_padded, boxes_padded_depth = [], [], []

        for idx_batch, batch_proposal in enumerate(proposals_match):
            batch_proposal_hands = batch_proposal[batch_proposal.gt_classes == self._id_hand]
            boxes.append(batch_proposal_hands.proposal_boxes)
            boxes_padded.append(Boxes(expand_box(batch_proposal_hands.proposal_boxes.tensor.detach().clone(), image_width, image_height, ratio = self._expand_hand_box_ratio)))
            depth_width, depth_height = batched_inputs[idx_batch]["image_for_depth_module"].shape[2], batched_inputs[idx_batch]["image_for_depth_module"].shape[1]
            tmp_boxes = Boxes(boxes_padded[-1].tensor.detach().clone())
            tmp_boxes.scale(scale_x=(depth_width / image_width), scale_y=(depth_height / image_height))
            boxes_padded_depth.append(tmp_boxes)
        rois = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], boxes)      
        self._c_hands_features = self.roi_heads.box_head(rois) 
        rois_padded = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], boxes_padded)      
        self._c_hands_features_padded = self.roi_heads.box_head(rois_padded)
        
        self._last_boxes_padded_rescaled = [box.tensor for box in boxes_padded_depth]

        if self._contact_state_modality != "rgb":
            rgb_images = torch.tensor(np.array([b["image_for_depth_module"] for b in batched_inputs])).to(self.device)
            rgb_images = kornia.color.bgr_to_rgb(rgb_images)
            
            channels_to_cat = []
            
            if "rgb" in self._contact_state_modality:
                roi_rgb = self._roi_align_cnn_contact_state(rgb_images, [box.tensor for box in boxes_padded_depth])
                channels_to_cat.append(roi_rgb)
            
            if "depth" in self._contact_state_modality:
                depths = torch.divide(self._depth_maps_predicted.detach().unsqueeze(1), 255)
                roi_depth = self._roi_align_cnn_contact_state(depths, [box.tensor for box in boxes_padded_depth])
                channels_to_cat.append(roi_depth)
            
            if "mask" in self._contact_state_modality:
                masks = extract_masks_and_resize(proposals_match, batched_inputs[0]["image_for_depth_module"].shape[1:], self._id_hand)
                if len(masks):
                    masks = torch.cat(masks).unsqueeze(1)
                    boxes_cat = torch.cat((torch.arange(0, masks.shape[0]).unsqueeze(1).to(self.device), torch.cat([box.tensor for box in boxes_padded_depth])), dim = 1)
                    roi_mask = self._roi_align_cnn_contact_state(masks, boxes_cat)
                else: 
                    num_hands = sum(len(b[b.gt_classes == self._id_hand]) for b in proposals_match)
                    roi_mask = torch.zeros(num_hands, 1, roi_rgb.shape[2], roi_rgb.shape[3], device=self.device)
                channels_to_cat.append(roi_mask)
            
            if "kpts" in self._contact_state_modality:
            #if self._use_kpts_in_contact_state:
                gt_kpts_list = [p[p.gt_classes == self._id_hand].gt_keypoints.tensor for p in proposals_match if len(p[p.gt_classes == self._id_hand]) > 0]
                hand_boxes_list = [p[p.gt_classes == self._id_hand].proposal_boxes.tensor for p in proposals_match if len(p[p.gt_classes == self._id_hand]) > 0]
                
                if len(gt_kpts_list) > 0:
                    gt_kpts = torch.cat(gt_kpts_list, dim=0)
                    hand_boxes = torch.cat(hand_boxes_list, dim=0)
                    roi_kpts = self.keypoint_heatmap_generator(gt_kpts, hand_boxes)
                else: 
                    num_hands = sum(len(b[b.gt_classes == self._id_hand]) for b in proposals_match)
                    roi_kpts = torch.zeros(num_hands, 1, roi_rgb.shape[2], roi_rgb.shape[3], device=self.device)
                channels_to_cat.append(roi_kpts)

            self._c_hands_features_cnn = torch.cat(channels_to_cat, dim=1)
            #self.debug_roi(self._c_hands_features_cnn, batched_inputs, phase="train")
        
    def _prepare_hands_features_inference(self, batched_inputs, instances_hands):
        image_width, image_height = batched_inputs[0]['width'], batched_inputs[0]['height']
        rois = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], [instances_hands.pred_boxes])
        self._c_hands_features = torch.squeeze(self.roi_heads.box_head(rois))
        boxes_padded = Boxes(expand_box(instances_hands.pred_boxes.tensor.detach().clone(), image_width, image_height, ratio = self._expand_hand_box_ratio))
        rois_padded = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], [boxes_padded])
        self._c_hands_features_padded = torch.squeeze(self.roi_heads.box_head(rois_padded))

        if self._contact_state_modality != "rgb":
            boxes_padded_rescaled = Boxes(boxes_padded.tensor.detach().clone())
            boxes_padded_rescaled.scale(scale_x=batched_inputs[0]["image_for_depth_module"].shape[2] / batched_inputs[0]["image"].shape[2] , scale_y=batched_inputs[0]["image_for_depth_module"].shape[1] / batched_inputs[0]["image"].shape[1])     
            self._last_boxes_padded_rescaled = [boxes_padded_rescaled.tensor]
            rgb_images = torch.tensor(np.array([b["image_for_depth_module"] for b in batched_inputs])).to(self.device)
            rgb_images = kornia.color.bgr_to_rgb(rgb_images)

            channels_to_cat = []

            if "rgb" in self._contact_state_modality:
                roi_rgb = self._roi_align_cnn_contact_state(rgb_images, [boxes_padded_rescaled.tensor])
                channels_to_cat.append(roi_rgb)

            if "depth" in self._contact_state_modality:
                depths = torch.divide(self._depth_maps_predicted.unsqueeze(1), 255)
                roi_depth = self._roi_align_cnn_contact_state(depths, [boxes_padded_rescaled.tensor])
                channels_to_cat.append(roi_depth)
            
            if "mask" in self._contact_state_modality:
                masks = extract_masks_and_resize([instances_hands], batched_inputs[0]["image_for_depth_module"].shape[1:], self._id_hand)
                if len(masks):
                    masks = torch.cat(masks).unsqueeze(1)
                    boxes_cat = torch.cat((torch.arange(0, masks.shape[0]).unsqueeze(1).to(self.device), boxes_padded_rescaled.tensor), dim = 1)
                    roi_mask  = self._roi_align_cnn_contact_state(masks, boxes_cat)
                else:
                    roi_mask = torch.zeros(len(instances_hands), 1, roi_rgb.shape[2], roi_rgb.shape[3], device=self.device)
                channels_to_cat.append(roi_mask)

            if "kpts" in self._contact_state_modality:
                #if self._use_kpts_in_contact_state:
                if instances_hands.has("pred_keypoints"):
                    pred_kpts = instances_hands.pred_keypoints
                    hand_boxes = instances_hands.pred_boxes.tensor
                    roi_kpts = self.keypoint_heatmap_generator(pred_kpts, hand_boxes)
                else: 
                    roi_kpts = torch.zeros(len(instances_hands), 1, roi_rgb.shape[2], roi_rgb.shape[3], device=self.device)
                channels_to_cat.append(roi_kpts)
            
            if len(channels_to_cat) > 0:
                self._c_hands_features_cnn = torch.cat(channels_to_cat, dim=1)
                self.debug_roi(self._c_hands_features_cnn, batched_inputs, phase="inference", img_to_save=3)

    def debug_roi(self, c_roi, batched_inputs, phase, img_to_save=5):
        if not hasattr(self, f"_debug_save_count_{phase}"):
            setattr(self, f"_debug_save_count_{phase}", 0)
        
        save_count = getattr(self, f"_debug_save_count_{phase}")
        if save_count >= img_to_save: return
        
        if c_roi is not None and c_roi.shape[0] > 0:
            first_hand_roi = c_roi[0].detach().cpu().numpy()
            
            file_name = batched_inputs[0].get("file_name", f"unknown_{save_count}.png")
            image_name = os.path.splitext(os.path.basename(file_name))[0]

            debug_dir = os.path.join(self.cfg.OUTPUT_DIR, "debug_rois")
            os.makedirs(debug_dir, exist_ok=True)
            
            channel_map = {}
            current_channel = 0
            if "rgb" in self._contact_state_modality:
                channel_map["rgb"] = slice(current_channel, current_channel + 3); current_channel += 3
            if "depth" in self._contact_state_modality:
                channel_map["depth"] = current_channel; current_channel += 1
            if "mask" in self._contact_state_modality:
                channel_map["mask"] = current_channel; current_channel += 1
            if "kpts" in self._contact_state_modality:
                channel_map["kpts"] = current_channel; current_channel += 1
            
            if "rgb" in channel_map and first_hand_roi.shape[0] >= channel_map["rgb"].stop:
                rgb_roi_chw = first_hand_roi[channel_map["rgb"]]
                rgb_roi_hwc = np.transpose(rgb_roi_chw, (1, 2, 0))
                rgb_uint8 = (rgb_roi_hwc * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(debug_dir, f"debug_{phase}_{image_name}_0_rgb.png"), rgb_uint8)

            if "depth" in channel_map and first_hand_roi.shape[0] > channel_map["depth"]:
                depth_roi = (first_hand_roi[3] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(debug_dir, f"debug_{phase}_{image_name}_1_depth.png"), depth_roi)
            
            if "mask" in channel_map and first_hand_roi.shape[0] > channel_map["mask"]:
                mask_roi_uint8 = (first_hand_roi[channel_map["mask"]] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(debug_dir, f"debug_{phase}_{image_name}_2_mask.png"), mask_roi_uint8)
            
            if "kpts" in channel_map and first_hand_roi.shape[0] > channel_map["kpts"]:
                kpt_heatmap = first_hand_roi[channel_map["kpts"]]
                kpt_heatmap_uint8 = (kpt_heatmap * 255).astype(np.uint8)
                kpt_heatmap_color = cv2.applyColorMap(kpt_heatmap_uint8, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(debug_dir, f"debug_{phase}_{image_name}_3_kpts_heatmap.png"), kpt_heatmap_color)

            setattr(self, f"_debug_save_count_{phase}", save_count + 1)