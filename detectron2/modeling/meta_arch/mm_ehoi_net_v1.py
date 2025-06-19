import logging
import torch
from typing import Dict, List, Optional
import numpy as np
import time
import kornia.augmentation as K
import kornia
import torch.nn.functional as F
import torchvision

from detectron2.structures import Instances
from detectron2.structures.boxes import Boxes
from detectron2.utils.custom_utils import expand_box, extract_masks_and_resize
from .rcnn import GeneralizedRCNN
from .additional_modules import *
from .ehoi_net import EhoiNet
from .build import META_ARCH_REGISTRY
from ..roi_heads import select_foreground_proposals
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference_in_training, mask_rcnn_loss
from detectron2.layers import cat

__all__ = ["MMEhoiNetv1"]
logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class MMEhoiNetv1(EhoiNet):
    """
    Multi-Modal EHOI Network with keypoint integration for contact state detection.
    
    Architecture flow:
    1. RGB Image -> Backbone -> Features
    2. Features -> ROI Head -> Object detection + Hand keypoints
    3. Hand features extraction (RGB, CNN depth/mask, Keypoints)
    4. Early fusion of all modalities for contact state classification
    """
    
    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)

        self._expand_hand_box_ratio = cfg.ADDITIONAL_MODULES.EXPAND_HAND_BOX_RATIO
        self.association_vector_regressor = AssociationVectorRegressor(cfg)

        input_size_cnn_contact_state = cfg.ADDITIONAL_MODULES.get('CONTACT_STATE_CNN_INPUT_SIZE', 128)
        self._roi_align_cnn_contact_state = torchvision.ops.RoIAlign((input_size_cnn_contact_state, input_size_cnn_contact_state), 1, -1)

        # Always use keypoints with early fusion
        self.keypoint_feature_extractor = KeypointFeatureExtractor(cfg)
        self.classification_contact_state = ContactStateFusionClassificationModule(cfg)

    def _extract_keypoints_from_instances(self, instances_list, image_size):
        """
        Extract keypoint coordinates from hand instances and convert to features.
        
        Flow: Hand instances -> Raw keypoints (21x3) -> Semantic features (128D)
        """
        all_keypoints = []
        
        for instances in instances_list:
            hand_instances = instances[instances.gt_classes == self._id_hand] if self.training else instances[instances.pred_classes == self._id_hand]
            
            if len(hand_instances) == 0:
                continue
                
            for i in range(len(hand_instances)):
                instance = hand_instances[i]
                if self.training and hasattr(instance, 'gt_keypoints') and instance.gt_keypoints is not None:
                    kpts = instance.gt_keypoints.tensor
                    if len(kpts.shape) == 3:
                        kpts = kpts.squeeze(0)
                elif not self.training and hasattr(instance, 'pred_keypoints') and instance.pred_keypoints is not None:
                    kpts = instance.pred_keypoints
                else:
                    kpts = torch.zeros(21, 3).to(self.device)
                
                all_keypoints.append(kpts)
        
        if len(all_keypoints) == 0:
            return torch.empty(0, 128).to(self.device)
            
        keypoints_batch = torch.stack(all_keypoints)
        return self.keypoint_feature_extractor(keypoints_batch, image_size)

    def mask_inference_in_training(self, pred_mask_logits, pred_instances):
        """Handle mask inference during training with proper tensor shape handling"""
        cls_agnostic_mask = pred_mask_logits.size(1) == 1
        
        if cls_agnostic_mask:
            mask_probs_pred = pred_mask_logits.sigmoid()
        else:
            all_gt_classes = [instances.gt_classes for instances in pred_instances if len(instances) > 0]
            if len(all_gt_classes) == 0:
                return pred_instances
            
            class_pred = cat(all_gt_classes)
            num_masks = pred_mask_logits.shape[0]
            
            min_size = min(len(class_pred), num_masks)
            class_pred = class_pred[:min_size]
            pred_mask_logits = pred_mask_logits[:min_size]
            
            max_class = pred_mask_logits.size(1) - 1
            class_pred = torch.clamp(class_pred, 0, max_class)
            
            indices = torch.arange(min_size, device=class_pred.device)
            mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
        
        num_boxes_per_image = [len(i) for i in pred_instances]
        total_expected = sum(num_boxes_per_image)
        
        if mask_probs_pred.shape[0] != total_expected:
            if mask_probs_pred.shape[0] < total_expected:
                padding_shape = (total_expected - mask_probs_pred.shape[0],) + mask_probs_pred.shape[1:]
                padding = torch.zeros(padding_shape, dtype=mask_probs_pred.dtype, device=mask_probs_pred.device)
                mask_probs_pred = torch.cat([mask_probs_pred, padding])
            else:
                mask_probs_pred = mask_probs_pred[:total_expected]
        
        mask_probs_pred_list = mask_probs_pred.split(num_boxes_per_image, dim=0)
        for prob, instances in zip(mask_probs_pred_list, pred_instances):
            if len(instances) > 0:
                instances.pred_masks = prob
        
        return pred_instances

    def _prepare_hands_features(self, batched_inputs, proposals_match):
        """
        Extract features from hands using multiple modalities.
        
        Flow: Hand proposals -> RGB features + CNN features + Keypoint features
        """
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
            
        # Extract RGB features from ROI Head
        rois = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], boxes)      
        self._c_hands_features = self.roi_heads.box_head(rois) 
        rois_padded = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], boxes_padded)      
        self._c_hands_features_padded = self.roi_heads.box_head(rois_padded)
        
        self._last_boxes_padded_rescaled = [box.tensor for box in boxes_padded_depth]
        
        # Extract CNN features (depth/mask modalities)
        if self._contact_state_modality != "rgb":
            rgb_images = torch.tensor(np.array([b["image_for_depth_module"] for b in batched_inputs])).to(self.device)
            rgb_images = kornia.color.bgr_to_rgb(rgb_images)
            if "depth" not in self._contact_state_modality:
                c_roi = self._roi_align_cnn_contact_state(rgb_images, [box.tensor for box in boxes_padded_depth])
            else:
                depths = torch.divide(self._depth_maps_predicted.detach().unsqueeze(1), 255)
                if rgb_images.shape[2:] != depths.shape[2:]:  
                    target_h, target_w = rgb_images.shape[2], rgb_images.shape[3]
                    depths = F.interpolate(depths, size=(target_h, target_w), mode='bilinear', align_corners=False)
                rgbd_images = torch.cat((rgb_images, depths), dim=1)
                c_roi = self._roi_align_cnn_contact_state(rgbd_images, [box.tensor for box in boxes_padded_depth])    
            if "mask" in self._contact_state_modality:
                masks = extract_masks_and_resize(proposals_match, batched_inputs[0]["image_for_depth_module"].shape[1:], self._id_hand)
                if len(masks):
                    masks = torch.cat(masks).unsqueeze(1)
                    boxes_cat = torch.cat((torch.arange(0, masks.shape[0]).unsqueeze(1).to(self.device), torch.cat([box.tensor for box in boxes_padded_depth])), dim = 1)
                    self._last_masks_roi = self._roi_align_cnn_contact_state(masks, boxes_cat)
                    c_roi = torch.cat((c_roi, self._last_masks_roi), dim=1)
                else:
                    self._last_masks_roi = torch.zeros(c_roi.shape)[:,:1].to(self.device)
                    c_roi = torch.cat((c_roi, self._last_masks_roi), dim=1)
            self._c_hands_features_cnn = c_roi
        else:
            self._c_hands_features_cnn = None

        # Extract keypoint features
        image_size = (image_height, image_width)
        self._c_hands_keypoint_features = self._extract_keypoints_from_instances(proposals_match, image_size)

    def _prepare_hands_features_inference(self, batched_inputs, instances_hands):
        """Extract hand features during inference - handles empty hands case"""
        if len(instances_hands) == 0:
            # No hands detected - set empty features
            self._c_hands_features = torch.empty(0, 1024).to(self.device)
            self._c_hands_features_padded = torch.empty(0, 1024).to(self.device)
            self._c_hands_features_cnn = torch.empty(0, 128, 128, 5).to(self.device) if self._contact_state_modality != "rgb" else None
            self._c_hands_keypoint_features = torch.empty(0, 128).to(self.device)
            return
            
        image_width, image_height = batched_inputs[0]['width'], batched_inputs[0]['height']
        
        rois = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], [instances_hands.pred_boxes])
        self._c_hands_features = torch.squeeze(self.roi_heads.box_head(rois))
        
        # Handle single hand case - ensure 2D
        if self._c_hands_features.dim() == 1:
            self._c_hands_features = self._c_hands_features.unsqueeze(0)
            
        boxes_padded = Boxes(expand_box(instances_hands.pred_boxes.tensor.detach().clone(), image_width, image_height, ratio = self._expand_hand_box_ratio))
        rois_padded = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], [boxes_padded])
        self._c_hands_features_padded = torch.squeeze(self.roi_heads.box_head(rois_padded))
        
        # Handle single hand case - ensure 2D
        if self._c_hands_features_padded.dim() == 1:
            self._c_hands_features_padded = self._c_hands_features_padded.unsqueeze(0)

        if self._contact_state_modality != "rgb":
            boxes_padded_rescaled = Boxes(boxes_padded.tensor.detach().clone())
            boxes_padded_rescaled.scale(scale_x=batched_inputs[0]["image_for_depth_module"].shape[2] / batched_inputs[0]["image"].shape[2] , scale_y=batched_inputs[0]["image_for_depth_module"].shape[1] / batched_inputs[0]["image"].shape[1])     
            self._last_boxes_padded_rescaled = [boxes_padded_rescaled.tensor]
            rgb_images = torch.tensor(np.array([b["image_for_depth_module"] for b in batched_inputs])).to(self.device)
            rgb_images = kornia.color.bgr_to_rgb(rgb_images)
            if "depth" not in self._contact_state_modality:
                c_roi = self._roi_align_cnn_contact_state(rgb_images, [boxes_padded_rescaled.tensor])
            else:
                depths = torch.divide(self._depth_maps_predicted.unsqueeze(1), 255)
                if rgb_images.shape[2:] != depths.shape[2:]:
                    target_h, target_w = rgb_images.shape[2], rgb_images.shape[3]                    
                    depths = F.interpolate(depths, size=(target_h, target_w), mode='bilinear', align_corners=False)
                rgbd_images = torch.cat((rgb_images, depths), dim=1)    
                c_roi = self._roi_align_cnn_contact_state(rgbd_images, [boxes_padded_rescaled.tensor])
            if "mask" in self._contact_state_modality:
                masks = extract_masks_and_resize([instances_hands], batched_inputs[0]["image_for_depth_module"].shape[1:], self._id_hand)
                if len(masks):
                    masks = torch.cat(masks).unsqueeze(1)
                    boxes_cat = torch.cat((torch.arange(0, masks.shape[0]).unsqueeze(1).to(self.device), boxes_padded_rescaled.tensor), dim = 1)
                    self._last_masks_roi = self._roi_align_cnn_contact_state(masks, boxes_cat)
                    c_roi = torch.cat((c_roi, self._last_masks_roi), dim=1)
                else:
                    self._last_masks_roi = torch.zeros(c_roi.shape)[:,:1].to(self.device)
                    c_roi = torch.cat((c_roi, self._last_masks_roi), dim=1)
            self._c_hands_features_cnn = c_roi
        else:
            self._c_hands_features_cnn = None

        # Extract keypoint features
        image_size = (image_height, image_width)
        self._c_hands_keypoint_features = self._extract_keypoints_from_instances([instances_hands], image_size)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Forward pass during training.
        
        Flow: Image -> Features -> ROI Head -> Hand features -> Contact classification
        """
        if not self.training: 
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs] if "instances" in batched_inputs[0] else None
        
        features = self.backbone(images.tensor)
        self._last_extracted_features["rgb"] = features

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        proposals_match, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        
        self._prepare_gt_labels(proposals_match)

        # Keypoint loss calculation
        keypoint_losses = {}
        hand_proposals = []
        hand_batch_indices = []
        
        for batch_idx, proposal_per_image in enumerate(proposals_match):
            hand_mask = proposal_per_image.gt_classes == self._id_hand
            if hand_mask.any():
                hand_instances = proposal_per_image[hand_mask]
                if not hasattr(hand_instances, 'proposal_boxes'):
                    hand_instances.proposal_boxes = hand_instances.gt_boxes
                hand_proposals.append(hand_instances)
                hand_batch_indices.append(batch_idx)
            
        total_hands = sum(len(p) for p in hand_proposals)
        
        if total_hands > 0 and hasattr(self.roi_heads, 'keypoint_head') and len(hand_proposals) > 0:
            hand_features = [features[f] for f in self.roi_heads.keypoint_in_features]
            hand_features_filtered = []
            
            for f in hand_features:
                filtered_f = f[hand_batch_indices]
                hand_features_filtered.append(filtered_f)
            
            keypoint_features = self.roi_heads.keypoint_pooler(
                hand_features_filtered, 
                [p.proposal_boxes for p in hand_proposals]
            )
            
            if len(hand_proposals) > 0:
                keypoint_logits = self.roi_heads.keypoint_head.layers(keypoint_features)
                from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_loss
                keypoint_loss = keypoint_rcnn_loss(keypoint_logits, hand_proposals, normalizer=None)
                keypoint_losses["loss_keypoint"] = keypoint_loss * self.roi_heads.keypoint_head.loss_weight
            else:
                keypoint_losses["loss_keypoint"] = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            keypoint_losses["loss_keypoint"] = torch.tensor(0.0, device=self.device, requires_grad=True)

        if self._use_depth_module:
            self._last_extracted_features["depth"], self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
            if "depth_gt" in batched_inputs[0]:
                gt_depth_maps = torch.tensor(np.array([e["depth_gt"] for e in batched_inputs])).to(self.device)
                loss_depth_estimation = self._loss_depth_f(self._depth_maps_predicted, gt_depth_maps)
            else:
                loss_depth_estimation = torch.tensor([0], dtype=torch.float32).to(self.device)

        if self._predict_mask:
            proposals_mask, _ = select_foreground_proposals(proposals_match, self._num_classes)
            total_proposals = sum(len(p) for p in proposals_mask)
            
            if total_proposals > 0:
                features_mask = self._mask_pooler([features[f] for f in self._mask_in_features], [x.proposal_boxes for x in proposals_mask])
                pred_mask_logits = self._mask_rcnn_head.layers(features_mask)
                
                if self._mask_gt:
                    mask_losses = {"loss_mask": mask_rcnn_loss(pred_mask_logits, proposals_mask) * self._mask_rcnn_head.loss_weight}
                
                proposals_match = self.mask_inference_in_training(pred_mask_logits, proposals_mask)
            else:
                if self._mask_gt:
                    mask_losses = {"loss_mask": torch.tensor(0.0, device=self.device, requires_grad=True)}

        self._prepare_hands_features(batched_inputs, proposals_match)
        self._last_proposals_match = proposals_match

        _, loss_classification_hand_lr = self.classification_hand_lr(self._c_hands_features, self._c_gt_hands_lr)

        # Contact state classification with early fusion
        _, loss_classification_contact_state = self.classification_contact_state(
            self._c_hands_features_padded,
            self._c_hands_features_cnn,
            self._c_hands_keypoint_features,
            self._c_gt_hands_contact_state
        )

        indexes_contact = [i for i, x in enumerate(self._c_gt_hands_contact_state) if x == 1]
        _, loss_regression_vector = self.association_vector_regressor(self._c_hands_features_padded[indexes_contact], np.array(self._c_gt_hands_dxdymagnitude)[indexes_contact])            

        total_loss = {}
        total_loss.update(detector_losses)
        total_loss.update(proposal_losses)
        if isinstance(loss_classification_contact_state, dict): 
            total_loss.update(loss_classification_contact_state)
        else: 
            total_loss['loss_classification_contact_state'] = loss_classification_contact_state
        total_loss['loss_classification_hand_lr'] = loss_classification_hand_lr
        total_loss['loss_regression_dxdymagn'] = loss_regression_vector
        if self._use_depth_module: 
            total_loss['loss_depth'] = loss_depth_estimation
        if self._mask_gt: 
            total_loss.update(mask_losses)
        
        total_loss.update(keypoint_losses)

        return total_loss

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True):
        """
        Inference pass.
        
        Flow: Image -> Features -> Detection -> Hand features -> Contact prediction
        """
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        self._last_inference_times = {k: 0 for k, v in self._last_inference_times.items()} 
        
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
        
        tmp_time = time.time()
        instances_hands = instances[instances.pred_classes == self._id_hand]
        
        # Handle case when no hands are detected
        if len(instances_hands) == 0:
            # Create empty outputs for consistency
            results[0]['additional_outputs'].set("boxes", torch.empty(0, 4))
            results[0]['additional_outputs'].set("sides", torch.empty(0, dtype=torch.int))
            results[0]['additional_outputs'].set("scores", torch.empty(0))
            results[0]['additional_outputs'].set("contact_states", torch.empty(0, dtype=torch.int))
            results[0]['additional_outputs'].set("dxdymagn_hand", torch.empty(0, 3))
            
            self._last_inference_times["data.prep.additional_modules"] = time.time() - tmp_time
            self._last_inference_times["classification_hand_lr"] = 0
            self._last_inference_times["association_vector_regressor"] = 0
            self._last_inference_times["classification_contact_state"] = 0
            
            _total = round(sum(self._last_inference_times.values()) * 1000, 2)
            self._last_inference_times = {k: round(v * 1000, 2) for k, v in self._last_inference_times.items()}
            self._last_inference_times["total"] = _total
            self._last_instances_hands = instances_hands
            return results
        
        self._prepare_hands_features_inference(batched_inputs, instances_hands)
        self._last_inference_times["data.prep.additional_modules"] = time.time() - tmp_time
    
        tmp_time = time.time()
        output_classification_side = torch.round(torch.sigmoid(self.classification_hand_lr(self._c_hands_features))).int()
        self._last_inference_times["classification_hand_lr"] = time.time() - tmp_time
        
        tmp_time = time.time()
        output_dxdymagn = self.association_vector_regressor(self._c_hands_features_padded)
        self._last_inference_times["association_vector_regressor"] = time.time() - tmp_time

        tmp_time = time.time()   
        # Contact state classification with early fusion
        self.scores_contact = self.classification_contact_state(
            self._c_hands_features_padded,
            self._c_hands_features_cnn,
            self._c_hands_keypoint_features
        )
        output_classification_contact = torch.round(self.scores_contact).int()
        self._last_inference_times["classification_contact_state"] = time.time() - tmp_time

        results[0]['additional_outputs'].set("boxes", instances_hands.pred_boxes.tensor.reshape(-1, 4).detach().cpu())
        results[0]['additional_outputs'].set("sides", output_classification_side.detach().cpu())
        results[0]['additional_outputs'].set("scores", instances_hands.scores.detach().cpu())
        results[0]['additional_outputs'].set("contact_states", output_classification_contact.detach().cpu())
        results[0]['additional_outputs'].set("dxdymagn_hand", output_dxdymagn.reshape(-1, 3).detach().cpu())
        if self._use_depth_module: 
            results[0]['depth_map'] = self._depth_maps_predicted

        _total = round(sum(self._last_inference_times.values()) * 1000, 2)
        self._last_inference_times = {k: round(v * 1000, 2) for k, v in self._last_inference_times.items()}
        self._last_inference_times["total"] = _total
        self._last_instances_hands = instances_hands
        return results

    def _prepare_gt_labels(self, proposals_match):
        """Prepare ground truth labels for training"""
        self._c_gt_hands_lr, self._c_gt_hands_contact_state, self._c_gt_hands_dxdymagnitude = [], [], []
        for batch_proposal in proposals_match:
            batch_proposal_hands = batch_proposal[batch_proposal.gt_classes == self._id_hand]
            for idx_proposal in range(len(batch_proposal_hands)): 
                self._c_gt_hands_lr.append(batch_proposal_hands[idx_proposal].gt_sides.item())
                self._c_gt_hands_contact_state.append(batch_proposal_hands[idx_proposal].gt_contact_states.item())
                self._c_gt_hands_dxdymagnitude.append(batch_proposal_hands[idx_proposal].gt_dxdymagn_hands.detach().cpu().numpy()[0])