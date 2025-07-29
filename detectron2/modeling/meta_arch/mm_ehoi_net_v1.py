import logging
import time
from typing import Dict, List, Optional

import kornia
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from detectron2.layers import cat
from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_loss
from detectron2.modeling.roi_heads.mask_head import (
    mask_rcnn_inference_in_training, mask_rcnn_loss)
from detectron2.structures import Instances
from detectron2.structures.boxes import Boxes
from detectron2.utils.custom_utils import expand_box, extract_masks_and_resize

from ..roi_heads import select_foreground_proposals
from .additional_modules import (AssociationVectorRegressor,
                                 ContactStateFusionClassificationModule,
                                 KeypointFeatureExtractor)
from .build import META_ARCH_REGISTRY
from .ehoi_net import EhoiNet
from .rcnn import GeneralizedRCNN

__all__ = ["MMEhoiNetv1"]
logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class MMEhoiNetv1(EhoiNet):
    """
    Multi-Modal Egocentric Human-Object Interaction Network (MMEhoiNetv1).
    """

    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)

        self._expand_hand_box_ratio = cfg.ADDITIONAL_MODULES.EXPAND_HAND_BOX_RATIO
        self.association_vector_regressor = AssociationVectorRegressor(cfg)
        self.keypoint_feature_extractor = KeypointFeatureExtractor(cfg)
        self.classification_contact_state = ContactStateFusionClassificationModule(cfg)

        input_size_cnn_contact_state = cfg.ADDITIONAL_MODULES.get('CONTACT_STATE_CNN_INPUT_SIZE', 128)
        self._roi_align_cnn_contact_state = torchvision.ops.RoIAlign(
            (input_size_cnn_contact_state, input_size_cnn_contact_state),
            spatial_scale=1.0,
            sampling_ratio=-1
        )

    def _extract_and_process_keypoints(self, instances_list: List[Instances], hand_boxes_list: List[Boxes]) -> torch.Tensor:
        """
        Extracts raw keypoints from hand instances and processes them into semantic features.
        This version robustly handles different keypoint tensor shapes from training and inference.
        """
        all_keypoints, all_hand_boxes_flat = [], []
        
        flat_boxes_tensors = [box for boxes_per_image in hand_boxes_list for box in boxes_per_image.tensor]
        box_idx = 0
        
        for instances in instances_list:
            is_hand = instances.gt_classes == self._id_hand if self.training else instances.pred_classes == self._id_hand
            hand_instances = instances[is_hand]

            if len(hand_instances) == 0:
                continue

            for i in range(len(hand_instances)):
                instance = hand_instances[i]
                
                # Determine the source of keypoints
                if self.training and instance.has("gt_keypoints"):
                    kpts = instance.gt_keypoints.tensor
                elif not self.training and instance.has("pred_keypoints"):
                    kpts = instance.pred_keypoints
                else:
                    kpts = torch.zeros(self.keypoint_feature_extractor.num_keypoints,
                                       self.keypoint_feature_extractor.keypoint_dim,
                                       device=self.device)
                
                # --- KEY FIX: Normalize keypoint tensor shape ---
                # Ensure kpts is always 3D [num_keypoints, 3] before appending.
                # It might come as [1, num_keypoints, 3] or even [1, 1, num_keypoints, 3].
                # We squeeze it to remove any singleton dimensions.
                if kpts.dim() > 2:
                    kpts = kpts.squeeze()
                
                # Final check to prevent errors with malformed squeezed tensors
                if kpts.dim() != 2 or kpts.shape[0] != self.keypoint_feature_extractor.num_keypoints:
                    # If shape is still wrong, fall back to zeros.
                    logger.warning(f"Malformed keypoint tensor detected with shape {kpts.shape}. Falling back to zeros.")
                    kpts = torch.zeros(self.keypoint_feature_extractor.num_keypoints,
                                       self.keypoint_feature_extractor.keypoint_dim,
                                       device=self.device)
                # --- END KEY FIX ---

                all_keypoints.append(kpts)
                if box_idx < len(flat_boxes_tensors):
                    all_hand_boxes_flat.append(flat_boxes_tensors[box_idx])
                    box_idx += 1
        
        if not all_keypoints or not all_hand_boxes_flat:
            return torch.empty(0, 128, device=self.device)
        
        # Now, keypoints_batch will reliably be [N, 21, 3]
        keypoints_batch = torch.stack(all_keypoints)
        hand_boxes_batch = torch.stack(all_hand_boxes_flat)

        return self.keypoint_feature_extractor(keypoints_batch, hand_boxes_batch)

    def _prepare_hands_features(self, batched_inputs: List[Dict], proposals_or_results: List[Instances]):
        """
        Centralized feature extraction logic for all modalities.
        Works for both training (with proposals_match) and inference (with results).
        """
        image_width, image_height = batched_inputs[0]['width'], batched_inputs[0]['height']
        instances_list = proposals_or_results
        
        boxes_hands, boxes_hands_padded = [], []
        for i, instances in enumerate(instances_list):
            is_hand = instances.gt_classes == self._id_hand if self.training else instances.pred_classes == self._id_hand
            hand_instances = instances[is_hand]
            
            if len(hand_instances) > 0:
                if self.training:
                    current_boxes = hand_instances.proposal_boxes if hand_instances.has("proposal_boxes") else hand_instances.gt_boxes
                else:
                    current_boxes = hand_instances.pred_boxes
                boxes_hands.append(current_boxes)
                padded_tensor = expand_box(current_boxes.tensor.clone(), image_width, image_height, ratio=self._expand_hand_box_ratio)
                boxes_hands_padded.append(Boxes(padded_tensor))
            else:
                boxes_hands.append(Boxes(torch.empty(0, 4, device=self.device)))
                boxes_hands_padded.append(Boxes(torch.empty(0, 4, device=self.device)))

        num_hands = sum(len(b) for b in boxes_hands)
        if num_hands > 0:
            rois_rgb = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], boxes_hands)
            self._c_hands_features = self.roi_heads.box_head(rois_rgb)
            rois_rgb_padded = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], boxes_hands_padded)
            self._c_hands_features_padded = self.roi_heads.box_head(rois_rgb_padded)
        else:
            self._c_hands_features = self._c_hands_features_padded = torch.empty(0, 1024, device=self.device)

        if self._contact_state_modality != "rgb" and num_hands > 0:
            boxes_padded_depth = []
            for i, padded_box in enumerate(boxes_hands_padded):
                depth_h, depth_w = batched_inputs[i]["image_for_depth_module"].shape[1:3]
                depth_scaled_boxes = padded_box.clone()
                depth_scaled_boxes.scale(scale_x=(depth_w / image_width), scale_y=(depth_h / image_height))
                boxes_padded_depth.append(depth_scaled_boxes)

            self._last_boxes_padded_rescaled = [b.tensor for b in boxes_padded_depth]
            
            rgb_images = torch.stack([torch.from_numpy(b["image_for_depth_module"]) for b in batched_inputs]).to(self.device)
            if not rgb_images.is_floating_point(): rgb_images = rgb_images.float() / 255.0

            modalities_to_cat = [rgb_images]
            if "depth" in self._contact_state_modality and hasattr(self, '_depth_maps_predicted'):
                depths = self._depth_maps_predicted.detach().unsqueeze(1)
                depths = F.interpolate(depths, size=rgb_images.shape[2:], mode='bilinear', align_corners=False)
                modalities_to_cat.append(depths / 255.0)
            
            cnn_input = torch.cat(modalities_to_cat, dim=1)
            batch_indices = torch.cat([torch.full((len(b),), i, device=self.device, dtype=torch.float) for i, b in enumerate(self._last_boxes_padded_rescaled)])
            boxes_with_indices = torch.cat([batch_indices.unsqueeze(1), torch.cat(self._last_boxes_padded_rescaled)], dim=1)
            c_roi = self._roi_align_cnn_contact_state(cnn_input, boxes_with_indices)
            
            if "mask" in self._contact_state_modality:
                masks = extract_masks_and_resize(instances_list, rgb_images.shape[2:], self._id_hand)
                if masks and any(len(m) > 0 for m in masks):
                    mask_tensor = torch.cat(masks).unsqueeze(1).float()
                    cropped_masks = self._roi_align_cnn_contact_state(mask_tensor, boxes_with_indices)
                    c_roi = torch.cat((c_roi, cropped_masks), dim=1)
                else:
                    c_roi = torch.cat((c_roi, torch.zeros_like(c_roi[:, :1, :, :])), dim=1)
            self._c_hands_features_cnn = c_roi
        else:
            self._c_hands_features_cnn = None

        self._c_hands_keypoint_features = self._extract_and_process_keypoints(instances_list, boxes_hands)

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)
        self._last_extracted_features["rgb"] = features
        
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        
        proposals_match, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        if self.roi_heads.mask_on:
            proposals_for_mask = [p for p in proposals_match if len(p) > 0]
            if len(proposals_for_mask) > 0:
                mask_features = self.roi_heads.mask_pooler(
                    [features[f] for f in self.roi_heads.mask_in_features],
                    [p.proposal_boxes for p in proposals_for_mask]
                )
                mask_logits = self.roi_heads.mask_head(mask_features)
                proposals_match = mask_rcnn_inference_in_training(mask_logits, proposals_for_mask)

        self._prepare_gt_labels(proposals_match)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        if self._use_depth_module:
            self._last_extracted_features["depth"], self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
            if "depth_gt" in batched_inputs[0]:
                gt_depth_maps = torch.stack([torch.from_numpy(e["depth_gt"]) for e in batched_inputs]).to(self.device)
                loss_depth = self._loss_depth_f(self._depth_maps_predicted, gt_depth_maps)
                losses['loss_depth'] = loss_depth
        
        self._prepare_hands_features(batched_inputs, proposals_match)
        
        if len(self._c_gt_hands_lr) > 0:
            _, loss_hand_lr = self.classification_hand_lr(self._c_hands_features, self._c_gt_hands_lr)
            losses['loss_classification_hand_lr'] = loss_hand_lr

            _, loss_contact_state_dict = self.classification_contact_state(
                self._c_hands_features_padded,
                self._c_hands_features_cnn,
                self._c_hands_keypoint_features,
                self._c_gt_hands_contact_state
            )
            losses.update(loss_contact_state_dict)

            contact_indices = [i for i, x in enumerate(self._c_gt_hands_contact_state) if x == 1]
            if contact_indices:
                gt_vectors = np.array(self._c_gt_hands_dxdymagnitude)[contact_indices]
                if self._c_hands_features_padded[contact_indices].numel() > 0:
                    _, loss_vector_reg = self.association_vector_regressor(self._c_hands_features_padded[contact_indices], gt_vectors)
                    losses['loss_regression_dxdymagn'] = loss_vector_reg
                else:
                    losses['loss_regression_dxdymagn'] = torch.tensor(0.0, device=self.device)
            else:
                losses['loss_regression_dxdymagn'] = torch.tensor(0.0, device=self.device)
        else:
            losses['loss_classification_hand_lr'] = torch.tensor(0.0, device=self.device)
            if 'loss_cs_fusion' not in losses: losses['loss_cs_fusion'] = sum(p.sum() for p in self.classification_contact_state.parameters()) * 0.0
            losses['loss_regression_dxdymagn'] = torch.tensor(0.0, device=self.device)
        
        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True):
        assert not self.training
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        self._last_extracted_features["rgb"] = features
        
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        instances = results[0]["instances"]
        
        if self._use_depth_module: 
            self._last_extracted_features["depth"], self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
            results[0]['depth_map'] = self._depth_maps_predicted
        
        self._prepare_hands_features(batched_inputs, [instances])
        
        instances_hands = instances[instances.pred_classes == self._id_hand]
        results[0]['additional_outputs'] = Instances(instances.image_size)

        if len(instances_hands) > 0:
            side_logits = self.classification_hand_lr(self._c_hands_features)
            contact_scores = self.classification_contact_state(
                self._c_hands_features_padded,
                self._c_hands_features_cnn,
                self._c_hands_keypoint_features
            )
            dxdymagn_vectors = self.association_vector_regressor(self._c_hands_features_padded)
            
            pred_sides = torch.round(torch.sigmoid(side_logits)).int()
            pred_contact_states = torch.round(contact_scores).int()

            results[0]['additional_outputs'].set("boxes", instances_hands.pred_boxes.tensor.detach().cpu())
            results[0]['additional_outputs'].set("sides", pred_sides.detach().cpu())
            results[0]['additional_outputs'].set("scores", instances_hands.scores.detach().cpu())
            results[0]['additional_outputs'].set("contact_states", pred_contact_states.detach().cpu())
            results[0]['additional_outputs'].set("dxdymagn_hand", dxdymagn_vectors.detach().cpu())
        
        self._last_instances_hands = instances_hands
        return results

    def _prepare_gt_labels(self, proposals_match: List[Instances]):
        self._c_gt_hands_lr, self._c_gt_hands_contact_state, self._c_gt_hands_dxdymagnitude = [], [], []
        for batch_proposal in proposals_match:
            hand_proposal_mask = batch_proposal.gt_classes == self._id_hand
            if not hand_proposal_mask.any():
                continue
                
            batch_proposal_hands = batch_proposal[hand_proposal_mask]
            for idx_proposal in range(len(batch_proposal_hands)):
                if batch_proposal_hands.has("gt_sides"):
                    self._c_gt_hands_lr.append(batch_proposal_hands[idx_proposal].gt_sides.item())
                if batch_proposal_hands.has("gt_contact_states"):
                    self._c_gt_hands_contact_state.append(batch_proposal_hands[idx_proposal].gt_contact_states.item())
                if batch_proposal_hands.has("gt_dxdymagn_hands"):
                    self._c_gt_hands_dxdymagnitude.append(batch_proposal_hands[idx_proposal].gt_dxdymagn_hands.detach().cpu().numpy()[0])