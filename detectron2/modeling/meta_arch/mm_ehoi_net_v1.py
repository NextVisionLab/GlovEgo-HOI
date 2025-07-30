import logging
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference_in_training
from detectron2.structures import Instances
from detectron2.structures.boxes import Boxes
from detectron2.utils.custom_utils import expand_box, extract_masks_and_resize

from .additional_modules import *
from .build import META_ARCH_REGISTRY
from .ehoi_net import EhoiNet
from .rcnn import GeneralizedRCNN

__all__ = ["MMEhoiNetv1"]
logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class MMEhoiNetv1(EhoiNet):
    """
    Multi-Modal Egocentric Human-Object Interaction Network (MMEhoiNetv1).

    This model extends a standard Faster R-CNN with additional heads to predict
    various attributes of detected hands, such as side (left/right), contact state,
    glove presence, and an association vector for interacting objects. It supports
    multi-modal inputs (RGB, Depth, Masks, Keypoints) for enhanced prediction accuracy.
    """

    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)

        # --- Initialize Custom Prediction Heads ---
        self._expand_hand_box_ratio = cfg.ADDITIONAL_MODULES.EXPAND_HAND_BOX_RATIO
        self.association_vector_regressor = AssociationVectorRegressor(cfg)
        self.keypoint_feature_extractor = KeypointFeatureExtractor(cfg)
        self.classification_contact_state = ContactStateFusionClassificationModule(cfg)
        self.gloves_classifier = GlovesClassificationModule(cfg)

        # --- Initialize RoIAligner for the Multi-Modal CNN Branch ---
        input_size_cnn_contact_state = cfg.ADDITIONAL_MODULES.get('CONTACT_STATE_CNN_INPUT_SIZE', 128)
        self._roi_align_cnn_contact_state = torchvision.ops.RoIAlign(
            (input_size_cnn_contact_state, input_size_cnn_contact_state),
            spatial_scale=1.0,
            sampling_ratio=-1
        )

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Defines the computation performed at every training step.

        Args:
            batched_inputs (list[dict]): A list of dicts, each dict corresponds to one image.

        Returns:
            dict[str: Tensor]: A dictionary of loss components.
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        # 1. Standard R-CNN Forward Pass
        features = self.backbone(images.tensor)
        self._last_extracted_features["rgb"] = features
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        proposals_match, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        # (Optional) Perform mask inference during training to get mask predictions for custom modules
        if self.roi_heads.mask_on:
            proposals_for_mask = [p for p in proposals_match if len(p) > 0]
            if len(proposals_for_mask) > 0:
                proposals_match = self.roi_heads.forward_with_given_boxes(features, proposals_for_mask)

        # 2. Compute Optional Auxiliary Losses (e.g., Depth Estimation)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self._use_depth_module:
            _, self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
            if "depth_gt" in batched_inputs[0]:
                gt_depth_maps = torch.stack([torch.from_numpy(e["depth_gt"]) for e in batched_inputs]).to(self.device)
                losses['loss_depth'] = self._loss_depth_f(self._depth_maps_predicted, gt_depth_maps)

        # 3. Centralized Feature Preparation for all Custom Heads
        self._prepare_hands_features(batched_inputs, proposals_match, images)

        # 4. Compute and Collect Losses from all Custom Heads
        if len(self._c_gt_hands_lr) > 0:
            losses['loss_classification_hand_lr'] = self.classification_hand_lr(self._c_hands_features, self._c_gt_hands_lr)[1]
            losses['loss_gloves'] = self.gloves_classifier(self._c_hands_features, self._c_gt_hands_gloves)[1]

            _, loss_contact_state_dict = self.classification_contact_state(
                self._c_hands_features_padded, self._c_hands_features_cnn,
                self._c_hands_keypoint_features, self._c_gt_hands_contact_state
            )
            losses.update(loss_contact_state_dict)

            # Association vector loss is computed only for hands in a "contact" state
            contact_indices = [i for i, x in enumerate(self._c_gt_hands_contact_state) if x == 1]
            if contact_indices and self._c_hands_features_padded.numel() > 0:
                gt_vectors = np.array(self._c_gt_hands_dxdymagnitude)[contact_indices]
                if self._c_hands_features_padded[contact_indices].numel() > 0:
                    losses['loss_regression_dxdymagn'] = self.association_vector_regressor(self._c_hands_features_padded[contact_indices], gt_vectors)[1]
            if 'loss_regression_dxdymagn' not in losses:
                losses['loss_regression_dxdymagn'] = torch.tensor(0.0, device=self.device)
        else:
            # 5. Handle Edge Case: No hands in batch (important for DDP stability)
            losses['loss_classification_hand_lr'] = torch.tensor(0.0, device=self.device)
            losses['loss_gloves'] = torch.tensor(0.0, device=self.device)
            losses['loss_cs_total'] = sum(p.sum() for p in self.classification_contact_state.parameters()) * 0.0
            losses['loss_regression_dxdymagn'] = torch.tensor(0.0, device=self.device)

        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], do_postprocess: bool = True):
        """
        Run inference on a single batch of images.
        """
        assert not self.training
        
        # 1. Standard R-CNN Inference
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        self._last_extracted_features["rgb"] = features
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        result = results[0]
        instances = result["instances"]
        
        # 2. Optional Auxiliary Predictions (e.g., Depth Map)
        if self._use_depth_module: 
            _, self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
            result['depth_map'] = self._depth_maps_predicted
        
        # 3. Centralized Feature Preparation for Custom Heads
        self._prepare_hands_features(batched_inputs, [instances], images)
        
        # 4. Run Custom Heads to get Predictions
        instances_hands = instances[instances.pred_classes == self._id_hand]
        result['additional_outputs'] = Instances(instances.image_size)

        if len(instances_hands) > 0:
            # Get raw predictions from all heads in parallel
            side_logits = self.classification_hand_lr(self._c_hands_features)
            contact_scores = self.classification_contact_state(
                self._c_hands_features_padded, self._c_hands_features_cnn, self._c_hands_keypoint_features
            )
            dxdymagn_vectors = self.association_vector_regressor(self._c_hands_features_padded)
            gloves_logits = self.gloves_classifier(self._c_hands_features)
            
            # 5. Post-process Predictions and Package into `additional_outputs`
            pred_sides = torch.round(torch.sigmoid(side_logits)).int()
            pred_contact_states = torch.round(contact_scores).int()
            pred_gloves = torch.round(torch.sigmoid(gloves_logits)).int()

            result['additional_outputs'].set("boxes", instances_hands.pred_boxes.tensor.detach().cpu())
            result['additional_outputs'].set("sides", pred_sides.detach().cpu())
            result['additional_outputs'].set("scores", instances_hands.scores.detach().cpu())
            result['additional_outputs'].set("contact_states", pred_contact_states.detach().cpu())
            result['additional_outputs'].set("dxdymagn_hand", dxdymagn_vectors.detach().cpu())
            result['additional_outputs'].set("gloves", pred_gloves.detach().cpu())
        
        return [result]

    def _prepare_hands_features(self, batched_inputs: List[Dict], proposals_or_results: List[Instances], images):
        """
        Extracts features and ground truth labels for all hand-specific heads.
        This is the central feature preparation hub before the final prediction heads.
        """
        # --- Initialize lists for batch-wide GT labels and features ---
        self._c_gt_hands_lr, self._c_gt_hands_contact_state, self._c_gt_hands_dxdymagnitude = [], [], []
        self._c_gt_hands_gloves = []
        all_hand_instances_flat = []
        per_image_hand_boxes_tensors = [[] for _ in range(len(proposals_or_results))]
        per_image_hand_boxes_padded_tensors = [[] for _ in range(len(proposals_or_results))]
        
        # --- Step 1: Collect GT Labels and Hand Boxes per Image ---
        for i, instances in enumerate(proposals_or_results):
            image_height, image_width = images.image_sizes[i] 
            is_hand_mask = instances.gt_classes == self._id_hand if self.training else instances.pred_classes == self._id_hand
            hand_instances = instances[is_hand_mask]
            
            if len(hand_instances) == 0:
                continue
            all_hand_instances_flat.append(hand_instances)

            for j in range(len(hand_instances)):
                hand_inst = hand_instances[j]
                if self.training:
                    self._c_gt_hands_lr.append(hand_inst.gt_sides.item() if hand_inst.has("gt_sides") else 0)
                    self._c_gt_hands_contact_state.append(hand_inst.gt_contact_states.item() if hand_inst.has("gt_contact_states") else 0)
                    dxdymagn_gt = hand_inst.gt_dxdymagn_hands.cpu().numpy()[0] if hand_inst.has("gt_dxdymagn_hands") else [0.0, 0.0, 0.0]
                    self._c_gt_hands_dxdymagnitude.append(dxdymagn_gt)
                    self._c_gt_hands_gloves.append(hand_inst.gt_gloves.item() if hand_inst.has("gt_gloves") else 0)

                current_boxes = hand_inst.proposal_boxes if self.training and hand_inst.has("proposal_boxes") else hand_inst.gt_boxes if self.training else hand_inst.pred_boxes
                per_image_hand_boxes_tensors[i].append(current_boxes.tensor)
                padded_tensor = expand_box(current_boxes.tensor.clone(), image_width, image_height, ratio=self._expand_hand_box_ratio)
                per_image_hand_boxes_padded_tensors[i].append(padded_tensor)

        pooled_boxes_tensors = [torch.cat(boxes) if len(boxes) > 0 else torch.empty(0, 4, device=self.device) for boxes in per_image_hand_boxes_tensors]
        pooled_boxes_padded_tensors = [torch.cat(boxes) if len(boxes) > 0 else torch.empty(0, 4, device=self.device) for boxes in per_image_hand_boxes_padded_tensors]
        num_hands = sum(len(b) for b in pooled_boxes_tensors)
        
        # --- Step 2: RGB Feature Extraction (Hand Feature Vector) ---
        if num_hands > 0:
            hand_boxes_for_pooler = [Boxes(t) for t in pooled_boxes_tensors]
            hand_boxes_padded_for_pooler = [Boxes(t) for t in pooled_boxes_padded_tensors]
            
            rois_rgb = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], hand_boxes_for_pooler)
            self._c_hands_features = self.roi_heads.box_head(rois_rgb)
            
            rois_rgb_padded = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.in_features], hand_boxes_padded_for_pooler)
            self._c_hands_features_padded = self.roi_heads.box_head(rois_rgb_padded)
        else:
            output_channels = self.roi_heads.box_head.output_shape.channels
            self._c_hands_features = self._c_hands_features_padded = torch.empty(0, output_channels, device=self.device)

        # --- Step 3: Multimodal CNN Feature Extraction (for Depth/Masks) ---
        if num_hands > 0 and self._contact_state_modality != "rgb":
            image_hand_counts = [len(boxes) for boxes in pooled_boxes_padded_tensors]
            flat_padded_boxes = torch.cat(pooled_boxes_padded_tensors)
            
            rgb_images_for_cnn = torch.stack([torch.from_numpy(b["image_for_depth_module"]) for b in batched_inputs]).to(self.device)
            if not rgb_images_for_cnn.is_floating_point(): 
                rgb_images_for_cnn = rgb_images_for_cnn.float() / 255.0

            modalities_to_cat = [rgb_images_for_cnn]
            if "depth" in self._contact_state_modality and hasattr(self, '_depth_maps_predicted'):
                depths = F.interpolate(self._depth_maps_predicted.detach().unsqueeze(1), size=rgb_images_for_cnn.shape[2:], mode='bilinear', align_corners=False)
                modalities_to_cat.append(depths / 255.0)
            cnn_input = torch.cat(modalities_to_cat, dim=1)
            
            batch_indices = torch.cat([torch.full((count,), i, device=self.device, dtype=torch.float) for i, count in enumerate(image_hand_counts) if count > 0])
            scaled_padded_boxes = flat_padded_boxes.clone()
            for i, (h, w) in enumerate(images.image_sizes):
                img_idx_mask = (batch_indices == i)
                if img_idx_mask.any():
                    cnn_h, cnn_w = rgb_images_for_cnn.shape[2:]
                    Boxes(scaled_padded_boxes[img_idx_mask]).scale(scale_x=(cnn_w / w), scale_y=(cnn_h / h))
            
            boxes_with_indices = torch.cat([batch_indices.unsqueeze(1), scaled_padded_boxes], dim=1)
            c_roi = self._roi_align_cnn_contact_state(cnn_input, boxes_with_indices)
            
            if "mask" in self._contact_state_modality:
                flat_instances = Instances.cat(all_hand_instances_flat)
                masks = extract_masks_and_resize([flat_instances], cnn_input.shape[2:], self._id_hand)
                if masks and any(len(m) > 0 for m in masks):
                    mask_tensor = torch.cat(masks).unsqueeze(1).float()
                    cropped_masks = self._roi_align_cnn_contact_state(mask_tensor, boxes_with_indices)
                    c_roi = torch.cat((c_roi, cropped_masks), dim=1)
                else:
                    c_roi = torch.cat((c_roi, torch.zeros_like(c_roi[:, :1, :, :])), dim=1)
            self._c_hands_features_cnn = c_roi
        else:
            self._c_hands_features_cnn = None

        # --- Step 4: Keypoint Feature Extraction ---
        if num_hands > 0:
            flat_hand_boxes_for_kpt = torch.cat(pooled_boxes_tensors)
            self._c_hands_keypoint_features = self._extract_and_process_keypoints(all_hand_instances_flat, flat_hand_boxes_for_kpt)
        else:
            self._c_hands_keypoint_features = torch.empty(0, 128, device=self.device)

    def _extract_and_process_keypoints(self, instances_list: List[Instances], hand_boxes_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encodes raw keypoints into a fixed-size feature vector.
        Handles missing keypoints and varying tensor shapes gracefully.
        """
        all_keypoints = []
        if not instances_list:
            return torch.empty(0, 128, device=self.device)
        flat_instances = Instances.cat(instances_list)
        
        if len(flat_instances) == 0 or len(hand_boxes_tensor) == 0:
            return torch.empty(0, 128, device=self.device)
        assert len(flat_instances) == len(hand_boxes_tensor), \
            f"Mismatch between number of instances ({len(flat_instances)}) and boxes ({len(hand_boxes_tensor)})"

        for i in range(len(flat_instances)):
            instance = flat_instances[i]
            if self.training and instance.has("gt_keypoints"):
                kpts = instance.gt_keypoints.tensor
            elif not self.training and instance.has("pred_keypoints"):
                kpts = instance.pred_keypoints
            else:
                kpts = torch.zeros(self.keypoint_feature_extractor.num_keypoints, self.keypoint_feature_extractor.keypoint_dim, device=self.device)
            
            if kpts.dim() > 2: kpts = kpts.squeeze(0)
            if kpts.dim() != 2 or kpts.shape[0] != self.keypoint_feature_extractor.num_keypoints:
                kpts = torch.zeros(self.keypoint_feature_extractor.num_keypoints, self.keypoint_feature_extractor.keypoint_dim, device=self.device)
            all_keypoints.append(kpts)
        
        if not all_keypoints:
            return torch.empty(0, 128, device=self.device)
        
        keypoints_batch = torch.stack(all_keypoints)
        return self.keypoint_feature_extractor(keypoints_batch, hand_boxes_tensor)