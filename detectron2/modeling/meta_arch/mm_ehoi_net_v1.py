import logging
import torch
import torchvision
from typing import Dict, List, Optional
import numpy as np
from detectron2.structures import Instances
from detectron2.structures.boxes import Boxes

# Import da Detectron2 per le operazioni delle ROI Head
from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_loss, keypoint_rcnn_inference
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_loss, mask_rcnn_inference

# Import specifici del progetto
from .rcnn import GeneralizedRCNN
from .ehoi_net import EhoiNet
from .build import META_ARCH_REGISTRY
from .additional_modules import AssociationVectorRegressor
from detectron2.utils.custom_utils import expand_box

__all__ = ["MMEhoiNetv1"]
logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class MMEhoiNetv1(EhoiNet):
    """
    Multimodal Egocentric Hand-Object Interaction Network (Version 1).

    This class provides the concrete implementation for the EhoiNet architecture.
    It orchestrates the forward pass for both training and inference, managing
    the flow of data through the standard detector components (backbone, RPN, ROI heads)
    and the custom, task-specific heads for EHOI (contact state, hand side, etc.).
    """

    def __init__(self, cfg, metadata):
        """
        Initializes the MMEhoiNetv1 model.

        Args:
            cfg (CfgNode): The configuration object.
            metadata (Metadata): The metadata for the dataset.
        """
        super().__init__(cfg, metadata)

        # Ratio for expanding hand bounding boxes to provide more context to auxiliary heads.
        self._expand_hand_box_ratio = cfg.ADDITIONAL_MODULES.EXPAND_HAND_BOX_RATIO

        # Custom module for regressing the hand-object association vector.
        self.association_vector_regressor = AssociationVectorRegressor(cfg)

        # RoIAlign operator configured for the early fusion module.
        # This will be used to extract fixed-size patches from various modalities.
        fusion_input_size = cfg.ADDITIONAL_MODULES.get("CONTACT_STATE_CNN_INPUT_SIZE", 128)
        self._roi_align_for_fusion = torchvision.ops.RoIAlign(
            output_size=(fusion_input_size, fusion_input_size), 
            spatial_scale=1.0, 
            sampling_ratio=-1,
            aligned=True
        )

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Defines the forward pass of the model.

        - During training, it computes losses from all branches (detection, mask, keypoint,
          and custom EHOI tasks) and returns them in a dictionary.
        - During inference, it routes the input to the `inference` method.
        
        Args:
            batched_inputs (list[dict]): A list of dicts, each representing an image
                and its annotations.

        Returns:
            dict[str, Tensor]: A dictionary of loss tensors during training.
            list[dict]: A list of dicts with predictions during inference.
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        # --- Section 1: Standard Detection Pipeline ---
        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # `proposals_match` contains proposals that have been matched to ground truth.
        proposals_match, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        # --- Section 2: Hand-Specific Head Processing (Training) ---
        # Filter proposals that have been matched to a "hand" ground truth.
        proposals_with_gt_hands = [p[p.gt_classes == self._id_hand] for p in proposals_match]
        num_hands = sum(len(p) for p in proposals_with_gt_hands)

        if num_hands > 0:
            # Filter out batches with no hands to simplify downstream processing.
            proposals_hands_in_batch = [p for p in proposals_with_gt_hands if len(p) > 0]

            # --- Mask Head ---
            if self._use_mask and proposals_hands_in_batch:
                features_mask = self.mask_pooler(
                    [features[f] for f in self.mask_in_features],
                    [p.proposal_boxes for p in proposals_hands_in_batch]
                )
                mask_logits = self.mask_head.layers(features_mask)
                loss_mask = mask_rcnn_loss(mask_logits, proposals_hands_in_batch)
                losses.update({"loss_mask": loss_mask * self.mask_head.loss_weight})
                
                # Generate soft masks for the fusion module.
                class_indices = torch.cat([p.gt_classes for p in proposals_hands_in_batch])
                indices = torch.arange(mask_logits.shape[0], device=mask_logits.device)
                mask_logits_per_instance = mask_logits[indices, class_indices]
                self._predicted_masks_for_fusion = mask_logits_per_instance.sigmoid().unsqueeze(1)
            
            # --- Keypoint Head ---
            if self._use_keypoints and proposals_hands_in_batch:
                features_kpt = self.keypoint_pooler(
                    [features[f] for f in self.keypoint_in_features], 
                    [p.proposal_boxes for p in proposals_hands_in_batch]
                )
                
                # The keypoint head's forward pass computes and returns the loss dictionary.
                loss_dict_kpt = self.keypoint_head(features_kpt, proposals_hands_in_batch)
                losses.update(loss_dict_kpt)
                
                # To get predictions for the fusion module, we must run inference logic separately.
                with torch.no_grad():
                    kpt_logits = self.keypoint_head.layers(features_kpt)
                    # The inference function requires `.pred_boxes`; we provide `.proposal_boxes`.
                    for p in proposals_hands_in_batch:
                        p.pred_boxes = p.proposal_boxes
                    # This function modifies the `proposals_hands_in_batch` list in-place.
                    keypoint_rcnn_inference(kpt_logits, proposals_hands_in_batch)
                    self._predicted_keypoints_for_fusion = proposals_hands_in_batch
            
            # --- Depth Module ---
            if self._use_depth_module:
                _, self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
                if "depth_gt" in batched_inputs[0]:
                    gt_depth_maps_list = [torch.from_numpy(e["depth_gt"]) for e in batched_inputs]
                    gt_depth_maps = torch.stack(gt_depth_maps_list).to(self.device).float()
                    
                    # Resize prediction to match ground truth dimensions for loss calculation.
                    target_h, target_w = gt_depth_maps.shape[-2:]
                    prediction_resized = torch.nn.functional.interpolate(
                        self._depth_maps_predicted.unsqueeze(1),
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    # Loss functions might expect [B, H, W] instead of [B, 1, H, W]
                    if gt_depth_maps.dim() == 3:
                        prediction_resized = prediction_resized.squeeze(1)
                    
                    losses["loss_depth"] = self.depth_loss_fn(prediction_resized, gt_depth_maps)

            # --- Section 3: Custom EHOI Task Processing ---
            # Prepare feature vectors and multimodal patches for the final heads.
            self._prepare_gt_labels(proposals_match)
            self._prepare_hands_features(batched_inputs, features, proposals_with_gt_hands)
            
            # --- Loss Calculation for Custom Modules ---
            if self._c_hands_features_padded.numel() > 0:
                if hasattr(self, 'classification_glove') and self._c_gt_gloves:
                    _, loss_glove = self.classification_glove(self._c_hands_features_padded, self._c_gt_gloves)
                    losses['loss_glove'] = loss_glove

                _, loss_lr = self.classification_hand_lr(self._c_hands_features, self._c_gt_hands_lr)
                losses['loss_classification_hand_lr'] = loss_lr
                
                # Association Vector Loss (only for hands in contact)
                indexes_contact = [i for i, x in enumerate(self._c_gt_hands_contact_state) if x == 1]
                if len(indexes_contact) > 0:
                    gt_vectors = np.array(self._c_gt_hands_dxdymagnitude)[indexes_contact]
                    features_in_contact = self._c_hands_features_padded[indexes_contact]
                    if features_in_contact.numel() > 0:
                        _, loss_regression_vector = self.association_vector_regressor(features_in_contact, gt_vectors)
                        losses['loss_regression_dxdymagn'] = loss_regression_vector

            # Contact State Loss (either RGB-based or Multimodal Fusion)
            if self._contact_state_modality == "rgb":
                if self._c_hands_features_padded.numel() > 0:
                    _, loss_cs = self.classification_contact_state(self._c_hands_features_padded, self._c_gt_hands_contact_state)
                    losses.update(loss_cs if isinstance(loss_cs, dict) else {'loss_classification_contact_state': loss_cs})
            else: # All other modalities use the fusion module
                if self._c_hands_features_cnn.numel() > 0:
                    _, loss_cs = self.classification_contact_state(self._c_hands_features_cnn, self._c_gt_hands_contact_state)
                    losses.update(loss_cs)

        return losses

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], do_postprocess: bool = True):
        """
        Defines the inference pass of the model.
        
        It runs the full detection pipeline and then applies all custom EHOI heads
        on the detected hand instances, enriching them with additional predictions.
        """
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        
        # --- Run Base Detector ---
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)
        
        if do_postprocess:
            results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        # --- Run Auxiliary Heads on Detected Instances ---
        final_result = results[0]
        instances = final_result["instances"]
        hand_instances = instances[instances.pred_classes == self._id_hand]

        if len(hand_instances) > 0:
            # Run Mask and Keypoint heads if enabled to get `.pred_masks` and `.pred_keypoints`.
            if self._use_mask:
                features_mask = self.mask_pooler([features[f] for f in self.mask_in_features], [hand_instances.pred_boxes])
                mask_logits = self.mask_head.layers(features_mask)
                mask_rcnn_inference(mask_logits, [hand_instances])

            if self._use_keypoints:
                features_kpt = self.keypoint_pooler([features[f] for f in self.keypoint_in_features], [hand_instances.pred_boxes])
                kpt_logits = self.keypoint_head.layers(features_kpt)
                keypoint_rcnn_inference(kpt_logits, [hand_instances])

            # Run Depth module if enabled.
            if self._use_depth_module:
                _, self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
                final_result['depth_map'] = self._depth_maps_predicted

            # --- Prepare Features and Run Custom EHOI Classifiers ---
            self._prepare_hands_features(batched_inputs, features, [hand_instances])

            # Add custom predictions to the final `hand_instances` object.
            if self._c_hands_features.numel() > 0:
                hand_instances.sides = torch.round(torch.sigmoid(self.classification_hand_lr(self._c_hands_features)))
            if self._c_hands_features_padded.numel() > 0:
                if hasattr(self, 'classification_glove'):
                    hand_instances.gloves = torch.round(torch.sigmoid(self.classification_glove(self._c_hands_features_padded)))
                hand_instances.dxdymagn_hand = self.association_vector_regressor(self._c_hands_features_padded)
            
            if self._contact_state_modality == "rgb":
                if self._c_hands_features_padded.numel() > 0:
                    contact_scores = torch.sigmoid(self.classification_contact_state(self._c_hands_features_padded))
                else:
                    contact_scores = torch.zeros(len(hand_instances), 1, device=self.device)
            else: # Fusion mode
                if self._c_hands_features_cnn.numel() > 0:
                    contact_scores = torch.sigmoid(self.classification_contact_state(self._c_hands_features_cnn))
                else:
                    contact_scores = torch.zeros(len(hand_instances), 1, device=self.device)
            hand_instances.contact_states = torch.round(contact_scores)
            
            # Ensure standard fields expected by the evaluator are present.
            hand_instances.set("boxes", hand_instances.pred_boxes)
            hand_instances.set("scores", hand_instances.scores)
            hand_instances.set("pred_classes", hand_instances.pred_classes)
            
            final_result['additional_outputs'] = hand_instances
        else:
            # If no hands are detected, return an empty Instances object.
            final_result['additional_outputs'] = Instances(images.image_sizes[0])

        return [final_result]
    
    def _prepare_gt_labels(self, proposals_match: List[Instances]):
        """
        Extracts and stores ground-truth labels for custom EHOI tasks from
        the matched proposals. This is done once per batch.
        """
        # Clear lists from the previous batch.
        self._c_gt_hands_lr, self._c_gt_hands_contact_state = [], []
        self._c_gt_hands_dxdymagnitude, self._c_gt_gloves = [], []
        
        for batch_proposal in proposals_match:
            hands = batch_proposal[batch_proposal.gt_classes == self._id_hand]
            for i in range(len(hands)):
                hand_proposal = hands[i]
                self._c_gt_hands_lr.append(hand_proposal.gt_sides.item())
                self._c_gt_hands_contact_state.append(hand_proposal.gt_contact_states.item())
                self._c_gt_hands_dxdymagnitude.append(hand_proposal.gt_dxdymagn_hands.cpu().numpy()[0])
                self._c_gt_gloves.append(hand_proposal.gt_gloves.item() if hasattr(hand_proposal, "gt_gloves") else 0)

    def _prepare_hands_features(self, batched_inputs, features, hand_proposals_or_instances):
        """
        Prepares all necessary feature representations for the hand-specific heads.
        This includes standard RoI-pooled feature vectors and multi-channel image patches
        for the early fusion module. This method is agnostic to training/inference mode.
        """
        # Filter out images in the batch that contain no hands.
        proposals_per_image = [p for p in hand_proposals_or_instances if len(p) > 0]
        if not proposals_per_image:
            # If no hands are present in the entire batch, initialize features as empty and return.
            self._c_hands_features = self._c_hands_features_padded = self._c_hands_features_cnn = torch.empty(0, device=self.device)
            return

        image_size = self.preprocess_image(batched_inputs).image_sizes[0]
        # Use `.proposal_boxes` in training and `.pred_boxes` in inference.
        box_field = "proposal_boxes" if self.training else "pred_boxes"

        # --- Part 1: Prepare 1D Feature Vectors ---
        # Standard RoI features for simple MLP-based heads.
        all_boxes = [p.get(box_field) for p in proposals_per_image]
        rois = self.roi_heads.box_pooler([features[f] for f in self.roi_heads.in_features], all_boxes)
        self._c_hands_features = self.roi_heads.box_head(rois)

        # Expanded RoI features to provide more spatial context.
        all_expanded_boxes = [Boxes(expand_box(p.get(box_field).tensor, image_size[1], image_size[0], ratio=self._expand_hand_box_ratio)) for p in proposals_per_image]
        rois_padded = self.roi_heads.box_pooler([features[f] for f in self.roi_heads.in_features], all_expanded_boxes)
        self._c_hands_features_padded = self.roi_heads.box_head(rois_padded)

        # If only RGB modality is used, no further processing is needed.
        if self._contact_state_modality == "rgb":
            return

        # --- Part 2: Prepare Multi-channel Patches for Early Fusion ---
        fusion_channels = []
        
        # Prepare boxes for RoIAlign. This requires careful handling of batch indices.
        boxes_for_align_list = []
        image_indices_with_hands = [i for i, p in enumerate(hand_proposals_or_instances) if len(p) > 0]
        for i, p_list in enumerate(proposals_per_image):
            original_batch_idx = image_indices_with_hands[i]
            target_h, target_w = batched_inputs[original_batch_idx]["image"].shape[1:3]
            
            scaled_boxes = all_expanded_boxes[i].clone()
            scaled_boxes.scale(target_w / image_size[1], target_h / image_size[0])
            
            # `RoIAlign` expects boxes in [batch_idx, x1, y1, x2, y2] format.
            batch_idx_tensor = torch.full((len(scaled_boxes), 1), i, device=self.device)
            boxes_for_align_list.append(torch.cat([batch_idx_tensor, scaled_boxes.tensor], dim=1))
        
        boxes_for_align = torch.cat(boxes_for_align_list, dim=0)
        images_with_hands = torch.stack([batched_inputs[i]["image"] for i in image_indices_with_hands]).float().to(self.device)
        
        # --- Channel: RGB ---
        if 'rgb' in self._contact_state_modality:
            rgb_patches = self._roi_align_for_fusion(images_with_hands, boxes_for_align)
            fusion_channels.append(rgb_patches / 255.0)

        # --- Channel: Depth ---
        if 'depth' in self._contact_state_modality:
            depth_maps_with_hands = self._depth_maps_predicted[image_indices_with_hands]
            target_h, target_w = images_with_hands.shape[-2:]
            depth_maps_resized = torch.nn.functional.interpolate(depth_maps_with_hands.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False)
            depth_patches = self._roi_align_for_fusion(depth_maps_resized, boxes_for_align)
            fusion_channels.append(depth_patches / 255.0)

        # --- Channel: Mask ---
        if 'mask' in self._contact_state_modality:
            if self.training:
                all_masks = self._predicted_masks_for_fusion
            else: # Inference
                all_masks = torch.cat([p.pred_masks for p in proposals_per_image])
            
            if all_masks.dim() == 3: all_masks = all_masks.unsqueeze(1)
            target_size = self._roi_align_for_fusion.output_size
            resized_masks = torch.nn.functional.interpolate(all_masks.float(), size=target_size, mode='bilinear', align_corners=False)
            fusion_channels.append(resized_masks)

        # --- Channel: Keypoints ---
        if 'keypoints' in self._contact_state_modality:
            if self.training:
                all_kpts = torch.cat([p.pred_keypoints for p in self._predicted_keypoints_for_fusion])
            else: # Inference
                all_kpts = torch.cat([p.pred_keypoints for p in proposals_per_image])

            all_boxes_tensor = torch.cat([b.tensor for b in all_expanded_boxes], dim=0)
            keypoint_heatmaps = self.keypoint_renderer(all_kpts, all_boxes_tensor)
            fusion_channels.append(keypoint_heatmaps)

        # --- Final Assembly ---
        # Concatenate all prepared channels along the channel dimension.
        if fusion_channels:
            self._c_hands_features_cnn = torch.cat(fusion_channels, dim=1)
        else:
            self._c_hands_features_cnn = torch.empty(0, device=self.device)