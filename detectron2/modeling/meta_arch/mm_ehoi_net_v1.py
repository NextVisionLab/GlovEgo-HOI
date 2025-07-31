import logging
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from detectron2.structures import Instances
from detectron2.structures.boxes import Boxes
from detectron2.utils.custom_utils import expand_box

from .additional_modules import (
    AssociationVectorRegressor,
    ContactStateFusionClassificationModule,
    GlovesClassificationModule,
    KeypointRenderer,
)
from .build import META_ARCH_REGISTRY
from .ehoi_net import EhoiNet
from .rcnn import GeneralizedRCNN
from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_loss, keypoint_rcnn_inference
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference_in_training, mask_rcnn_loss

__all__ = ["MMEhoiNetv1"]
logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class MMEhoiNetv1(EhoiNet):
    """
    Multi-Modal EHOI Network using a fused input for contact state prediction.
    - Side, Glove, Offset Vector heads use a standard RoI-pooled RGB feature vector (HFV).
    - Contact State head uses a dedicated CNN on a fused tensor from:
      [RGB (3), Depth (1), Mask (1), Keypoint Heatmap (1)].
    """

    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)

        # --- Parameters ---
        self._expand_hand_box_ratio = cfg.ADDITIONAL_MODULES.EXPAND_HAND_BOX_RATIO
        self.association_vector_regressor = AssociationVectorRegressor(cfg)

        # --- RoIAligner for Cropping Patches ---
        input_size_cnn_contact_state = cfg.ADDITIONAL_MODULES.CONTACT_STATE_CNN_INPUT_SIZE
        self._roi_align_cnn_contact_state = torchvision.ops.RoIAlign(
            (input_size_cnn_contact_state, input_size_cnn_contact_state), 
            spatial_scale=1.0, 
            sampling_ratio=-1, 
            aligned=True
        )

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        
        features = self.backbone(images.tensor)
        self._last_extracted_features["rgb"] = features

        # --- Depth Loss ---
        loss_depth = torch.tensor(0.0, device=self.device)
        if self._use_depth_module:
            _, self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
            
            if "depth_gt" in batched_inputs[0]:
                gt_depth_maps = torch.as_tensor(np.array([e["depth_gt"] for e in batched_inputs])).to(self.device)
                
                # <<< INIZIO BLOCCO MODIFICATO >>>
                ### MODIFICA: Assicuriamo che la predizione abbia 4 dimensioni ###
                # The depth module might return a tensor of shape [N, H, W].
                # F.interpolate requires a 4D input of shape [N, C, H, W].
                predicted_depth_4d = self._depth_maps_predicted.unsqueeze(1) # Adds the channel dimension
                
                # Resize the prediction to match the ground truth size
                predicted_depth_resized = F.interpolate(
                    predicted_depth_4d, 
                    size=gt_depth_maps.shape[1:], # Target size is [H_gt, W_gt]
                    mode='bilinear', 
                    align_corners=False
                )
                
                # The loss function expects [N, H, W], so we remove the channel dimension
                loss_depth = self.depth_loss_fn(predicted_depth_resized.squeeze(1), gt_depth_maps)
                # <<< FINE BLOCCO MODIFICATO >>>

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        proposals_match, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        
        self._prepare_hands_features(batched_inputs, proposals_match)
        
        if not self._c_gt_hands_lr:
            losses = {**detector_losses, **proposal_losses}
            if self._use_depth_module:
                losses['loss_depth'] = loss_depth
            return losses

        # --- Calculate Custom Losses ---
        _, loss_lr = self.classification_hand_lr(self._c_hands_features, self._c_gt_hands_lr)
        _, loss_glove = self.classification_glove(self._c_hands_features, self._c_gt_hands_glove)
        
        if self._contact_state_modality == "rgb":
            _, loss_contact = self.classification_contact_state(self._c_hands_features_padded, self._c_gt_hands_contact_state)
        else:
            _, loss_contact = self.classification_contact_state(self._c_hands_features_cnn, self._c_gt_hands_contact_state)
        
        contact_indices = [i for i, x in enumerate(self._c_gt_hands_contact_state) if x == 1]
        loss_regression_vector = torch.tensor(0.0, device=self.device)
        if contact_indices:
            contact_features = self._c_hands_features_padded[contact_indices]
            gt_vectors = np.array(self._c_gt_hands_dxdymagnitude)[contact_indices]
            if contact_features.numel() > 0:
                _, loss_regression_vector = self.association_vector_regressor(contact_features, gt_vectors)
        
        # --- Aggregate All Losses ---
        total_loss = {}
        total_loss.update(detector_losses)
        total_loss.update(proposal_losses)
        total_loss['loss_hand_lr'] = loss_lr
        total_loss['loss_glove'] = loss_glove
        total_loss['loss_dxdymagn'] = loss_regression_vector
        if self._use_depth_module: 
            total_loss['loss_depth'] = loss_depth
        if isinstance(loss_contact, dict):
            total_loss.update(loss_contact)
        else:
            total_loss['loss_contact_state'] = loss_contact

        return total_loss
    
    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], detected_instances: Optional[List[Instances]] = None, do_postprocess: bool = True):  
        assert not self.training
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        self._last_extracted_features["rgb"] = features
        
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        final_result = results[0]
        all_instances = final_result["instances"]
        
        if self._use_depth_module: 
            _, self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
            final_result['depth_map'] = self._depth_maps_predicted
        
        hand_instances = all_instances[all_instances.pred_classes == self._id_hand]
        if len(hand_instances) > 0:
            # Run all auxiliary heads on the prepared features
            self._prepare_hands_features(batched_inputs, [hand_instances])
            side_logits = self.classification_hand_lr(self._c_hands_features)
            glove_logits = self.classification_glove(self._c_hands_features)
            offset_vectors = self.association_vector_regressor(self._c_hands_features_padded)
            
            if self._contact_state_modality == "rgb":
                contact_logits = self.classification_contact_state(self._c_hands_features_padded)
            else:
                contact_logits = self.classification_contact_state(self._c_hands_features_cnn)

            # Attach predictions to a new Instances object
            additional_outputs = Instances(all_instances.image_size)
            additional_outputs.pred_boxes = hand_instances.pred_boxes
            additional_outputs.scores = hand_instances.scores
            additional_outputs.pred_classes = hand_instances.pred_classes
            additional_outputs.pred_sides = torch.round(torch.sigmoid(side_logits))
            additional_outputs.pred_gloves = torch.round(torch.sigmoid(glove_logits))
            additional_outputs.pred_contact_states = torch.round(torch.sigmoid(contact_logits))
            additional_outputs.pred_dxdymagn = offset_vectors
            
            if hand_instances.has("pred_keypoints"):
                additional_outputs.pred_keypoints = hand_instances.pred_keypoints
        else:
            additional_outputs = Instances(all_instances.image_size)

        final_result['additional_outputs'] = additional_outputs
        return [final_result] 

    def _prepare_hands_features(self, batched_inputs: List[Dict], instances_per_image: List[Instances]):
        """
        Filters hand instances, extracts GT labels (during training), and prepares
        all required feature tensors for both training and inference.
        """
        # --- 1. Filter Hand Instances and Group by Image ---
        hand_instances_by_image = [[] for _ in range(len(instances_per_image))]
        if self.training:
            self._c_gt_hands_lr, self._c_gt_hands_contact_state, self._c_gt_hands_dxdymagnitude, self._c_gt_hands_glove = [], [], [], []

        for i, instances in enumerate(instances_per_image):
            id_field = "gt_classes" if self.training and instances.has("gt_classes") else "pred_classes"
            if not instances.has(id_field): continue

            keep_mask = getattr(instances, id_field) == self._id_hand
            hands_in_image = instances[keep_mask]
            
            for j in range(len(hands_in_image)):
                hand_inst = hands_in_image[j]
                hand_instances_by_image[i].append(hand_inst)
                if self.training:
                    self._c_gt_hands_lr.append(hand_inst.gt_sides.item())
                    self._c_gt_hands_contact_state.append(hand_inst.gt_contact_states.item())
                    self._c_gt_hands_dxdymagnitude.append(hand_inst.gt_dxdymagn_hands.cpu().numpy()[0])
                    self._c_gt_hands_glove.append(hand_inst.gt_gloves.item())

        # Flatten the list for GTs and individual processing later
        hand_instances_flat = [inst for sublist in hand_instances_by_image for inst in sublist]
        if not hand_instances_flat:
            self._c_hands_features = self._c_hands_features_padded = self._c_hands_features_cnn = torch.empty(0, device=self.device)
            return

        # --- 2. Prepare Standard Feature Vectors (HFV) ---
        box_field = "proposal_boxes" if self.training else "pred_boxes"
        
        hand_boxes_per_image = [
            Boxes.cat([inst.get(box_field) for inst in hand_list])
            if hand_list else Boxes(torch.empty(0, 4, device=self.device))
            for hand_list in hand_instances_by_image
        ]

        rois = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.box_in_features], hand_boxes_per_image)
        self._c_hands_features = self.roi_heads.box_head(rois)
        
        image_height, image_width = self.preprocess_image(batched_inputs).image_sizes[0]
        
        hand_boxes_padded_per_image = [
            Boxes.cat([Boxes(expand_box(inst.get(box_field).tensor, image_height, image_width, ratio=self._expand_hand_box_ratio)) for inst in hand_list])
            if hand_list else Boxes(torch.empty(0, 4, device=self.device))
            for hand_list in hand_instances_by_image
        ]
        rois_padded = self.roi_heads.box_pooler([self._last_extracted_features["rgb"][f] for f in self.roi_heads.box_in_features], hand_boxes_padded_per_image)
        self._c_hands_features_padded = self.roi_heads.box_head(rois_padded)


        # --- 3. Prepare Fused Multimodal Patches (if needed) ---
        if self._contact_state_modality != "rgb":
            flat_padded_boxes_list = []
            batch_indices = []
            for i, hand_list in enumerate(hand_instances_by_image):
                if not hand_list: continue
                
                padded_boxes_tensor = hand_boxes_padded_per_image[i].tensor
                flat_padded_boxes_list.append(padded_boxes_tensor)
                batch_indices.extend([i] * len(padded_boxes_tensor))
            
            if not flat_padded_boxes_list:
                self._c_hands_features_cnn = torch.empty(0, device=self.device)
            else:
                flat_padded_boxes_tensor = torch.cat(flat_padded_boxes_list)
                boxes_with_indices = torch.cat([
                    torch.tensor(batch_indices, device=self.device, dtype=torch.float).view(-1, 1),
                    flat_padded_boxes_tensor
                ], dim=1)

                modalities_to_fuse = []
                
                # <<< INIZIO BLOCCO MODIFICATO >>>
                # --- MODIFICA: Aggiungiamo fallback con tensori di zeri ---
            
                # Otteniamo la forma di riferimento dalle impostazioni della RoIAlign
                num_hands = len(flat_padded_boxes_tensor)
                h, w = self._roi_align_cnn_contact_state.output_size
                device = self.device
                
                # RGB Patches
                if 'rgb' in self._contact_state_modality:
                    # Assumiamo che le immagini del batch abbiano la stessa dimensione dopo il pre-processing
                    rgb_images_batch = torch.stack([x["image"].to(self.device).float() / 255.0 for x in batched_inputs])
                    rgb_patches = self._roi_align_cnn_contact_state(rgb_images_batch, boxes_with_indices)
                    modalities_to_fuse.append(rgb_patches)

                # Depth Patches
                if 'depth' in self._contact_state_modality:
                    if hasattr(self, '_depth_maps_predicted') and self._depth_maps_predicted is not None:
                        depth_maps_batch = self._depth_maps_predicted.detach().unsqueeze(1)
                        # Assicuriamoci che 'rgb_images_batch' esista se necessario per il resize
                        if 'rgb_images_batch' not in locals():
                            rgb_images_batch = torch.stack([x["image"].to(self.device) for x in batched_inputs])
                        resized_depths = F.interpolate(depth_maps_batch, size=rgb_images_batch.shape[2:], mode='bilinear', align_corners=False)
                        modalities_to_fuse.append(self._roi_align_cnn_contact_state(resized_depths / 255.0, boxes_with_indices))
                    else:
                        logger.warning("Depth modality requested but '_depth_maps_predicted' not found. Fusing zeros.")
                        modalities_to_fuse.append(torch.zeros(num_hands, 1, h, w, device=device))

                # Mask Patches
                if 'mask' in self._contact_state_modality:
                    mask_field = "pred_masks"
                    if all(inst.has(mask_field) for inst in hand_instances_flat):
                        # Nota: si presume che le pred_masks siano a grandezza di immagine
                        masks = torch.stack([inst.get(mask_field) for inst in hand_instances_flat]).unsqueeze(1).float()
                        # Si esegue RoIAlign sulle maschere per coerenza dimensionale
                        modalities_to_fuse.append(self._roi_align_cnn_contact_state(masks, boxes_with_indices))
                    else:
                        logger.warning(f"Mask modality requested but '{mask_field}' not found for all hands. Fusing zeros.")
                        modalities_to_fuse.append(torch.zeros(num_hands, 1, h, w, device=device))
                
                # Keypoint Heatmap Patches
                if 'keypoints' in self._contact_state_modality:
                    keypoint_field = "pred_keypoints" if not self.training else "gt_keypoints"
                    if all(inst.has(keypoint_field) for inst in hand_instances_flat):
                        kpt_tensor_source = [
                            inst.get(keypoint_field).tensor if self.training and inst.get(keypoint_field) is not None else inst.get(keypoint_field)
                            for inst in hand_instances_flat
                        ]
                        raw_keypoints = torch.cat(kpt_tensor_source)
                        keypoint_heatmaps = self.keypoint_renderer(raw_keypoints, flat_padded_boxes_tensor)
                        modalities_to_fuse.append(keypoint_heatmaps)
                    else:
                        logger.warning(f"Keypoint modality requested but '{keypoint_field}' not found for all hands. Fusing zeros.")
                        # Assumendo che il keypoint renderer produca 1 canale
                        modalities_to_fuse.append(torch.zeros(num_hands, 1, h, w, device=device))
                
                if modalities_to_fuse:
                    self._c_hands_features_cnn = torch.cat(modalities_to_fuse, dim=1)
                else:
                    self._c_hands_features_cnn = torch.empty(0, device=self.device)
                # <<< FINE BLOCCO MODIFICATO >>>