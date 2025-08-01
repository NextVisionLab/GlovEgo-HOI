import logging
import torch
import torchvision
from typing import Dict, List, Optional
import numpy as np
from .rcnn import GeneralizedRCNN

from detectron2.structures import Instances
from detectron2.structures.boxes import Boxes
from detectron2.utils.custom_utils import expand_box
from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_loss, keypoint_rcnn_inference
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference, mask_rcnn_loss

from .ehoi_net import EhoiNet
from .build import META_ARCH_REGISTRY
from ..roi_heads import select_foreground_proposals
from .additional_modules import AssociationVectorRegressor

__all__ = ["MMEhoiNetv1"]
logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class MMEhoiNetv1(EhoiNet):
    def __init__(self, cfg, metadata):
        super().__init__(cfg, metadata)

        ### PARAMS
        self._expand_hand_box_ratio = cfg.ADDITIONAL_MODULES.EXPAND_HAND_BOX_RATIO

        ### ADDITIONAL MODULES
        self.association_vector_regressor = AssociationVectorRegressor(cfg)

        ### ROI ALIGN FOR FUSION MODULE
        input_size_cnn_contact_state = cfg.ADDITIONAL_MODULES.get("CONTACT_STATE_CNN_INPUT_SIZE", 128)
        self._roi_align_for_fusion = torchvision.ops.RoIAlign(
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

        # --- Base Detector ---
        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        proposals_match, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        
        # --- Prepara GT per tutti i task ausiliari ---
        self._prepare_gt_labels(proposals_match)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        
        # --- Filtra solo le proposte di mano per i task specifici ---
        proposals_with_gt_hands = [p[p.gt_classes == self._id_hand] for p in proposals_match]
        num_hands = sum(len(p) for p in proposals_with_gt_hands)

        if num_hands > 0:
            # --- Esegui le Head Ausiliarie SOLO sulle mani ---

            # MASK HEAD (se abilitato)
            if self._use_mask:
                proposals_hands_in_batch = [p for p in proposals_with_gt_hands if len(p) > 0]
                if proposals_hands_in_batch:
                    features_mask = self.mask_pooler(
                        [features[f] for f in self.mask_in_features],
                        [p.proposal_boxes for p in proposals_hands_in_batch]
                    )
                    mask_logits = self.mask_head.layers(features_mask)
                    loss_mask = mask_rcnn_loss(mask_logits, proposals_hands_in_batch)
                    losses.update({"loss_mask": loss_mask * self.mask_head.loss_weight})
                    
                    class_indices = torch.cat([p.gt_classes for p in proposals_hands_in_batch])
                    num_proposals = mask_logits.shape[0]
                    indices = torch.arange(num_proposals, device=mask_logits.device)
                    mask_logits_per_instance = mask_logits[indices, class_indices]
                    soft_masks = mask_logits_per_instance.sigmoid().unsqueeze(1)
                    self._predicted_masks_for_fusion = soft_masks
                else:
                    self._predicted_masks_for_fusion = torch.empty(0)

            # KEYPOINT HEAD (se abilitato)
            if self._use_keypoints:
                proposals_hands_in_batch = [p for p in proposals_with_gt_hands if len(p) > 0]
                if proposals_hands_in_batch:
                    features_kpt = self.keypoint_pooler(
                        [features[f] for f in self.keypoint_in_features], 
                        [p.proposal_boxes for p in proposals_hands_in_batch]
                    )
                    
                    loss_dict_kpt = self.keypoint_head(features_kpt, proposals_hands_in_batch)
                    losses.update(loss_dict_kpt)
                    
                    with torch.no_grad():
                        kpt_logits = self.keypoint_head.layers(features_kpt)
                        for p in proposals_hands_in_batch:
                            p.pred_boxes = p.proposal_boxes
                        keypoint_rcnn_inference(kpt_logits, proposals_hands_in_batch)
                        self._predicted_keypoints_for_fusion = proposals_hands_in_batch
                else:
                    self._predicted_keypoints_for_fusion = []

            # ### INIZIO BLOCCO FIXATO ###
            # DEPTH MODULE (se abilitato)
            if self._use_depth_module:
                _, self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
                
                if "depth_gt" in batched_inputs[0]:
                    # Converti ogni array a tensore PRIMA di usare torch.stack
                    gt_depth_maps_list = [torch.from_numpy(e["depth_gt"]) for e in batched_inputs]
                    gt_depth_maps = torch.stack(gt_depth_maps_list).to(self.device).float() # Aggiunto .float() per sicurezza
                    
                    # --- FIX: Ridimensiona la predizione per farla combaciare con il ground truth ---
                    # Prendi la shape target dal ground truth
                    target_h, target_w = gt_depth_maps.shape[-2:]
                    
                    # Ridimensiona la predizione
                    # Aggiunge una dimensione canale (C) se assente, necessario per interpolate
                    # Il formato atteso da interpolate è (N, C, H, W)
                    if self._depth_maps_predicted.dim() == 3:
                        prediction_to_resize = self._depth_maps_predicted.unsqueeze(1)
                    else:
                        prediction_to_resize = self._depth_maps_predicted

                    prediction_resized = torch.nn.functional.interpolate(
                        prediction_to_resize,
                        size=(target_h, target_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    # Rimuovi la dimensione canale se la loss non la vuole e se l'avevamo aggiunta noi
                    if prediction_to_resize.shape != prediction_resized.shape and gt_depth_maps.dim() == 3:
                        prediction_resized = prediction_resized.squeeze(1)

                    # Calcola la loss usando la predizione ridimensionata
                    if hasattr(self, '_loss_depth_f'):
                        losses["loss_depth"] = self._loss_depth_f(prediction_resized, gt_depth_maps)
                    else:
                        losses["loss_depth"] = self.depth_loss_fn(prediction_resized, gt_depth_maps)
            # ### FINE BLOCCO FIXATO ###

            # --- Prepara tutte le feature per i classificatori finali ---
            self._prepare_hands_features(batched_inputs, features, proposals_with_gt_hands)
            
            # --- Calcola le Loss dei Moduli Custom ---
            # GLOVE CLASSIFICATION LOSS
            if hasattr(self, 'classification_glove') and self._c_gt_gloves:
                _, loss_glove = self.classification_glove(self._c_hands_features_padded, self._c_gt_gloves)
                losses['loss_glove'] = loss_glove

            # SIDE CLASSIFICATION LOSS
            _, loss_lr = self.classification_hand_lr(self._c_hands_features, self._c_gt_hands_lr)
            losses['loss_classification_hand_lr'] = loss_lr

            # CONTACT STATE LOSS
            if self._contact_state_modality == "rgb":
                _, loss_cs = self.classification_contact_state(self._c_hands_features_padded, self._c_gt_hands_contact_state)
            else: # Tutte le altre modalità usano la fusione
                _, loss_cs = self.classification_contact_state(self._c_hands_features_cnn, self._c_gt_hands_contact_state)
            
            if isinstance(loss_cs, dict): losses.update(loss_cs)
            else: losses['loss_classification_contact_state'] = loss_cs

            # ASSOCIATION VECTOR LOSS
            indexes_contact = [i for i, x in enumerate(self._c_gt_hands_contact_state) if x == 1]
            if len(indexes_contact) > 0:
                gt_vectors = np.array(self._c_gt_hands_dxdymagnitude)[indexes_contact]
                if self._c_hands_features_padded[indexes_contact].numel() > 0:
                    _, loss_regression_vector = self.association_vector_regressor(self._c_hands_features_padded[indexes_contact], gt_vectors)
                    losses['loss_regression_dxdymagn'] = loss_regression_vector

        return losses

    # SOSTITUISCI QUESTA INTERA FUNZIONE IN mm_ehoi_net_v1.py

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]], do_postprocess: bool = True):
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        
        # --- Base Detector ---
        features = self.backbone(images.tensor)
        proposals, _ = self.proposal_generator(images, features, None)
        results, _ = self.roi_heads(images, features, proposals, None)
        
        if do_postprocess:
            results = GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)

        # --- Esegui Head Ausiliarie sulle istanze rilevate ---
        final_result = results[0]
        instances = final_result["instances"]
        hand_instances = instances[instances.pred_classes == self._id_hand]

        if len(hand_instances) > 0:
            # MASK HEAD (se abilitato)
            if self._use_mask:
                features_mask = self.mask_pooler(
                    [features[f] for f in self.mask_in_features], 
                    [hand_instances.pred_boxes]
                )
                mask_logits = self.mask_head.layers(features_mask)
                mask_rcnn_inference(mask_logits, [hand_instances])

            # KEYPOINT HEAD (se abilitato)
            if self._use_keypoints:
                features_kpt = self.keypoint_pooler(
                    [features[f] for f in self.keypoint_in_features], 
                    [hand_instances.pred_boxes]
                )
                kpt_logits = self.keypoint_head.layers(features_kpt)
                keypoint_rcnn_inference(kpt_logits, [hand_instances])

            # DEPTH MODULE (se abilitato)
            if self._use_depth_module:
                _, self._depth_maps_predicted = self.depth_module.extract_features_maps(batched_inputs)
                final_result['depth_map'] = self._depth_maps_predicted

            # --- Prepara Features e Esegui Classificatori Custom ---
            self._prepare_hands_features(
                batched_inputs, 
                features, 
                [hand_instances]
            )

            # --- Aggiungi predizioni custom all'oggetto hand_instances ---
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
            else: 
                if self._c_hands_features_cnn.numel() > 0:
                    contact_scores = torch.sigmoid(self.classification_contact_state(self._c_hands_features_cnn))
                else:
                    contact_scores = torch.zeros(len(hand_instances), 1, device=self.device)
            hand_instances.contact_states = torch.round(contact_scores)
                        
            # Aggiungi i campi standard che l'evaluator si aspetta
            hand_instances.set("boxes", hand_instances.pred_boxes)
            hand_instances.set("scores", hand_instances.scores)
            hand_instances.set("pred_classes", hand_instances.pred_classes)
            
            final_result['additional_outputs'] = hand_instances
        else:
            final_result['additional_outputs'] = Instances(images.image_sizes[0])

        return [final_result]
    
    def _prepare_gt_labels(self, proposals_match):
        self._c_gt_hands_lr, self._c_gt_hands_contact_state = [], []
        self._c_gt_hands_dxdymagnitude, self._c_gt_gloves = [], []
        
        for batch_proposal in proposals_match:
            hands = batch_proposal[batch_proposal.gt_classes == self._id_hand]
            
            for i in range(len(hands)):
                hand_proposal = hands[i] 
                
                self._c_gt_hands_lr.append(hand_proposal.gt_sides.item())
                self._c_gt_hands_contact_state.append(hand_proposal.gt_contact_states.item())
                self._c_gt_hands_dxdymagnitude.append(hand_proposal.gt_dxdymagn_hands.cpu().numpy()[0])
                
                if hasattr(hand_proposal, "gt_gloves"):
                    self._c_gt_gloves.append(hand_proposal.gt_gloves.item())
                else:
                    self._c_gt_gloves.append(0)

    def _prepare_hands_features(self, batched_inputs, features, hand_proposals_or_instances):
        """
        Prepara tutte le feature necessarie per le head specifiche della mano.
        Funziona sia in training (con proposals) che in inference (con instances).
        """
        proposals_per_image = [p for p in hand_proposals_or_instances if len(p) > 0]
        if not proposals_per_image:
            self._c_hands_features = self._c_hands_features_padded = self._c_hands_features_cnn = torch.empty(0, device=self.device)
            return

        image_size = self.preprocess_image(batched_inputs).image_sizes[0]
        # Determina quale campo usare per i box a seconda della modalità
        box_field = "proposal_boxes" if self.training else "pred_boxes"

        # Estrai feature vettoriali di base dalle RoI
        all_boxes = [p.get(box_field) for p in proposals_per_image]
        rois = self.roi_heads.box_pooler([features[f] for f in self.roi_heads.in_features], all_boxes)
        self._c_hands_features = self.roi_heads.box_head(rois)

        all_expanded_boxes = [Boxes(expand_box(p.get(box_field).tensor, image_size[1], image_size[0], ratio=self._expand_hand_box_ratio)) for p in proposals_per_image]
        rois_padded = self.roi_heads.box_pooler([features[f] for f in self.roi_heads.in_features], all_expanded_boxes)
        self._c_hands_features_padded = self.roi_heads.box_head(rois_padded)

        # Se non serve la fusione, esci
        if self._contact_state_modality == "rgb":
            return

        # --- Prepara i canali per l'Early Fusion ---
        fusion_channels = []
        
        # Prepara i box per RoIAlign su immagini a piena risoluzione
        boxes_for_align_list = []
        image_indices_with_hands = [i for i, p in enumerate(hand_proposals_or_instances) if len(p) > 0]
        
        for i, p_list in enumerate(proposals_per_image):
            original_batch_idx = image_indices_with_hands[i]
            target_h, target_w = batched_inputs[original_batch_idx]["image"].shape[1:3]
            
            expanded_boxes_single_image = all_expanded_boxes[i]
            scaled_boxes = expanded_boxes_single_image.clone()
            scaled_boxes.scale(target_w / image_size[1], target_h / image_size[0])
            
            batch_idx_tensor = torch.full((len(scaled_boxes), 1), i, device=self.device)
            boxes_for_align_list.append(torch.cat([batch_idx_tensor, scaled_boxes.tensor], dim=1))
        
        boxes_for_align = torch.cat(boxes_for_align_list, dim=0)

        # Prepara immagini e depth map per RoIAlign
        images_with_hands = torch.stack([batched_inputs[i]["image"] for i in image_indices_with_hands]).float().to(self.device)
        
        # Canale RGB
        if 'rgb' in self._contact_state_modality:
            rgb_patches = self._roi_align_for_fusion(images_with_hands, boxes_for_align)
            fusion_channels.append(rgb_patches / 255.0)

        # Canale Depth
        if 'depth' in self._contact_state_modality:
            depth_maps_with_hands = self._depth_maps_predicted[image_indices_with_hands]
            target_h, target_w = images_with_hands.shape[-2:]
            depth_maps_resized = torch.nn.functional.interpolate(depth_maps_with_hands.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False)
            depth_patches = self._roi_align_for_fusion(depth_maps_resized, boxes_for_align)
            fusion_channels.append(depth_patches / 255.0)

        # Canale Mask
        if 'mask' in self._contact_state_modality:
            if self.training:
                all_masks = self._predicted_masks_for_fusion
            else:
                all_masks = torch.cat([p.pred_masks for p in proposals_per_image])
            
            # Aggiungi dimensione canale se necessario
            if all_masks.dim() == 3: all_masks = all_masks.unsqueeze(1)

            target_size = self._roi_align_for_fusion.output_size
            resized_masks = torch.nn.functional.interpolate(all_masks.float(), size=target_size, mode='bilinear', align_corners=False)
            fusion_channels.append(resized_masks)

        # Canale Keypoints
        if 'keypoints' in self._contact_state_modality:
            if self.training:
                all_kpts = torch.cat([p.pred_keypoints for p in self._predicted_keypoints_for_fusion])
            else:
                all_kpts = torch.cat([p.pred_keypoints for p in proposals_per_image])

            all_boxes_tensor = torch.cat([b.tensor for b in all_expanded_boxes], dim=0)
            keypoint_heatmaps = self.keypoint_renderer(all_kpts, all_boxes_tensor)
            fusion_channels.append(keypoint_heatmaps)

        if fusion_channels:
            self._c_hands_features_cnn = torch.cat(fusion_channels, dim=1)
        else:
            self._c_hands_features_cnn = torch.empty(0, device=self.device)