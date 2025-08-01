import logging
from abc import ABC, abstractmethod
from typing import List

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.roi_heads import build_mask_head, build_keypoint_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_loss, keypoint_rcnn_inference

from ..poolers import ROIPooler
from .additional_modules import (
    SideLRClassificationModule, 
    GlovesClassificationModule,
    ContactStateRGBClassificationModule,
    ContactStateFusionClassificationModule,
    KeypointRenderer,
    DepthModule
)
from .MiDaS.midas.midas_loss_simple import DepthLoss

__all__ = ["EhoiNet"]
logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class EhoiNet(GeneralizedRCNN, ABC):
    """
    Ego-centric Hand-Object Interaction Network (EhoiNet).
    This class extends GeneralizedRCNN to handle EHOI-specific tasks.
    It is configured via a single modality string for maximum flexibility in ablation studies.
    """
    def __init__(self, cfg, metadata): 
        base_modules = GeneralizedRCNN.from_config(cfg)
        super().__init__(**base_modules)
        
        thing_classes = metadata.thing_classes
        self._id_hand = thing_classes.index("hand") if "hand" in thing_classes else thing_classes.index("mano")
        self.metadata = metadata
        
        # --- Modality & Task Configuration ---
        self._contact_state_modality = cfg.ADDITIONAL_MODULES.CONTACT_STATE_MODALITY
        self._use_depth_module = 'depth' in self._contact_state_modality
        self._use_keypoints = 'keypoints' in self._contact_state_modality
        self._use_mask = 'mask' in self._contact_state_modality

        self._last_extracted_features = {}
        self._last_inference_times = {}
        
        logger.info(f"EhoiNet initialized with contact state modality: '{self._contact_state_modality}'")
        logger.info(f"Derived module usage: Depth={self._use_depth_module}, Keypoints={self._use_keypoints}, Mask={self._use_mask}")

        # --- Build All EHOI-Specific Modules ---
        self._build_modules(cfg)

    def _build_modules(self, cfg):
        """Builds all required neural network modules based on the configuration."""
        
        # Auxiliary Task Heads (Always Active)
        self.classification_hand_lr = SideLRClassificationModule(cfg)
        self.classification_glove = GlovesClassificationModule(cfg)
        
        # Core Contact State Head (dynamically configured)
        self.classification_contact_state = self._build_contact_state_head(cfg)
        
        # Input Modality-Specific Modules
        if self._use_mask:
            self.mask_in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
            self.mask_head, self.mask_pooler = self._build_mask_head(cfg, self.mask_in_features)
            logger.info("Mask head built.")

        if self._use_depth_module: 
            self.depth_module = DepthModule(cfg)
            self.depth_loss_fn = DepthLoss()
            logger.info("Depth module built.")

        if self._use_keypoints:
            self.keypoint_in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
            self.keypoint_head, self.keypoint_pooler = self._build_keypoint_head(cfg, self.keypoint_in_features)
            self.keypoint_renderer = KeypointRenderer(cfg)
            logger.info("Keypoint head and renderer built.")

    def _build_mask_head(self, cfg, in_features):
        """Builds the standard Detectron2 mask head and its pooler."""
        input_shape = self.backbone.output_shape()
        pooler = ROIPooler(
            output_size=cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION,
            scales=tuple(1.0 / input_shape[k].stride for k in in_features),
            sampling_ratio=cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO,
            pooler_type=cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE,
        )
        shape = ShapeSpec(
            channels=input_shape[in_features[0]].channels,
            width=cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION,
            height=cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION,
        )
        return build_mask_head(cfg, shape), pooler
        
    def _build_keypoint_head(self, cfg, in_features):
        """Builds the standard Detectron2 keypoint head and its pooler."""
        input_shape = self.backbone.output_shape()
        pooler = ROIPooler(
            output_size=cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION,
            scales=tuple(1.0 / input_shape[k].stride for k in in_features),
            sampling_ratio=cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO,
            pooler_type=cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE,
        )
        shape = ShapeSpec(
            channels=input_shape[in_features[0]].channels,
            width=cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION,
            height=cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION,
        )
        return build_keypoint_head(cfg, shape), pooler

    def _build_contact_state_head(self, cfg):
        modality_str = self._contact_state_modality
        logger.info(f"Building contact state head for modality string: '{modality_str}'")

        if modality_str == "rgb":
            return ContactStateRGBClassificationModule(cfg)

        n_channels = 0
        if 'rgb' in modality_str:
            n_channels += 3
        if 'depth' in modality_str:
            n_channels += 1
        if 'mask' in modality_str:
            n_channels += 1
        if 'keypoints' in modality_str:
            n_channels += 1
        
        if modality_str == "cnn_rgb":
            n_channels = 3

        if n_channels == 0:
            raise ValueError(f"Could not determine any input channels from modality string: '{modality_str}'. "
                             "Check for valid components (rgb, depth, mask, keypoints).")

        logger.info(f"Determined {n_channels} input channels for the fusion module.")
        return ContactStateFusionClassificationModule(cfg, n_channels=n_channels)
    
    @abstractmethod
    def forward(self, batched_inputs):
        pass
        
    @abstractmethod
    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        pass