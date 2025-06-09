import logging
from abc import abstractmethod
import torchvision

from detectron2.modeling.roi_heads.mask_head import build_mask_head
from .rcnn import GeneralizedRCNN
from .build import META_ARCH_REGISTRY
from .additional_modules import *
from ..roi_heads.keypoint_head import build_keypoint_head
from ..poolers import ROIPooler
from detectron2.layers import ShapeSpec
from .MiDaS.midas.midas_loss import ScaleAndShiftInvariantLoss
from .MiDaS.midas.midas_loss_simple import DepthLoss

__all__ = ["EhoiNet"]
logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class EhoiNet(GeneralizedRCNN):
    def __init__(self, cfg, metadata):
        GeneralizedRCNN_modules = GeneralizedRCNN.from_config(cfg)
        super().__init__(
            backbone = GeneralizedRCNN_modules["backbone"],
            proposal_generator = GeneralizedRCNN_modules["proposal_generator"],
            roi_heads = GeneralizedRCNN_modules["roi_heads"],
            pixel_mean = GeneralizedRCNN_modules["pixel_mean"],
            pixel_std = GeneralizedRCNN_modules["pixel_std"],
            input_format = GeneralizedRCNN_modules["input_format"],
            vis_period = GeneralizedRCNN_modules["vis_period"])

        # Basic attributes
        thing_classes = metadata.as_dict()["thing_classes"]
        self._thing_classes = thing_classes
        self._id_hand = thing_classes.index("hand") if "hand" in thing_classes else thing_classes.index("mano")
        self._num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self._contact_state_modality = cfg.ADDITIONAL_MODULES.CONTACT_STATE_MODALITY
        self._predict_mask = cfg.ADDITIONAL_MODULES.USE_MASK
        self._mask_gt = cfg.ADDITIONAL_MODULES.USE_MASK_GT
        self._use_depth_module = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.USE_DEPTH_MODULE

        # NEW: Keypoint attributes
        self._use_keypoints = cfg.ADDITIONAL_MODULES.USE_KEYPOINTS if hasattr(cfg.ADDITIONAL_MODULES, 'USE_KEYPOINTS') else False
        self._use_keypoint_early_fusion = cfg.ADDITIONAL_MODULES.USE_KEYPOINT_EARLY_FUSION if hasattr(cfg.ADDITIONAL_MODULES, 'USE_KEYPOINT_EARLY_FUSION') else False
        
        # Legacy attributes for backward compatibility
        self._predict_keypoints = self._use_keypoints  # Keep old name for compatibility
        self._keypoints_gt = cfg.ADDITIONAL_MODULES.KEYPOINTS_GT if hasattr(cfg.ADDITIONAL_MODULES, 'KEYPOINTS_GT') else False

        # Utils
        self._last_extracted_features = {}
        self._last_inference_times = {}

        # Additional modules
        self.classification_hand_lr = SideLRClassificationModule(cfg)
        self.classification_contact_state = self.build_contact_state_head(cfg)
        
        # Build mask module if needed
        if self._predict_mask: 
            self.build_mask_module(cfg)

        # Build keypoint module if needed
        if self._use_keypoints: 
            self.build_keypoint_module(cfg)

        # Depth module
        if self._use_depth_module: 
            self.depth_module = DepthModule(cfg)
            self._loss_depth_f = DepthLoss()
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
        
    @abstractmethod
    def inference(self, *args, **kwargs):
        pass
    
    def build_keypoint_module(self, cfg):
        """Build keypoint module for hand keypoint detection"""
        try:
            in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
            input_shape = self.backbone.output_shape()
            pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION if hasattr(cfg.MODEL.ROI_KEYPOINT_HEAD, 'POOLER_RESOLUTION') else 14
            pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
            sampling_ratio = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO if hasattr(cfg.MODEL.ROI_KEYPOINT_HEAD, 'POOLER_SAMPLING_RATIO') else 0
            pooler_type = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE if hasattr(cfg.MODEL.ROI_KEYPOINT_HEAD, 'POOLER_TYPE') else "ROIAlignV2"
            in_channels = [input_shape[f].channels for f in in_features][0]
            
            self._keypoint_in_features = in_features
            self._keypoint_pooler = ROIPooler(
                output_size=pooler_resolution, 
                scales=pooler_scales, 
                sampling_ratio=sampling_ratio, 
                pooler_type=pooler_type
            )
            
            shape = ShapeSpec(
                channels=in_channels, 
                width=pooler_resolution, 
                height=pooler_resolution
            )
            self._keypoint_head = build_keypoint_head(cfg, shape)
            
            logger.info(f"Keypoint module built successfully with {cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS} keypoints")
            
        except Exception as e:
            logger.error(f"Failed to build keypoint module: {e}")
            # Fallback: disable keypoints if building fails
            self._use_keypoints = False
            self._predict_keypoints = False
            logger.warning("Keypoints disabled due to build failure")

    def build_contact_state_head(self, cfg):
        """Build contact state head with keypoint support"""
        modality = self._contact_state_modality
        
        # NEW: Keypoint-specific modalities
        if modality == "keypoints":
            return ContactStateKeypointOnlyClassificationModule(cfg)
        elif "keypoints" in modality and "fusion" in modality:
            return ContactStateKeypointFusionClassificationModule(cfg)
        
        # Original modalities
        elif modality == "rgb": 
            return ContactStateRGBClassificationModule(cfg)
        elif modality == "cnn_rgb": 
            return ContactStateCNNClassificationModule(cfg, n_channels=3)
        elif modality == "depth": 
            return ContactStateCNNClassificationModule(cfg, n_channels=1, use_pretrain_first_layer=False)
        elif modality == "mask": 
            return ContactStateCNNClassificationModule(cfg, n_channels=1, use_pretrain_first_layer=False)
        elif modality == "rgb+depth": 
            return ContactStateCNNClassificationModule(cfg, n_channels=4)
        elif modality == "mask+rgb": 
            return ContactStateCNNClassificationModule(cfg, n_channels=4)
        elif modality == "mask+depth": 
            return ContactStateCNNClassificationModule(cfg, n_channels=2, use_pretrain_first_layer=False)
        elif modality == "mask+rgb+depth": 
            return ContactStateCNNClassificationModule(cfg, n_channels=5)
        elif modality == "mask+rgb+depth+fusion": 
            return ContactStateFusionClassificationModule(cfg, n_channels=5)
        elif modality == "mask+rgb+fusion": 
            return ContactStateFusionClassificationModule(cfg, n_channels=4)
        elif modality == "rgb+depth+fusion": 
            return ContactStateFusionClassificationModule(cfg, n_channels=4)
        elif modality == "rgb+fusion": 
            return ContactStateFusionClassificationModule(cfg, n_channels=3)
        
        # Extended modalities with keypoints (fallback to fusion modules)
        elif "keypoints" in modality:
            # Determine number of channels based on other modalities
            n_channels = 3  # Default RGB
            if "mask" in modality:
                n_channels += 1
            if "depth" in modality:
                n_channels += 1
            
            if "fusion" in modality:
                return ContactStateKeypointFusionClassificationModule(cfg)
            else:
                return ContactStateCNNClassificationModule(cfg, n_channels=n_channels)
        
        else:
            raise ValueError(f"Unknown contact state modality: {modality}")

    def build_mask_module(self, cfg):
        """Build mask module"""
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        input_shape = self.backbone.output_shape()
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        in_channels = [input_shape[f].channels for f in in_features][0]
        
        self._mask_in_features = in_features            
        self._mask_pooler = ROIPooler(
            output_size=pooler_resolution, 
            scales=pooler_scales, 
            sampling_ratio=sampling_ratio, 
            pooler_type=pooler_type
        )
        
        shape = ShapeSpec(
            channels=in_channels, 
            width=pooler_resolution, 
            height=pooler_resolution
        )
        self._mask_rcnn_head = build_mask_head(cfg, shape)

    def extract_features_maps(self, batched_inputs):
        """Extract feature maps from backbone and depth module"""
        images = self.preprocess_image(batched_inputs)
        self._last_extracted_features["rgb"] = self.backbone(images.tensor)
        
        if self._use_depth_module: 
            self._last_extracted_features["depth"] = self.depth_module.extract_features_maps(batched_inputs)[0]

    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def _prepare_gt_labels(self, proposals_match):
        """Prepare ground truth labels for training"""
        pass

    @abstractmethod
    def _prepare_hands_features(self, *args, **kwargs):
        """Prepare hand features for additional modules"""
        pass

    # NEW: Utility methods for keypoint handling
    def _extract_keypoint_features(self, instances_list, features):
        """Extract keypoint features from instances (to be used by subclasses)"""
        if not self._use_keypoints:
            return None
            
        try:
            hand_instances = []
            for instances in instances_list:
                hand_mask = instances.gt_classes == self._id_hand if self.training else instances.pred_classes == self._id_hand
                hand_instances.extend(instances[hand_mask])
            
            if len(hand_instances) == 0:
                return None
                
            # Extract ROI features for keypoint detection
            boxes = [inst.proposal_boxes if self.training else inst.pred_boxes for inst in hand_instances]
            keypoint_features = self._keypoint_pooler([features[f] for f in self._keypoint_in_features], boxes)
            
            return keypoint_features
            
        except Exception as e:
            logger.warning(f"Failed to extract keypoint features: {e}")
            return None

    def get_keypoint_loss(self, pred_keypoint_logits, instances):
        """Calculate keypoint loss (to be used by subclasses)"""
        if not self._use_keypoints or pred_keypoint_logits is None:
            return {}
            
        try:
            from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_loss
            normalizer = None  # Use default normalization
            loss_keypoint = keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer)
            return {"loss_keypoint": loss_keypoint * cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT}
        except Exception as e:
            logger.warning(f"Failed to calculate keypoint loss: {e}")
            return {}

    def predict_keypoints(self, keypoint_features, instances):
        """Predict keypoints from features (to be used by subclasses)"""
        if not self._use_keypoints or keypoint_features is None:
            return instances
            
        try:
            pred_keypoint_logits = self._keypoint_head(keypoint_features)
            if not self.training:
                from detectron2.modeling.roi_heads.keypoint_head import keypoint_rcnn_inference
                keypoint_rcnn_inference(pred_keypoint_logits, instances)
            return instances, pred_keypoint_logits
        except Exception as e:
            logger.warning(f"Failed to predict keypoints: {e}")
            return instances, None

    @property
    def device(self):
        """Get device of the model"""
        return next(self.parameters()).device