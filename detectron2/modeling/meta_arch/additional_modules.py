import torch
from torch import nn
import numpy as np
import torchvision
import torch.nn.functional as F
from .MiDaS.midas.midas_net import MidasNet

# =================================================================================
# Standard Classification/Regression Heads
# These modules operate on the 1024-dim feature vector from the RoI head.
# =================================================================================

class SideLRClassificationModule(nn.Module):
    def __init__(self, cfg):
        super(SideLRClassificationModule, self).__init__()
        self.layer_1 = nn.Linear(1024, 256)
        self.layer_2 = nn.Linear(256, 1)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p = cfg.ADDITIONAL_MODULES.SIDE_LR_CLASSIFICATION_MODULE_DROPOUT)
    def forward(self, x, gt = None):
        output_1 = self.layer_1(x)
        output_1 = self.relu(output_1)
        output_1 = self.dropout(output_1)
        output_2 = self.layer_2(output_1)
        if gt is None: return output_2
        if len(gt) == 0: return None, torch.tensor([0], dtype=torch.float32).to(self.device)
        gt_tensor = torch.from_numpy(np.array(gt, np.float32)).unsqueeze(1).to(self.device)
        loss = nn.functional.binary_cross_entropy_with_logits(output_2, gt_tensor)
        loss = torch.tensor([0], dtype=torch.float32).to(self.device) if torch.isnan(loss) else loss
        return output_2, loss
    @property
    def device(self):
        return next(self.parameters()).device

class AssociationVectorRegressor(nn.Module):
    def __init__(self, cfg):
        super(AssociationVectorRegressor, self).__init__()
        self.layer_1 = nn.Linear(1024, 256)
        self.layer_2 = nn.Linear(256, 3)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p = cfg.ADDITIONAL_MODULES.ASSOCIATION_VECTOR_REGRESSOR_MODULE_DROPOUT)
    def forward(self, x, gt = None):
        output_1 = self.layer_1(x)
        output_1 = self.relu(output_1)
        output_1 = self.dropout(output_1)
        output_2 = self.layer_2(output_1).float()
        if gt is None: return output_2
        if len(gt) == 0: return None, torch.tensor([0], dtype=torch.float32).to(self.device)
        gt_tensor = torch.from_numpy(np.array(gt)).float().to(self.device)
        loss = nn.functional.mse_loss(output_2, gt_tensor)
        loss = torch.tensor([0], dtype=torch.float32).to(self.device) if torch.isnan(loss) else loss
        return output_2, loss
    @property
    def device(self):
        return next(self.parameters()).device

class GlovesClassificationModule(nn.Module):
    """ Classifies if the detected hand is wearing gloves. """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p = cfg.ADDITIONAL_MODULES.GLOVES_CLASSIFICATION_MODULE_DROPOUT),
            nn.Linear(256, 1)
        )

    def forward(self, x, gt_gloves=None):
        if x.numel() == 0:
            return torch.empty(0, 1, device=self.device) if not self.training else (torch.empty(0, 1, device=self.device), torch.tensor(0.0, device=self.device))

        logits = self.layers(x)

        if not self.training:
            return logits

        if gt_gloves is None or len(gt_gloves) == 0:
            return logits, torch.tensor(0.0, device=self.device)
        
        gt_tensor = torch.tensor(gt_gloves, dtype=torch.float32, device=self.device).unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, gt_tensor)
        return logits, loss

    @property
    def device(self):
        return next(self.parameters()).device

# =================================================================================
# Monocular Depth Estimation Module
# =================================================================================

class DepthModule(MidasNet):
    def __init__(self, cfg):
        self.path_weights = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.WEIGHT_PATH 
        try:
            self.pretrained = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.PRETRAIN
        except:
            self.pretrained = False
        self.features = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.FEATURES 
        self.non_negative = cfg.ADDITIONAL_MODULES.DEPTH_MODULE.NON_NEGATIVE 
        super().__init__(self.path_weights, self.features, self.non_negative, use_pretrained = self.pretrained)

    def preprocess_batch(self, x):
        batch_images = np.array([e["image_for_depth_module"] for e in x])
        return torch.from_numpy(batch_images).to(self.device)

    def forward(self, x):
        images = self.preprocess_batch(x)
        return super().forward(images)

    def extract_features_maps(self, x):
        images = self.preprocess_batch(x)

        layer_1 = self.pretrained.layer1(images)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        features_maps = {
            "res_5": path_4,
            "res_4": path_3,
            "res_3": path_2,
            "res_2": path_1
        }

        depth = torch.squeeze(self.scratch.output_conv(path_1),  dim=1)
        return features_maps, depth

    @property
    def device(self):
        return next(self.parameters()).device

# =================================================================================
# Keypoint Heatmap Generation
# =================================================================================

class KeypointRenderer(nn.Module):
    """
    Renders hand keypoints into a 2D heatmap.
    This module takes absolute keypoint coordinates and their bounding box,
    normalizes them, and generates a Gaussian heatmap for each visible keypoint.
    """
    def __init__(self, cfg):
        super().__init__()
        # Load params from config for easier hyperparameter tuning.
        self.image_size = cfg.MODEL.ROI_KEYPOINT_HEAD.HEATMAP_SIZE
        self.sigma = cfg.MODEL.ROI_KEYPOINT_HEAD.SIGMA

    def forward(self, keypoints: torch.Tensor, hand_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keypoints (torch.Tensor): Tensor of shape [N, K, 3] with (x, y, visibility) for N hands.
            hand_boxes (torch.Tensor): Tensor of shape [N, 4] with (x1, y1, x2, y2) boxes.
        Returns:
            torch.Tensor: A batch of heatmaps of shape [N, 1, H, W].
        """
        if len(keypoints) == 0:
            return torch.empty(0, 1, *self.image_size, device=keypoints.device)

        batch_heatmaps = []
        for i in range(keypoints.shape[0]):
            kpts_single_hand = keypoints[i]
            box_single_hand = hand_boxes[i]
            
            x0, y0, x1, y1 = box_single_hand
            box_w, box_h = x1 - x0, y1 - y0
            
            # Normalize keypoints to be relative to the bounding box [0, 1].
            # This makes the representation object-centric and scale-invariant.
            kpts_normalized = kpts_single_hand.clone()
            kpts_normalized[:, 0] = (kpts_single_hand[:, 0] - x0) / box_w
            kpts_normalized[:, 1] = (kpts_single_hand[:, 1] - y0) / box_h

            # Scale normalized keypoints to the target heatmap dimensions.
            kpts_scaled = kpts_normalized
            kpts_scaled[:, 0] *= self.image_size[1]  # width
            kpts_scaled[:, 1] *= self.image_size[0]  # height

            heatmap = self._generate_heatmap(kpts_scaled)
            batch_heatmaps.append(heatmap)
            
        return torch.stack(batch_heatmaps).unsqueeze(1)

    def _generate_heatmap(self, keypoints_scaled: torch.Tensor) -> torch.Tensor:
        H, W = self.image_size
        heatmap = torch.zeros((H, W), device=keypoints_scaled.device)
        
        for x, y, visibility_score in keypoints_scaled:
            # Only render visible keypoints.
            if visibility_score > 0.1:
                # This block robustly handles keypoints near the image border,
                # where the Gaussian kernel might extend outside the heatmap.
                # It calculates the valid intersection between the heatmap and the
                # Gaussian patch to prevent out-of-bounds errors.
                
                # Define the bounds of the Gaussian kernel.
                ul = [int(x - 3 * self.sigma), int(y - 3 * self.sigma)]
                br = [int(x + 3 * self.sigma + 1), int(y + 3 * self.sigma + 1)]
                
                # Calculate the intersection area on the heatmap.
                h_start, h_end = max(0, ul[1]), min(H, br[1])
                w_start, w_end = max(0, ul[0]), min(W, br[0])

                if h_start >= h_end or w_start >= w_end:
                    continue

                # Create the full Gaussian kernel.
                size = 6 * self.sigma + 1
                xx = torch.arange(0, size, 1, dtype=torch.float32, device=heatmap.device)
                yy = xx[:, None]
                x_mu = y_mu = size // 2
                g = torch.exp(-((xx - x_mu) ** 2 + (yy - y_mu) ** 2) / (2 * self.sigma ** 2))
                
                # Select the correct slice from the Gaussian kernel.
                g_h_start, g_h_end = h_start - ul[1], h_end - ul[1]
                g_w_start, g_w_end = w_start - ul[0], w_end - ul[0]
                
                # Place the Gaussian on the heatmap using torch.max to handle overlaps.
                heatmap_slice = heatmap[h_start:h_end, w_start:w_end]
                gaussian_slice = g[g_h_start:g_h_end, g_w_start:g_w_end]
                
                heatmap[h_start:h_end, w_start:w_end] = torch.max(heatmap_slice, gaussian_slice)
                
        return heatmap

# =================================================================================
# Multimodal Fusion Module for Contact State
# =================================================================================

class ContactStateRGBClassificationModule(nn.Module):
    def __init__(self, cfg):
        super(ContactStateRGBClassificationModule, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p = cfg.ADDITIONAL_MODULES.CONTACT_STATE_CLASSIFICATION_MODULE_DROPOUT),
            nn.Linear(256, 1))
    def forward(self, x, gt = None):
        output = self.layers(x)
        if gt is None: return output
        if len(gt) == 0: return None, torch.tensor([0], dtype=torch.float32).to(self.device)
        gt_tensor = torch.from_numpy(np.array(gt, np.float32)).unsqueeze(1).to(self.device)
        loss = nn.functional.binary_cross_entropy_with_logits(output, gt_tensor)
        loss = torch.tensor([0], dtype=torch.float32).to(self.device) if torch.isnan(loss) else loss
        return output, loss
    @property
    def device(self):
        return next(self.parameters()).device

class ContactStateCNNClassificationModule(nn.Module):
    def __init__(self, cfg, n_channels = 5, use_pretrain_first_layer = True):
        super(ContactStateCNNClassificationModule, self).__init__()
        self.layers_1 = torchvision.models.efficientnet_v2_s(weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)   
        weight = self.layers_1.features[0][0].weight.clone()
        self.layers_1.features[0][0] = nn.Conv2d(n_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        if use_pretrain_first_layer:
            with torch.no_grad():
                self.layers_1.features[0][0].weight[:,:3,:,:].data[...] = weight
        self.layers_1.classifier.add_module("3", nn.Linear(1000, 1))

    def forward(self, x1, gt = None):
        output_1 = self.layers_1(x1)

        if gt is None: return output_1
        if len(gt) == 0: return None, torch.tensor([0], dtype=torch.float32).to(self.device)
        gt_tensor = torch.from_numpy(np.array(gt, np.float32)).unsqueeze(1).to(self.device)

        loss_1 = nn.functional.binary_cross_entropy_with_logits(output_1, gt_tensor)
        loss_1 = torch.tensor([0], dtype=torch.float32).to(self.device) if torch.isnan(loss_1) else loss_1

        loss_dict = {"loss_cs_multi": loss_1}
        return output_1, loss_dict

    @property
    def device(self):
        return next(self.parameters()).device
    
class ContactStateFusionClassificationModule(nn.Module):
    """
    Predicts hand contact state via early fusion of multiple modalities.
    Architecture: [RGB (3) + Depth (1) + Mask (1) + Keypoints (1)] -> CNN -> Classification
    This module is designed to be flexible and supports ablation studies by accepting
    a dynamic number of input channels.
    """
    def __init__(self, cfg, n_channels: int):
        super().__init__()
        
        self.cnn_branch = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)
        
        # --- Dynamic Input Layer ---
        original_first_layer = self.cnn_branch.features[0][0]
        self.cnn_branch.features[0][0] = nn.Conv2d(
            n_channels, 
            original_first_layer.out_channels, 
            kernel_size=original_first_layer.kernel_size, 
            stride=original_first_layer.stride, 
            padding=original_first_layer.padding, 
            bias=False
        )

        # --- Smart Weight Initialization (Transfer Learning) ---
        with torch.no_grad():
            if n_channels >= 3:
                self.cnn_branch.features[0][0].weight[:, :3, :, :] = original_first_layer.weight.clone()
            if n_channels > 3:
                self.cnn_branch.features[0][0].weight[:, 3:, :, :].normal_(0, 0.01)

        classifier_input_features = self.cnn_branch.classifier[1].in_features
        self.cnn_branch.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(classifier_input_features, 1)
        )

    def forward(self, fused_multimodal_patch, gt=None):
        """
        Args:
            fused_multimodal_patch (torch.Tensor): Tensor of shape [N, C, H, W],
                                                   where C is the dynamic number of channels.
            gt (list or None): Ground truth labels for training.
        """
        if fused_multimodal_patch is None or fused_multimodal_patch.numel() == 0:
            if not self.training: return torch.empty(0, 1, device=self.device)
            else: return torch.empty(0, 1, device=self.device), {"loss_cs_fusion": torch.tensor(0.0, device=self.device)}
            
        logits = self.cnn_branch(fused_multimodal_patch)
        
        if not self.training:
            return torch.sigmoid(logits)
            
        if gt is None or len(gt) == 0:
            return torch.sigmoid(logits), {"loss_cs_fusion": torch.tensor(0.0, device=self.device)}

        gt_tensor = torch.tensor(gt, dtype=torch.float32, device=self.device).unsqueeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, gt_tensor)
        
        return torch.sigmoid(logits), {"loss_cs_fusion": loss}

    @property
    def device(self):
        return next(self.parameters()).device