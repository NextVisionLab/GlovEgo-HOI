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
    Renders hand keypoints into a 2D heatmap representation suitable for a CNN input.

    This module takes absolute keypoint coordinates, normalizes them relative to their
    bounding box, and generates a Gaussian heatmap for each visible keypoint. This
    representation is object-centric and scale-invariant.
    """
    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg.MODEL.ROI_KEYPOINT_HEAD.HEATMAP_SIZE
        self.sigma = cfg.MODEL.ROI_KEYPOINT_HEAD.SIGMA

    def forward(self, keypoints: torch.Tensor, hand_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keypoints (torch.Tensor): Tensor of shape [N, K, 3] with (x, y, visibility)
                                      for N hands and K keypoints.
            hand_boxes (torch.Tensor): Tensor of shape [N, 4] with (x1, y1, x2, y2) boxes.
        
        Returns:
            torch.Tensor: A batch of heatmaps of shape [N, 1, H, W].
        """
        if keypoints.numel() == 0:
            return torch.empty(0, 1, *self.image_size, device=keypoints.device)

        batch_heatmaps = []
        for i in range(keypoints.shape[0]):
            kpts_single_hand = keypoints[i]
            box_single_hand = hand_boxes[i]
            
            x0, y0, x1, y1 = box_single_hand
            # Add a small epsilon to avoid division by zero for zero-sized boxes.
            box_w, box_h = (x1 - x0).clamp(min=1e-6), (y1 - y0).clamp(min=1e-6)
            
            # Normalize keypoints to be relative to the bounding box [0, 1].
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
        device = keypoints_scaled.device
        
        # Create a grid of coordinates for the heatmap.
        # Shape: [H, W, 2]
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device),
            indexing='ij' # Explicitly use 'ij' indexing for consistency
        )
        coords_grid = torch.stack([x_coords, y_coords], dim=-1).float()

        # Initialize an empty heatmap for this instance.
        heatmap = torch.zeros((H, W), device=device)
        
        # Filter for visible keypoints
        visible_keypoints = keypoints_scaled[keypoints_scaled[:, 2] > 0.1]
        if visible_keypoints.numel() == 0:
            return heatmap

        # Extract centers and reshape for broadcasting.
        # Shape: [num_visible_kpts, 1, 1, 2]
        mu = visible_keypoints[:, :2].unsqueeze(1).unsqueeze(1)
        
        # Calculate squared distance from each pixel to each keypoint center.
        # Broadcasting: [H, W, 2] - [k, 1, 1, 2] -> [k, H, W, 2]
        squared_dist = torch.sum((coords_grid.unsqueeze(0) - mu) ** 2, dim=-1)
        
        # Calculate the Gaussian value for each keypoint at each pixel.
        # Shape: [num_visible_kpts, H, W]
        gaussian = torch.exp(-squared_dist / (2 * self.sigma ** 2))
        
        # Combine the heatmaps for all keypoints by taking the maximum value at each pixel.
        # This handles overlaps correctly.
        if gaussian.numel() > 0:
            heatmap = torch.max(gaussian, dim=0).values
            
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
    Predicts hand contact state by combining two distinct architectural branches
    via Late Fusion, reflecting the reference architecture.

    - Branch "eff" (EfficientNet): A CNN backbone operating on a multi-channel patch
      composed of fused modalities (RGB, Depth, Mask, Keypoints) via Early Fusion.
    - Branch "res" (Residual-like): An MLP operating on the 1024-dim RoI feature 
      vector (HFV) extracted from the detector's backbone.

    The final prediction is an average of the outputs from both branches.
    """
    def __init__(self, cfg, n_channels: int):
        super().__init__()
        
        # --- Branch "eff": CNN for Early Fusion of Multimodal Patches ---
        self.cnn_branch = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)
        
        original_first_layer = self.cnn_branch.features[0][0]
        self.cnn_branch.features[0][0] = nn.Conv2d(
            n_channels, original_first_layer.out_channels, 
            kernel_size=original_first_layer.kernel_size, stride=original_first_layer.stride, 
            padding=original_first_layer.padding, bias=False
        )
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

        # --- Branch "res": MLP for Hand Feature Vector (HFV) ---
        self.vector_branch = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=cfg.ADDITIONAL_MODULES.CONTACT_STATE_CLASSIFICATION_MODULE_DROPOUT),
            nn.Linear(256, 1)
        )

    def forward(self, fused_multimodal_patch, hand_feature_vector, gt=None):
        """
        Args:
            fused_multimodal_patch (torch.Tensor): Tensor of shape [N, C, H, W] for the CNN branch.
            hand_feature_vector (torch.Tensor): Tensor of shape [N, 1024] for the MLP branch.
            gt (list or None): Ground truth labels for training.
        """
        # --- Branch Predictions ---
        logits_cnn = self.cnn_branch(fused_multimodal_patch)
        logits_vector = self.vector_branch(hand_feature_vector)
        
        # --- Late Fusion ---
        # The final prediction is the average of the probabilities from each branch.
        prob_cnn = torch.sigmoid(logits_cnn)
        prob_vector = torch.sigmoid(logits_vector)
        final_prob = (prob_cnn + prob_vector) / 2.0

        if not self.training:
            return final_prob

        # --- Loss Calculation ---
        if gt is None or len(gt) == 0:
            # Return zero loss if no ground truth is available.
            loss_dict = {
                "loss_cs_multi": torch.tensor(0.0, device=self.device),
                "loss_cs_eff": torch.tensor(0.0, device=self.device),
                "loss_cs_res": torch.tensor(0.0, device=self.device)
            }
            return final_prob, loss_dict

        gt_tensor = torch.tensor(gt, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Calculate loss for each branch to ensure both are trained effectively.
        loss_cnn = F.binary_cross_entropy_with_logits(logits_cnn, gt_tensor, reduction="mean")
        loss_vector = F.binary_cross_entropy_with_logits(logits_vector, gt_tensor, reduction="mean")
        
        # The main loss is the average of the two, encouraging joint optimization.
        total_loss = (loss_cnn + loss_vector) / 2.0
        
        # --- FIX: Use consistent loss keys as requested ---
        loss_dict = {
            "loss_cs_multi": total_loss,
            "loss_cs_eff": loss_cnn,        # Loss from the EfficientNet (Early Fusion) branch A
            "loss_cs_res": loss_vector      # Loss from the ResNet-features (HFV) branch B
        }
        
        return final_prob, loss_dict

    @property
    def device(self):
        return next(self.parameters()).device