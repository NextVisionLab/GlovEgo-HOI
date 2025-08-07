import kornia
import torch
from torch import nn
import numpy as np
import torchvision

from .MiDaS.midas.midas_net import MidasNet

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

class KeypointHeatmapGenerator(nn.Module):
    """
    Generates a single-channel heatmap image from keypoint coordinates.
    This heatmap can be concatenated with other image-based features (RGB, depth, mask)
    for early fusion into a CNN.
    """
    def __init__(self, cfg):
        super(KeypointHeatmapGenerator, self).__init__()
        self.output_size = (
            cfg.ADDITIONAL_MODULES.CONTACT_STATE_CNN_INPUT_SIZE,
            cfg.ADDITIONAL_MODULES.CONTACT_STATE_CNN_INPUT_SIZE
        )
        self.confidence_threshold = cfg.ADDITIONAL_MODULES.get('KEYPOINT_CONFIDENCE_THRESHOLD', 0.1)
        
        self.sigma = cfg.ADDITIONAL_MODULES.get('KEYPOINT_HEATMAP_SIGMA', 1.5)

    def forward(self, keypoints: torch.Tensor, hand_boxes: torch.Tensor) -> torch.Tensor:
        """
        Processes a batch of hand keypoints to generate heatmaps.
        """
        if len(keypoints) == 0:
            return torch.empty(0, 1, *self.output_size, device=hand_boxes.device)
        
        # --- Normalizzazione (invariata) ---
        x0, y0 = hand_boxes[:, 0].unsqueeze(1), hand_boxes[:, 1].unsqueeze(1)
        box_w = (hand_boxes[:, 2] - x0.squeeze(1)).unsqueeze(1)
        box_h = (hand_boxes[:, 3] - y0.squeeze(1)).unsqueeze(1)
        box_w = torch.clamp(box_w, min=1e-6)
        box_h = torch.clamp(box_h, min=1e-6)

        norm_keypoints = keypoints.clone()
        norm_keypoints[:, :, 0] = (keypoints[:, :, 0] - x0) / box_w
        norm_keypoints[:, :, 1] = (keypoints[:, :, 1] - y0) / box_h
        norm_keypoints[:, :, 0] *= self.output_size[1] # width
        norm_keypoints[:, :, 1] *= self.output_size[0] # height
        
        batch_size = keypoints.shape[0]
        heatmaps = torch.zeros(batch_size, self.output_size[0], self.output_size[1], device=keypoints.device)
        
        # --- Creazione griglia (invariata) ---
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.output_size[0], device=keypoints.device, dtype=keypoints.dtype),
            torch.arange(self.output_size[1], device=keypoints.device, dtype=keypoints.dtype),
            indexing='ij' # Fondamentale per avere H, W
        ) # grid_y shape: [H, W], grid_x shape: [H, W]

        for i in range(batch_size):
            visible_kps = norm_keypoints[i][norm_keypoints[i, :, 2] > 0]
            
            if visible_kps.numel() > 0:
                # --- INIZIO LOGICA DI BROADCASTING DEFINITIVA ---
                
                kps_x = visible_kps[:, 0] # Shape: [N_kps]
                kps_y = visible_kps[:, 1] # Shape: [N_kps]
                
                # Sottrazione esplicita per asse
                # grid_x[:, :, None] -> [H, W, 1]
                # kps_x[None, None, :] -> [1, 1, N_kps]
                # Risultato -> [H, W, N_kps]
                dist_x_sq = (grid_x.unsqueeze(-1) - kps_x) ** 2
                dist_y_sq = (grid_y.unsqueeze(-1) - kps_y) ** 2
                
                squared_dist = dist_x_sq + dist_y_sq # Shape: [H, W, N_kps]
                
                # --- FINE LOGICA DI BROADCASTING DEFINITIVA ---
                
                gaussian_maps = torch.exp(-squared_dist / (2 * self.sigma**2))
                heatmaps[i], _ = torch.max(gaussian_maps, dim=2)
                
        return heatmaps.unsqueeze(1)
    
class ContactStateRGBClassificationModule(nn.Module):
    def __init__(self, cfg):
        super(ContactStateRGBClassificationModule, self).__init__()
        self.layer_1 = nn.Linear(1024, 256)
        self.layer_2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = cfg.ADDITIONAL_MODULES.CONTACT_STATE_CLASSIFICATION_MODULE_DROPOUT)

    def forward(self, x, gt = None):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.layer_2(x)
        
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
    def __init__(self, cfg, n_channels = 5, use_pretrain_first_layer = True):
        super(ContactStateFusionClassificationModule, self).__init__()
        self.layers_1 = torchvision.models.efficientnet_v2_s(weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)   
        weight = self.layers_1.features[0][0].weight.clone()
        self.layers_1.features[0][0] = nn.Conv2d(n_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        if use_pretrain_first_layer:
            with torch.no_grad():
                self.layers_1.features[0][0].weight[:,:3,:,:].data[...] = weight
        self.layers_1.classifier.add_module("3", nn.Linear(1000, 1))

        self.layers_2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p = cfg.ADDITIONAL_MODULES.CONTACT_STATE_CLASSIFICATION_MODULE_DROPOUT),
            nn.Linear(256, 1))

    def forward(self, x1, x2, gt = None):
        output_1 = self.layers_1(x1)
        output_2 = self.layers_2(x2)
        output = torch.mean(torch.stack( (torch.sigmoid(output_1), torch.sigmoid(output_2.reshape(-1, 1))) ), dim = 0)
        if gt is None: return output
        if len(gt) == 0: return None, torch.tensor([0], dtype=torch.float32).to(self.device)
        gt_tensor = torch.from_numpy(np.array(gt, np.float32)).unsqueeze(1).to(self.device)

        loss_1 = nn.functional.binary_cross_entropy_with_logits(output_1, gt_tensor)
        loss_1 = torch.tensor([0], dtype=torch.float32).to(self.device) if torch.isnan(loss_1) else loss_1

        loss_2 = nn.functional.binary_cross_entropy_with_logits(output_2, gt_tensor)
        loss_2 = torch.tensor([0], dtype=torch.float32).to(self.device) if torch.isnan(loss_2) else loss_2

        loss_dict = {"loss_cs_multi": (loss_1 + loss_2) / 2, "loss_cs_eff": loss_1, "loss_cs_res": loss_2}
        return output, loss_dict

    @property
    def device(self):
        return next(self.parameters()).device
