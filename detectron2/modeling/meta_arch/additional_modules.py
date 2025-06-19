import os
import cv2
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
        """Preprocesses the batch of images for the depth module"""
        batch_images = []
        
        for batch_input in x:
            if "image_for_depth_module" in batch_input:
                image_data = batch_input["image_for_depth_module"]
                
                if isinstance(image_data, np.ndarray):
                    if image_data.shape[0] == 3:
                        image_hwc = image_data.transpose(1, 2, 0)
                    else:
                        image_hwc = image_data
                    
                    image_resized = cv2.resize(image_hwc, (384, 384))
                    
                    if image_resized.max() > 1.0:
                        image_resized = image_resized.astype(np.float32) / 255.0
                    else:
                        image_resized = image_resized.astype(np.float32)
                    
                    image_chw = image_resized.transpose(2, 0, 1)
                    image_tensor = torch.from_numpy(image_chw)
                    batch_images.append(image_tensor)
                    
                else:
                    image_tensor = image_data
                    
                    if image_tensor.shape != (3, 384, 384):
                        if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
                            image_hwc = image_tensor.permute(1, 2, 0).numpy()
                            image_resized = cv2.resize(image_hwc, (384, 384))
                            
                            if image_resized.max() > 1.0:
                                image_resized = image_resized.astype(np.float32) / 255.0
                            else:
                                image_resized = image_resized.astype(np.float32)
                            
                            image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1))
                        else:
                            image_tensor = torch.zeros(3, 384, 384)
                    
                    batch_images.append(image_tensor)
                    
            elif "file_name" in batch_input:
                file_path = batch_input["file_name"]
                
                if not file_path.startswith('./data/'):
                    full_path = f"./data/egoism-hoi-dataset/images/{file_path}"
                else:
                    full_path = file_path
                
                if not os.path.exists(full_path):
                    base_name = os.path.basename(full_path)
                    
                    if base_name.startswith('camera_'):
                        id_part = base_name.replace('camera_', '').replace('.png', '')
                        alt_path = f"./data/egoism-hoi-dataset/images/rgb_{id_part}.png"
                        if os.path.exists(alt_path):
                            full_path = alt_path
                            
                    elif base_name.startswith('rgb_'):
                        id_part = base_name.replace('rgb_', '').replace('.png', '')
                        alt_path = f"./data/egoism-hoi-dataset/images/camera_{id_part}.png"
                        if os.path.exists(alt_path):
                            full_path = alt_path
                    else:
                        if not base_name.startswith(('rgb_', 'camera_')):
                            alt_path = f"./data/egoism-hoi-dataset/images/rgb_{base_name}"
                            if os.path.exists(alt_path):
                                full_path = alt_path
                
                image = cv2.imread(full_path)
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_resized = cv2.resize(image_rgb, (384, 384))
                    image_normalized = image_resized.astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_normalized.transpose(2, 0, 1))
                    batch_images.append(image_tensor)
                    batch_input["image_for_depth_module"] = image_tensor.numpy()
                else:
                    batch_images.append(torch.zeros(3, 384, 384))
            else:
                batch_images.append(torch.zeros(3, 384, 384))
        
        result_tensor = torch.stack(batch_images).to(self.device)
        return result_tensor

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

class KeypointFeatureExtractor(nn.Module):
    """
    Extract semantic features from hand keypoints.
    
    Architecture: Raw coordinates (21x3) -> Semantic features (128D)
    """
    def __init__(self, cfg):
        super(KeypointFeatureExtractor, self).__init__()
        self.num_keypoints = 21
        self.keypoint_dim = 3  # x, y, visibility
        self.normalize_coords = cfg.ADDITIONAL_MODULES.get('NORMALIZE_KEYPOINT_COORDS', True)
        
        # Transform raw coordinates to semantic features
        self.keypoint_encoder = nn.Sequential(
            nn.Linear(self.num_keypoints * self.keypoint_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
    def forward(self, keypoints, image_size=None):
        """
        Args:
            keypoints: Tensor [N, 21, 3] or [N, 63] with (x,y,score) per keypoint
            image_size: Tuple (height, width) for coordinate normalization
        Returns:
            features: Tensor [N, 128] semantic features
        """
        if len(keypoints) == 0:
            return torch.empty(0, 128).to(self.device)
            
        if len(keypoints.shape) == 3:
            keypoints = keypoints.view(keypoints.size(0), -1)
        
        if self.normalize_coords and image_size is not None:
            kpts_reshaped = keypoints.view(-1, self.num_keypoints, 3)
            kpts_reshaped[:, :, 0] /= image_size[1]  # normalize x
            kpts_reshaped[:, :, 1] /= image_size[0]  # normalize y
            keypoints = kpts_reshaped.view(keypoints.size(0), -1)
        
        return self.keypoint_encoder(keypoints)
    
    @property
    def device(self):
        return next(self.parameters()).device

class ContactStateFusionClassificationModule(nn.Module):
    """
    Early fusion module for contact state classification.
    
    Architecture: RGB + CNN + Keypoint features -> Contact classification
    """
    def __init__(self, cfg, n_channels=5, use_pretrain_first_layer=True):
        super(ContactStateFusionClassificationModule, self).__init__()
        
        # CNN branch for depth/mask features
        self.cnn_branch = torchvision.models.efficientnet_v2_s(weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT)   
        weight = self.cnn_branch.features[0][0].weight.clone()
        self.cnn_branch.features[0][0] = nn.Conv2d(n_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        if use_pretrain_first_layer:
            with torch.no_grad():
                self.cnn_branch.features[0][0].weight[:,:3,:,:].data[...] = weight
        self.cnn_branch.classifier.add_module("3", nn.Linear(1000, 256))

        # RGB branch
        self.rgb_branch = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=cfg.ADDITIONAL_MODULES.get('CONTACT_STATE_CLASSIFICATION_MODULE_DROPOUT', 0.1))
        )
        
        # Keypoint branch
        self.keypoint_branch = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(768, 512),  # 256*3 modalities
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, rgb_features, cnn_features, keypoint_features, gt=None):
        """
        Early fusion of RGB, CNN and keypoint features for contact state classification.
        
        Args:
            rgb_features: RGB features from ROI head [N, 1024]
            cnn_features: CNN features from depth/mask [N, H, W, C]
            keypoint_features: Keypoint features [N, 128]
            gt: Ground truth labels for training
        """
        features_list = []
        
        # Process RGB features
        if rgb_features is not None and len(rgb_features) > 0:
            rgb_out = self.rgb_branch(rgb_features)
            features_list.append(rgb_out)
        
        # Process CNN features
        if cnn_features is not None and len(cnn_features) > 0:
            cnn_out = self.cnn_branch(cnn_features)
            features_list.append(cnn_out)
        
        # Process keypoint features
        if keypoint_features is not None and len(keypoint_features) > 0:
            kpt_out = self.keypoint_branch(keypoint_features)
            features_list.append(kpt_out)
        
        # Early fusion: concatenate all features
        if len(features_list) > 0:
            # Ensure all features have same batch size
            min_batch_size = min(f.shape[0] for f in features_list)
            features_list = [f[:min_batch_size] for f in features_list]
            fused_features = torch.cat(features_list, dim=1)
        else:
            # Fallback: create zero features
            device = next(self.parameters()).device
            fused_features = torch.zeros(1, 768, device=device)
        
        # Final classification
        output = self.fusion_net(fused_features)
        
        if gt is None: 
            return torch.sigmoid(output)
            
        if len(gt) == 0: 
            return None, torch.tensor([0], dtype=torch.float32).to(self.device)
            
        gt_tensor = torch.from_numpy(np.array(gt, np.float32)).unsqueeze(1).to(self.device)
        
        # Calculate individual branch losses for monitoring
        loss_dict = {}
        
        # Main fusion loss
        loss_fusion = nn.functional.binary_cross_entropy_with_logits(output, gt_tensor)
        loss_fusion = torch.tensor([0], dtype=torch.float32).to(self.device) if torch.isnan(loss_fusion) else loss_fusion
        
        # Individual branch losses if multiple modalities are present
        if len(features_list) > 1:
            if rgb_features is not None and len(rgb_features) > 0:
                rgb_classifier = nn.Linear(256, 1).to(self.device)
                rgb_pred = rgb_classifier(features_list[0][:min_batch_size])
                loss_rgb = nn.functional.binary_cross_entropy_with_logits(rgb_pred, gt_tensor[:min_batch_size])
                loss_dict["loss_cs_rgb"] = loss_rgb if not torch.isnan(loss_rgb) else torch.tensor([0], dtype=torch.float32).to(self.device)
            
            if cnn_features is not None and len(cnn_features) > 0 and len(features_list) > 1:
                cnn_idx = 1 if rgb_features is not None else 0
                cnn_classifier = nn.Linear(256, 1).to(self.device)
                cnn_pred = cnn_classifier(features_list[cnn_idx][:min_batch_size])
                loss_cnn = nn.functional.binary_cross_entropy_with_logits(cnn_pred, gt_tensor[:min_batch_size])
                loss_dict["loss_cs_cnn"] = loss_cnn if not torch.isnan(loss_cnn) else torch.tensor([0], dtype=torch.float32).to(self.device)
            
            if keypoint_features is not None and len(keypoint_features) > 0:
                kpt_idx = len(features_list) - 1
                kpt_classifier = nn.Linear(256, 1).to(self.device)
                kpt_pred = kpt_classifier(features_list[kpt_idx][:min_batch_size])
                loss_kpt = nn.functional.binary_cross_entropy_with_logits(kpt_pred, gt_tensor[:min_batch_size])
                loss_dict["loss_cs_keypoint"] = loss_kpt if not torch.isnan(loss_kpt) else torch.tensor([0], dtype=torch.float32).to(self.device)
        
        loss_dict["loss_cs_fusion"] = loss_fusion
        
        # Total loss is average of all components if multiple, otherwise just fusion loss
        if len(loss_dict) > 1:
            total_loss = sum(loss_dict.values()) / len(loss_dict)
        else:
            total_loss = loss_fusion
            
        loss_dict["loss_cs_total"] = total_loss

        return output, loss_dict

    @property
    def device(self):
        return next(self.parameters()).device