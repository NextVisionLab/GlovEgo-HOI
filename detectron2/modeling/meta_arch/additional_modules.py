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
        """
        Preprocesses the batch of images for the depth module.
        """
        batch_images = []
        
        for batch_input in x:
            if "image_for_depth_module" in batch_input:
                image_data = batch_input["image_for_depth_module"]
                
                if isinstance(image_data, np.ndarray):
                    if image_data.shape[0] == 3:
                        image_hwc = image_data.transpose(1, 2, 0)  # (H, W, C)
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
    """Extractor di features dai keypoints per il modulo Contact State"""
    def __init__(self, cfg):
        super(KeypointFeatureExtractor, self).__init__()
        self.num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS if hasattr(cfg.MODEL, 'ROI_KEYPOINT_HEAD') else 21
        self.keypoint_dim = 3  # x, y, visibility
        self.normalize_coords = cfg.ADDITIONAL_MODULES.NORMALIZE_KEYPOINT_COORDS if hasattr(cfg.ADDITIONAL_MODULES, 'NORMALIZE_KEYPOINT_COORDS') else True
        
        # Feature extractor per keypoints
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
            keypoints: Tensor [N, num_keypoints, 3] or [N, num_keypoints*3]
            image_size: Tuple (height, width) 
        Returns:
            features: Tensor [N, 128]
        """
        if len(keypoints) == 0:
            return torch.empty(0, 128).to(self.device)
            
        if len(keypoints.shape) == 3:
            # Flatten keypoints [N, num_keypoints, 3] to [N, num_keypoints*3]
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

class ContactStateKeypointFusionClassificationModule(nn.Module):
    """Early Fusion Module per Contact State con Keypoints, RGB e CNN"""
    def __init__(self, cfg):
        super(ContactStateKeypointFusionClassificationModule, self).__init__()
        
        self.rgb_dim = 1024  # from ROI Head
        self.cnn_dim = 1000  # from CNN (depth/mask)
        self.keypoint_dim = 128  # from Keypoint Feature Extractor
        
        self.use_rgb = "rgb" in cfg.ADDITIONAL_MODULES.CONTACT_STATE_MODALITY
        self.use_cnn = any(x in cfg.ADDITIONAL_MODULES.CONTACT_STATE_MODALITY for x in ["depth", "mask", "cnn"])
        self.use_keypoints = "keypoints" in cfg.ADDITIONAL_MODULES.CONTACT_STATE_MODALITY
        
        total_dim = 0
        if self.use_rgb:
            total_dim += self.rgb_dim
        if self.use_cnn:
            total_dim += self.cnn_dim
        if self.use_keypoints:
            total_dim += self.keypoint_dim
            
        if total_dim == 0:
            raise ValueError("Almeno una modalità deve essere attiva per la fusion")
        
        # Fusion Network
        self.fusion_net = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(cfg.ADDITIONAL_MODULES.CONTACT_STATE_CLASSIFICATION_MODULE_DROPOUT),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, rgb_features=None, cnn_features=None, keypoint_features=None, gt=None):
        import torch.nn as nn
        import torch.nn.functional as F
        
        reduced_features = []
        target_dim = 256  # Reduce all features to this size
        
        # Process each feature type
        features = [
            ("rgb", rgb_features),
            ("cnn", cnn_features),
            ("keypoint", keypoint_features)
        ]
        
        for name, feat in features:
            if feat is not None:
                # Check if this feature type should be used
                if name == "rgb" and not self.use_rgb:
                    continue
                if name == "cnn" and not self.use_cnn:
                    continue
                if name == "keypoint" and not self.use_keypoints:
                    continue
                    
                # Flatten if needed
                if feat.dim() > 2:
                    feat = torch.flatten(feat, start_dim=1)
                elif feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                
                # Reduce dimension if too large
                if feat.shape[1] > target_dim:
                    # Create reducer if not exists
                    reducer_attr = f'{name}_reducer'
                    if not hasattr(self, reducer_attr):
                        setattr(self, reducer_attr, 
                            nn.Linear(feat.shape[1], target_dim).to(feat.device))
                    feat = getattr(self, reducer_attr)(feat)
                
                reduced_features.append(feat)
        
        # Concatenate
        if reduced_features:
            fused_features = torch.cat(reduced_features, dim=1)
        else:
            device = next(self.parameters()).device
            fused_features = torch.zeros(1, target_dim, device=device)
                
        # Rebuild network if input size doesn't match
        expected_input = fused_features.shape[1]
        if self.fusion_net[0].in_features != expected_input:
            self.fusion_net = nn.Sequential(
                nn.Linear(expected_input, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            ).to(fused_features.device)
        
        output = self.fusion_net(fused_features)
        
        if gt is None: 
            return output
            
        if len(gt) == 0: 
            return None, torch.tensor([0], dtype=torch.float32).to(self.device)
            
        gt_tensor = torch.from_numpy(np.array(gt, np.float32)).unsqueeze(1).to(self.device)
        loss = nn.functional.binary_cross_entropy_with_logits(output, gt_tensor)
        loss = torch.tensor([0], dtype=torch.float32).to(self.device) if torch.isnan(loss) else loss
        
        return output, loss        

    @property
    def device(self):
        return next(self.parameters()).device

class ContactStateKeypointOnlyClassificationModule(nn.Module):
    """Classification Module per Contact State con solo Keypoints"""
    def __init__(self, cfg):
        super(ContactStateKeypointOnlyClassificationModule, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(128, 256),  
            nn.ReLU(),
            nn.Dropout(cfg.ADDITIONAL_MODULES.CONTACT_STATE_CLASSIFICATION_MODULE_DROPOUT),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, keypoint_features, gt=None):
        """
        Args:
            keypoint_features: Features from keypoints [N, 128]
            gt: Ground truth labels
        """
        output = self.layers(keypoint_features)
        
        if gt is None: 
            return output
            
        if len(gt) == 0: 
            return None, torch.tensor([0], dtype=torch.float32).to(self.device)
            
        gt_tensor = torch.from_numpy(np.array(gt, np.float32)).unsqueeze(1).to(self.device)
        loss = nn.functional.binary_cross_entropy_with_logits(output, gt_tensor)
        loss = torch.tensor([0], dtype=torch.float32).to(self.device) if torch.isnan(loss) else loss
        
        return output, loss
    
    @property
    def device(self):
        return next(self.parameters()).device

def build_contact_state_module(cfg):
    """Factory function per creare il modulo contact state appropriato"""
    modality = cfg.ADDITIONAL_MODULES.CONTACT_STATE_MODALITY
    
    if "keypoints" in modality and "fusion" in modality:
        # Early Fusion with keypoints
        return ContactStateKeypointFusionClassificationModule(cfg)
    elif modality == "keypoints":
        # Solo keypoints
        return ContactStateKeypointOnlyClassificationModule(cfg)
    elif modality == "rgb":
        # Solo RGB 
        return ContactStateRGBClassificationModule(cfg)
    elif "fusion" in modality:
        # Fusion senza keypoints 
        return ContactStateFusionClassificationModule(cfg)
    else:
        # CNN only 
        return ContactStateCNNClassificationModule(cfg)

def extract_keypoints_from_annotation(annotation, image_size=None):
    """
    Extract keypoints from COCO-style annotation
    Args:
        annotation: Dict with 'keypoints' field
        image_size: Tuple (height, width) for normalization
    """
    if 'keypoints' not in annotation or len(annotation['keypoints']) == 0:
        return torch.zeros(21, 3)
    
    kpts = np.array(annotation['keypoints']).reshape(-1, 3)
    kpts_tensor = torch.tensor(kpts, dtype=torch.float32)
    
    if image_size is not None:
        kpts_tensor[:, 0] /= image_size[1]  # x
        kpts_tensor[:, 1] /= image_size[0]  # y
    
    return kpts_tensor

def validate_keypoints(keypoints, threshold=0.3):
    """
    Validate keypoints based on visibility threshold
    Args:
        keypoints: Tensor of shape [N, num_keypoints, 3] or [N, num_keypoints*3]
        threshold: Visibility threshold for keypoints
    """
    if len(keypoints) == 0:
        return torch.tensor([], dtype=torch.bool)
    
    visible_count = (keypoints[:, :, 2] > threshold).sum(dim=1)
    
    valid_mask = visible_count >= 10
    
    return valid_mask