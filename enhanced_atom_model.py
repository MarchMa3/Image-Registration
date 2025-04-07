__author__ = 'Jingnan Ma'

import torch
import torch.nn as nn
from vit_2d_sa import own_model as create_2d_model
from vit_3d_ga1 import own_model as create_3d_ga1_model
from vit_3d_ga2 import own_model as create_3d_ga2_model
from feature_fusion import FeatureFusionModule

class EnhancedATOMModel(nn.Module):
    def __init__(self, brainmvp_extractor=None, brainmvp_embed_dim=768, fusion_dim=1000, use_brainmvp=True):
        super().__init__()
        
        # BrainMVP feature extractor
        self.brainmvp_extractor = brainmvp_extractor
        self.use_brainmvp = use_brainmvp
        
        # Original 2D encoder
        self.vit_2d = create_2d_model()
        
        # Feature fusion module
        if use_brainmvp:
            self.fusion_module = FeatureFusionModule(
                brainmvp_dim=brainmvp_embed_dim,
                atom_2d_dim=5,  # 2D encoder feature dimension, adjust based on specific implementation
                fusion_dim=fusion_dim
            )
        
        # Modified 3D encoder GA1
        self.vit_3d_ga1 = create_3d_ga1_model()
        if use_brainmvp:
            # Configure parameters to support BrainMVP features
            self.vit_3d_ga1.use_brainmvp = True
            self.vit_3d_ga1.brainmvp_adapter = nn.Sequential(
                nn.Linear(brainmvp_embed_dim, 1000),
                nn.LayerNorm(1000),
                nn.GELU()
            )
            self.vit_3d_ga1.block1.use_brainmvp = True
            self.vit_3d_ga1.block1.attn.use_brainmvp = True
            
        # Modified 3D encoder GA2
        self.vit_3d_ga2 = create_3d_ga2_model()
        if use_brainmvp:
            # Configure parameters to support BrainMVP features
            self.vit_3d_ga2.use_brainmvp = True
            self.vit_3d_ga2.brainmvp_adapter = nn.Sequential(
                nn.Linear(brainmvp_embed_dim, 6*6*6),
                nn.LayerNorm(6*6*6),
                nn.GELU()
            )
            self.vit_3d_ga2.block1.use_brainmvp = True
            self.vit_3d_ga2.block1.attn.use_brainmvp = True
    
    def extract_subvolume(self, volume_3d, pred1_out):
        """
        Extract subvolumes from the large volume
        
        Args:
            volume_3d: Input 3D volume with shape [batch_size, channels, depth, height, width]
            pred1_out: First stage prediction result indicating subvolume indices
            
        Returns:
            Extracted subvolumes with shape [batch_size, channels, 10, 10, 10]
        """
        batch_size = volume_3d.shape[0]
        subvolumes = torch.zeros((batch_size, 1, 10, 10, 10), device=volume_3d.device)
        
        for b in range(batch_size):
            # Get predicted block index
            idx = pred1_out[b][0].item()
            
            # Calculate 3D index
            i = idx // 9
            j = (idx - i * 9) // 3
            k = idx - i * 9 - j * 3
            
            # Extract subvolume
            subvolumes[b] = volume_3d[b, :, i*5:i*5+10, j*5:j*5+10, k*5:k*5+10]
        
        return subvolumes
    
    def forward(self, images_3d, images_2d):
        """
        Forward pass
        
        Args:
            images_3d: 3D input images with shape [batch_size, channels, depth, height, width]
            images_2d: 2D input images with shape [batch_size, channels, height, width]
            
        Returns:
            Predictions from first and second stages
        """
        # Extract BrainMVP features
        brainmvp_features = None
        if self.use_brainmvp and self.brainmvp_extractor is not None:
            with torch.no_grad():  # Don't compute gradients
                brainmvp_features = self.brainmvp_extractor(images_3d)
        
        # 2D encoder forward pass
        x, k, v = self.vit_2d(images_2d)
        
        # 3D encoder GA1 forward pass
        pred1, pred_class, pred_class_k1, x_set = self.vit_3d_ga1(images_3d, k, v, brainmvp_features=brainmvp_features)
        
        # Extract subvolumes
        subvolumes = self.extract_subvolume(images_3d, pred_class_k1)
        
        # 3D encoder GA2 forward pass
        pred_new1, pred_class_new, pred_class_k1_new = self.vit_3d_ga2(subvolumes, k, v, brainmvp_features=brainmvp_features)
        
        return (pred1, pred_class, pred_class_k1, x_set), (pred_new1, pred_class_new, pred_class_k1_new)


def create_enhanced_model(brainmvp_model=None, brainmvp_embed_dim=768, fusion_dim=1000, use_brainmvp=True):
    """
    Factory function to create enhanced ATOM model
    
    Args:
        brainmvp_model: Pre-trained BrainMVP model
        brainmvp_embed_dim: BrainMVP feature dimension
        fusion_dim: Fusion feature dimension
        use_brainmvp: Whether to use BrainMVP features
        
    Returns:
        Enhanced ATOM model instance
    """
    model = EnhancedATOMModel(
        brainmvp_extractor=brainmvp_model,
        brainmvp_embed_dim=brainmvp_embed_dim,
        fusion_dim=fusion_dim,
        use_brainmvp=use_brainmvp
    )
    
    return model