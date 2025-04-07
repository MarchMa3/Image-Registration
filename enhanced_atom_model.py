import torch
import torch.nn as nn
from vit_2d_sa import own_model as create_2d_model
from vit_3d_ga1 import own_model as create_3d_ga1_model
from vit_3d_ga2 import own_model as create_3d_ga2_model
from feature_fusion import FeatureFusionModule

class EnhancedATOMModel(nn.Module):
    def __init__(self, brainmvp_extractor=None, brainmvp_embed_dim=768, fusion_dim=1000, use_brainmvp=True):
        super().__init__()
        
        self.brainmvp_extractor = brainmvp_extractor
        self.use_brainmvp = use_brainmvp
        
        self.vit_2d = create_2d_model()
        
        if use_brainmvp:
            self.fusion_module = FeatureFusionModule(
                brainmvp_dim=brainmvp_embed_dim,
                atom_2d_dim=5,  
                fusion_dim=fusion_dim
            )
        
        self.vit_3d_ga1 = create_3d_ga1_model()
        if use_brainmvp:
            self.vit_3d_ga1.use_brainmvp = True
            self.vit_3d_ga1.brainmvp_adapter = nn.Sequential(
                nn.Linear(brainmvp_embed_dim, 1000),
                nn.LayerNorm(1000),
                nn.GELU()
            )
            self.vit_3d_ga1.block1.use_brainmvp = True
            self.vit_3d_ga1.block1.attn.use_brainmvp = True
            
        self.vit_3d_ga2 = create_3d_ga2_model()
        if use_brainmvp:
            self.vit_3d_ga2.use_brainmvp = True
            self.vit_3d_ga2.brainmvp_adapter = nn.Sequential(
                nn.Linear(brainmvp_embed_dim, 6*6*6),
                nn.LayerNorm(6*6*6),
                nn.GELU()
            )
            self.vit_3d_ga2.block1.use_brainmvp = True
            self.vit_3d_ga2.block1.attn.use_brainmvp = True
    
    def extract_subvolume(self, volume_3d, pred1_out):
        batch_size = volume_3d.shape[0]
        subvolumes = []
        
        for b in range(batch_size):
            idx = pred1_out[b][0].item()
            i = idx // 9
            j = (idx - i * 9) // 3
            k = idx - i * 9 - j * 3
            start