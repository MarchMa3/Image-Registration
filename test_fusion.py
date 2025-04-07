import torch
import unittest
from feature_fusion import FeatureFusionModule

class TestFeatureFusionModule(unittest.TestCase):
    def setUp(self):
        # 设置测试参数
        self.batch_size = 4
        self.brainmvp_dim = 256      # BrainMVP 特征通道数
        self.atom_2d_dim = 64        # 2D 特征通道数
        self.fusion_dim = 128        # 融合后的特征维度
        self.d, self.h, self.w = 10, 10, 10  # 3D 特征图尺寸
        
        # 初始化融合模块
        self.fusion_module = FeatureFusionModule(
            brainmvp_dim=self.brainmvp_dim,
            atom_2d_dim=self.atom_2d_dim,
            fusion_dim=self.fusion_dim,
            num_heads=4  # 使用默认值
        )
        
    def test_forward_pass(self):
        # 创建模拟输入
        brainmvp_feat = torch.randn(self.batch_size, self.brainmvp_dim, self.d, self.h, self.w)
        
        # 根据源代码实现，atom_2d_features 应该是 2D 特征，形状与 brainmvp_features 重塑后匹配
        # 重塑后 brainmvp_features 的形状为 [batch_size, d*h*w, brainmvp_dim]
        # atom_2d_features 应为 [batch_size, d*h*w, atom_2d_dim]
        atom_2d_feat = torch.randn(self.batch_size, self.d*self.h*self.w, self.atom_2d_dim)
        
        # 执行前向传播
        fused_feat = self.fusion_module(brainmvp_feat, atom_2d_feat)
        
        # 检查输出维度是否正确
        # 根据源代码，输出维度应该是 [batch_size, d*h*w, fusion_dim]
        expected_shape = (self.batch_size, self.d*self.h*self.w, self.fusion_dim)
        self.assertEqual(fused_feat.shape, expected_shape, 
                         f"Output shape {fused_feat.shape} does not match expected shape {expected_shape}")
        
        # 确保输出不包含 NaN 值
        self.assertFalse(torch.isnan(fused_feat).any(), 
                         "Output contains NaN values")
    
    def test_gradient_flow(self):
        # 创建需要梯度的输入
        brainmvp_feat = torch.randn(self.batch_size, self.brainmvp_dim, self.d, self.h, self.w, requires_grad=True)
        atom_2d_feat = torch.randn(self.batch_size, self.d*self.h*self.w, self.atom_2d_dim, requires_grad=True)
        
        # 执行前向传播
        fused_feat = self.fusion_module(brainmvp_feat, atom_2d_feat)
        
        # 计算损失并反向传播
        loss = fused_feat.sum()
        loss.backward()
        
        # 检查梯度是否正确传播
        self.assertIsNotNone(brainmvp_feat.grad, "No gradient propagated to brainmvp_feat")
        self.assertIsNotNone(atom_2d_feat.grad, "No gradient propagated to atom_2d_feat")
        
    def test_fusion_weights(self):
        # 创建模拟输入
        brainmvp_feat = torch.randn(self.batch_size, self.brainmvp_dim, self.d, self.h, self.w)
        atom_2d_feat = torch.randn(self.batch_size, self.d*self.h*self.w, self.atom_2d_dim)
        
        # 将一个输入全部设为0，测试融合逻辑
        zero_brainmvp = torch.zeros_like(brainmvp_feat)
        
        # 当 brainmvp_feat 为 0 时，测试融合模块行为
        fused_feat_zero_brainmvp = self.fusion_module(zero_brainmvp, atom_2d_feat)
        
        # 非零输入的情况
        fused_feat_normal = self.fusion_module(brainmvp_feat, atom_2d_feat)
        
        # 确保两种情况下的输出都是合理的（不包含 NaN）
        self.assertFalse(torch.isnan(fused_feat_zero_brainmvp).any(), 
                         "Output with zero brainmvp_feat contains NaN values")
        self.assertFalse(torch.isnan(fused_feat_normal).any(), 
                         "Output with normal inputs contains NaN values")
        
        # 验证两种情况下输出是不同的
        self.assertFalse(torch.allclose(fused_feat_zero_brainmvp, fused_feat_normal),
                         "Fusion should produce different outputs for different inputs")

if __name__ == '__main__':
    unittest.main()