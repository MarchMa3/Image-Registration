import unittest
import numpy as np
import torch
from unittest.mock import MagicMock

class TestExtractFeatures(unittest.TestCase):
    def test_extract_features(self):
        """测试extract_features函数的基本功能"""
        # 创建模拟对象
        mock_self = MagicMock()
        
        # 创建测试输入数据
        test_images = np.ones((2, 20, 20, 20, 1), dtype=np.float32)
        
        # 测试场景1: use_brainmvp=False
        mock_self.use_brainmvp = False
        result = extract_features(mock_self, test_images)
        # 应该直接返回输入图像
        np.testing.assert_array_equal(result, test_images)
        
        # 测试场景2: use_brainmvp=True
        mock_self.use_brainmvp = True
        mock_self.n_channels = 1
        mock_self.apply_aug_before_feature = False
        
        # 设置不同level的模拟特征图
        mock_x0 = torch.ones((2, 32, 20, 20, 20))
        mock_x1 = torch.ones((2, 64, 10, 10, 10))
        mock_x2 = torch.ones((2, 128, 5, 5, 5)) 
        mock_x3 = torch.ones((2, 320, 3, 3, 3))
        mock_x4 = torch.ones((2, 512, 2, 2, 2))
        mock_self.feature_extractor.return_value = (mock_x0, mock_x1, mock_x2, mock_x3, mock_x4)
        
        # 测试不同return_features_level的返回值
        feature_levels = [0, 1, 2, 3, 4]
        expected_shapes = [
            (2, 32, 20, 20, 20),
            (2, 64, 10, 10, 10),
            (2, 128, 5, 5, 5),
            (2, 320, 3, 3, 3),
            (2, 512, 2, 2, 2)
        ]
        
        for level, expected_shape in zip(feature_levels, expected_shapes):
            mock_self.return_features_level = level
            result = extract_features(mock_self, test_images)
            
            # 验证结果是正确层级的特征图
            self.assertTrue(torch.is_tensor(result))
            self.assertEqual(result.shape, expected_shape)


# 被测试的函数
def extract_features(self, images_3d):
    """
    Extract features from images_3d
    
    input: images_3d: [batch_size, *dim, n_channels]
    output: features: return different levels of features according to various 'return_features_level' parameter
    """
    if not self.use_brainmvp:
        return images_3d
    
    if self.apply_aug_before_feature:
        images_3d = self.augment_before_feature_extraction(images_3d)
    
    images = torch.from_numpy(images_3d).float()
    
    if images.shape[-1] == self.n_channels:
        images = images.permute(0, 4, 1, 2, 3)
    
    x_0, x_enc1, x_enc2, x_enc3, x_enc4 = self.feature_extractor(images)
    
    feature_maps = [x_0, x_enc1, x_enc2, x_enc3, x_enc4]
    return feature_maps[self.return_features_level]


if __name__ == '__main__':
    unittest.main()