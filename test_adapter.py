def test_brainmvp_extractor():
    """
    测试BrainMVP特征提取器
    """
    import torch
    
    # 初始化提取器
    extractor = BrainMVPExtractor(
        img_dir='',
        split='train',
        batchSize=4,
        use_brainmvp=True,
        brainmvp_model_path='path/to/pretrained/model.pth',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        return_features_level=4,  # 返回最深层特征
        augmented=True,  # 启用数据增强
        apply_aug_before_feature=True  # 在特征提取前应用数据增强
    )
    
    # 获取一批数据
    features, image_2d, labels1, labels1_loss1, labels2, labels2_loss = extractor[0]
    
    # 验证输出形状
    print(f"BrainMVP特征形状: {features.shape}")  # 应为特征图形状
    print(f"2D切片形状: {image_2d.shape}")        # 应为[batch_size, 4, 4]
    print(f"标签1形状: {labels1_loss1.shape}")    # 应为[batch_size, 27]
    
    # 测试不同层级特征
    extractor.return_features_level = 2  # 测试中间层级特征
    features2, _, _, _, _, _ = extractor[0]
    print(f"中间层级特征形状: {features2.shape}")
    
    # 测试数据增强选项
    extractor.apply_aug_before_feature = False  # 禁用特征提取前的数据增强
    extractor.augmented = True  # 启用常规数据增强
    features3, _, _, _, _, _ = extractor[0]
    print(f"常规数据增强后的特征形状: {features3.shape}")
    
    # 禁用BrainMVP功能测试
    extractor.use_brainmvp = False
    images_3d, _, _, _, _, _ = extractor[0]
    print(f"原始3D图像形状: {images_3d.shape}")    # 应为[batch_size, 20, 20, 20, 1]

if __name__ == "__main__":
    test_brainmvp_extractor()