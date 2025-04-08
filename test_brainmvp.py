import sys
import torch

# Test introducing brainmvpGenerator
try:
    from brainmvpGenerator import BrainMVPFeatureExtractor, BrainMVPExtractor
    print("成功导入brainmvpGenerator模块")
except Exception as e:
    print(f"导入brainmvpGenerator失败: {e}")
    sys.exit(1)

# Test loading pretrained model
try:
    feature_extractor = BrainMVPFeatureExtractor(
        num_phase=1,
        pretrained_path="./pretrained/brainmvp_uniformer.pth",
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("成功加载BrainMVP预训练模型")
    
    # Test forcast
    test_input = torch.randn(2, 1, 20, 20, 20)
    if torch.cuda.is_available():
        test_input = test_input.cuda()
    with torch.no_grad():
        outputs = feature_extractor(test_input)
    print(f"模型前向传播成功，输出特征形状: {[o.shape for o in outputs]}")
except Exception as e:
    print(f"加载或测试模型失败: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成")
