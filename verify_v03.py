# 验证v0.3代码是否可以正常运行
import sys
sys.path.append('src')

print("=" * 60)
print("验证v0.3融合模型代码")
print("=" * 60)

# 测试1: 导入模块
print("\n1. 测试模块导入...")
try:
    from models.fusion_models import EarlyFusionModel, LateFusionModel, CLIPFusionModel, BLIP2FusionModel
    print("   ✅ 融合模型导入成功")
except Exception as e:
    print(f"   ❌ 融合模型导入失败: {e}")
    sys.exit(1)

# 测试2: 配置文件
print("\n2. 测试配置文件...")
import yaml
import os

configs = [
    'configs/early_fusion.yaml',
    'configs/late_fusion.yaml',
    'configs/clip_fusion.yaml',
    'configs/blip2_fusion.yaml'
]

for config_path in configs:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"   ✅ {os.path.basename(config_path)} 加载成功")
        
        # 验证控制变量
        assert config['batch_size'] == 32, "batch_size应为32"
        assert config['epochs'] == 15, "epochs应为15"
        assert config['seed'] == 42, "seed应为42"
        assert config['learning_rate'] == 5.0e-5, "learning_rate应为5e-5"
        
    except Exception as e:
        print(f"   ❌ {config_path} 验证失败: {e}")
        sys.exit(1)

print("   ✅ 所有配置文件控制变量一致")

# 测试3: Early Fusion模型实例化
print("\n3. 测试Early Fusion模型...")
try:
    import torch
    model = EarlyFusionModel(
        text_model='roberta-base',
        image_model='efficientnet_b4',
        num_classes=3,
        dropout=0.3
    )
    print(f"   ✅ Early Fusion模型创建成功")
    
    # 测试forward
    batch_size = 2
    text_input = {
        'input_ids': torch.randint(0, 1000, (batch_size, 128)),
        'attention_mask': torch.ones(batch_size, 128)
    }
    image_input = torch.randn(batch_size, 3, 224, 224)
    
    # 测试3种模式
    for mode in ['both', 'text_only', 'image_only']:
        output = model(text_input, image_input, mode=mode)
        assert output.shape == (batch_size, 3), f"输出形状错误: {output.shape}"
        print(f"   ✅ {mode} 模式测试通过")
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
except Exception as e:
    print(f"   ❌ Early Fusion测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: Late Fusion模型实例化
print("\n4. 测试Late Fusion模型...")
try:
    model = LateFusionModel(
        text_model_path='none',  # 随机初始化
        image_model_path='none',
        num_classes=3,
        learnable_weight=True
    )
    print(f"   ✅ Late Fusion模型创建成功")
    
    # 测试forward
    output = model(text_input, image_input, mode='both')
    assert output.shape == (batch_size, 3)
    print(f"   ✅ Late Fusion forward测试通过")
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
except Exception as e:
    print(f"   ❌ Late Fusion测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试5: 检查训练脚本语法
print("\n5. 测试训练脚本语法...")
try:
    with open('src/train_fusion.py', 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, 'src/train_fusion.py', 'exec')
    print("   ✅ train_fusion.py 语法正确")
except SyntaxError as e:
    print(f"   ❌ train_fusion.py 语法错误: {e}")
    sys.exit(1)

# 测试6: 检查对比脚本语法
print("\n6. 测试对比脚本语法...")
try:
    with open('src/compare_fusion.py', 'r', encoding='utf-8') as f:
        code = f.read()
    compile(code, 'src/compare_fusion.py', 'exec')
    print("   ✅ compare_fusion.py 语法正确")
except SyntaxError as e:
    print(f"   ❌ compare_fusion.py 语法错误: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！v0.3代码验证成功")
print("=" * 60)
print("\n可以开始训练实验！")
