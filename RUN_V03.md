# v0.3 多模态融合方法对比实验 - 运行指令

## 环境准备

确保已安装所有依赖：
```bash
pip install transformers==4.30.0
pip install torch==2.0.1 torchvision==0.15.2
```

## 前置条件

**重要：需要先训练v0.2的单模态模型（Late Fusion需要）**

如果还未训练v0.2，先运行：
```bash
# 训练文本模型
python src/train.py --config configs/text_only.yaml

# 训练图像模型  
python src/train.py --config configs/image_only.yaml
```

## 运行v0.3对比实验（4种融合方法）

### 方案A：逐个训练（推荐，便于调试）

```bash
# 1. Early Fusion (预计2.5小时)
python src/train_fusion.py --config configs/early_fusion.yaml

# 2. Late Fusion (预计10分钟，复用v0.2模型)
python src/train_fusion.py --config configs/late_fusion.yaml

# 3. CLIP Fusion (预计1小时)
python src/train_fusion.py --config configs/clip_fusion.yaml

# 4. BLIP-2 Fusion (预计1.5小时)
python src/train_fusion.py --config configs/blip2_fusion.yaml
```

### 方案B：批量运行（睡觉前一次性启动）

创建批处理文件 `run_all_fusion.bat`:
```batch
@echo off
echo 开始训练v0.3所有融合模型...
echo.

echo [1/4] 训练Early Fusion...
python src/train_fusion.py --config configs/early_fusion.yaml
if %errorlevel% neq 0 (
    echo Early Fusion训练失败！
    pause
    exit /b 1
)

echo.
echo [2/4] 训练Late Fusion...
python src/train_fusion.py --config configs/late_fusion.yaml
if %errorlevel% neq 0 (
    echo Late Fusion训练失败！
    pause
    exit /b 1
)

echo.
echo [3/4] 训练CLIP Fusion...
python src/train_fusion.py --config configs/clip_fusion.yaml
if %errorlevel% neq 0 (
    echo CLIP Fusion训练失败！
    pause
    exit /b 1
)

echo.
echo [4/4] 训练BLIP-2 Fusion...
python src/train_fusion.py --config configs/blip2_fusion.yaml
if %errorlevel% neq 0 (
    echo BLIP-2 Fusion训练失败！
    pause
    exit /b 1
)

echo.
echo ========================================
echo 所有模型训练完成！
echo ========================================
echo.
echo 开始生成对比报告...
python src/compare_fusion.py --output_dir experiments/fusion_comparison

echo.
echo 全部完成！请查看 experiments/fusion_comparison 目录
pause
```

然后运行：
```bash
./run_all_fusion.bat
```

## 生成对比报告

训练完成后，运行对比脚本：
```bash
python src/compare_fusion.py --output_dir experiments/fusion_comparison
```

输出文件：
- `experiments/fusion_comparison/comparison_results.csv` - 对比数据表格
- `experiments/fusion_comparison/fusion_comparison.png` - 对比可视化图
- `experiments/fusion_comparison/*_confusion_matrix.png` - 各模型混淆矩阵

## 预期训练时间（GeForce RTX 3090）

| 模型 | 训练时间 | 显存占用 |
|------|---------|---------|
| Early Fusion | ~2.5小时 | ~8GB |
| Late Fusion | ~10分钟 | ~6GB |
| CLIP Fusion | ~1小时 | ~10GB |
| BLIP-2 Fusion | ~1.5小时 | ~12GB |

**总计：约5.5小时**

## 训练监控

使用TensorBoard实时监控：
```bash
tensorboard --logdir=experiments/logs
```

然后访问 http://localhost:6006

## 常见问题

### 1. 显存不足 (CUDA out of memory)
解决方案：
- 减小batch_size（在yaml文件中修改，如32→16）
- 使用更小的BLIP-2模型（改为'Salesforce/blip2-opt-2.7b'）

### 2. Late Fusion找不到checkpoint
确保v0.2的模型已训练并保存在：
- `checkpoints/text_only/best_model.pth`
- `checkpoints/image_only/best_model.pth`

### 3. CLIP/BLIP-2下载失败
使用镜像源：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 检查点说明

训练完成后的checkpoints位置：
```
checkpoints/
├── early_fusion/
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
├── late_fusion/
│   └── best_model.pth
├── clip_fusion/
│   └── best_model.pth
└── blip2_fusion/
    └── best_model.pth
```

## 下一步

训练完成后，查看对比结果，选择最佳模型进入v0.4深化优化。
