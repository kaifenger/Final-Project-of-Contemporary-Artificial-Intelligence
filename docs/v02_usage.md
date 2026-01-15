# v0.2版本使用说明

## 概述
v0.2实现了两个单模态基线模型用于消融实验。

## 已实现内容

### 模型
1. **纯文本模型** (`src/models/text_encoder.py`)
   - 基于BERT-base-uncased
   - 使用[CLS] token进行分类
   - 双层MLP分类头

2. **纯图像模型** (`src/models/image_encoder.py`)
   - 基于ResNet50（预训练）
   - 双层MLP分类头
   - 支持数据增强

### 脚本
1. **训练脚本** (`src/train.py`)
   - 支持文本/图像单模态训练
   - TensorBoard日志记录
   - 自动保存最佳模型
   - 学习率余弦退火

2. **评估脚本** (`src/evaluate.py`)
   - 计算准确率、F1分数
   - 生成分类报告
   - 绘制混淆矩阵

## 使用方法

### 方法1：使用运行脚本（推荐）
```powershell
.\run_v02.ps1
```
然后按照菜单选择操作。

### 方法2：直接命令行

**训练文本模型：**
```powershell
C:/Users/Lenovo/AppData/Local/Programs/Python/Python312/python.exe src/train.py --config configs/text_only.yaml
```

**训练图像模型：**
```powershell
C:/Users/Lenovo/AppData/Local/Programs/Python/Python312/python.exe src/train.py --config configs/image_only.yaml
```

**评估文本模型：**
```powershell
C:/Users/Lenovo/AppData/Local/Programs/Python/Python312/python.exe src/evaluate.py --config configs/text_only.yaml --checkpoint checkpoints/text_only/best_model.pth
```

**评估图像模型：**
```powershell
C:/Users/Lenovo/AppData/Local/Programs/Python/Python312/python.exe src/evaluate.py --config configs/image_only.yaml --checkpoint checkpoints/image_only/best_model.pth
```

## 配置文件

### 文本模型配置 (`configs/text_only.yaml`)
- 模型: BERT-base-uncased
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 10
- 不使用图像数据增强

### 图像模型配置 (`configs/image_only.yaml`)
- 模型: ResNet50
- Batch size: 32
- Learning rate: 1e-4
- Epochs: 20
- 使用图像数据增强

## 输出结果

### 训练过程
- 检查点保存在 `checkpoints/[model_type]/`
- TensorBoard日志在 `experiments/logs/[model_type]/`

### 评估结果
- 混淆矩阵保存在 `results/evaluations/`
- 控制台输出详细指标

## 查看TensorBoard
```powershell
tensorboard --logdir experiments/logs
```
然后访问 http://localhost:6006

## 消融实验

**Exp-1: 纯文本基线**
- 只使用文本数据
- 评估文本模态的判别能力

**Exp-2: 纯图像基线**
- 只使用图像数据
- 评估图像模态的判别能力

## 预期性能
根据数据集特点，预期：
- 文本模型准确率: 60-70%
- 图像模型准确率: 45-55%
- 文本模态对情感分类贡献更大

## 注意事项
1. 首次运行会下载预训练模型（约500MB）
2. 需要GPU支持（建议4GB以上显存）
3. 训练时间：文本模型约30分钟，图像模型约1小时（视硬件而定）
