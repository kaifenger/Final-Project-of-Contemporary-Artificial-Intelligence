# 多模态融合模型训练脚本
# 支持3种融合方法：Early Fusion, Late Fusion, Cross-Attention Fusion

print("=" * 60)
print("启动训练脚本...")
print("=" * 60)

import os
import sys

# 设置环境变量避免多线程冲突
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import yaml
print("正在导入PyTorch...")
import torch
torch.set_num_threads(1)  # 限制线程数
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import numpy as np

print("正在导入自定义模块...")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_models import EarlyFusionModel, LateFusionModel, CrossAttentionFusion
from dataset import MultimodalDataset, TextPreprocessor, get_image_transforms
from utils import set_seed, get_device, AverageMeter
print("导入完成！")


class EarlyStopping:
    """Early Stopping工具"""
    
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                return False
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                return False
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False



def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, config):
    # 训练一个epoch
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]')
    
    for batch_idx, batch in enumerate(pbar):
        # 从字典中提取数据
        text = batch['text']
        images = batch['image']
        labels = batch['label']
        
        # 移动到设备
        images = images.to(device)
        labels = labels.to(device)
        text_input = {k: v.to(device) for k, v in text.items()}
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(text_input, images, mode='both')
        loss = criterion(logits, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()  # 正确样本数
        batch_acc = correct / images.size(0)  # 当前batch准确率
        
        # 更新统计
        losses.update(loss.item(), images.size(0))
        accuracies.update(batch_acc, images.size(0))  # 传入准确率比例，进行加权平均
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.4f}'
        })
        
        # TensorBoard记录
        global_step = epoch * len(train_loader) + batch_idx
        batch_acc = correct / images.size(0)
        writer.add_scalar('Train/Loss', losses.avg, global_step)
        writer.add_scalar('Train/Acc', batch_acc, global_step)
        
        # 释放中间变量内存
        del text_input, images, labels, logits, loss, preds
        if batch_idx % 10 == 0:  # 每10个batch清理一次
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return losses.avg, accuracies.avg


def validate(model, val_loader, criterion, device, mode='both'):
    # 验证模型
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]'):
            # 从字典中提取数据
            text = batch['text']
            images = batch['image']
            labels = batch['label']
            
            images = images.to(device)
            labels = labels.to(device)
            text_input = {k: v.to(device) for k, v in text.items()}
            
            # 前向传播
            logits = model(text_input, images, mode=mode)
            loss = criterion(logits, labels)
            
            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()  # 正确样本数
            batch_acc = correct / images.size(0)  # 当前batch准确率
            
            # 更新统计
            losses.update(loss.item(), images.size(0))
            accuracies.update(batch_acc, images.size(0))  # 传入准确率比例，进行加权平均
            
            # 收集预测结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return losses.avg, accuracies.avg, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train multimodal fusion model')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')
    args = parser.parse_args()
    
    print("开始加载配置...")
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设置设备
    device = get_device()
    print(f"使用设备: {device}")
    
    print("正在加载tokenizer...")
    # 创建tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    print("正在创建数据集...")
    # 创建数据集
    text_preprocessor = TextPreprocessor(max_length=config['max_text_length'])
    train_transform = get_image_transforms(config['image_size'], augment=config['augment'])
    val_transform = get_image_transforms(config['image_size'], augment=False)
    
    train_dataset = MultimodalDataset(
        csv_file=config['train_file'],
        data_dir=config['data_dir'],
        tokenizer=tokenizer,
        text_transform=text_preprocessor,
        image_transform=train_transform,
        max_text_length=config['max_text_length']
    )
    
    val_dataset = MultimodalDataset(
        csv_file=config['val_file'],
        data_dir=config['data_dir'],
        tokenizer=tokenizer,
        text_transform=text_preprocessor,
        image_transform=val_transform,
        max_text_length=config['max_text_length']
    )
    
    # 创建数据加载器 (CPU训练时关闭pin_memory避免警告)
    use_pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=use_pin_memory
    )
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 创建模型
    fusion_type = config['fusion_type']
    print(f"融合方法: {fusion_type}")
    
    if fusion_type == 'early':
        model = EarlyFusionModel(
            text_model=config['text_model'],
            image_model=config['image_model'],
            num_classes=config['num_classes'],
            dropout=config['dropout'],
            pretrained=config.get('pretrained', True),
            freeze_backbone=config.get('freeze_backbone', True)
        )
    elif fusion_type == 'late':
        model = LateFusionModel(
            text_model=config['text_model'],
            image_model=config['image_model'],
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.1),
            pretrained=config.get('pretrained', True),
            freeze_backbone=config.get('freeze_backbone', True)
        )
    elif fusion_type == 'cross_attention':
        model = CrossAttentionFusion(
            text_model=config['text_model'],
            image_model=config['image_model'],
            num_classes=config['num_classes'],
            dropout=config['dropout'],
            pretrained=config.get('pretrained', True),
            freeze_backbone=config.get('freeze_backbone', True),
            num_heads=config.get('num_heads', 8)
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    model = model.to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"冻结参数量: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    
    # 损失函数和优化器（分层学习率）
    criterion = nn.CrossEntropyLoss()
    
    # 参数分组：backbone用小学习率微调，projection和classifier用大学习率
    backbone_params = []
    projection_params = []
    classifier_params = []
    
    # 收集各部分参数
    if hasattr(model, 'text_encoder'):
        backbone_params.extend(model.text_encoder.encoder.parameters())
        projection_params.extend(model.text_encoder.projection.parameters())
        if hasattr(model.text_encoder, 'classifier'):
            classifier_params.extend(model.text_encoder.classifier.parameters())
    
    if hasattr(model, 'image_encoder'):
        backbone_params.extend(model.image_encoder.encoder.parameters())
        projection_params.extend(model.image_encoder.projection.parameters())
        if hasattr(model.image_encoder, 'classifier'):
            classifier_params.extend(model.image_encoder.classifier.parameters())
    
    # 收集融合层和其他参数
    fusion_params = []
    for name, param in model.named_parameters():
        if 'encoder' not in name and 'projection' not in name and 'text_encoder.classifier' not in name and 'image_encoder.classifier' not in name:
            fusion_params.append(param)
    
    # 创建参数组
    param_groups = [
        {'params': backbone_params, 'lr': config['backbone_lr']},
        {'params': projection_params, 'lr': config['projection_lr']},
        {'params': classifier_params + fusion_params, 'lr': config['classifier_lr']}
    ]
    
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=config['weight_decay']
    )
    
    print(f"📊 学习率配置: backbone={config['backbone_lr']}, projection={config['projection_lr']}, classifier={config['classifier_lr']}")
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )
    
    # TensorBoard
    os.makedirs(config['log_dir'], exist_ok=True)
    writer = SummaryWriter(config['log_dir'])
    
    # 创建checkpoint目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Early Stopping
    early_stopping = None
    if config.get('early_stopping', {}).get('enabled', False):
        early_stopping = EarlyStopping(
            patience=config['early_stopping']['patience'],
            min_delta=config['early_stopping']['min_delta'],
            mode='max'
        )
        print(f"📉 Early stopping enabled: patience={early_stopping.patience}")
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        print(f"正在从checkpoint恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('val_acc', 0.0)
        print(f"✅ 从epoch {start_epoch}恢复训练，之前最佳准确率: {best_val_acc:.4f}")
    
    # 训练循环
    
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, config
        )
        
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # 验证
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device, mode='both')
        
        # 计算实际准确率（用于验证）
        import numpy as np
        actual_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # 打印预测分布，帮助诊断问题
        from collections import Counter
        pred_dist = Counter(all_preds)
        label_dist = Counter(all_labels)
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.6f} (实际: {actual_acc:.6f})")
        print(f"预测分布: {dict(pred_dist)}, 真实分布: {dict(label_dist)}")
        
        # 垃圾回收
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # TensorBoard记录
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/Train_Acc', train_acc, epoch)
        writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
        writer.add_scalar('Epoch/Val_Acc', val_acc, epoch)
        writer.add_scalar('Epoch/LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 学习率调整
        scheduler.step()
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_path)
            print(f"✅ 保存最佳模型 (Val Acc: {val_acc:.4f})")
        
        # 定期保存checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_path)
        
        # Early Stopping检查
        if early_stopping:
            if early_stopping(val_acc):
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"   Best val acc: {early_stopping.best_score:.4f}")
                break
    
    writer.close()
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
