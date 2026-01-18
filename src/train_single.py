#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单模态模型训练脚本
支持Text-only和Image-only训练，用于消融实验
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gc
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from data.dataloader import MultimodalDataset
from utils.metrics import AverageMeter


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


def train_epoch(model, dataloader, criterion, optimizer, device, modality):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        text_input = {k: v.to(device) for k, v in batch['text'].items()}
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        batch_size = images.size(0)
        
        # 根据模态选择输入
        if modality == 'text':
            logits = model(text_input['input_ids'], text_input['attention_mask'])
        else:  # image
            logits = model(images)
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        _, preds = torch.max(logits, 1)
        correct = (preds == labels).sum().item()
        batch_acc = correct / batch_size
        
        # 更新统计
        losses.update(loss.item(), batch_size)
        accuracies.update(batch_acc, batch_size)
        
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.4f}'
        })
        
        # 清理内存
        del logits, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return losses.avg, accuracies.avg


def validate(model, dataloader, criterion, device, modality):
    """验证"""
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            text_input = {k: v.to(device) for k, v in batch['text'].items()}
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            batch_size = images.size(0)
            
            # 根据模态选择输入
            if modality == 'text':
                logits = model(text_input['input_ids'], text_input['attention_mask'])
            else:  # image
                logits = model(images)
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 计算准确率
            _, preds = torch.max(logits, 1)
            correct = (preds == labels).sum().item()
            batch_acc = correct / batch_size
            
            # 更新统计
            losses.update(loss.item(), batch_size)
            accuracies.update(batch_acc, batch_size)
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{accuracies.avg:.4f}'
            })
    
    return losses.avg, accuracies.avg


def main():
    parser = argparse.ArgumentParser(description='Train single-modal model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    modality = config['modality']  # 'text' or 'image'
    if modality == 'text':
        model = TextEncoder(
            model_name=config['text_model'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        )
        print(f"✅ Created Text-only model: {config['text_model']}")
    else:
        model = ImageEncoder(
            model_name=config['image_model'],
            num_classes=config['num_classes'],
            pretrained=config['pretrained'],
            dropout=config['dropout']
        )
        print(f"✅ Created Image-only model: {config['image_model']}")
    
    model = model.to(device)
    
    # 使用分层学习率策略（无需冻结backbone）
    print("🎯 使用分层学习率: backbone微调(1e-5), projection+classifier训练(1e-3)")
    
    # 数据加载
    train_dataset = MultimodalDataset(
        csv_file=config['train_file'],
        data_dir=config['data_dir'],
        text_model=config.get('text_model', 'roberta-base'),
        max_text_length=config['max_text_length'],
        image_size=config['image_size'],
        augment=config['augment']
    )
    
    val_dataset = MultimodalDataset(
        csv_file=config['val_file'],
        data_dir=config['data_dir'],
        text_model=config.get('text_model', 'roberta-base'),
        max_text_length=config['max_text_length'],
        image_size=config['image_size'],
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    print(f"📊 Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # 优化器和损失函数（分层学习率）
    # 参数分组：backbone用小学习率微调，projection和classifier用大学习率
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': config['backbone_lr']},
        {'params': model.projection.parameters(), 'lr': config['projection_lr']},
        {'params': model.classifier.parameters(), 'lr': config['classifier_lr']}
    ]
    
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=config['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"📊 学习率配置: backbone={config['backbone_lr']}, projection={config['projection_lr']}, classifier={config['classifier_lr']}")
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )
    
    # TensorBoard
    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # 检查点目录
    checkpoint_dir = config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resumed from epoch {start_epoch}")
    
    # 训练循环
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, config['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"{'='*60}")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, modality
        )
        
        # 验证
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, modality
        )
        
        # 学习率调度
        scheduler.step()
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\n📊 Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"📊 Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"💾 Saved best model (acc: {val_acc:.4f})")
        
        # 保存最新checkpoint
        latest_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, latest_path)
        
        # Early Stopping检查
        if early_stopping:
            if early_stopping(val_acc):
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"   Best val acc: {early_stopping.best_score:.4f}")
                break
        
        # 内存清理
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    writer.close()
    print(f"\n✅ Training completed! Best val acc: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
