# 多模态融合模型训练脚本
# 支持4种融合方法：Early Fusion, Late Fusion, CLIP-based, BLIP-2

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_models import EarlyFusionModel, LateFusionModel, CLIPFusionModel, BLIP2FusionModel
from dataset import MultimodalDataset, TextPreprocessor, get_image_transforms
from utils import set_seed, get_device, AverageMeter


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, config):
    # 训练一个epoch
    model.train()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
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
        acc = (preds == labels).float().mean().item()
        
        # 更新统计
        losses.update(loss.item(), images.size(0))
        accuracies.update(acc, images.size(0))
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc': f'{accuracies.avg:.4f}'
        })
        
        # TensorBoard记录
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('Train/Loss', loss.item(), global_step)
        writer.add_scalar('Train/Acc', acc, global_step)
    
    return losses.avg, accuracies.avg


def validate(model, val_loader, criterion, device, mode='both'):
    # 验证模型
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
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
            acc = (preds == labels).float().mean().item()
            
            # 更新统计
            losses.update(loss.item(), images.size(0))
            accuracies.update(acc, images.size(0))
            
            # 收集预测结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return losses.avg, accuracies.avg, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description='Train multimodal fusion model')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的checkpoint路径')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 设置设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
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
            dropout=config['dropout']
        )
    elif fusion_type == 'late':
        model = LateFusionModel(
            text_model_path=config.get('text_model_path', 'none'),
            image_model_path=config.get('image_model_path', 'none'),
            num_classes=config['num_classes'],
            learnable_weight=config.get('learnable_weight', True)
        )
    elif fusion_type == 'clip':
        model = CLIPFusionModel(
            model_name=config.get('clip_model', 'openai/clip-vit-base-patch32'),
            num_classes=config['num_classes'],
            freeze_clip=config.get('freeze_clip', True),
            dropout=config['dropout']
        )
    elif fusion_type == 'blip2':
        model = BLIP2FusionModel(
            model_name=config.get('blip2_model', 'Salesforce/blip2-opt-2.7b'),
            num_classes=config['num_classes'],
            freeze_blip=config.get('freeze_blip', True),
            dropout=config['dropout']
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    model = model.to(device)
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
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
    
    # 训练循环
    best_val_acc = 0.0
    start_epoch = 0
    
    for epoch in range(start_epoch, config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, config
        )
        
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # 验证
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device, mode='both')
        
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
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
    
    writer.close()
    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
