# 训练脚本
# 支持文本、图像和多模态模型的训练

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from dataset import MultimodalDataset, TextPreprocessor, get_image_transforms
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from utils import set_seed, create_dirs, get_device, AverageMeter


class Trainer:
    # 训练器类
    
    def __init__(self, config):
        # 初始化训练器
        # Args:
        #   config: 配置字典
        self.config = config
        set_seed(config['seed'])
        self.device = get_device()
        
        # 创建必要目录
        create_dirs([
            config['checkpoint_dir'],
            config['log_dir']
        ])
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(config['log_dir'])
        
        # 加载数据
        self.train_loader, self.val_loader = self._prepare_data()
        
        # 初始化模型
        self.model = self._build_model()
        self.model = self.model.to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs']
        )
        
        self.best_acc = 0.0
        self.best_epoch = 0
    
    def _prepare_data(self):
        # 准备数据加载器
        config = self.config
        
        # 文本预处理
        text_preprocessor = TextPreprocessor(max_length=config['max_text_length'])
        
        # 图像预处理
        train_image_transform = get_image_transforms(
            image_size=config['image_size'],
            augment=config['augment']
        )
        val_image_transform = get_image_transforms(
            image_size=config['image_size'],
            augment=False
        )
        
        # 训练集
        train_dataset = MultimodalDataset(
            csv_file=config['train_file'],
            data_dir=config['data_dir'],
            text_transform=text_preprocessor,
            image_transform=train_image_transform
        )
        
        # 验证集
        val_dataset = MultimodalDataset(
            csv_file=config['val_file'],
            data_dir=config['data_dir'],
            text_transform=text_preprocessor,
            image_transform=val_image_transform
        )
        
        # 数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _build_model(self):
        # 构建模型
        config = self.config
        model_type = config['model_type']
        
        if model_type == 'text_only':
            model = TextEncoder(
                model_name=config['text_model'],
                num_classes=config['num_classes'],
                dropout=config['dropout']
            )
        elif model_type == 'image_only':
            model = ImageEncoder(
                model_name=config['image_model'],
                num_classes=config['num_classes'],
                pretrained=config['pretrained'],
                dropout=config['dropout']
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    
    def train_epoch(self, epoch):
        # 训练一个epoch
        self.model.train()
        losses = AverageMeter()
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        
        for batch in pbar:
            texts = batch['text']
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            if self.config['model_type'] == 'text_only':
                # 文本模型
                encoding = self.model.encode_text(texts, device=self.device)
                logits = self.model(encoding['input_ids'], encoding['attention_mask'])
            else:
                # 图像模型
                logits = self.model(images)
            
            loss = self.criterion(logits, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            losses.update(loss.item(), labels.size(0))
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        train_acc = 100. * correct / total
        return losses.avg, train_acc
    
    def validate(self):
        # 验证模型
        self.model.eval()
        losses = AverageMeter()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                texts = batch['text']
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                if self.config['model_type'] == 'text_only':
                    encoding = self.model.encode_text(texts, device=self.device)
                    logits = self.model(encoding['input_ids'], encoding['attention_mask'])
                else:
                    logits = self.model(images)
                
                loss = self.criterion(logits, labels)
                
                # 统计
                losses.update(loss.item(), labels.size(0))
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        return losses.avg, val_acc
    
    def train(self):
        # 完整训练流程
        print(f"\nTraining {self.config['model_type']} model...")
        print(f"Device: {self.device}")
        
        for epoch in range(self.config['epochs']):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 学习率调度
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录到TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # 打印结果
            print(f'\nEpoch {epoch+1}/{self.config["epochs"]}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Learning Rate: {current_lr:.6f}')
            
            # 保存最佳模型
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch + 1
                self.save_checkpoint('best_model.pth')
                print(f'  Best model saved! (Acc: {val_acc:.2f}%)')
            
            # 保存最新模型
            self.save_checkpoint('last_model.pth')
        
        print(f'\nTraining completed!')
        print(f'Best validation accuracy: {self.best_acc:.2f}% (Epoch {self.best_epoch})')
        
        self.writer.close()
    
    def save_checkpoint(self, filename):
        # 保存检查点
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'config': self.config
        }
        
        filepath = os.path.join(self.config['checkpoint_dir'], filename)
        torch.save(checkpoint, filepath)


def main():
    parser = argparse.ArgumentParser(description='Train sentiment classification model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 训练
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
