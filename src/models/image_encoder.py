# 图像编码器模型
# 基于EfficientNet/ResNet/ViT的图像分类模型

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    # 纯图像情感分类模型
    
    def __init__(self, model_name='efficientnet_b4', num_classes=3, pretrained=True, dropout=0.1):
        # 初始化图像编码器
        # Args:
        #   model_name: 模型名称 (efficientnet_b4, resnet50, resnet101, vit等)
        #   num_classes: 分类类别数
        #   pretrained: 是否使用预训练权重
        #   dropout: dropout比率
        super(ImageEncoder, self).__init__()
        
        self.model_name = model_name
        
        # 加载预训练模型
        if model_name == 'efficientnet_b4':
            self.encoder = models.efficientnet_b4(pretrained=pretrained)
            feature_dim = self.encoder.classifier[1].in_features
            self.encoder.classifier = nn.Identity()  # 移除原始分类头
            
        elif model_name == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()  # 移除原始分类头
            
        elif model_name == 'resnet101':
            self.encoder = models.resnet101(pretrained=pretrained)
            feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
            
        elif model_name == 'vit_b_16':
            self.encoder = models.vit_b_16(pretrained=pretrained)
            feature_dim = self.encoder.heads.head.in_features
            self.encoder.heads.head = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # 投影层 (1792 → 512，统一多模态特征维度)
        self.projection = nn.Linear(feature_dim, 512)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, images):
        # 前向传播
        # Args:
        #   images: 图像张量 (batch_size, 3, H, W)
        # Returns:
        #   logits: 分类logits
        features = self.encoder(images)
        
        # 投影到512维
        projected = self.projection(features)
        
        logits = self.classifier(projected)
        return logits
