# 多模态融合模型
# 实现4种融合方法：Early Fusion, Late Fusion, CLIP-based, BLIP-2

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, Blip2Model, Blip2Processor
from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder


class EarlyFusionModel(nn.Module):
    # 早期融合：特征级拼接
    
    def __init__(self, text_model='roberta-base', image_model='efficientnet_b4', 
                 num_classes=3, dropout=0.3, pretrained=True, freeze_backbone=True):
        # 初始化早期融合模型
        # Args:
        #   text_model: 文本模型名称
        #   image_model: 图像模型名称
        #   num_classes: 分类数
        #   dropout: dropout比率
        #   pretrained: 是否使用预训练权重
        #   freeze_backbone: 是否冻结预训练backbone
        super(EarlyFusionModel, self).__init__()
        
        # 文本编码器
        self.text_encoder = TextEncoder(text_model, num_classes=num_classes, dropout=0.1)
        # 移除分类头，只保留编码器
        if 'roberta' in text_model.lower():
            text_feat_dim = 768
        else:
            text_feat_dim = 768
        
        # 图像编码器
        self.image_encoder = ImageEncoder(image_model, num_classes=num_classes, 
                                         pretrained=pretrained, dropout=0.1)
        # 获取图像特征维度
        if image_model == 'efficientnet_b4':
            image_feat_dim = 1792
        elif 'resnet' in image_model:
            image_feat_dim = 2048
        else:
            image_feat_dim = 1792
        
        # 冻结预训练模型参数(小数据集标准做法)
        if freeze_backbone:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        
        # 融合层（特征拼接后的MLP）
        self.fusion = nn.Sequential(
            nn.Linear(text_feat_dim + image_feat_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        self.text_feat_dim = text_feat_dim
        self.image_feat_dim = image_feat_dim
    
    def forward(self, text_input, image_input, mode='both'):
        # 前向传播，支持消融实验
        # Args:
        #   text_input: dict with input_ids, attention_mask
        #   image_input: 图像张量
        #   mode: 'both', 'text_only', 'image_only'
        # Returns:
        #   logits
        
        if mode == 'both':
            # 提取文本特征
            text_outputs = self.text_encoder.encoder(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask']
            )
            text_feat = text_outputs.pooler_output
            
            # 提取图像特征
            image_feat = self.image_encoder.encoder(image_input)
            
            # 拼接
            fused_feat = torch.cat([text_feat, image_feat], dim=1)
            
        elif mode == 'text_only':
            # 只用文本
            text_outputs = self.text_encoder.encoder(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask']
            )
            text_feat = text_outputs.pooler_output
            
            # 图像部分用零向量
            batch_size = text_feat.size(0)
            image_feat = torch.zeros(batch_size, self.image_feat_dim, 
                                    device=text_feat.device)
            
            fused_feat = torch.cat([text_feat, image_feat], dim=1)
            
        elif mode == 'image_only':
            # 只用图像
            image_feat = self.image_encoder.encoder(image_input)
            
            # 文本部分用零向量
            batch_size = image_feat.size(0)
            text_feat = torch.zeros(batch_size, self.text_feat_dim,
                                   device=image_feat.device)
            
            fused_feat = torch.cat([text_feat, image_feat], dim=1)
        
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # 融合分类
        logits = self.fusion(fused_feat)
        
        return logits


class LateFusionModel(nn.Module):
    # 晚期融合：决策级加权
    
    def __init__(self, text_model_path, image_model_path, num_classes=3, learnable_weight=True, freeze_backbone=True):
        # 初始化晚期融合模型
        # Args:
        #   text_model_path: 已训练的文本模型路径
        #   image_model_path: 已训练的图像模型路径
        #   num_classes: 分类数
        #   learnable_weight: 是否使用可学习权重
        #   freeze_backbone: 是否冻结预训练backbone
        super(LateFusionModel, self).__init__()
        
        # 加载已训练的单模态模型
        self.text_classifier = self._load_model(text_model_path, 'text')
        self.image_classifier = self._load_model(image_model_path, 'image')
        
        # 冻结单模态模型
        if freeze_backbone:
            for param in self.text_classifier.parameters():
                param.requires_grad = False
            for param in self.image_classifier.parameters():
                param.requires_grad = False
        
        # 融合权重
        if learnable_weight:
            self.alpha = nn.Parameter(torch.tensor(0.5))  # 文本权重
        else:
            self.register_buffer('alpha', torch.tensor(0.5))
        
        self.learnable_weight = learnable_weight
    
    def _load_model(self, model_path, model_type):
        # 加载模型
        if model_type == 'text':
            model = TextEncoder(model_name='roberta-base', num_classes=3)
        else:
            model = ImageEncoder(model_name='efficientnet_b4', num_classes=3)
        
        # 加载权重
        if model_path and model_path != 'none':
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded {model_type} model from {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load {model_type} model: {e}")
                print(f"   Using random initialization")
        
        return model
    
    def forward(self, text_input, image_input, mode='both'):
        # 前向传播
        # Args:
        #   text_input: dict with input_ids, attention_mask
        #   image_input: 图像张量
        #   mode: 'both', 'text_only', 'image_only'
        
        if mode == 'text_only':
            return self.text_classifier(
                text_input['input_ids'], 
                text_input['attention_mask']
            )
        
        elif mode == 'image_only':
            return self.image_classifier(image_input)
        
        else:  # both
            text_logits = self.text_classifier(
                text_input['input_ids'],
                text_input['attention_mask']
            )
            image_logits = self.image_classifier(image_input)
            
            # 加权融合
            alpha = torch.sigmoid(self.alpha) if self.learnable_weight else self.alpha
            fused_logits = alpha * text_logits + (1 - alpha) * image_logits
            
            return fused_logits


class CLIPFusionModel(nn.Module):
    # 基于CLIP的融合模型
    
    def __init__(self, model_name='openai/clip-vit-base-patch32', num_classes=3, 
                 freeze_clip=True, dropout=0.3):
        # 初始化CLIP融合模型
        # Args:
        #   model_name: CLIP模型名称
        #   num_classes: 分类数
        #   freeze_clip: 是否冻结CLIP参数
        #   dropout: dropout比率
        super(CLIPFusionModel, self).__init__()
        
        # 加载CLIP模型
        self.clip = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # 是否冻结CLIP
        if freeze_clip:
            self.clip.requires_grad_(False)
        
        # 获取CLIP特征维度
        clip_dim = self.clip.config.projection_dim  # 512 for base model
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.freeze_clip = freeze_clip
    
    def forward(self, text_input, image_input, mode='both'):
        # 前向传播
        
        if self.freeze_clip:
            with torch.no_grad():
                outputs = self.clip(
                    input_ids=text_input['input_ids'],
                    attention_mask=text_input['attention_mask'],
                    pixel_values=image_input,
                    return_dict=True
                )
        else:
            outputs = self.clip(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                pixel_values=image_input,
                return_dict=True
            )
        
        # 使用图文联合特征
        if mode == 'both':
            # 取文本和图像特征的平均
            text_embeds = outputs.text_embeds
            image_embeds = outputs.image_embeds
            joint_embeds = (text_embeds + image_embeds) / 2
        elif mode == 'text_only':
            joint_embeds = outputs.text_embeds
        elif mode == 'image_only':
            joint_embeds = outputs.image_embeds
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # 分类
        logits = self.classifier(joint_embeds)
        
        return logits


class BLIP2FusionModel(nn.Module):
    # 基于BLIP-2的融合模型
    
    def __init__(self, model_name='Salesforce/blip2-opt-2.7b', num_classes=3,
                 freeze_blip=True, dropout=0.3):
        # 初始化BLIP-2融合模型
        # Args:
        #   model_name: BLIP-2模型名称
        #   num_classes: 分类数
        #   freeze_blip: 是否冻结BLIP-2参数
        #   dropout: dropout比率
        super(BLIP2FusionModel, self).__init__()
        
        # 加载BLIP-2模型
        self.blip = Blip2Model.from_pretrained(model_name)
        self.processor = Blip2Processor.from_pretrained(model_name)
        
        # 是否冻结BLIP-2
        if freeze_blip:
            self.blip.requires_grad_(False)
        
        # 获取BLIP-2特征维度
        blip_dim = self.blip.config.text_config.hidden_size  # 通常是768
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(blip_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.freeze_blip = freeze_blip
    
    def forward(self, text_input, image_input, mode='both'):
        # 前向传播
        
        if self.freeze_blip:
            with torch.no_grad():
                outputs = self.blip(
                    input_ids=text_input['input_ids'],
                    attention_mask=text_input['attention_mask'],
                    pixel_values=image_input,
                    return_dict=True
                )
        else:
            outputs = self.blip(
                input_ids=text_input['input_ids'],
                attention_mask=text_input['attention_mask'],
                pixel_values=image_input,
                return_dict=True
            )
        
        # 使用[CLS] token的特征
        features = outputs.last_hidden_state[:, 0, :]
        
        # 分类
        logits = self.classifier(features)
        
        return logits
