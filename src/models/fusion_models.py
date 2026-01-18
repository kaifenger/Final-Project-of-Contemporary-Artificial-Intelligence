# 多模态融合模型
# 实现3种融合方法：Early Fusion, Late Fusion, Cross-Attention Fusion

import torch
import torch.nn as nn
import torch.nn.functional as F
from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder


class EarlyFusionModel(nn.Module):
    """早期融合：特征级拼接（使用512维投影特征）"""
    
    def __init__(self, text_model='roberta-base', image_model='efficientnet_b4', 
                 num_classes=3, dropout=0.3, pretrained=True, freeze_backbone=False):
        super(EarlyFusionModel, self).__init__()
        
        # 文本编码器（已包含768→512投影）
        self.text_encoder = TextEncoder(text_model, num_classes=num_classes, dropout=0.1)
        
        # 图像编码器（已包含1792→512投影）
        self.image_encoder = ImageEncoder(image_model, num_classes=num_classes, 
                                         pretrained=pretrained, dropout=0.1)
        
        # 使用分层学习率策略，无需冻结backbone
        # backbone将用小学习率1e-5微调，projection和fusion用1e-3训练
        
        # 融合层（512+512=1024 → 3）
        self.fusion = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, text_input, image_input):
        """前向传播"""
        # 提取文本特征（768维）
        text_outputs = self.text_encoder.encoder(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask']
        )
        text_feat = text_outputs.pooler_output  # (batch, 768)
        
        # 投影到512维
        text_proj = self.text_encoder.projection(text_feat)  # (batch, 512)
        
        # 提取图像特征（1792维）
        image_feat = self.image_encoder.encoder(image_input)  # (batch, 1792)
        
        # 投影到512维
        image_proj = self.image_encoder.projection(image_feat)  # (batch, 512)
        
        # 拼接融合特征
        fused_feat = torch.cat([text_proj, image_proj], dim=1)  # (batch, 1024)
        
        # 分类
        logits = self.fusion(fused_feat)
        
        return logits


class LateFusionModel(nn.Module):
    """晚期融合：决策级加权（使用512维投影特征）"""
    
    def __init__(self, text_model='roberta-base', image_model='efficientnet_b4',
                 num_classes=3, dropout=0.1, pretrained=True, freeze_backbone=False):
        super(LateFusionModel, self).__init__()
        
        # 文本分支（encoder + 投影 + 分类头）
        self.text_encoder = TextEncoder(text_model, num_classes=num_classes, dropout=dropout)
        
        # 图像分支（encoder + 投影 + 分类头）
        self.image_encoder = ImageEncoder(image_model, num_classes=num_classes, 
                                         pretrained=pretrained, dropout=dropout)
        
        # 使用分层学习率策略，无需冻结backbone
        # backbone将用小学习率1e-5微调，projection和classifier用1e-3训练
        
        # 可学习融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 文本权重
    
    def forward(self, text_input, image_input):
        """前向传播"""
        # 文本分类logits
        text_logits = self.text_encoder(
            text_input['input_ids'],
            text_input['attention_mask']
        )
        
        # 图像分类logits
        image_logits = self.image_encoder(image_input)
        
        # 加权融合
        alpha = torch.sigmoid(self.alpha)
        fused_logits = alpha * text_logits + (1 - alpha) * image_logits
        
        return fused_logits


class CrossAttentionFusion(nn.Module):
    """Cross-Attention融合：文本和图像特征互相注意力"""
    
    def __init__(self, text_model='roberta-base', image_model='efficientnet_b4',
                 num_classes=3, dropout=0.3, pretrained=True, freeze_backbone=False,
                 num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        
        # 文本编码器（encoder + 512维投影）
        self.text_encoder = TextEncoder(text_model, num_classes=num_classes, dropout=0.1)
        
        # 图像编码器（encoder + 512维投影）
        self.image_encoder = ImageEncoder(image_model, num_classes=num_classes,
                                         pretrained=pretrained, dropout=0.1)
        
        # 使用分层学习率策略，无需冻结backbone
        # backbone将用小学习率1e-5微调，projection和attention+classifier用1e-3训练
        
        # Cross-Attention层（512维）
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization
        self.text_norm = nn.LayerNorm(512)
        self.image_norm = nn.LayerNorm(512)
        
        # 融合分类头（1024 → 3）
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, text_input, image_input):
        """前向传播"""
        # 提取文本特征并投影到512维
        text_outputs = self.text_encoder.encoder(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask']
        )
        text_feat = text_outputs.pooler_output  # (batch, 768)
        text_proj = self.text_encoder.projection(text_feat)  # (batch, 512)
        
        # 提取图像特征并投影到512维
        image_feat = self.image_encoder.encoder(image_input)  # (batch, 1792)
        image_proj = self.image_encoder.projection(image_feat)  # (batch, 512)
        
        # 为attention添加序列维度
        text_seq = text_proj.unsqueeze(1)  # (batch, 1, 512)
        image_seq = image_proj.unsqueeze(1)  # (batch, 1, 512)
        
        # Text attends to Image
        text_attended, _ = self.text_to_image_attn(
            query=text_seq,
            key=image_seq,
            value=image_seq
        )
        text_attended = self.text_norm(text_seq + text_attended)  # Residual
        
        # Image attends to Text
        image_attended, _ = self.image_to_text_attn(
            query=image_seq,
            key=text_seq,
            value=text_seq
        )
        image_attended = self.image_norm(image_seq + image_attended)  # Residual
        
        # 移除序列维度并拼接
        text_final = text_attended.squeeze(1)  # (batch, 512)
        image_final = image_attended.squeeze(1)  # (batch, 512)
        fused_feat = torch.cat([text_final, image_final], dim=1)  # (batch, 1024)
        
        # 分类
        logits = self.classifier(fused_feat)
        
        return logits

