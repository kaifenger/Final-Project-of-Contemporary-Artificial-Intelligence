# 文本编码器模型
# 基于BERT的文本分类模型

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer


class TextEncoder(nn.Module):
    # 纯文本情感分类模型
    
    def __init__(self, model_name='bert-base-uncased', num_classes=3, dropout=0.1):
        # 初始化文本编码器
        # Args:
        #   model_name: 预训练模型名称
        #   num_classes: 分类类别数
        #   dropout: dropout比率
        super(TextEncoder, self).__init__()
        
        self.model_name = model_name
        
        # 加载预训练模型
        if 'roberta' in model_name.lower():
            self.encoder = RobertaModel.from_pretrained(model_name)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        else:
            self.encoder = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        hidden_size = self.encoder.config.hidden_size
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        # 前向传播
        # Args:
        #   input_ids: token ids
        #   attention_mask: attention mask
        # Returns:
        #   logits: 分类logits
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS] token的输出
        pooled_output = outputs.pooler_output
        
        # 分类
        logits = self.classifier(pooled_output)
        
        return logits
    
    def encode_text(self, texts, max_length=128, device='cuda'):
        # 编码文本
        # Args:
        #   texts: 文本列表
        #   max_length: 最大长度
        #   device: 设备
        # Returns:
        #   encoding: 编码后的字典
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {k: v.to(device) for k, v in encoding.items()}
