# 自定义Dataset类
# 用于加载文本和图像数据

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from typing import Optional, Tuple


class MultimodalDataset(Dataset):
    # 多模态情感分类数据集
    
    def __init__(self, 
                 csv_file: str,
                 data_dir: str,
                 text_transform=None,
                 image_transform=None,
                 max_text_length: int = 128):
        # 初始化数据集
        # Args:
        #   csv_file: CSV标签文件路径
        #   data_dir: 数据文件夹路径
        #   text_transform: 文本预处理函数
        #   image_transform: 图像预处理函数
        #   max_text_length: 最大文本长度
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.text_transform = text_transform
        self.image_transform = image_transform
        self.max_text_length = max_text_length
        
        # 标签映射
        self.label_map = {
            'positive': 0,
            'neutral': 1,
            'negative': 2
        }
        self.id2label = {v: k for k, v in self.label_map.items()}
        
    def __len__(self):
        # 返回数据集大小
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取单个样本
        # Args:
        #   idx: 样本索引
        # Returns:
        #   sample: 包含text, image, label的字典
        row = self.data.iloc[idx]
        guid = row['guid']
        
        # 读取文本
        text = self._load_text(guid)
        
        # 读取图像
        image = self._load_image(guid)
        
        # 获取标签
        label_str = row['tag']
        if label_str == 'null' or pd.isna(label_str):
            label = -1  # 测试集标签为-1
        else:
            label = self.label_map[label_str]
        
        # 应用预处理
        if self.text_transform:
            text = self.text_transform(text)
        
        if self.image_transform:
            image = self.image_transform(image)
        
        sample = {
            'guid': guid,
            'text': text,
            'image': image,
            'label': label
        }
        
        return sample
    
    def _load_text(self, guid) -> str:
        # 加载文本文件
        # Args:
        #   guid: 数据唯一标识
        # Returns:
        #   文本内容
        text_path = os.path.join(self.data_dir, f'{guid}.txt')
        
        try:
            with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Text file not found for guid {guid}")
            text = ""
        
        return text
    
    def _load_image(self, guid) -> Image.Image:
        # 加载图像文件
        # Args:
        #   guid: 数据唯一标识
        # Returns:
        #   PIL Image对象
        image_path = os.path.join(self.data_dir, f'{guid}.jpg')
        
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image file not found for guid {guid}")
            # 创建空白图像
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        
        return image


class TextPreprocessor:
    # 文本预处理类
    
    def __init__(self, max_length: int = 128):
        # 初始化文本预处理器
        # Args:
        #   max_length: 最大文本长度
        self.max_length = max_length
    
    def __call__(self, text: str) -> str:
        # 基础文本清洗
        # Args:
        #   text: 原始文本
        # Returns:
        #   清洗后的文本
        # 去除多余空白
        text = ' '.join(text.split())
        
        # 限制长度
        if len(text) > self.max_length * 5:  # 粗略估计字符数
            text = text[:self.max_length * 5]
        
        return text


def get_image_transforms(image_size: int = 224, augment: bool = False):
    # 获取图像预处理transforms
    # Args:
    #   image_size: 图像尺寸
    #   augment: 是否进行数据增强
    # Returns:
    #   torchvision transforms
    from torchvision import transforms
    
    if augment:
        # 训练集数据增强
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # 验证集/测试集
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def test_dataset():
    # 测试数据集加载
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 设置路径
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TRAIN_SPLIT = os.path.join(BASE_DIR, 'data_split', 'train_split.csv')
    
    # 检查文件是否存在
    if not os.path.exists(TRAIN_SPLIT):
        print(f"Error: {TRAIN_SPLIT} not found. Please run data_loader.py first.")
        return
    
    # 创建数据集
    text_preprocessor = TextPreprocessor(max_length=128)
    image_transform = get_image_transforms(image_size=224, augment=False)
    
    dataset = MultimodalDataset(
        csv_file=TRAIN_SPLIT,
        data_dir=DATA_DIR,
        text_transform=text_preprocessor,
        image_transform=image_transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Label mapping: {dataset.label_map}")
    
    # 测试加载几个样本
    print("\nTesting sample loading:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  GUID: {sample['guid']}")
        print(f"  Text (first 100 chars): {sample['text'][:100]}...")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Label: {sample['label']} ({dataset.id2label.get(sample['label'], 'unknown')})")
    
    print("\nDataset test completed successfully!")


if __name__ == '__main__':
    test_dataset()
