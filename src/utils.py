# 工具函数模块
# 提供项目中常用的辅助功能

import os
import random
import numpy as np
import torch
import json
from typing import Dict, Any


def set_seed(seed: int = 42):
    # 设置随机种子以确保实验可复现
    # Args:
    #   seed: 随机种子值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def create_dirs(dirs: list):
    # 创建必要的目录
    # Args:
    #   dirs: 目录路径列表
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        

def save_json(data: Dict[Any, Any], filepath: str):
    # 保存数据为JSON文件
    # Args:
    #   data: 要保存的数据
    #   filepath: 保存路径
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(filepath: str) -> Dict[Any, Any]:
    # 加载JSON文件
    # Args:
    #   filepath: 文件路径
    # Returns:
    #   加载的数据
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_device():
    # 获取可用的计算设备
    # Returns:
    #   torch.device对象
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


class AverageMeter:
    # 计算并存储平均值和当前值
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
