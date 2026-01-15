# 数据加载和统计分析模块
# 负责读取数据、生成统计报告、划分训练集和验证集

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from utils import set_seed, create_dirs, save_json


class DataLoader:
    # 数据加载和处理类
    
    def __init__(self, data_dir: str, train_file: str, test_file: str, seed: int = 42):
        # 初始化数据加载器
        # Args:
        #   data_dir: 数据文件夹路径
        #   train_file: 训练标签文件路径
        #   test_file: 测试标签文件路径
        #   seed: 随机种子
        self.data_dir = data_dir
        self.train_file = train_file
        self.test_file = test_file
        self.seed = seed
        set_seed(seed)
        
        # 读取数据
        self.train_df = pd.read_csv(train_file)
        self.test_df = pd.read_csv(test_file)
        
    def get_data_statistics(self):
        # 获取数据集统计信息
        # Returns:
        #   统计信息字典
        stats = {}
        
        # 训练集统计
        stats['train_size'] = len(self.train_df)
        stats['test_size'] = len(self.test_df)
        stats['total_size'] = stats['train_size'] + stats['test_size']
        
        # 类别分布
        label_counts = self.train_df['tag'].value_counts().to_dict()
        stats['label_distribution'] = label_counts
        
        # 类别比例
        total = stats['train_size']
        stats['label_percentage'] = {
            label: f"{count/total*100:.2f}%" 
            for label, count in label_counts.items()
        }
        
        # 检查数据完整性
        stats['missing_text_files'] = self._check_missing_files('.txt')
        stats['missing_image_files'] = self._check_missing_files('.jpg')
        
        return stats
    
    def _check_missing_files(self, extension: str):
        # 检查缺失的数据文件
        # Args:
        #   extension: 文件扩展名 (.txt 或 .jpg)
        # Returns:
        #   缺失文件的guid列表
        missing = []
        all_guids = pd.concat([self.train_df['guid'], self.test_df['guid']])
        
        for guid in all_guids:
            filepath = os.path.join(self.data_dir, f"{guid}{extension}")
            if not os.path.exists(filepath):
                missing.append(guid)
        
        return missing
    
    def print_statistics(self, stats: dict):
        # 打印数据统计信息
        # Args:
        #   stats: 统计信息字典
        print("=" * 60)
        print("Dataset Statistics")
        print("=" * 60)
        print(f"Training samples: {stats['train_size']}")
        print(f"Test samples: {stats['test_size']}")
        print(f"Total samples: {stats['total_size']}")
        print("\nLabel Distribution:")
        for label, count in stats['label_distribution'].items():
            percentage = stats['label_percentage'][label]
            print(f"  {label}: {count} ({percentage})")
        
        if stats['missing_text_files']:
            print(f"\nWarning: {len(stats['missing_text_files'])} text files missing")
        else:
            print("\nAll text files present")
            
        if stats['missing_image_files']:
            print(f"Warning: {len(stats['missing_image_files'])} image files missing")
        else:
            print("All image files present")
        print("=" * 60)
    
    def visualize_distribution(self, save_path: str = None):
        # 可视化类别分布
        # Args:
        #   save_path: 保存图片的路径
        label_counts = self.train_df['tag'].value_counts()
        
        plt.figure(figsize=(10, 6))
        
        # 饼图
        plt.subplot(1, 2, 1)
        colors = ['#66b3ff', '#99ff99', '#ff9999']
        plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        plt.title('Label Distribution')
        
        # 柱状图
        plt.subplot(1, 2, 2)
        bars = plt.bar(label_counts.index, label_counts.values, color=colors)
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Label Distribution')
        
        # 在柱状图上显示数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def split_train_val(self, val_ratio: float = 0.2, stratify: bool = True):
        # 划分训练集和验证集
        # Args:
        #   val_ratio: 验证集比例
        #   stratify: 是否进行分层采样
        # Returns:
        #   train_df, val_df: 训练集和验证集的DataFrame
        if stratify:
            train_df, val_df = train_test_split(
                self.train_df,
                test_size=val_ratio,
                random_state=self.seed
            )
        
        # 重置索引
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        
        print(f"\nData Split (seed={self.seed}):")
        print(f"  Training set: {len(train_df)} samples")
        print(f"  Validation set: {len(val_df)} samples")
        print(f"  Split ratio: {1-val_ratio:.0%} / {val_ratio:.0%}")
        
        # 验证分层采样效果
        if stratify:
            print("\nValidation Set Label Distribution:")
            val_dist = val_df['tag'].value_counts()
            for label, count in val_dist.items():
                percentage = count / len(val_df) * 100
                print(f"  {label}: {count} ({percentage:.2f}%)")
        
        return train_df, val_df
    
    def save_split(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                   output_dir: str = 'data_split'):
        # 保存数据划分结果
        # Args:
        #   train_df: 训练集DataFrame
        #   val_df: 验证集DataFrame
        #   output_dir: 输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存CSV文件
        train_path = os.path.join(output_dir, 'train_split.csv')
        val_path = os.path.join(output_dir, 'val_split.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        
        print(f"\nData split saved:")
        print(f"  Training set: {train_path}")
        print(f"  Validation set: {val_path}")
        
        # 保存划分信息
        split_info = {
            'seed': self.seed,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'val_ratio': len(val_df) / (len(train_df) + len(val_df)),
            'train_label_dist': train_df['tag'].value_counts().to_dict(),
            'val_label_dist': val_df['tag'].value_counts().to_dict()
        }
        
        info_path = os.path.join(output_dir, 'split_info.json')
        save_json(split_info, info_path)
        print(f"  Split info: {info_path}")


def main():
    # 主函数：执行数据加载和统计分析
    
    # 设置路径
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TRAIN_FILE = os.path.join(BASE_DIR, 'train.txt')
    TEST_FILE = os.path.join(BASE_DIR, 'test_without_label.txt')
    
    # 创建输出目录
    output_dirs = [
        os.path.join(BASE_DIR, 'results'),
        os.path.join(BASE_DIR, 'results', 'visualizations'),
        os.path.join(BASE_DIR, 'data_split')
    ]
    create_dirs(output_dirs)
    
    # 初始化数据加载器
    loader = DataLoader(DATA_DIR, TRAIN_FILE, TEST_FILE, seed=42)
    
    # 获取并打印统计信息
    stats = loader.get_data_statistics()
    loader.print_statistics(stats)
    
    # 保存统计信息
    stats_path = os.path.join(BASE_DIR, 'results', 'data_statistics.json')
    save_json(stats, stats_path)
    print(f"\nStatistics saved to {stats_path}")
    
    # 可视化类别分布
    vis_path = os.path.join(BASE_DIR, 'results', 'visualizations', 'label_distribution.png')
    loader.visualize_distribution(save_path=vis_path)
    
    # 划分训练集和验证集
    train_df, val_df = loader.split_train_val(val_ratio=0.2, stratify=True)
    
    # 保存划分结果
    loader.save_split(train_df, val_df, output_dir=os.path.join(BASE_DIR, 'data_split'))
    
    print("\nData preparation completed successfully!")


if __name__ == '__main__':
    main()
