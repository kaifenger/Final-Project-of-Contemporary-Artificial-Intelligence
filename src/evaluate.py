# 评估脚本
# 评估模型性能并生成详细报告

import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from dataset import MultimodalDataset, TextPreprocessor, get_image_transforms
from models.text_encoder import TextEncoder
from models.image_encoder import ImageEncoder
from utils import set_seed, get_device


class Evaluator:
    # 评估器类
    
    def __init__(self, config, checkpoint_path):
        # 初始化评估器
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = get_device()
        
        # 加载数据
        self.val_loader = self._prepare_data()
        
        # 加载模型
        self.model = self._load_model()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.label_names = ['positive', 'neutral', 'negative']
    
    def _prepare_data(self):
        # 准备数据加载器
        config = self.config
        
        text_preprocessor = TextPreprocessor(max_length=config['max_text_length'])
        image_transform = get_image_transforms(
            image_size=config['image_size'],
            augment=False
        )
        
        val_dataset = MultimodalDataset(
            csv_file=config['val_file'],
            data_dir=config['data_dir'],
            text_transform=text_preprocessor,
            image_transform=image_transform
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        return val_loader
    
    def _load_model(self):
        # 加载模型
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
                pretrained=False,
                dropout=config['dropout']
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # 加载权重
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {self.checkpoint_path}")
        print(f"Best validation accuracy: {checkpoint.get('best_acc', 'N/A'):.2f}%")
        
        return model
    
    def evaluate(self):
        # 评估模型
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                texts = batch['text']
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 前向传播
                if self.config['model_type'] == 'text_only':
                    encoding = self.model.encode_text(texts, device=self.device)
                    logits = self.model(encoding['input_ids'], encoding['attention_mask'])
                else:
                    logits = self.model(images)
                
                probs = torch.softmax(logits, dim=1)
                _, predicted = logits.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def print_results(self, preds, labels):
        # 打印评估结果
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        
        # 准确率
        acc = accuracy_score(labels, preds)
        print(f"Accuracy: {acc * 100:.2f}%")
        
        # F1分数
        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')
        print(f"F1-score (macro): {f1_macro:.4f}")
        print(f"F1-score (weighted): {f1_weighted:.4f}")
        
        # 分类报告
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=self.label_names))
        
        print("=" * 60)
    
    def plot_confusion_matrix(self, preds, labels, save_path):
        # 绘制混淆矩阵
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_names,
                    yticklabels=self.label_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {self.config["model_type"]}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def run(self, output_dir):
        # 运行评估
        os.makedirs(output_dir, exist_ok=True)
        
        # 评估
        preds, labels, probs = self.evaluate()
        
        # 打印结果
        self.print_results(preds, labels)
        
        # 保存混淆矩阵
        cm_path = os.path.join(output_dir, f'confusion_matrix_{self.config["model_type"]}.png')
        self.plot_confusion_matrix(preds, labels, cm_path)
        
        # 保存预测结果
        results = {
            'predictions': preds.tolist(),
            'labels': labels.tolist(),
            'probabilities': probs.tolist()
        }
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate sentiment classification model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output', type=str, default='results/evaluations', help='Output directory')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 评估
    evaluator = Evaluator(config, args.checkpoint)
    evaluator.run(args.output)


if __name__ == '__main__':
    main()
