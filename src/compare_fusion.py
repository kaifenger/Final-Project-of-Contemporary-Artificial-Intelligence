# 多模态融合方法对比实验脚本
# 对比Early Fusion, Late Fusion, CLIP, BLIP-2四种方法

import os
import sys
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import argparse
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fusion_models import EarlyFusionModel, LateFusionModel, CLIPFusionModel, BLIP2FusionModel
from dataset import MultimodalDataset, TextPreprocessor, get_image_transforms
from torch.utils.data import DataLoader


def load_model(config_path, checkpoint_path, device):
    # 加载模型
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    fusion_type = config['fusion_type']
    
    # 创建模型
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
            num_classes=config['num_classes']
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
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def evaluate_model(model, val_loader, device, mode='both'):
    # 评估模型
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for text, images, labels in tqdm(val_loader, desc=f'Evaluating ({mode})'):
            images = images.to(device)
            labels = labels.to(device)
            text_input = {k: v.to(device) for k, v in text.items()}
            
            # 前向传播
            logits = model(text_input, images, mode=mode)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算准确率
    accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    
    return all_preds, all_labels, all_probs, accuracy


def main():
    parser = argparse.ArgumentParser(description='Compare fusion methods')
    parser.add_argument('--output_dir', type=str, default='../experiments/fusion_comparison',
                       help='输出目录')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 定义要对比的模型
    models_to_compare = [
        {
            'name': 'Early Fusion',
            'config': '../configs/early_fusion.yaml',
            'checkpoint': '../checkpoints/early_fusion/best_model.pth'
        },
        {
            'name': 'Late Fusion',
            'config': '../configs/late_fusion.yaml',
            'checkpoint': '../checkpoints/late_fusion/best_model.pth'
        },
        {
            'name': 'CLIP',
            'config': '../configs/clip_fusion.yaml',
            'checkpoint': '../checkpoints/clip_fusion/best_model.pth'
        },
        {
            'name': 'BLIP-2',
            'config': '../configs/blip2_fusion.yaml',
            'checkpoint': '../checkpoints/blip2_fusion/best_model.pth'
        }
    ]
    
    # 准备验证数据
    with open('../configs/early_fusion.yaml', 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    text_preprocessor = TextPreprocessor(max_length=base_config['max_text_length'])
    val_transform = get_image_transforms(base_config['image_size'], augment=False)
    
    val_dataset = MultimodalDataset(
        csv_file=base_config['val_file'],
        data_dir=base_config['data_dir'],
        text_transform=text_preprocessor,
        image_transform=val_transform,
        max_text_length=base_config['max_text_length']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # 标签映射
    label_names = ['negative', 'neutral', 'positive']
    
    # 存储结果
    results = []
    
    # 对比每个模型
    for model_info in models_to_compare:
        print(f"\n{'='*60}")
        print(f"评估: {model_info['name']}")
        print(f"{'='*60}")
        
        # 检查checkpoint是否存在
        if not os.path.exists(model_info['checkpoint']):
            print(f"⚠️ Checkpoint不存在: {model_info['checkpoint']}")
            print(f"   跳过该模型")
            continue
        
        # 加载模型
        model, config = load_model(model_info['config'], model_info['checkpoint'], device)
        
        # 评估完整模型
        preds, labels, probs, acc_both = evaluate_model(model, val_loader, device, mode='both')
        print(f"\n完整模型准确率: {acc_both:.4f}")
        
        # 消融实验：只用文本
        _, _, _, acc_text = evaluate_model(model, val_loader, device, mode='text_only')
        print(f"只用文本准确率: {acc_text:.4f}")
        
        # 消融实验：只用图像
        _, _, _, acc_image = evaluate_model(model, val_loader, device, mode='image_only')
        print(f"只用图像准确率: {acc_image:.4f}")
        
        # 分类报告
        print(f"\n分类报告:")
        print(classification_report(labels, preds, target_names=label_names, digits=4))
        
        # 混淆矩阵
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_names, yticklabels=label_names)
        plt.title(f'Confusion Matrix - {model_info["name"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'{model_info["name"].replace(" ", "_")}_confusion_matrix.png'))
        plt.close()
        
        # 保存结果
        results.append({
            'Model': model_info['name'],
            'Both Acc': acc_both,
            'Text Only Acc': acc_text,
            'Image Only Acc': acc_image,
            'Improvement (Both vs Text)': acc_both - acc_text,
            'Improvement (Both vs Image)': acc_both - acc_image
        })
    
    # 创建对比表格
    results_df = pd.DataFrame(results)
    print(f"\n\n{'='*80}")
    print("融合方法对比结果")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    # 保存结果
    results_df.to_csv(os.path.join(args.output_dir, 'comparison_results.csv'), index=False)
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 准确率对比
    ax1 = axes[0]
    x = range(len(results))
    width = 0.25
    
    ax1.bar([i - width for i in x], results_df['Both Acc'], width, label='Both', alpha=0.8)
    ax1.bar([i for i in x], results_df['Text Only Acc'], width, label='Text Only', alpha=0.8)
    ax1.bar([i + width for i in x], results_df['Image Only Acc'], width, label='Image Only', alpha=0.8)
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Fusion Methods Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Model'], rotation=15)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 提升幅度对比
    ax2 = axes[1]
    ax2.bar([i - width/2 for i in x], results_df['Improvement (Both vs Text)'], 
            width, label='vs Text Only', alpha=0.8)
    ax2.bar([i + width/2 for i in x], results_df['Improvement (Both vs Image)'], 
            width, label='vs Image Only', alpha=0.8)
    
    ax2.set_ylabel('Improvement')
    ax2.set_title('Fusion Improvement over Single Modality')
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df['Model'], rotation=15)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'fusion_comparison.png'), dpi=300)
    plt.close()
    
    print(f"\n✅ 对比结果已保存到: {args.output_dir}")
    print(f"   - comparison_results.csv")
    print(f"   - fusion_comparison.png")
    print(f"   - *_confusion_matrix.png")


if __name__ == '__main__':
    main()
