#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键训练所有多模态融合模型
顺序训练：Early Fusion → Late Fusion → Cross-Attention
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# 训练配置列表
FUSION_CONFIGS = [
    {
        'name': 'Early Fusion',
        'config': 'configs/early_fusion.yaml',
        'description': '早期融合（特征拼接）'
    },
    {
        'name': 'Late Fusion',
        'config': 'configs/late_fusion.yaml',
        'description': '晚期融合（决策加权）'
    },
    {
        'name': 'Cross-Attention Fusion',
        'config': 'configs/cross_attention.yaml',
        'description': '跨模态注意力融合'
    }
]


def print_section(title):
    """打印分隔线"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def train_model(config_info, index, total):
    """训练单个模型"""
    name = config_info['name']
    config_file = config_info['config']
    description = config_info['description']
    
    print_section(f"[{index}/{total}] 开始训练: {name}")
    print(f"📝 描述: {description}")
    print(f"⚙️  配置文件: {config_file}")
    print(f"🕐 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查配置文件是否存在
    if not os.path.exists(config_file):
        print(f"❌ 错误: 配置文件不存在 - {config_file}")
        return False
    
    # 构建训练命令
    cmd = [sys.executable, 'src/train_fusion.py', '--config', config_file]
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 执行训练
        print(f"\n🚀 执行命令: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, check=True)
        
        # 计算训练时长
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"\n✅ {name} 训练完成!")
        print(f"⏱️  用时: {hours}小时 {minutes}分钟 {seconds}秒")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        
        print(f"\n❌ {name} 训练失败!")
        print(f"⏱️  已用时: {hours}小时 {minutes}分钟")
        print(f"💥 错误信息: {e}")
        return False
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  用户中断训练: {name}")
        print("⏸️  训练已暂停")
        return False


def main():
    """主函数"""
    print_section("🎯 多模态融合模型 - 批量训练脚本")
    
    print("📋 训练计划:")
    for i, config in enumerate(FUSION_CONFIGS, 1):
        print(f"  {i}. {config['name']: <25} - {config['description']}")
    
    print(f"\n📊 总计: {len(FUSION_CONFIGS)} 个模型")
    print("⚙️  训练策略: 分层学习率 (backbone: 1e-5, projection/classifier: 1e-3)")
    print("📉 Early Stopping: patience=3, min_delta=0.001")
    
    # 询问用户确认
    print("\n" + "-" * 70)
    response = input("❓ 确认开始训练? [Y/n]: ").strip().lower()
    if response and response != 'y' and response != 'yes':
        print("❌ 训练已取消")
        return
    
    # 记录总开始时间
    total_start_time = time.time()
    
    # 训练结果统计
    results = []
    
    # 依次训练每个模型
    for i, config in enumerate(FUSION_CONFIGS, 1):
        success = train_model(config, i, len(FUSION_CONFIGS))
        results.append({
            'name': config['name'],
            'success': success
        })
        
        # 如果训练失败，询问是否继续
        if not success and i < len(FUSION_CONFIGS):
            print("\n" + "-" * 70)
            response = input("❓ 当前模型训练失败，是否继续训练下一个模型? [Y/n]: ").strip().lower()
            if response and response != 'y' and response != 'yes':
                print("⏸️  批量训练已终止")
                break
    
    # 计算总用时
    total_elapsed = time.time() - total_start_time
    total_hours = int(total_elapsed // 3600)
    total_minutes = int((total_elapsed % 3600) // 60)
    total_seconds = int(total_elapsed % 60)
    
    # 打印训练总结
    print_section("📊 训练总结")
    
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count
    
    print("训练结果:")
    for result in results:
        status = "✅ 成功" if result['success'] else "❌ 失败"
        print(f"  • {result['name']: <25} {status}")
    
    print(f"\n统计:")
    print(f"  • 成功: {success_count}/{len(results)}")
    print(f"  • 失败: {fail_count}/{len(results)}")
    print(f"  • 总用时: {total_hours}小时 {total_minutes}分钟 {total_seconds}秒")
    
    if success_count == len(FUSION_CONFIGS):
        print("\n🎉 所有模型训练完成!")
        print("\n📁 检查点保存位置:")
        print("  • checkpoints/early_fusion/")
        print("  • checkpoints/late_fusion/")
        print("  • checkpoints/cross_attention/")
        print("\n📈 TensorBoard日志:")
        print("  • experiments/logs/early_fusion/")
        print("  • experiments/logs/late_fusion/")
        print("  • experiments/logs/cross_attention/")
        print("\n💡 查看训练结果:")
        print("  tensorboard --logdir experiments/logs")
    else:
        print("\n⚠️  部分模型训练失败，请检查错误信息")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断批量训练")
        print("👋 再见!")
        sys.exit(0)
