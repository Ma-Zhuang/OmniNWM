#!/usr/bin/env python3
"""
Stage 1 TMI特征质量评估脚本
评估融合模态特征的质量和训练效果
"""

import numpy as np
import os
import json
from pathlib import Path
import sys

print("=" * 80)
print("Stage 1 TMI特征质量评估")
print("=" * 80)

# 1. 检查特征文件基本情况
print("\n1. 特征文件基本检查")
print("-" * 60)

feature_dir = "/code/VLA/datasets/fused_features/train/features"
all_feature_files = list(Path(feature_dir).glob("*.npy"))
print(f"找到 {len(all_feature_files)} 个训练特征文件")

# 验证集特征
val_feature_dir = "/code/VLA/datasets/fused_features/val/features"
if Path(val_feature_dir).exists():
    val_feature_files = list(Path(val_feature_dir).glob("*.npy"))
    print(f"找到 {len(val_feature_files)} 个验证特征文件")

# 2. 分析特征统计信息
print("\n2. 特征统计分析（随机采样10个文件）")
print("-" * 60)

import random
sample_files = random.sample(all_feature_files, min(10, len(all_feature_files)))

all_stats = []
for i, f in enumerate(sample_files):
    feat = np.load(f)
    
    # 基本统计
    stats = {
        'file': f.name,
        'shape': feat.shape,
        'mean': float(feat.mean()),
        'std': float(feat.std()),
        'min': float(feat.min()),
        'max': float(feat.max()),
        'has_nan': bool(np.isnan(feat).any()),
        'has_inf': bool(np.isinf(feat).any()),
        'zero_ratio': float((feat == 0).mean()),
        'activation_ratio': float((np.abs(feat) > 0.1).mean())
    }
    all_stats.append(stats)
    
    print(f"\n样本 {i+1}: {f.name[:40]}")
    print(f"  形状: {feat.shape}")
    print(f"  均值: {stats['mean']:.4f}, 标准差: {stats['std']:.4f}")
    print(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  零值比例: {stats['zero_ratio']:.2%}")
    print(f"  激活比例(|x|>0.1): {stats['activation_ratio']:.2%}")
    
    if stats['has_nan'] or stats['has_inf']:
        print(f"  ⚠️ 警告: 包含NaN={stats['has_nan']}, Inf={stats['has_inf']}")

# 3. 总体统计
print("\n3. 总体统计")
print("-" * 60)

all_means = [s['mean'] for s in all_stats]
all_stds = [s['std'] for s in all_stats]
all_zeros = [s['zero_ratio'] for s in all_stats]

print(f"平均均值: {np.mean(all_means):.4f} (±{np.std(all_means):.4f})")
print(f"平均标准差: {np.mean(all_stds):.4f} (±{np.std(all_stds):.4f})")
print(f"平均零值比例: {np.mean(all_zeros):.2%}")

# 判断问题
issues = []
if np.mean(all_stds) < 0.01:
    issues.append("⚠️ 特征标准差过小，可能存在梯度消失")
if np.mean(all_zeros) > 0.5:
    issues.append("⚠️ 超过50%的值为零，稀疏度过高")
if np.std(all_means) < 0.001:
    issues.append("⚠️ 不同样本的均值几乎相同，可能存在模式坍塌")

# 4. 特征多样性分析
print("\n4. 特征多样性分析（20个样本）")
print("-" * 60)

try:
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 采样20个特征计算相似度
    sample_features = random.sample(all_feature_files, min(20, len(all_feature_files)))
    
    pooled_features = []
    for f in sample_features:
        feat = np.load(f)
        # 去掉batch维度 [1, 1168, 3584] -> [1168, 3584]
        feat = feat.squeeze(0) if feat.shape[0] == 1 else feat
        # 使用平均池化获得全局特征
        pooled = feat.mean(axis=0)  # [3584]
        pooled_features.append(pooled)
    
    pooled_features = np.stack(pooled_features)
    
    # 计算余弦相似度
    sim_matrix = cosine_similarity(pooled_features)
    
    # 排除对角线
    mask = np.ones_like(sim_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    off_diagonal = sim_matrix[mask]
    
    print(f"平均特征相似度: {off_diagonal.mean():.4f}")
    print(f"相似度范围: [{off_diagonal.min():.4f}, {off_diagonal.max():.4f}]")
    print(f"相似度标准差: {off_diagonal.std():.4f}")
    
    # 判断多样性
    if off_diagonal.mean() > 0.95:
        issues.append("⚠️ 特征相似度过高(>0.95)，严重模式坍塌!")
    elif off_diagonal.mean() > 0.85:
        issues.append("⚠️ 特征相似度偏高(>0.85)，多样性不足")
    elif off_diagonal.mean() < 0.3:
        print("✅ 特征多样性良好")
    
except ImportError:
    print("sklearn未安装，跳过相似度分析")

# 5. 死神经元检测
print("\n5. 神经元激活分析")
print("-" * 60)

# 采样更多文件进行死神经元检测
sample_for_dead = random.sample(all_feature_files, min(50, len(all_feature_files)))
all_features = []
for f in sample_for_dead[:50]:
    feat = np.load(f)
    # 去掉batch维度 [1, 1168, 3584] -> [1168, 3584]
    feat = feat.squeeze(0) if feat.shape[0] == 1 else feat
    all_features.append(feat.reshape(-1, feat.shape[-1]))  # [N, 3584]

all_features = np.concatenate(all_features, axis=0)

# 计算每个神经元的激活统计
neuron_stds = all_features.std(axis=0)
neuron_means = np.abs(all_features.mean(axis=0))

dead_neurons = (neuron_stds < 0.01).sum()
low_activation = (neuron_means < 0.01).sum()

print(f"总神经元数: {len(neuron_stds)}")
print(f"死神经元(std<0.01): {dead_neurons} ({dead_neurons/len(neuron_stds):.1%})")
print(f"低激活神经元(|mean|<0.01): {low_activation} ({low_activation/len(neuron_stds):.1%})")

if dead_neurons > len(neuron_stds) * 0.2:
    issues.append(f"⚠️ {dead_neurons/len(neuron_stds):.0%}的神经元几乎没有激活")

# 6. 特征分布可视化
print("\n6. 特征值分布（ASCII直方图）")
print("-" * 60)

sample_feat = np.load(sample_files[0])
flat_values = sample_feat.flatten()

# 创建直方图
hist, bins = np.histogram(flat_values, bins=20)
hist = hist / hist.max() * 40  # 归一化到40字符宽度

print(f"样本: {sample_files[0].name[:40]}")
print("\n值分布:")
for i, (h, b) in enumerate(zip(hist, bins[:-1])):
    bar = '█' * int(h)
    spaces = ' ' * (40 - int(h))
    percentage = hist[i] / hist.sum() * 100
    print(f"  {b:7.3f} |{bar}{spaces}| {percentage:5.1f}%")

# 7. 与训练配置对比
print("\n7. 训练配置分析")
print("-" * 60)

print("TMI模块配置:")
print("  - fusion_hidden_size: 2048")
print("  - fusion_num_layers: 4")
print("  - fusion_type: mamba")
print("  - 输出维度: 3584 (与Qwen hidden_size匹配)")

print("\nStage 1训练配置:")
print("  - 学习率: 5e-6 (非常保守)")
print("  - Batch size: 1 x 8 GPU x 4梯度累积 = 32")
print("  - Warmup: 40%")
print("  - 梯度裁剪: 0.5")

# 8. 诊断总结
print("\n" + "=" * 80)
print("诊断总结")
print("=" * 80)

if not issues:
    print("✅ 特征质量基本正常")
else:
    print("发现以下问题:")
    for issue in issues:
        print(f"  {issue}")

print("\n可能的原因分析:")
print("-" * 60)

# 分析Loss波动小的原因
print("\n为什么eval_loss只有0.01的波动？")
print("1. 学习率过小 (5e-6):")
print("   - 正常LoRA微调用2e-5，这里只有1/4")
print("   - TMI模块参数更新太慢")

print("\n2. TMI模块容量问题:")
print("   - Mamba层数只有4层")
print("   - 可能不足以学习复杂的三模态融合")

print("\n3. 数据问题:")
print("   - 20000个样本可能不够")
print("   - RGB/Depth/Semantic的对齐可能有问题")

print("\n建议改进方案:")
print("-" * 60)
print("1. 增大学习率到 2e-5 或 3e-5")
print("2. 增加TMI的Mamba层数到 6-8 层")
print("3. 减少warmup比例到 0.1")
print("4. 使用更大的batch size (如果显存允许)")
print("5. 检查三种模态的预处理是否正确")

print("\n" + "=" * 80)
print("评估完成！")
print("=" * 80)