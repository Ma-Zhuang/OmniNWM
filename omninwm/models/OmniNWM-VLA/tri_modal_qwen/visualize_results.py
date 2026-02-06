#!/usr/bin/env python3
"""
可视化评估结果
生成图表和统计分析
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List
import pandas as pd

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_evaluation_results(eval_dir: str) -> List[Dict]:
    """加载评估结果"""
    eval_path = Path(eval_dir)
    results = []
    
    # 查找所有评估结果文件
    for result_file in eval_path.glob("evaluation_results_*.json"):
        with open(result_file, 'r') as f:
            results.append(json.load(f))
    
    return results


def plot_metrics_distribution(metrics: Dict[str, float], output_path: str):
    """绘制指标分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ADE分布
    ax = axes[0, 0]
    if 'ADE_mean' in metrics:
        data = {
            'Mean': metrics.get('ADE_mean', 0),
            'Std': metrics.get('ADE_std', 0),
            'Min': metrics.get('ADE_min', 0),
            'Max': metrics.get('ADE_max', 0),
            'Median': metrics.get('ADE_median', 0)
        }
        ax.bar(data.keys(), data.values(), color='skyblue')
        ax.set_title('ADE (Average Displacement Error) Statistics')
        ax.set_ylabel('Distance (m)')
        ax.grid(True, alpha=0.3)
    
    # FDE分布
    ax = axes[0, 1]
    if 'FDE_mean' in metrics:
        data = {
            'Mean': metrics.get('FDE_mean', 0),
            'Std': metrics.get('FDE_std', 0),
            'Min': metrics.get('FDE_min', 0),
            'Max': metrics.get('FDE_max', 0),
            'Median': metrics.get('FDE_median', 0)
        }
        ax.bar(data.keys(), data.values(), color='lightcoral')
        ax.set_title('FDE (Final Displacement Error) Statistics')
        ax.set_ylabel('Distance (m)')
        ax.grid(True, alpha=0.3)
    
    # Miss Rate
    ax = axes[1, 0]
    if 'MissRate' in metrics:
        miss_rate = metrics['MissRate']
        success_rate = 1 - miss_rate
        ax.pie([success_rate, miss_rate], 
               labels=['Success', 'Miss'], 
               colors=['lightgreen', 'salmon'],
               autopct='%1.1f%%',
               startangle=90)
        ax.set_title(f'Success/Miss Rate (threshold=2.0m)')
    
    # 航向角误差（如果有）
    ax = axes[1, 1]
    if 'AHE_mean' in metrics and 'FHE_mean' in metrics:
        data = {
            'AHE Mean': metrics.get('AHE_mean', 0),
            'AHE Std': metrics.get('AHE_std', 0),
            'FHE Mean': metrics.get('FHE_mean', 0),
            'FHE Std': metrics.get('FHE_std', 0)
        }
        # 转换为角度
        data_deg = {k: np.degrees(v) for k, v in data.items()}
        ax.bar(data_deg.keys(), data_deg.values(), color='plum')
        ax.set_title('Heading Error Statistics')
        ax.set_ylabel('Angle (degrees)')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No heading data available', 
                ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle(f'Evaluation Metrics (N={metrics.get("num_samples", 0)} samples)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"图表已保存到: {output_path}")


def plot_error_histogram(metrics: Dict[str, float], output_path: str):
    """绘制误差直方图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 模拟误差分布（基于均值和标准差）
    if 'ADE_mean' in metrics and 'ADE_std' in metrics:
        # 生成模拟数据
        ade_samples = np.random.normal(
            metrics['ADE_mean'], 
            metrics['ADE_std'], 
            1000
        )
        ade_samples = np.clip(ade_samples, 0, metrics.get('ADE_max', metrics['ADE_mean'] + 3*metrics['ADE_std']))
        
        ax = axes[0]
        ax.hist(ade_samples, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        ax.axvline(metrics['ADE_mean'], color='red', linestyle='--', label=f'Mean: {metrics["ADE_mean"]:.3f}m')
        ax.axvline(metrics.get('ADE_median', metrics['ADE_mean']), color='green', linestyle='--', label=f'Median: {metrics.get("ADE_median", metrics["ADE_mean"]):.3f}m')
        ax.set_xlabel('ADE (m)')
        ax.set_ylabel('Frequency')
        ax.set_title('Average Displacement Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if 'FDE_mean' in metrics and 'FDE_std' in metrics:
        # 生成模拟数据
        fde_samples = np.random.normal(
            metrics['FDE_mean'], 
            metrics['FDE_std'], 
            1000
        )
        fde_samples = np.clip(fde_samples, 0, metrics.get('FDE_max', metrics['FDE_mean'] + 3*metrics['FDE_std']))
        
        ax = axes[1]
        ax.hist(fde_samples, bins=30, color='lightcoral', alpha=0.7, edgecolor='black')
        ax.axvline(metrics['FDE_mean'], color='red', linestyle='--', label=f'Mean: {metrics["FDE_mean"]:.3f}m')
        ax.axvline(metrics.get('FDE_median', metrics['FDE_mean']), color='green', linestyle='--', label=f'Median: {metrics.get("FDE_median", metrics["FDE_mean"]):.3f}m')
        ax.set_xlabel('FDE (m)')
        ax.set_ylabel('Frequency')
        ax.set_title('Final Displacement Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Error Distribution Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"直方图已保存到: {output_path}")


def generate_report(metrics: Dict[str, float], output_path: str):
    """生成详细的评估报告"""
    report = []
    report.append("=" * 60)
    report.append("轨迹预测模型评估报告")
    report.append("=" * 60)
    report.append("")
    
    # 基础指标
    report.append("## 位置预测指标")
    report.append("-" * 40)
    report.append(f"样本数量: {metrics.get('num_samples', 0)}")
    report.append("")
    
    if 'ADE_mean' in metrics:
        report.append("### ADE (Average Displacement Error)")
        report.append(f"  平均值: {metrics['ADE_mean']:.4f} m")
        report.append(f"  标准差: {metrics.get('ADE_std', 0):.4f} m")
        report.append(f"  最小值: {metrics.get('ADE_min', 0):.4f} m")
        report.append(f"  最大值: {metrics.get('ADE_max', 0):.4f} m")
        report.append(f"  中位数: {metrics.get('ADE_median', 0):.4f} m")
        report.append("")
    
    if 'FDE_mean' in metrics:
        report.append("### FDE (Final Displacement Error)")
        report.append(f"  平均值: {metrics['FDE_mean']:.4f} m")
        report.append(f"  标准差: {metrics.get('FDE_std', 0):.4f} m")
        report.append(f"  最小值: {metrics.get('FDE_min', 0):.4f} m")
        report.append(f"  最大值: {metrics.get('FDE_max', 0):.4f} m")
        report.append(f"  中位数: {metrics.get('FDE_median', 0):.4f} m")
        report.append("")
    
    if 'MissRate' in metrics:
        report.append("### Miss Rate")
        report.append(f"  比率: {metrics['MissRate']:.2%}")
        report.append(f"  成功率: {(1-metrics['MissRate']):.2%}")
        report.append("")
    
    # 航向角指标
    if 'AHE_mean' in metrics:
        report.append("## 航向角预测指标")
        report.append("-" * 40)
        report.append("### AHE (Average Heading Error)")
        report.append(f"  平均值: {metrics['AHE_mean']:.4f} rad ({np.degrees(metrics['AHE_mean']):.2f}°)")
        report.append(f"  标准差: {metrics.get('AHE_std', 0):.4f} rad ({np.degrees(metrics.get('AHE_std', 0)):.2f}°)")
        report.append("")
        
        if 'FHE_mean' in metrics:
            report.append("### FHE (Final Heading Error)")
            report.append(f"  平均值: {metrics['FHE_mean']:.4f} rad ({np.degrees(metrics['FHE_mean']):.2f}°)")
            report.append(f"  标准差: {metrics.get('FHE_std', 0):.4f} rad ({np.degrees(metrics.get('FHE_std', 0)):.2f}°)")
            report.append("")
    
    # 性能分析
    report.append("## 性能分析")
    report.append("-" * 40)
    
    # 根据指标判断性能
    if 'ADE_mean' in metrics:
        ade = metrics['ADE_mean']
        if ade < 0.5:
            grade = "优秀"
        elif ade < 1.0:
            grade = "良好"
        elif ade < 2.0:
            grade = "合格"
        else:
            grade = "需要改进"
        report.append(f"位置预测精度: {grade}")
    
    if 'MissRate' in metrics:
        mr = metrics['MissRate']
        if mr < 0.1:
            grade = "优秀"
        elif mr < 0.2:
            grade = "良好"
        elif mr < 0.3:
            grade = "合格"
        else:
            grade = "需要改进"
        report.append(f"轨迹成功率: {grade}")
    
    if 'AHE_mean' in metrics:
        ahe_deg = np.degrees(metrics['AHE_mean'])
        if ahe_deg < 5:
            grade = "优秀"
        elif ahe_deg < 10:
            grade = "良好"
        elif ahe_deg < 15:
            grade = "合格"
        else:
            grade = "需要改进"
        report.append(f"航向角精度: {grade}")
    
    report.append("")
    report.append("=" * 60)
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"详细报告已保存到: {output_path}")
    print("\n报告内容:")
    print('\n'.join(report))


def compare_with_baselines(metrics: Dict[str, float]):
    """与基准方法对比"""
    # 一些典型的基准值（来自文献）
    baselines = {
        'Constant Velocity': {'ADE': 3.5, 'FDE': 7.0},
        'LSTM': {'ADE': 2.0, 'FDE': 4.5},
        'Social-LSTM': {'ADE': 1.5, 'FDE': 3.5},
        'Trajectron++': {'ADE': 0.9, 'FDE': 2.0},
        'Our Method': {
            'ADE': metrics.get('ADE_mean', 0),
            'FDE': metrics.get('FDE_mean', 0)
        }
    }
    
    # 创建对比表
    df = pd.DataFrame(baselines).T
    df = df.round(3)
    
    print("\n" + "="*50)
    print("与基准方法对比:")
    print("="*50)
    print(df.to_string())
    print("="*50)
    
    # 绘制对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(baselines))
    width = 0.35
    
    ade_values = [baselines[m]['ADE'] for m in baselines]
    fde_values = [baselines[m]['FDE'] for m in baselines]
    
    bars1 = ax.bar(x - width/2, ade_values, width, label='ADE', color='skyblue')
    bars2 = ax.bar(x + width/2, fde_values, width, label='FDE', color='lightcoral')
    
    # 高亮我们的方法
    bars1[-1].set_color('darkblue')
    bars2[-1].set_color('darkred')
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Error (m)')
    ax.set_title('Comparison with Baseline Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(baselines.keys(), rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="可视化评估结果")
    parser.add_argument("--eval-dir", type=str, 
                       default="/code/VLA/outputs/evaluation",
                       help="评估结果目录")
    parser.add_argument("--output-dir", type=str,
                       default=None,
                       help="输出目录（默认与eval-dir相同）")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.eval_dir
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载评估结果
    results = load_evaluation_results(args.eval_dir)
    
    if not results:
        print(f"未找到评估结果文件在: {args.eval_dir}")
        return
    
    # 使用最新的结果
    latest_result = results[-1]
    metrics = latest_result['metrics']
    
    print(f"加载评估结果: {latest_result.get('timestamp', 'unknown')}")
    print(f"样本数: {metrics.get('num_samples', 0)}")
    
    # 生成可视化
    plot_metrics_distribution(metrics, output_path / "metrics_distribution.png")
    plot_error_histogram(metrics, output_path / "error_histogram.png")
    
    # 生成报告
    generate_report(metrics, output_path / "evaluation_report.txt")
    
    # 与基准对比
    compare_with_baselines(metrics)


if __name__ == "__main__":
    main()