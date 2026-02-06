#!/usr/bin/env python3
"""
过拟合监控脚本
实时监控训练，自动检测过拟合并提供建议
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List
import argparse


class OverfittingMonitor:
    """过拟合监控器"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
        
    def analyze(self) -> Dict:
        """分析训练状态"""
        # 读取训练日志
        self._load_history()
        
        results = {
            'is_overfitting': False,
            'overfitting_epoch': None,
            'suggestions': [],
            'metrics': {}
        }
        
        if len(self.history['train_loss']) < 5:
            results['suggestions'].append("需要更多epoch才能判断")
            return results
        
        # 1. 检测过拟合
        train_loss = np.array(self.history['train_loss'])
        val_loss = np.array(self.history['val_loss'])
        
        # 计算loss差距
        loss_gap = val_loss - train_loss
        recent_gap = loss_gap[-5:].mean()  # 最近5个epoch
        
        # 验证集loss趋势
        val_trend = np.polyfit(range(len(val_loss[-10:])), val_loss[-10:], 1)[0]
        
        # 判断标准
        if recent_gap > 0.5 and val_trend > 0:
            results['is_overfitting'] = True
            results['overfitting_epoch'] = self._find_overfitting_point()
            
        # 2. 计算指标
        results['metrics'] = {
            'train_loss': train_loss[-1],
            'val_loss': val_loss[-1],
            'loss_gap': recent_gap,
            'val_trend': val_trend,
            'best_val_loss': val_loss.min(),
            'best_epoch': val_loss.argmin() + 1
        }
        
        # 3. 生成建议
        results['suggestions'] = self._generate_suggestions(results)
        
        return results
    
    def _find_overfitting_point(self) -> int:
        """找到开始过拟合的epoch"""
        val_loss = self.history['val_loss']
        
        # 找到验证loss开始上升的点
        for i in range(5, len(val_loss)):
            if all(val_loss[i] > val_loss[i-j] for j in range(1, 4)):
                return i
        return len(val_loss)
    
    def _generate_suggestions(self, analysis: Dict) -> List[str]:
        """生成改进建议"""
        suggestions = []
        metrics = analysis['metrics']
        
        if analysis['is_overfitting']:
            suggestions.append(f"⚠️ 检测到过拟合！从epoch {analysis['overfitting_epoch']}开始")
            suggestions.append("建议立即采取以下措施：")
            
            # 根据gap大小给出建议
            gap = metrics['loss_gap']
            if gap > 1.0:
                suggestions.append("• 严重过拟合：增加dropout到0.4-0.5")
                suggestions.append("• 减少学习率50%")
                suggestions.append("• 考虑早停")
            elif gap > 0.5:
                suggestions.append("• 中度过拟合：增加weight_decay到0.1")
                suggestions.append("• 增加数据增强")
                suggestions.append("• 减少模型复杂度")
            
        else:
            # 检查是否欠拟合
            if metrics['train_loss'] > 1.0:
                suggestions.append("• 训练loss仍然很高，可能欠拟合")
                suggestions.append("• 增加模型容量或训练更长时间")
            
            # 检查学习是否停滞
            if abs(metrics['val_trend']) < 0.001:
                suggestions.append("• 学习停滞，考虑调整学习率")
        
        # TMI特定建议
        suggestions.append("\n📊 TMI模块特定建议：")
        suggestions.append(f"• 最佳checkpoint: epoch {metrics['best_epoch']}")
        suggestions.append(f"• 当前loss gap: {metrics['loss_gap']:.3f}")
        
        if metrics['loss_gap'] > 0.3:
            suggestions.append("• 考虑冻结更多TMI层，只训练融合核心")
            suggestions.append("• 使用R-Drop或其他一致性正则化")
        
        return suggestions
    
    def plot_curves(self, save_path: str = "training_curves.png"):
        """绘制训练曲线"""
        if not self.history['train_loss']:
            print("没有数据可绘制")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss曲线
        axes[0].plot(self.history['epoch'], self.history['train_loss'], 
                    label='Train Loss', marker='o')
        axes[0].plot(self.history['epoch'], self.history['val_loss'], 
                    label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training vs Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss Gap
        gap = np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
        axes[1].plot(self.history['epoch'], gap, 
                    label='Val-Train Gap', marker='d', color='red')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].axhline(y=0.3, color='orange', linestyle='--', 
                       alpha=0.5, label='Overfitting Threshold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss Gap')
        axes[1].set_title('Overfitting Indicator')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📈 训练曲线已保存到 {save_path}")
    
    def _load_history(self):
        """从checkpoint加载训练历史"""
        # 这里简化处理，实际需要从tensorboard或训练日志读取
        # 示例：从trainer_state.json读取
        state_file = self.checkpoint_dir / "trainer_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                # 解析训练历史
                for entry in state.get('log_history', []):
                    if 'loss' in entry:
                        self.history['train_loss'].append(entry['loss'])
                    if 'eval_loss' in entry:
                        self.history['val_loss'].append(entry['eval_loss'])
                    if 'epoch' in entry:
                        self.history['epoch'].append(entry['epoch'])


def main():
    parser = argparse.ArgumentParser(description='过拟合监控')
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='/code/VLA/outputs/stage1_tmi_fixed',
        help='Checkpoint目录'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='绘制训练曲线'
    )
    
    args = parser.parse_args()
    
    # 创建监控器
    monitor = OverfittingMonitor(args.checkpoint_dir)
    
    # 分析
    print("\n" + "="*60)
    print("🔍 过拟合分析报告")
    print("="*60)
    
    results = monitor.analyze()
    
    # 打印结果
    if results['is_overfitting']:
        print("⚠️  状态: 过拟合！")
        print(f"   开始epoch: {results['overfitting_epoch']}")
    else:
        print("✅ 状态: 正常训练")
    
    print("\n📊 关键指标:")
    for key, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    print("\n💡 建议:")
    for suggestion in results['suggestions']:
        print(f"   {suggestion}")
    
    # 绘图
    if args.plot:
        monitor.plot_curves()
    
    print("="*60)


if __name__ == '__main__':
    main()