#!/usr/bin/env python3
"""
自定义训练器for LLaMA Factory
在训练时动态注入TMI支持，让标准Qwen可以使用TMI特征
包含ADE/FDE轨迹评估指标
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import re
from typing import Dict, Any, Optional, List

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "scripts"))

# 动态导入inject模块，因为它在scripts目录下
try:
    from inject_tmi_to_qwen import inject_tmi_support, load_tmi_adapter
except ImportError:
    # 如果直接导入失败，尝试从scripts目录导入
    import importlib.util
    inject_module_path = project_root / "scripts" / "inject_tmi_to_qwen.py"
    spec = importlib.util.spec_from_file_location("inject_tmi_to_qwen", str(inject_module_path))
    inject_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inject_module)
    inject_tmi_support = inject_module.inject_tmi_support
    load_tmi_adapter = inject_module.load_tmi_adapter


class TMIDataCollator:
    """
    自定义数据整理器，处理TMI特征
    支持训练集和验证集使用不同的特征目录
    """
    
    def __init__(self, tokenizer, feature_dir: str, eval_feature_dir: str = None):
        self.tokenizer = tokenizer
        self.feature_dir = Path(feature_dir)
        # 如果没有指定验证集特征目录，使用训练集目录
        self.eval_feature_dir = Path(eval_feature_dir) if eval_feature_dir else self.feature_dir
        self.is_training = True  # 默认为训练模式
    
    def set_eval_mode(self, is_eval: bool = True):
        """切换训练/评估模式"""
        self.is_training = not is_eval
    
    def __call__(self, features):
        # 先提取TMI特征路径，避免传给tokenizer
        tmi_feature_paths = []
        for feature in features:
            if 'tmi_features' in feature:
                tmi_feature_paths.append(feature.pop('tmi_features'))
            else:
                tmi_feature_paths.append(None)
        
        # 手动处理padding以确保所有序列长度一致
        # 使用固定的max_length=2048
        max_length = 2048
        
        # 手动padding每个feature
        for feature in features:
            for key in ['input_ids', 'attention_mask', 'labels']:
                if key in feature:
                    current_len = len(feature[key])
                    if current_len < max_length:
                        # padding
                        if key == 'labels':
                            # labels使用-100 padding（忽略的token）
                            feature[key] = feature[key] + [-100] * (max_length - current_len)
                        elif key == 'attention_mask':
                            # attention_mask使用0 padding
                            feature[key] = feature[key] + [0] * (max_length - current_len)
                        else:
                            # input_ids使用pad_token_id padding
                            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                            feature[key] = feature[key] + [pad_token_id] * (max_length - current_len)
                    elif current_len > max_length:
                        # 截断（保留前max_length个token）
                        feature[key] = feature[key][:max_length]
        
        # 转换为tensor
        batch = {}
        for key in ['input_ids', 'attention_mask', 'labels']:
            if key in features[0]:
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.long)
        
        # 处理其他可能的字段（如token_type_ids等）
        for key in features[0].keys():
            if key not in ['input_ids', 'attention_mask', 'labels']:
                # 对于其他字段，尝试直接转换
                try:
                    values = [f[key] for f in features]
                    # 检查是否需要padding
                    if isinstance(values[0], list):
                        # 需要padding的列表
                        max_len = max(len(v) for v in values)
                        padded_values = []
                        for v in values:
                            if len(v) < max_len:
                                v = v + [0] * (max_len - len(v))
                            padded_values.append(v)
                        batch[key] = torch.tensor(padded_values)
                    else:
                        # 标量值，直接stack
                        batch[key] = torch.tensor(values)
                except:
                    # 无法处理的字段，跳过
                    pass
        
        # 根据模式选择特征目录
        current_feature_dir = self.feature_dir if self.is_training else self.eval_feature_dir
        
        # 处理TMI特征
        tmi_features = []
        
        for i, feature_path_str in enumerate(tmi_feature_paths):
            if feature_path_str:
                # 已经有完整路径
                feature_path = Path(feature_path_str)
            else:
                # 如果没有路径，跳过
                continue
            
            # 检查文件是否存在
            if not feature_path.exists():
                raise FileNotFoundError(f"TMI特征文件不存在: {feature_path}")
            
            # 加载特征
            feat = np.load(feature_path)
            tmi_features.append(torch.from_numpy(feat))
        
        if tmi_features:
            # Padding特征到相同长度
            max_len = max(f.shape[0] for f in tmi_features)
            padded_features = []
            for feat in tmi_features:
                if feat.shape[0] < max_len:
                    padding = torch.zeros(max_len - feat.shape[0], feat.shape[1])
                    feat = torch.cat([feat, padding], dim=0)
                padded_features.append(feat)
            
            batch['tmi_features'] = torch.stack(padded_features)
        
        return batch


def setup_model_for_llamafactory(model_args, training_args):
    """
    LLaMA Factory调用的钩子函数
    用于在模型加载后注入TMI支持
    """
    
    # 这个函数会被LLaMA Factory在加载模型后调用
    def model_post_init(model):
        # 注入TMI支持 (MIDI三模态: Qwen2.5-7B hidden_size=3584)
        tmi_hidden_size = getattr(model_args, 'tmi_hidden_size', 3584)  # MIDI输出维度
        model = inject_tmi_support(model, tmi_hidden_size)
        
        # 如果有预训练的TMI适配器，加载它
        if hasattr(model_args, 'tmi_adapter_path') and model_args.tmi_adapter_path:
            model = load_tmi_adapter(model, model_args.tmi_adapter_path)
        
        return model
    
    return model_post_init


def get_custom_data_collator(tokenizer, data_args):
    """
    LLaMA Factory调用的钩子函数
    返回自定义的数据整理器
    """
    
    if hasattr(data_args, 'tmi_feature_dir'):
        # 检查是否有单独的验证集特征目录
        eval_feature_dir = getattr(data_args, 'eval_tmi_feature_dir', None)
        return TMIDataCollator(
            tokenizer, 
            feature_dir=data_args.tmi_feature_dir,
            eval_feature_dir=eval_feature_dir
        )
    else:
        # 返回默认的collator
        from transformers import DataCollatorForSeq2Seq
        return DataCollatorForSeq2Seq(tokenizer)


class TrajectoryMetrics:
    """
    轨迹评估指标计算
    """
    
    @staticmethod
    def parse_trajectory_from_text(text: str) -> Optional[np.ndarray]:
        """从生成的文本中解析轨迹坐标"""
        # 查找PLANNING标签
        planning_pattern = r'<PLANNING>(.*?)</PLANNING>'
        planning_match = re.search(planning_pattern, text, re.DOTALL)
        
        if planning_match:
            planning_content = planning_match.group(1)
            # 解析[x, y, heading]格式
            point_pattern = r'\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]'
            matches = re.findall(point_pattern, planning_content)
            
            if len(matches) == 36:  # 确保36个点 (3秒 x 12Hz)
                trajectory = np.array([[float(m[0]), float(m[1]), float(m[2])] for m in matches])
                return trajectory
        
        return None
    
    @staticmethod
    def compute_ade(predictions: List[np.ndarray], ground_truths: List[np.ndarray]) -> float:
        """计算平均位移误差 (Average Displacement Error)"""
        if not predictions or not ground_truths:
            return float('inf')
        
        total_error = 0
        count = 0
        
        for pred, gt in zip(predictions, ground_truths):
            if pred.shape[0] == gt.shape[0]:  # 确保形状匹配
                distances = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
                total_error += np.mean(distances)
                count += 1
        
        return total_error / max(count, 1)
    
    @staticmethod
    def compute_fde(predictions: List[np.ndarray], ground_truths: List[np.ndarray]) -> float:
        """计算最终位移误差 (Final Displacement Error)"""
        if not predictions or not ground_truths:
            return float('inf')
        
        total_error = 0
        count = 0
        
        for pred, gt in zip(predictions, ground_truths):
            if pred.shape[0] == gt.shape[0]:
                final_distance = np.linalg.norm(pred[-1, :2] - gt[-1, :2])
                total_error += final_distance
                count += 1
        
        return total_error / max(count, 1)


def compute_trajectory_metrics(trainer):
    """
    LLaMA Factory调用的钩子函数
    计算轨迹评估指标
    """
    
    def custom_compute_metrics(eval_preds):
        predictions, labels = eval_preds
        
        # 标准指标（如perplexity）
        metrics = {}
        
        # 如果启用轨迹评估
        if hasattr(trainer.args, 'compute_trajectory_metrics') and trainer.args.compute_trajectory_metrics:
            all_predictions = []
            all_ground_truths = []
            
            # 生成预测并解析轨迹
            for pred_ids, label_ids in zip(predictions, labels):
                # 解码预测文本
                pred_text = trainer.tokenizer.decode(pred_ids, skip_special_tokens=True)
                pred_traj = TrajectoryMetrics.parse_trajectory_from_text(pred_text)
                
                if pred_traj is not None:
                    all_predictions.append(pred_traj)
                    
                    # 从标签中解析真实轨迹（如果存在）
                    label_text = trainer.tokenizer.decode(label_ids, skip_special_tokens=True)
                    gt_traj = TrajectoryMetrics.parse_trajectory_from_text(label_text)
                    if gt_traj is not None:
                        all_ground_truths.append(gt_traj)
            
            # 计算ADE/FDE
            if all_predictions and all_ground_truths:
                ade = TrajectoryMetrics.compute_ade(all_predictions, all_ground_truths)
                fde = TrajectoryMetrics.compute_fde(all_predictions, all_ground_truths)
                metrics['ade'] = ade
                metrics['fde'] = fde
        
        return metrics
    
    return custom_compute_metrics


# 导出给LLaMA Factory使用的接口
__all__ = [
    'setup_model_for_llamafactory',
    'get_custom_data_collator',
    'TMIDataCollator',
    'compute_trajectory_metrics',
    'TrajectoryMetrics'
]