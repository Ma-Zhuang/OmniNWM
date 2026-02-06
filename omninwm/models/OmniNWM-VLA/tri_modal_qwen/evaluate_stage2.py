#!/usr/bin/env python3
"""
Stage 2模型评估脚本
评估使用MIDI特征训练的Qwen2.5-VL模型
"""

import os
import sys
import json
import torch
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass, field
import argparse
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from transformers import AutoTokenizer
from peft import PeftModel


class EvaluationDataset(Dataset):
    """评估数据集"""
    def __init__(self, data: List[Dict], feature_dir: str):
        self.data = data
        self.feature_dir = Path(feature_dir)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # 确保特征路径正确
        if 'tmi_features' in sample:
            feature_path = sample['tmi_features']
            if not Path(feature_path).is_absolute():
                feature_path = self.feature_dir / Path(feature_path).name
                sample['tmi_features'] = str(feature_path)
        
        return sample


@dataclass
class TrajectoryMetrics:
    """轨迹评估指标"""
    ade_values: List[float] = field(default_factory=list)  # Average Displacement Error
    fde_values: List[float] = field(default_factory=list)  # Final Displacement Error
    ahe_values: List[float] = field(default_factory=list)  # Average Heading Error
    fhe_values: List[float] = field(default_factory=list)  # Final Heading Error
    miss_rates: List[float] = field(default_factory=list)  # Miss Rate (FDE > threshold)
    # L2分时指标 (12Hz采样: 1s=12帧, 2s=24帧, 3s=36帧)
    l2_1s_values: List[float] = field(default_factory=list)  # L2 error at 1 second
    l2_2s_values: List[float] = field(default_factory=list)  # L2 error at 2 seconds
    l2_3s_values: List[float] = field(default_factory=list)  # L2 error at 3 seconds
    
    def add_sample(self, pred_traj: np.ndarray, gt_traj: np.ndarray, miss_threshold: float = 2.0):
        """
        添加一个样本的评估结果
        
        Args:
            pred_traj: 预测轨迹 [N, 3] (x, y, heading)
            gt_traj: 真实轨迹 [N, 3] (x, y, heading)
            miss_threshold: Miss Rate的阈值（米）
        """
        # 确保轨迹长度一致
        min_len = min(len(pred_traj), len(gt_traj))
        pred_traj = pred_traj[:min_len]
        gt_traj = gt_traj[:min_len]
        
        # 计算位置误差
        position_errors = np.sqrt((pred_traj[:, 0] - gt_traj[:, 0])**2 + 
                                  (pred_traj[:, 1] - gt_traj[:, 1])**2)
        
        # ADE: 平均位移误差
        ade = np.mean(position_errors)
        self.ade_values.append(ade)
        
        # FDE: 最终位移误差
        fde = position_errors[-1] if len(position_errors) > 0 else 0
        self.fde_values.append(fde)
        
        # Miss Rate
        self.miss_rates.append(1.0 if fde > miss_threshold else 0.0)
        
        # L2分时指标 (12Hz采样率)
        # 1秒 = 12帧 = 索引11 (0-based)
        # 2秒 = 24帧 = 索引23
        # 3秒 = 36帧 = 索引35
        if len(position_errors) > 11:  # 至少有1秒的数据
            l2_1s = position_errors[11]
            self.l2_1s_values.append(l2_1s)
        else:
            # 如果轨迹太短，使用最后一个点
            self.l2_1s_values.append(position_errors[-1] if len(position_errors) > 0 else 0)
        
        if len(position_errors) > 23:  # 至少有2秒的数据
            l2_2s = position_errors[23]
            self.l2_2s_values.append(l2_2s)
        else:
            self.l2_2s_values.append(position_errors[-1] if len(position_errors) > 0 else 0)
        
        if len(position_errors) > 35:  # 有完整3秒的数据
            l2_3s = position_errors[35]
            self.l2_3s_values.append(l2_3s)
        else:
            # 使用最后一个点（应该就是FDE）
            self.l2_3s_values.append(fde)
        
        # 如果有航向角信息
        if pred_traj.shape[1] >= 3 and gt_traj.shape[1] >= 3:
            # 计算航向角误差（考虑角度的环形特性）
            heading_errors = np.abs(np.arctan2(
                np.sin(pred_traj[:, 2] - gt_traj[:, 2]),
                np.cos(pred_traj[:, 2] - gt_traj[:, 2])
            ))
            
            # AHE: 平均航向角误差
            ahe = np.mean(heading_errors)
            self.ahe_values.append(ahe)
            
            # FHE: 最终航向角误差
            fhe = heading_errors[-1] if len(heading_errors) > 0 else 0
            self.fhe_values.append(fhe)
    
    def compute_statistics(self) -> Dict[str, float]:
        """计算统计指标"""
        stats = {}
        
        if self.ade_values:
            stats['ADE_mean'] = np.mean(self.ade_values)
            stats['ADE_std'] = np.std(self.ade_values)
            stats['ADE_min'] = np.min(self.ade_values)
            stats['ADE_max'] = np.max(self.ade_values)
            stats['ADE_median'] = np.median(self.ade_values)
            
        if self.fde_values:
            stats['FDE_mean'] = np.mean(self.fde_values)
            stats['FDE_std'] = np.std(self.fde_values)
            stats['FDE_min'] = np.min(self.fde_values)
            stats['FDE_max'] = np.max(self.fde_values)
            stats['FDE_median'] = np.median(self.fde_values)
            
        # L2分时指标统计
        if self.l2_1s_values:
            stats['L2_1s_mean'] = np.mean(self.l2_1s_values)
            stats['L2_1s_std'] = np.std(self.l2_1s_values)
            stats['L2_1s_min'] = np.min(self.l2_1s_values)
            stats['L2_1s_max'] = np.max(self.l2_1s_values)
            stats['L2_1s_median'] = np.median(self.l2_1s_values)
            
        if self.l2_2s_values:
            stats['L2_2s_mean'] = np.mean(self.l2_2s_values)
            stats['L2_2s_std'] = np.std(self.l2_2s_values)
            stats['L2_2s_min'] = np.min(self.l2_2s_values)
            stats['L2_2s_max'] = np.max(self.l2_2s_values)
            stats['L2_2s_median'] = np.median(self.l2_2s_values)
            
        if self.l2_3s_values:
            stats['L2_3s_mean'] = np.mean(self.l2_3s_values)
            stats['L2_3s_std'] = np.std(self.l2_3s_values)
            stats['L2_3s_min'] = np.min(self.l2_3s_values)
            stats['L2_3s_max'] = np.max(self.l2_3s_values)
            stats['L2_3s_median'] = np.median(self.l2_3s_values)
        
        # 计算L2平均值（与论文中的Avg. L2对应）
        if self.l2_1s_values and self.l2_2s_values and self.l2_3s_values:
            l2_avg = (np.mean(self.l2_1s_values) + np.mean(self.l2_2s_values) + np.mean(self.l2_3s_values)) / 3
            stats['L2_avg'] = l2_avg
            
        if self.miss_rates:
            stats['MissRate'] = np.mean(self.miss_rates)
            
        if self.ahe_values:
            stats['AHE_mean'] = np.mean(self.ahe_values)
            stats['AHE_std'] = np.std(self.ahe_values)
            stats['FHE_mean'] = np.mean(self.fhe_values) if self.fhe_values else 0
            stats['FHE_std'] = np.std(self.fhe_values) if self.fhe_values else 0
            
        # 添加样本数
        stats['num_samples'] = len(self.ade_values)
        
        return stats


def parse_trajectory_from_text(text: str) -> Optional[np.ndarray]:
    """
    从生成的文本中解析轨迹坐标
    
    Args:
        text: 生成的文本，包含轨迹信息
        
    Returns:
        轨迹数组 [N, 3] (x, y, heading) 或 None
    """
    # 如果文本包含<PLANNING>标签，先提取标签内的内容
    if '<PLANNING>' in text and '</PLANNING>' in text:
        start = text.find('<PLANNING>') + len('<PLANNING>')
        end = text.find('</PLANNING>')
        text = text[start:end]
    
    # 尝试多种格式
    patterns = [
        # 格式1: [3.03, -0.00, 0.004] - 实际数据格式（支持科学计数法）
        r'\[([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\]',
        # 格式1b: 支持额外字段的版本 - 匹配至少3个数字，但只提取前3个
        r'\[([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)[,\s]*(?:[^]]*)\]',
        # 格式2: [x: 0.5, y: 0.1, heading: 0.01]
        r'\[x:\s*([-\d.]+),\s*y:\s*([-\d.]+),\s*(?:heading|h):\s*([-\d.]+)\]',
        # 格式3: (0.5, 0.1, 0.01)
        r'\(([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)',
        # 格式4: x=0.5, y=0.1, heading=0.01
        r'x\s*=\s*([-\d.]+)[,\s]+y\s*=\s*([-\d.]+)[,\s]+(?:heading|h)\s*=\s*([-\d.]+)',
    ]
    
    trajectory = []
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for match in matches:
                try:
                    x, y, heading = map(float, match)
                    trajectory.append([x, y, heading])
                except:
                    continue
            if trajectory:
                break
    
    # 验证轨迹点数量
    if len(trajectory) == 36:
        return np.array(trajectory)
    elif len(trajectory) > 30:  # 如果有超过30个点（允许小误差），仍然返回前36个
        # 打印警告但仍然返回
        if len(trajectory) != 36:
            pass  # 静默处理，避免过多输出
        return np.array(trajectory[:36]) if len(trajectory) > 36 else np.array(trajectory)
    elif trajectory:  # 如果轨迹太短，可能是生成不完整
        # 返回None表示失败
        return None
    
    return None


def load_model_and_tokenizer(checkpoint_path: str, device: str = "cuda"):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: checkpoint路径
        device: 设备 (可以是 "cuda", "cuda:0", "cuda:1" 等)
        
    Returns:
        model, tokenizer
    """
    # 解析设备ID
    if device.startswith("cuda:"):
        gpu_id = int(device.split(":")[1])
    else:
        gpu_id = 0 if device == "cuda" else None
    
    print(f"在设备 {device} 上加载模型: {checkpoint_path}")
    
    # 加载基座模型 (与Stage 2训练使用的模型一致)
    base_model_path = "/code/VLA/models/Qwen2.5-VL-7B-Instruct"
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载Qwen2.5-VL模型 - 必须使用ForConditionalGeneration版本才能生成文本
    print(f"加载Qwen2.5-VL模型到GPU {gpu_id}...")
    try:
        # 尝试使用Qwen2_5_VLForConditionalGeneration
        from transformers import Qwen2_5_VLForConditionalGeneration
        # 将模型放在指定的GPU上
        if gpu_id is not None:
            device_map = {"": gpu_id}  # 将整个模型放在指定GPU上
        else:
            device_map = None
            
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
            device_map=device_map,
            trust_remote_code=True
        )
        print("成功加载Qwen2_5_VLForConditionalGeneration模型")
    except ImportError:
        # 如果没有这个类，尝试动态导入
        print("尝试动态加载模型类...")
        from transformers.models.auto import modeling_auto
        # 动态注册模型类
        modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["qwen2_5_vl"] = "Qwen2_5_VLForConditionalGeneration"
        
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if "cuda" in device else torch.float32,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="eager"  # 避免flash attention问题
        )
    
    # 加载LoRA adapter
    if Path(checkpoint_path).exists():
        print("加载LoRA weights...")
        model = PeftModel.from_pretrained(
            model,
            checkpoint_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        # 不要merge_and_unload！保持PeftModel状态，与训练时一致
        # model = model.merge_and_unload()  # 注释掉这行
    
    model.eval()
    
    # 注入TMI支持（在PeftModel上注入，与训练时一致）
    sys.path.append(str(Path(__file__).parent / "scripts"))
    from inject_tmi_to_qwen import inject_tmi_support
    # MIDI输出维度为3584 (Qwen2.5-7B hidden_size)，与Qwen2.5-VL-7B完美对齐
    model = inject_tmi_support(model, tmi_hidden_size=3584)
    
    return model, tokenizer


def load_evaluation_data(data_path: str, feature_dir: str) -> List[Dict]:
    """
    加载评估数据
    
    Args:
        data_path: 数据文件路径
        feature_dir: TMI特征目录
        
    Returns:
        数据列表
    """
    print(f"加载数据: {data_path}")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # 确保每个样本都有特征路径
    for sample in data:
        if 'tmi_features' in sample:
            feature_path = sample['tmi_features']
            # 如果是相对路径，转换为绝对路径
            if not Path(feature_path).is_absolute():
                feature_path = Path(feature_dir) / Path(feature_path).name
                sample['tmi_features'] = str(feature_path)
    
    return data


def evaluate_model(
    model,
    tokenizer,
    eval_data: List[Dict],
    max_samples: int = None,
    device: str = "cuda",
    batch_size: int = 1,
    num_workers: int = 0
) -> TrajectoryMetrics:
    """
    评估模型
    
    Args:
        model: 模型
        tokenizer: 分词器
        eval_data: 评估数据
        max_samples: 最大评估样本数
        device: 设备
        
    Returns:
        评估指标
    """
    metrics = TrajectoryMetrics()
    
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    print(f"评估 {len(eval_data)} 个样本...")
    
    skip_reasons = {"no_input": 0, "no_gt_traj": 0, "no_tmi": 0, "error": 0}
    successful_evals = 0
    
    for sample_idx, sample in enumerate(tqdm(eval_data, desc="评估中")):
        try:
            # 获取输入和真实答案
            messages = sample['messages']
            
            # 提取用户输入和真实轨迹
            user_input = None
            ground_truth = None
            
            for msg in messages:
                if msg['role'] == 'user':
                    user_input = msg['content']
                elif msg['role'] == 'assistant':
                    ground_truth = msg['content']
            
            if not user_input or not ground_truth:
                skip_reasons["no_input"] += 1
                if sample_idx < 3:  # 打印前3个跳过的样本
                    print(f"\n样本 {sample_idx} 跳过: 无输入或真实值")
                continue
            
            # 解析真实轨迹
            gt_trajectory = parse_trajectory_from_text(ground_truth)
            if gt_trajectory is None:
                skip_reasons["no_gt_traj"] += 1
                if sample_idx < 3:
                    print(f"\n样本 {sample_idx} 跳过: 无法解析真实轨迹")
                    print(f"  真实文本: {ground_truth[:200]}...")
                continue
            
            # 准备输入
            # 清理用户输入中的<image>标记
            user_input_clean = user_input.replace('<image>', '').strip()
            
            # 获取模型实际所在的设备
            model_device = next(model.parameters()).device
            
            # Tokenize输入
            inputs = tokenizer(
                user_input_clean,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding=True
            ).to(model_device)
            
            # 加载TMI特征
            tmi_features_path = sample.get('tmi_features')
            if tmi_features_path and Path(tmi_features_path).exists():
                tmi_features = np.load(tmi_features_path)
                # 调试：打印TMI特征信息
                if sample_idx == 0:
                    print(f"\n样本 {sample_idx} TMI特征加载:")
                    print(f"  特征文件: {tmi_features_path}")
                    print(f"  特征形状: {tmi_features.shape}")
                    print(f"  特征范围: [{tmi_features.min():.4f}, {tmi_features.max():.4f}]")
                    print(f"  特征均值: {tmi_features.mean():.4f}")
                    print(f"  特征标准差: {tmi_features.std():.4f}")
                # 确保TMI特征在模型所在的设备上
                model_device = next(model.parameters()).device
                tmi_features = torch.from_numpy(tmi_features).to(model_device).to(model.dtype)
            else:
                skip_reasons["no_tmi"] += 1
                if sample_idx < 3:
                    print(f"\n样本 {sample_idx} 跳过: TMI特征文件不存在")
                    print(f"  路径: {tmi_features_path}")
                continue
            
            # 生成预测 - 使用与训练时相同的forward调用方式
            # PeftModel的forward接受labels参数
            with torch.no_grad():
                # 构建输入提示 - 使用与训练时相同的chat template格式
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input_clean}
                ]
                
                # 应用chat template
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True  # 添加<|im_start|>assistant标记
                )
                
                prompt_inputs = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    max_length=2048,  # 增加到2048，确保输入不被截断
                    truncation=True,
                    padding=True
                ).to(model_device)
                
                # 准备labels（使用-100表示忽略的位置）
                # 注意：由于TMI特征会增加10个tokens，需要扩展labels
                tmi_token_count = 10  # TMI特征的token数量
                batch_size = prompt_inputs.input_ids.shape[0]
                
                # 创建扩展的labels：前10个是TMI tokens（忽略），后面是原始文本的labels（也忽略）
                ignore_labels = torch.full(
                    (batch_size, prompt_inputs.input_ids.shape[1] + tmi_token_count),
                    -100,
                    dtype=torch.long,
                    device=model_device
                )
                
                # 首次forward调用，获取初始logits（与训练时相同的调用方式）
                outputs = model(
                    input_ids=prompt_inputs.input_ids,
                    attention_mask=prompt_inputs.attention_mask,
                    labels=ignore_labels,  # PeftModel需要labels参数，维度已调整
                    tmi_features=tmi_features
                )
                
                # 方法2: 自回归生成 - 基于logits逐token生成
                if hasattr(outputs, 'logits'):
                    generated_ids = []
                    current_ids = prompt_inputs.input_ids
                    current_mask = prompt_inputs.attention_mask
                    
                    for _ in range(1024):  # 最多生成1024个token，确保完整生成36个轨迹点
                        # 为当前序列准备labels（包含TMI tokens）
                        current_labels = torch.full(
                            (batch_size, current_ids.shape[1] + tmi_token_count),
                            -100,
                            dtype=torch.long,
                            device=model_device
                        )
                        
                        # 使用当前序列进行forward
                        outputs = model(
                            input_ids=current_ids,
                            attention_mask=current_mask,
                            labels=current_labels,
                            tmi_features=tmi_features
                        )
                        
                        # 获取最后一个token的logits
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                        
                        # 检查是否是结束token
                        if next_token.item() == tokenizer.eos_token_id:
                            break
                        
                        generated_ids.append(next_token.item())
                        
                        # 更新输入序列
                        current_ids = torch.cat([current_ids, next_token], dim=1)
                        current_mask = torch.cat([
                            current_mask,
                            torch.ones((1, 1), device=current_mask.device, dtype=current_mask.dtype)
                        ], dim=1)
                        
                        # 限制最大长度
                        if current_ids.shape[1] > 2048:
                            break
                    
                    # 只解码生成的tokens
                    if generated_ids:
                        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    else:
                        generated_text = ""
                else:
                    # 如果没有logits，返回空字符串
                    generated_text = ""
            
            # 解析预测轨迹
            pred_trajectory = parse_trajectory_from_text(generated_text)
            
            if pred_trajectory is not None:
                # 计算指标
                metrics.add_sample(pred_trajectory, gt_trajectory)
                successful_evals += 1
                
                # 打印第一个成功的预测
                if successful_evals == 1:
                    print(f"\n第一个成功预测:")
                    print(f"  生成文本: {generated_text[:200]}...")
                    print(f"  预测轨迹形状: {pred_trajectory.shape}")
                    print(f"  真实轨迹形状: {gt_trajectory.shape}")
                
        except Exception as e:
            skip_reasons["error"] += 1
            if sample_idx < 3:
                print(f"\n样本 {sample_idx} 处理出错: {e}")
            continue
    
    # 打印统计
    print(f"\n评估统计:")
    print(f"  成功评估: {successful_evals}")
    print(f"  跳过原因:")
    for reason, count in skip_reasons.items():
        if count > 0:
            print(f"    {reason}: {count}")
    
    return metrics


def evaluate_worker(
    gpu_id: int,
    checkpoint_path: str,
    data_chunk: List[Dict],
    feature_dir: str,
    return_dict: dict
):
    """
    单个GPU上的评估工作进程
    
    Args:
        gpu_id: GPU编号
        checkpoint_path: 模型路径
        data_chunk: 该GPU处理的数据块
        feature_dir: 特征目录
        return_dict: 用于返回结果的共享字典
    """
    # 设置当前进程使用的GPU
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    
    print(f"GPU {gpu_id}: 开始处理 {len(data_chunk)} 个样本")
    
    # 每个进程独立加载模型（避免模型共享问题）
    model, tokenizer = load_model_and_tokenizer(checkpoint_path, device=device)
    
    # 评估指标
    metrics = TrajectoryMetrics()
    skip_reasons = {"no_input": 0, "no_gt_traj": 0, "no_tmi": 0, "error": 0}
    successful_evals = 0
    
    # 处理数据
    for sample_idx, sample in enumerate(tqdm(data_chunk, desc=f"GPU {gpu_id}", position=gpu_id)):
        try:
            # 获取输入和真实答案
            messages = sample['messages']
            
            # 提取用户输入和真实轨迹
            user_input = None
            ground_truth = None
            
            for msg in messages:
                if msg['role'] == 'user':
                    user_input = msg['content']
                elif msg['role'] == 'assistant':
                    ground_truth = msg['content']
            
            if not user_input or not ground_truth:
                skip_reasons["no_input"] += 1
                continue
            
            # 解析真实轨迹
            gt_trajectory = parse_trajectory_from_text(ground_truth)
            if gt_trajectory is None:
                skip_reasons["no_gt_traj"] += 1
                continue
            
            # 准备输入
            user_input_clean = user_input.replace('<image>', '').strip()
            
            # 加载TMI特征
            tmi_features_path = sample.get('tmi_features')
            if tmi_features_path and Path(tmi_features_path).exists():
                tmi_features = np.load(tmi_features_path)
                # 与单卡保持一致：不需要unsqueeze
                tmi_features = torch.from_numpy(tmi_features).to(device).to(model.dtype)
            else:
                skip_reasons["no_tmi"] += 1
                continue
            
            # 构建prompt - 使用与训练时相同的chat template格式
            # 训练时LLaMA Factory会自动添加这些格式，评估时我们需要手动添加
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input_clean}
            ]
            
            # 应用chat template，添加特殊标记
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # 添加<|im_start|>assistant标记，提示模型开始生成
            )
            
            # Tokenize - 增加max_length避免截断
            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                max_length=2048,  # 增加到2048，确保输入不被截断
                truncation=True,
                padding=True
            ).to(device)
            
            # 生成预测
            with torch.no_grad():
                try:
                    # 重要：model.generate()不支持tmi_features参数
                    # 必须使用自定义的forward方法生成来确保TMI特征被使用
                    
                    # 直接使用forward方法（确保使用TMI）
                    raise Exception("使用forward方法以确保TMI特征被使用")
                    
                except Exception as e:
                    # 使用forward方法进行自回归生成（确保使用TMI特征）
                    if sample_idx == 0:
                        print(f"\nGPU {gpu_id} 第一个样本TMI特征:")
                        print(f"  特征形状: {tmi_features.shape}")
                        print(f"  开始自回归生成（最多1024步）...")
                    
                    tmi_token_count = 10
                    generated_ids = []
                    current_ids = inputs.input_ids
                    current_mask = inputs.attention_mask
                    
                    # 自回归生成（类似训练时的方式）
                    for _ in range(1024):  # 与单卡保持一致：生成1024步
                        batch_size = current_ids.shape[0]
                        
                        # 准备labels（包含TMI tokens的扩展）
                        current_labels = torch.full(
                            (batch_size, current_ids.shape[1] + tmi_token_count),
                            -100,
                            dtype=torch.long,
                            device=device
                        )
                        
                        # Forward with TMI features
                        outputs = model(
                            input_ids=current_ids,
                            attention_mask=current_mask,
                            labels=current_labels,
                            tmi_features=tmi_features  # 关键：这里真正使用了TMI特征！
                        )
                        
                        if hasattr(outputs, 'logits'):
                            # 获取最后一个token的预测
                            next_token_logits = outputs.logits[:, -1, :]
                            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                            
                            # 检查是否生成了结束标记
                            if next_token.item() == tokenizer.eos_token_id:
                                break
                            
                            generated_ids.append(next_token.item())
                            
                            # 更新输入序列
                            current_ids = torch.cat([current_ids, next_token], dim=1)
                            current_mask = torch.cat([
                                current_mask,
                                torch.ones((1, 1), device=device, dtype=current_mask.dtype)
                            ], dim=1)
                            
                            # 限制最大长度
                            if current_ids.shape[1] > 2048:
                                break
                        else:
                            break
                    
                    # 解码完整序列，然后提取assistant生成的部分
                    # current_ids包含: prompt tokens + 生成的tokens
                    if generated_ids:
                        # Decode完整序列（包含所有special tokens以便分割）
                        full_text = tokenizer.decode(current_ids[0], skip_special_tokens=False)
                        
                        # 提取<|im_start|>assistant之后的内容
                        if '<|im_start|>assistant' in full_text:
                            # 分割出assistant部分
                            assistant_part = full_text.split('<|im_start|>assistant')[-1]
                            # 移除可能的<|im_end|>标记
                            assistant_part = assistant_part.replace('<|im_end|>', '').strip()
                            generated_text = assistant_part
                        else:
                            # 备用：如果没有找到assistant标记，只decode新生成的部分
                            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    else:
                        # 如果没有生成新tokens，返回空字符串
                        generated_text = ""
            
            # generated_text现在只包含生成的部分，不需要去除输入
            
            # 解析预测轨迹
            pred_trajectory = parse_trajectory_from_text(generated_text)
            
            if pred_trajectory is not None:
                metrics.add_sample(pred_trajectory, gt_trajectory)
                successful_evals += 1
                
                # 打印第一个成功的样本，检查生成质量
                if successful_evals == 1:
                    print(f"\nGPU {gpu_id} 第一个成功样本:")
                    print(f"  生成文本长度: {len(generated_text)}")
                    print(f"  生成文本前200字符: {generated_text[:200]}...")
                    print(f"  解析出的轨迹点数: {len(pred_trajectory)}")
                    print(f"  预测前3个点: {pred_trajectory[:3]}")
                    print(f"  真实前3个点: {gt_trajectory[:3]}")
                    
                # 每100个成功样本打印一次统计
                if successful_evals % 100 == 0:
                    print(f"GPU {gpu_id}: 已成功评估 {successful_evals} 个样本")
            else:
                # 调试：打印解析失败的原因
                if sample_idx < 3:  # 只打印前3个失败样本
                    print(f"GPU {gpu_id} 样本{sample_idx} 轨迹解析失败:")
                    print(f"  生成文本长度: {len(generated_text)} 字符")
                    print(f"  生成文本前100字符: {generated_text[:100]}...")
                    # 检查是否包含PLANNING标签
                    if '<PLANNING>' in generated_text:
                        print(f"  包含<PLANNING>标签")
                        if '</PLANNING>' in generated_text:
                            print(f"  包含</PLANNING>标签")
                        else:
                            print(f"  缺少</PLANNING>结束标签")
                    else:
                        print(f"  缺少<PLANNING>标签")
                
        except Exception as e:
            skip_reasons["error"] += 1
            if sample_idx < 3:  # 只打印前3个错误
                print(f"GPU {gpu_id} 样本{sample_idx} 处理出错: {str(e)[:200]}")
            continue
    
    # 返回结果
    return_dict[gpu_id] = {
        'metrics': metrics,
        'skip_reasons': skip_reasons,
        'successful_evals': successful_evals
    }
    
    print(f"GPU {gpu_id}: 完成处理")


def evaluate_model_parallel(
    checkpoint_path: str,
    eval_data: List[Dict],
    feature_dir: str,
    max_samples: int = None,
    num_gpus: int = 8
) -> TrajectoryMetrics:
    """
    真正的多GPU并行评估
    
    Args:
        checkpoint_path: 模型checkpoint路径
        eval_data: 评估数据
        feature_dir: TMI特征目录
        max_samples: 最大评估样本数
        num_gpus: 使用的GPU数量
        
    Returns:
        合并后的评估指标
    """
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    print(f"使用 {num_gpus} 个GPU并行评估 {len(eval_data)} 个样本...")
    
    # 将数据分块，每个GPU处理一部分
    chunk_size = len(eval_data) // num_gpus
    data_chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        if i == num_gpus - 1:
            # 最后一个GPU处理剩余的所有数据
            end_idx = len(eval_data)
        else:
            end_idx = (i + 1) * chunk_size
        data_chunks.append(eval_data[start_idx:end_idx])
    
    # 打印数据分配情况
    for i, chunk in enumerate(data_chunks):
        print(f"  GPU {i}: {len(chunk)} 个样本")
    
    # 使用多进程管理器创建共享字典
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # 创建进程池
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=evaluate_worker,
            args=(gpu_id, checkpoint_path, data_chunks[gpu_id], feature_dir, return_dict)
        )
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 合并所有GPU的结果
    combined_metrics = TrajectoryMetrics()
    total_skip_reasons = {"no_input": 0, "no_gt_traj": 0, "no_tmi": 0, "error": 0}
    total_successful = 0
    
    for gpu_id in range(num_gpus):
        if gpu_id in return_dict:
            result = return_dict[gpu_id]
            gpu_metrics = result['metrics']
            
            # 合并指标
            combined_metrics.ade_values.extend(gpu_metrics.ade_values)
            combined_metrics.fde_values.extend(gpu_metrics.fde_values)
            combined_metrics.ahe_values.extend(gpu_metrics.ahe_values)
            combined_metrics.fhe_values.extend(gpu_metrics.fhe_values)
            combined_metrics.miss_rates.extend(gpu_metrics.miss_rates)
            # 合并L2分时指标
            combined_metrics.l2_1s_values.extend(gpu_metrics.l2_1s_values)
            combined_metrics.l2_2s_values.extend(gpu_metrics.l2_2s_values)
            combined_metrics.l2_3s_values.extend(gpu_metrics.l2_3s_values)
            
            # 合并统计
            for reason, count in result['skip_reasons'].items():
                total_skip_reasons[reason] += count
            total_successful += result['successful_evals']
    
    # 打印总体统计
    print(f"\n总体评估统计:")
    print(f"  成功评估: {total_successful}")
    print(f"  跳过原因:")
    for reason, count in total_skip_reasons.items():
        if count > 0:
            print(f"    {reason}: {count}")
    
    return combined_metrics


def main():
    parser = argparse.ArgumentParser(description="评估Stage 2模型")
    parser.add_argument("--checkpoint", type=str, 
                       default="/code/VLA/outputs/stage2_llama_factory/checkpoint-1800",
                       help="模型checkpoint路径")
    parser.add_argument("--data_path", type=str,
                       default="/code/VLA/datasets/fused_features/val/val_with_tmi_cleaned.json",
                       help="验证数据路径")
    parser.add_argument("--feature_dir", type=str,
                       default="/code/VLA/datasets/fused_features/val/features",
                       help="TMI特征目录")
    parser.add_argument("--output_dir", type=str,
                       default="/code/VLA/outputs/evaluation",
                       help="输出目录")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大评估样本数")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="设备")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批处理大小")
    parser.add_argument("--use_parallel", action="store_true",
                       help="使用并行评估（批处理）")
    parser.add_argument("--num_gpus", type=int, default=8,
                       help="并行使用的GPU数量")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    eval_data = load_evaluation_data(args.data_path, args.feature_dir)
    
    # 评估
    if args.use_parallel:
        print(f"使用多GPU并行评估模式 (num_gpus={args.num_gpus})")
        # 多GPU并行评估不需要预先加载模型，每个进程会自己加载
        metrics = evaluate_model_parallel(
            checkpoint_path=args.checkpoint,
            eval_data=eval_data,
            feature_dir=args.feature_dir,
            max_samples=args.max_samples,
            num_gpus=args.num_gpus
        )
    else:
        print("使用串行评估模式")
        # 串行模式需要先加载模型
        model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.device)
        metrics = evaluate_model(
            model, tokenizer, eval_data, 
            max_samples=args.max_samples,
            device=args.device
        )
    
    # 计算统计
    stats = metrics.compute_statistics()
    
    # 打印结果
    print("\n" + "="*50)
    print("评估结果:")
    print("="*50)
    
    # 打印主要指标
    print("\n位置误差指标:")
    print("-"*30)
    if 'ADE_mean' in stats:
        print(f"ADE (平均位移误差): {stats['ADE_mean']:.4f} ± {stats['ADE_std']:.4f} m")
    if 'FDE_mean' in stats:
        print(f"FDE (最终位移误差): {stats['FDE_mean']:.4f} ± {stats['FDE_std']:.4f} m")
    
    # 打印L2分时指标
    print("\nL2分时指标 (与论文对比):")
    print("-"*30)
    if 'L2_1s_mean' in stats:
        print(f"L2 @ 1s: {stats['L2_1s_mean']:.4f} ± {stats['L2_1s_std']:.4f} m")
    if 'L2_2s_mean' in stats:
        print(f"L2 @ 2s: {stats['L2_2s_mean']:.4f} ± {stats['L2_2s_std']:.4f} m")
    if 'L2_3s_mean' in stats:
        print(f"L2 @ 3s: {stats['L2_3s_mean']:.4f} ± {stats['L2_3s_std']:.4f} m")
    if 'L2_avg' in stats:
        print(f"L2 Avg: {stats['L2_avg']:.4f} m")
    
    # 打印其他指标
    print("\n其他指标:")
    print("-"*30)
    if 'MissRate' in stats:
        print(f"Miss Rate: {stats['MissRate']:.2%}")
    if 'AHE_mean' in stats:
        print(f"AHE (平均航向误差): {stats['AHE_mean']:.4f} rad")
    if 'FHE_mean' in stats:
        print(f"FHE (最终航向误差): {stats['FHE_mean']:.4f} rad")
    
    print(f"\n评估样本数: {stats.get('num_samples', 0)}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"evaluation_results_{timestamp}.json"
    
    with open(result_file, 'w') as f:
        json.dump({
            'checkpoint': args.checkpoint,
            'data_path': args.data_path,
            'metrics': stats,
            'timestamp': timestamp,
            'num_samples_evaluated': stats.get('num_samples', 0)
        }, f, indent=2)
    
    print(f"\n结果已保存到: {result_file}")
    
    # 生成详细报告
    report_file = output_dir / f"evaluation_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Stage 2 模型评估报告\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"模型: {args.checkpoint}\n")
        f.write(f"数据: {args.data_path}\n")
        f.write(f"时间: {timestamp}\n\n")
        
        f.write("主要指标:\n")
        f.write("-"*40 + "\n")
        f.write(f"ADE (平均位移误差): {stats.get('ADE_mean', 0):.4f} ± {stats.get('ADE_std', 0):.4f} m\n")
        f.write(f"FDE (最终位移误差): {stats.get('FDE_mean', 0):.4f} ± {stats.get('FDE_std', 0):.4f} m\n")
        f.write(f"Miss Rate: {stats.get('MissRate', 0):.2%}\n")
        
        # 添加L2分时指标
        f.write(f"\nL2分时指标 (与论文对比):\n")
        f.write("-"*40 + "\n")
        f.write(f"L2 @ 1s: {stats.get('L2_1s_mean', 0):.4f} ± {stats.get('L2_1s_std', 0):.4f} m\n")
        f.write(f"  中位数: {stats.get('L2_1s_median', 0):.4f} m\n")
        f.write(f"L2 @ 2s: {stats.get('L2_2s_mean', 0):.4f} ± {stats.get('L2_2s_std', 0):.4f} m\n")
        f.write(f"  中位数: {stats.get('L2_2s_median', 0):.4f} m\n")
        f.write(f"L2 @ 3s: {stats.get('L2_3s_mean', 0):.4f} ± {stats.get('L2_3s_std', 0):.4f} m\n")
        f.write(f"  中位数: {stats.get('L2_3s_median', 0):.4f} m\n")
        f.write(f"L2 Avg: {stats.get('L2_avg', 0):.4f} m\n")
        
        if 'AHE_mean' in stats:
            f.write(f"\n航向角指标:\n")
            f.write("-"*40 + "\n")
            f.write(f"AHE (平均航向误差): {stats.get('AHE_mean', 0):.4f} ± {stats.get('AHE_std', 0):.4f} rad\n")
            f.write(f"FHE (最终航向误差): {stats.get('FHE_mean', 0):.4f} ± {stats.get('FHE_std', 0):.4f} rad\n")
        
        f.write(f"\n评估样本数: {stats.get('num_samples', 0)}\n")
    
    print(f"详细报告已保存到: {report_file}")


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main()