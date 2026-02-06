#!/usr/bin/env python3
"""
动态注入TMI特征到Qwen模型
在不修改原始Qwen代码的情况下，让它能接受TMI特征
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any
from transformers import Qwen2VLForConditionalGeneration
from torch.nn import CrossEntropyLoss


def create_tmi_aware_forward(original_forward, tmi_projection_layer):
    """
    创建一个新的forward函数，可以处理TMI特征
    """
    
    def forward_with_tmi(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        **kwargs
    ):
        # 检查是否有TMI特征输入
        tmi_features = kwargs.pop('tmi_features', None)
        tmi_features_path = kwargs.pop('tmi_features_path', None)
        
        if tmi_features_path is not None:
            # 从文件加载TMI特征
            tmi_features = np.load(tmi_features_path)
            # 转换为tensor并移动到设备，保持与模型相同的dtype
            tmi_features = torch.from_numpy(tmi_features).to(self.device)
            # 确保与模型dtype一致（BF16或FP32）
            if hasattr(self, 'dtype'):
                tmi_features = tmi_features.to(self.dtype)
        
        if tmi_features is not None:
            # 使用TMI特征替代视觉编码
            # 1. 投影TMI特征（如果需要）
            tmi_features_projected = tmi_projection_layer(tmi_features)
            
            # 2. 确保特征形状正确
            # TMI特征: [batch_size, seq_len, hidden_size] 或 [seq_len, hidden_size]
            if tmi_features_projected.dim() == 2:
                tmi_features_projected = tmi_features_projected.unsqueeze(0)
            
            batch_size = input_ids.shape[0] if input_ids is not None else tmi_features_projected.shape[0]
            
            # 如果batch size不匹配，扩展TMI特征
            if tmi_features_projected.shape[0] == 1 and batch_size > 1:
                tmi_features_projected = tmi_features_projected.expand(batch_size, -1, -1)
            
            # 3. 获取文本嵌入
            if hasattr(self, 'model') and hasattr(self.model, 'embed_tokens'):
                text_embeds = self.model.embed_tokens(input_ids)
            elif hasattr(self, 'get_input_embeddings'):
                text_embeds = self.get_input_embeddings()(input_ids)
            else:
                # 对于Qwen2VL，嵌入层在model.model.embed_tokens
                text_embeds = self.model.model.embed_tokens(input_ids)
            
            # 4. 合并TMI特征和文本特征
            # TMI特征应该插入到文本嵌入的开始位置（模拟视觉token）
            combined_embeds = torch.cat([tmi_features_projected, text_embeds], dim=1)
            
            # 5. 更新attention_mask
            if attention_mask is not None:
                tmi_seq_len = tmi_features_projected.shape[1]
                tmi_mask = torch.ones(
                    batch_size, tmi_seq_len,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
                attention_mask = torch.cat([tmi_mask, attention_mask], dim=1)
            
            # 6. 调用模型的语言模型部分
            # 对于Qwen2VL，语言模型在self.model
            model_outputs = self.model(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                **kwargs
            )
            
            # 7. 计算语言模型损失（如果有标签）
            labels = kwargs.get('labels', None)
            if labels is not None:
                # Shift标签以对齐预测
                shift_logits = model_outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # 计算交叉熵损失
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                model_outputs.loss = loss
            
            return model_outputs
        else:
            # 没有TMI特征，使用原始forward
            return original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                **kwargs
            )
    
    return forward_with_tmi


def inject_tmi_support(model: Qwen2VLForConditionalGeneration, tmi_hidden_size: int = 3584):
    """
    给标准Qwen模型注入TMI支持
    
    Args:
        model: 标准的Qwen2.5-VL模型
        tmi_hidden_size: TMI输出的特征维度（应该是3584，与Qwen隐藏层维度一致）
    """
    
    # 1. 创建投影层
    # 重要修正：TMI输出已经是3584维（与Qwen的hidden_size一致）
    # 不需要额外投影，或者只需要一个恒等映射
    llm_hidden_size = model.config.hidden_size  # 3584 for Qwen2.5-VL-7B
    
    # 如果TMI输出维度与LLM隐藏维度相同，使用恒等映射
    if tmi_hidden_size == llm_hidden_size:
        tmi_projection = nn.Identity()
    else:
        # 否则创建投影层
        tmi_projection = nn.Linear(tmi_hidden_size, llm_hidden_size)
    
    # 将投影层移动到正确的设备和dtype
    tmi_projection = tmi_projection.to(model.device)
    if hasattr(model, 'dtype'):
        tmi_projection = tmi_projection.to(model.dtype)
    
    # 2. 保存原始forward
    original_forward = model.forward
    
    # 3. 替换forward方法
    model.forward = create_tmi_aware_forward(original_forward, tmi_projection).__get__(model, type(model))
    
    # 4. 添加投影层到模型（这样可以一起保存）
    model.tmi_projection = tmi_projection
    
    print("✓ TMI支持已注入到Qwen模型")
    return model


def save_tmi_adapter(model, save_path: str):
    """
    保存TMI适配器权重
    """
    adapter_state = {
        'tmi_projection': model.tmi_projection.state_dict()
    }
    torch.save(adapter_state, save_path)
    print(f"✓ TMI适配器已保存到: {save_path}")


def load_tmi_adapter(model, adapter_path: str):
    """
    加载TMI适配器权重
    """
    adapter_state = torch.load(adapter_path, map_location='cpu')
    model.tmi_projection.load_state_dict(adapter_state['tmi_projection'])
    print(f"✓ TMI适配器已加载: {adapter_path}")
    return model