"""
自定义数据集处理器
让LLaMA-Factory能够加载三模态数据
"""

import json
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Any
from pathlib import Path


def preprocess_tri_modal_dataset(
    examples: Dict[str, List[Any]],
    tokenizer,
    processor,
    **kwargs
) -> Dict[str, List[Any]]:
    """
    预处理三模态数据集
    
    这个函数会被LLaMA-Factory调用来处理数据
    """
    
    model_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "pixel_values": [],
        "depth_pixel_values": [],
        "semantic_pixel_values": [],
        "labels": []
    }
    
    for i in range(len(examples["messages"])):
        # 1. 处理文本
        messages = examples["messages"][i]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=2048
        )
        
        model_inputs["input_ids"].append(text)
        model_inputs["attention_mask"].append([1] * len(text))
        
        # 2. 处理RGB图像
        if "images" in examples and examples["images"][i]:
            image_paths = examples["images"][i]
            # 加载第一张图像（前视图）
            rgb_image = Image.open(image_paths[0])
            pixel_values = processor.image_processor(rgb_image, return_tensors="pt")["pixel_values"]
            model_inputs["pixel_values"].append(pixel_values)
        else:
            # 创建空图像
            model_inputs["pixel_values"].append(torch.zeros(3, 392, 392))
        
        # 3. 处理深度图
        if "depth_maps" in examples and examples["depth_maps"][i]:
            depth_paths = examples["depth_maps"][i]
            # 加载第一张深度图
            depth_map = np.load(depth_paths[0])
            depth_tensor = torch.from_numpy(depth_map).float().unsqueeze(0)  # [1, H, W]
            # Resize到392x392
            depth_tensor = torch.nn.functional.interpolate(
                depth_tensor.unsqueeze(0), 
                size=(392, 392), 
                mode='bilinear'
            ).squeeze(0)
            model_inputs["depth_pixel_values"].append(depth_tensor)
        else:
            model_inputs["depth_pixel_values"].append(torch.zeros(1, 392, 392))
        
        # 4. 处理语义图
        if "semantic_maps" in examples and examples["semantic_maps"][i]:
            semantic_paths = examples["semantic_maps"][i]
            # 加载第一张语义图
            semantic_map = np.load(semantic_paths[0])
            # 转换为one-hot编码
            num_classes = 17  # nuScenes语义类别数
            h, w = semantic_map.shape
            semantic_onehot = np.zeros((num_classes, h, w))
            for c in range(num_classes):
                semantic_onehot[c] = (semantic_map == c)
            semantic_tensor = torch.from_numpy(semantic_onehot).float()
            # Resize到392x392
            semantic_tensor = torch.nn.functional.interpolate(
                semantic_tensor.unsqueeze(0), 
                size=(392, 392), 
                mode='nearest'
            ).squeeze(0)
            model_inputs["semantic_pixel_values"].append(semantic_tensor)
        else:
            model_inputs["semantic_pixel_values"].append(torch.zeros(17, 392, 392))
        
        # 5. 处理标签
        # 提取轨迹预测部分作为标签
        labels = text.copy()
        # 只保留assistant的回复部分作为标签
        # 这里需要根据实际的tokenizer来调整
        model_inputs["labels"].append(labels)
    
    return model_inputs


# 注册到LLaMA-Factory
def register_tri_modal_dataset():
    """
    注册三模态数据集到LLaMA-Factory
    """
    return {
        "tri_modal_trajectory_train": {
            "file_name": "nuscenes_sharegpt_train.json",
            "file_path": "/code/VLA/datasets/sharegpt_data/nuscenes_sharegpt_train.json",
            "preprocessing_func": preprocess_tri_modal_dataset,
            "columns": {
                "messages": "messages",
                "images": "images",
                "depth_maps": "depth_maps",
                "semantic_maps": "semantic_maps"
            }
        }
    }