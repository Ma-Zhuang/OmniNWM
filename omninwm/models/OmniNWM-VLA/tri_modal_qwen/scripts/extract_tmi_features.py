#!/usr/bin/env python3
"""
使用训练好的TMI模块或SSR-MIDI提取融合特征
为Stage 2的LLaMA Factory训练准备数据

支持两种模式：
1. TMI模式：使用原有的TMI模型提取特征
2. MIDI模式：使用SSR-MIDI-7B提取特征
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import argparse
import base64
import io
from PIL import Image

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

# 根据模式导入不同的模块
USE_MIDI_MODE = os.environ.get('USE_MIDI_MODE', 'false').lower() == 'true'

if not USE_MIDI_MODE:
    # TMI模式导入
    from tri_modal_qwen.modeling.modeling_tri_modal_qwen import TriModalQwenForCausalLM
    from tri_modal_qwen.modeling.configuration_tri_modal_qwen import TriModalQwenConfig
    from tri_modal_qwen.data.processor import TriModalProcessor
    from tri_modal_qwen.data.path_mapper import PathMapper
    from transformers import AutoTokenizer
else:
    # MIDI模式导入
    sys.path.append(str(Path("/Users/pengbaorui/Desktop/VLA/SSR-main")))
    sys.path.append(str(Path("/code/VLA/SSR-main")))
    from ssr.models.midi import MIDI
    from ssr.utils.prompt import SSRSpecialToken, insert_tor
    from ssr.utils.misc import quiet, freeze_module
    from transformers import (
        AutoTokenizer,
        CLIPProcessor, CLIPVisionModel,
        SiglipProcessor, SiglipVisionModel
    )


def insert_tor_intelligent(text: str, num_tor: int = 10) -> str:
    """
    空间锚定的TOR插入策略：专门为MIDI的空间推理能力优化
    将TOR tokens锚定在空间语义关键位置，让MIDI能够编码不同的空间信息
    """
    import re
    
    # 如果可以使用SSR的原始insert_tor且不是MIDI模式，使用原始方法
    if USE_MIDI_MODE:
        try:
            # 对于MIDI，我们使用自定义的空间锚定策略
            pass
        except:
            pass
    
    # 空间锚定策略：为不同类型的空间信息分配TOR
    tor_positions = []
    
    # 1. 在全景图描述的各个视角后插入TOR（6个视角，分配6个TOR）
    view_keywords = [
        ('front,', 1), ('front-left,', 1), ('front-right', 1),
        ('rear,', 1), ('rear-left,', 1), ('rear-right', 1)
    ]
    
    for keyword, tor_count in view_keywords:
        if keyword in text.lower():
            # 找到关键词位置
            idx = text.lower().index(keyword)
            # 在关键词后插入
            tor_positions.append(idx + len(keyword))
    
    # 2. 在历史轨迹时间点后插入TOR（最重要的2个时间点）
    # 匹配 "(t-X.XXXs)" 模式
    time_matches = list(re.finditer(r'\(t-[\d.]+s\)', text))
    # 取最近的2个时间点（最相关的历史）
    for match in time_matches[-2:]:
        tor_positions.append(match.end())
    
    # 3. 在关键空间动作词后插入TOR（2个）
    action_keywords = ['Predict', 'Task:', 'trajectory', 'spatial']
    action_count = 0
    for keyword in action_keywords:
        if action_count >= 2:
            break
        if keyword in text:
            idx = text.index(keyword) + len(keyword)
            if idx not in tor_positions:
                tor_positions.append(idx)
                action_count += 1
    
    # 去重并排序
    tor_positions = sorted(list(set(tor_positions)))
    
    # 如果位置不够10个，在合适的位置补充
    if len(tor_positions) < num_tor:
        # 计算需要补充的数量
        needed = num_tor - len(tor_positions)
        
        # 在句子边界补充
        sentence_ends = [m.end() for m in re.finditer(r'\. ', text)]
        for end in sentence_ends[:needed]:
            if end not in tor_positions:
                tor_positions.append(end)
        
        # 如果还不够，均匀分布
        if len(tor_positions) < num_tor:
            text_len = len(text)
            step = text_len // (num_tor - len(tor_positions) + 1)
            for i in range(1, num_tor - len(tor_positions) + 1):
                pos = i * step
                if pos not in tor_positions:
                    tor_positions.append(pos)
    
    # 限制到num_tor个
    tor_positions = sorted(tor_positions)[:num_tor]
    
    # 插入TOR tokens（从后向前，避免位置偏移）
    result = text
    for pos in reversed(tor_positions):
        # 使用<tor>（小写）以兼容SSR
        result = result[:pos] + " <tor>" + result[pos:]
    
    # 清理多余空格
    result = re.sub(r'\s+', ' ', result)
    
    # 验证TOR数量
    tor_count = result.lower().count('<tor>')
    if tor_count < num_tor:
        # 在末尾补充
        for _ in range(num_tor - tor_count):
            result += " <tor>"
    
    return result.strip()


def supplement_tor_tokens(text: str, remaining: int) -> str:
    """
    补充TOR tokens到指定数量
    在文本末尾或合适位置添加剩余的TOR
    """
    words = text.split()
    
    # 策略1：在句子边界插入
    sentence_ends = []
    for i, word in enumerate(words):
        if word.endswith('.') or word.endswith('!') or word.endswith('?'):
            sentence_ends.append(i + 1)
    
    # 在句子边界插入
    for pos in sentence_ends[:remaining]:
        if pos < len(words):
            words.insert(pos, "<TOR>")
            remaining -= 1
    
    # 策略2：如果还需要更多，均匀分布在文本中
    if remaining > 0:
        step = max(1, len(words) // (remaining + 1))
        inserted = 0
        for i in range(step, len(words), step):
            if inserted >= remaining:
                break
            if i < len(words) and words[i] != "<TOR>":
                words.insert(i, "<TOR>")
                inserted += 1
    
    # 策略3：如果还不够，在末尾添加
    current_tor = words.count("<TOR>")
    target_tor = current_tor + remaining
    while words.count("<TOR>") < target_tor:
        words.append("<TOR>")
    
    return " ".join(words)


def load_trained_model(checkpoint_path: str, device: str = "cuda"):
    """加载Stage 1训练好的完整模型"""
    if USE_MIDI_MODE:
        # MIDI模式不需要这个函数
        return None
    
    print(f"加载训练好的模型: {checkpoint_path}")
    
    # 确保路径是目录
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path.parent
    
    # 方法1: 尝试直接使用from_pretrained（最优方式）
    try:
        print("尝试使用from_pretrained直接加载模型...")
        
        # 设置设备映射
        device_map = "auto" if device == "cuda" else None
        
        # 直接从checkpoint加载完整的训练好的模型
        # 使用与训练时相同的dtype策略
        if device == "cuda" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16  # 优先使用BF16
            print("使用BF16精度（与训练时一致）")
        else:
            torch_dtype = torch.float32  # 否则使用FP32
            print("使用FP32精度")
        
        model = TriModalQwenForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
        print("✓ 模型加载成功 (from_pretrained)")
        print(f"  - TMI模块参数: {sum(p.numel() for p in model.tmi_module.parameters()) / 1e6:.1f}M")
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"from_pretrained加载失败: {e}")
        print("使用备用方法加载...")
    
    # 方法2: 手动加载权重（备用方式）
    index_file = checkpoint_path / "model.safetensors.index.json"
    if index_file.exists():
        print("检测到分片的safetensors格式，手动加载...")
        
        try:
            from safetensors import safe_open
            from safetensors.torch import load_file
            
            # 加载配置
            config_file = checkpoint_path / "config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                # 确保base_model_name_or_path正确
                if 'base_model_name_or_path' in config_dict:
                    if 'checkpoint' in config_dict.get('base_model_name_or_path', '') or 'best_model' in config_dict.get('base_model_name_or_path', ''):
                        config_dict['base_model_name_or_path'] = '/code/VLA/models/Qwen2.5-VL-7B-Instruct'
                config = TriModalQwenConfig(**config_dict)
            else:
                config = TriModalQwenConfig(
                    base_model_name_or_path='/code/VLA/models/Qwen2.5-VL-7B-Instruct'
                )
            
            # 创建模型
            print("创建模型结构...")
            
            # 使用与训练时相同的dtype策略
            if device == "cuda" and torch.cuda.is_bf16_supported():
                print("使用BF16精度（与训练时一致）")
                # 配置中设置dtype，让模型的_setup_dtype方法正确处理
                config.torch_dtype = "bfloat16"
            else:
                print("使用FP32精度")
                config.torch_dtype = "float32"
            
            with torch.cuda.device(device if device != "cpu" else None):
                model = TriModalQwenForCausalLM(config)
            
            if device != "cpu":
                model = model.to(device)
            
            # 加载权重
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            state_dict = {}
            weight_files = set(index['weight_map'].values())
            
            for weight_file in weight_files:
                weight_path = checkpoint_path / weight_file
                if weight_path.exists():
                    print(f"加载权重文件: {weight_file}")
                    with safe_open(weight_path, framework="pt", device=device) as f:
                        for key in f.keys():
                            state_dict[key] = f.get_tensor(key)
            
            # 加载权重到模型
            print("加载权重到模型...")
            result = model.load_state_dict(state_dict, strict=False)
            
            if result.missing_keys:
                print(f"缺失的权重键: {len(result.missing_keys)} 个")
                # 只显示前几个缺失的键用于调试
                for key in list(result.missing_keys)[:5]:
                    print(f"  - {key}")
            if result.unexpected_keys:
                print(f"意外的权重键: {len(result.unexpected_keys)} 个")
            
            # 验证TMI权重
            tmi_keys = [k for k in state_dict.keys() if 'tmi' in k.lower()]
            print(f"✓ 加载了 {len(tmi_keys)} 个TMI权重")
            
            print("✓ 模型加载成功 (手动加载)")
            print(f"  - TMI模块参数: {sum(p.numel() for p in model.tmi_module.parameters()) / 1e6:.1f}M")
            
            # 验证关键组件
            assert hasattr(model, 'tmi_module'), "TMI模块未找到"
            assert hasattr(model, 'model'), "基座模型未找到"
            print("✓ 所有必要组件已加载")
            
            model.eval()
            return model
            
        except Exception as e:
            print(f"手动加载失败: {e}")
            raise e
    
    # 尝试其他格式
    try:
        # 直接加载完整的TriModalQwenForCausalLM模型
        model = TriModalQwenForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        
        print("✓ 模型加载成功")
        print(f"  - TMI模块参数: {sum(p.numel() for p in model.tmi_module.parameters()) / 1e6:.1f}M")
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        
        # 如果都失败了，尝试手动构建
        config = TriModalQwenConfig(
            base_model_name_or_path='/code/VLA/models/Qwen2.5-VL-7B-Instruct'
        )
        model = TriModalQwenForCausalLM(config)
        
        # 尝试加载单个权重文件
        for name in ["pytorch_model.bin", "model.safetensors", "model.pt"]:
            model_file = checkpoint_path / name
            if model_file.exists():
                print(f"找到权重文件: {name}")
                state_dict = torch.load(model_file, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
                print("✓ 手动加载模型成功")
                break
        else:
            print("警告: 未找到模型权重文件，将使用随机初始化的模型")
        
        model.eval()
        return model


def prepare_image_inputs(rgb_paths: List[str], image_size: int = 378) -> torch.Tensor:
    """准备RGB图像输入
    
    将6个摄像头图像合并为全景图，并转换为模型期望的格式
    """
    from PIL import Image
    from torchvision import transforms
    
    # 加载并调整图像大小
    images = []
    for path in rgb_paths:
        img = Image.open(path).convert('RGB')
        # 调整到标准尺寸
        img = img.resize((image_size * 16 // 9, image_size))  # 保持16:9比例
        images.append(img)
    
    # 创建2x3网格全景图
    panorama_width = images[0].width * 3
    panorama_height = images[0].height * 2
    panorama = Image.new('RGB', (panorama_width, panorama_height))
    
    # 按照nuscenes的摄像头顺序排列: 
    # 上排: FRONT_LEFT, FRONT, FRONT_RIGHT
    # 下排: BACK_LEFT, BACK, BACK_RIGHT
    positions = [
        (0, 0), (images[0].width, 0), (images[0].width * 2, 0),
        (0, images[0].height), (images[0].width, images[0].height), (images[0].width * 2, images[0].height)
    ]
    
    for img, pos in zip(images, positions):
        panorama.paste(img, pos)
    
    # 转换为tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pixel_values = transform(panorama).unsqueeze(0)  # [1, 3, H, W]
    
    return pixel_values, panorama


def process_sample(
    sample: Dict,
    model,  # 移除类型注解，支持TMI和MIDI模式
    device: str = "cuda",
    output_dir: str = None
) -> Dict:
    """处理单个样本，提取TMI融合特征"""
    
    try:
        # 1. 处理6个摄像头的图像路径
        rgb_paths = sample.get('images', [])
        depth_paths = sample.get('depth_maps', [])
        semantic_paths = sample.get('semantic_maps', [])
        
        # 修复路径（确保路径以/开头）
        depth_paths = ['/' + p if not p.startswith('/') else p for p in depth_paths]
        semantic_paths = ['/' + p if not p.startswith('/') else p for p in semantic_paths]
        
        # 确保有6个摄像头
        if len(rgb_paths) != 6 or len(depth_paths) != 6 or len(semantic_paths) != 6:
            print(f"跳过样本: 摄像头数量不是6")
            return None
        
        # 2. 准备RGB输入（创建全景图）
        pixel_values, rgb_panorama = prepare_image_inputs(rgb_paths, image_size=378)
        pixel_values = pixel_values.to(device).to(model.dtype)
        
        # 3. 准备深度和语义输入
        from PIL import Image
        import numpy as np
        
        # 加载深度图像并创建全景图
        depth_images = [Image.open(p).convert('L').resize((672, 378)) for p in depth_paths]
        depth_panorama = Image.new('L', (672*3, 378*2))
        positions = [(0,0), (672,0), (1344,0), (0,378), (672,378), (1344,378)]
        for img, pos in zip(depth_images, positions):
            depth_panorama.paste(img, pos)
        
        # 加载语义图像并创建全景图
        # 重要：语义图是类别索引图，需要转换为one-hot编码
        semantic_images = []
        for p in semantic_paths:
            # 加载为灰度图（类别索引）
            sem_img = Image.open(p)
            if sem_img.mode != 'L':
                # 如果是RGB，取第一个通道作为类别索引
                sem_img = sem_img.convert('L')
            sem_img = sem_img.resize((672, 378), Image.NEAREST)  # 使用最近邻插值避免改变类别值
            semantic_images.append(sem_img)
        
        # 创建语义全景图
        semantic_panorama = Image.new('L', (672*3, 378*2))
        for img, pos in zip(semantic_images, positions):
            semantic_panorama.paste(img, pos)
        
        # 转换为tensor并应用one-hot编码（与训练时一致）
        # 深度图：保持float32，稍后转为模型dtype
        depth_tensor = torch.from_numpy(np.array(depth_panorama)).unsqueeze(0).unsqueeze(0).float().to(device)
        # 深度图可以安全地使用BFloat16
        depth_tensor = depth_tensor.to(model.dtype)
        
        # 语义图转为tensor并进行one-hot编码
        semantic_array = np.array(semantic_panorama)
        semantic_tensor = torch.from_numpy(semantic_array).long()  # 类别索引应该是long类型
        
        # 转换为one-hot编码（150个类别）
        num_classes = 150
        semantic_one_hot = torch.nn.functional.one_hot(semantic_tensor, num_classes=num_classes)
        # 重新排列维度: [H, W, C] -> [C, H, W]
        semantic_one_hot = semantic_one_hot.permute(2, 0, 1).float()
        # 添加batch维度: [C, H, W] -> [1, C, H, W]
        semantic_tensor = semantic_one_hot.unsqueeze(0).to(device)
        
        # 重要：语义特征保持float32！
        # one_hot操作不支持BFloat16，且语义编码器可能需要float32输入
        # 让TMI模块内部处理dtype转换
        # semantic_tensor = semantic_tensor.to(model.dtype)  # 不要转换！
        
        # 4. 使用模型提取RGB特征（使用训练时相同的方法）
        with torch.no_grad():
            # 使用模型内部的_extract_rgb_features方法
            # 这确保与训练时的处理完全一致
            rgb_features = model._extract_rgb_features(
                pixel_values=pixel_values,
                image_grid_thw=None  # 对于单张图像，不需要grid信息
            )
            
            # 5. 通过TMI模块生成融合特征
            fused_features = model.tmi_module(
                rgb_features=rgb_features,
                depth_pixel_values=depth_tensor,
                semantic_pixel_values=semantic_tensor
            )
        
        # 6. 保存TMI融合特征
        sample_id = sample.get('id', 'unknown')
        
        # 保存特征文件
        # 重要：numpy不支持BFloat16，需要转换为float32
        features_to_save = fused_features.cpu().float().numpy()  # 转为float32再保存
        
        if output_dir:
            feature_path = os.path.join(output_dir, f"{sample_id}_features.npy")
            np.save(feature_path, features_to_save)
            
            # 同时保存RGB全景图供可视化
            image_path = os.path.join(output_dir, f"{sample_id}_panorama.png")
            rgb_panorama.save(image_path)
        else:
            feature_path = f"/tmp/{sample_id}_features.npy"
            np.save(feature_path, features_to_save)
            image_path = f"/tmp/{sample_id}_panorama.png"
            rgb_panorama.save(image_path)
        
        # 7. 创建输出格式
        messages = sample.get("messages", [])
        
        processed_sample = {
            "id": sample.get("id", ""),
            "messages": messages,  # 保持原始的对话格式
            "tmi_features": feature_path,  # TMI特征文件路径
            "panorama_image": image_path,  # 全景图供可视化
            "feature_shape": list(fused_features.shape)  # 特征形状信息
        }
        
        return processed_sample
        
    except Exception as e:
        print(f"处理样本失败: {e}")
        return None


def process_single_sample_midi(
    sample: Dict,
    midi_model,
    clip_model,
    clip_processor,
    siglip_model,
    siglip_processor,
    segformer_model,
    segformer_processor,
    mamba_tokenizer,
    tor_token_id: int,
    output_dir: str,
    device: str = "cuda",
    create_panorama: bool = True
) -> Dict:
    """使用MIDI模型处理单个样本，提取融合特征（三模态：RGB + Depth + Semantic）"""
    
    try:
        # 1. 处理6个摄像头的图像路径
        rgb_paths = sample.get('images', [])
        depth_paths = sample.get('depth_maps', [])
        semantic_paths = sample.get('semantic_maps', [])
        
        # 修复路径（确保路径以/开头）
        depth_paths = ['/' + p if not p.startswith('/') else p for p in depth_paths]
        semantic_paths = ['/' + p if not p.startswith('/') else p for p in semantic_paths]
        
        # 确保有6个摄像头
        if len(rgb_paths) != 6 or len(depth_paths) != 6 or len(semantic_paths) != 6:
            print(f"跳过样本: 摄像头数量不是6 (RGB:{len(rgb_paths)}, Depth:{len(depth_paths)}, Semantic:{len(semantic_paths)})")
            return None
        
        if create_panorama:
            # 创建全景图（与TMI一致的行为）
            positions = [(0,0), (672,0), (1344,0), (0,378), (672,378), (1344,378)]
            
            # RGB全景图
            rgb_images = [Image.open(p).convert('RGB').resize((672, 378)) for p in rgb_paths]
            rgb_panorama = Image.new('RGB', (672*3, 378*2))
            for img, pos in zip(rgb_images, positions):
                rgb_panorama.paste(img, pos)
            
            # 深度全景图
            depth_images = [Image.open(p).convert('L').resize((672, 378)) for p in depth_paths]
            depth_panorama = Image.new('L', (672*3, 378*2))
            for img, pos in zip(depth_images, positions):
                depth_panorama.paste(img, pos)
            
            # 语义全景图 (三模态新增)
            semantic_images = [Image.open(p).convert('RGB').resize((672, 378)) for p in semantic_paths]
            semantic_panorama = Image.new('RGB', (672*3, 378*2))
            for img, pos in zip(semantic_images, positions):
                semantic_panorama.paste(img, pos)
            
            # 转为numpy用于编码
            rgb_array = np.array(rgb_panorama)
            depth_array = np.array(depth_panorama)
            semantic_array = np.array(semantic_panorama)
            # 深度图转为RGB格式（SigLIP需要）
            depth_array_rgb = np.stack([depth_array] * 3, axis=-1)
        else:
            # 只使用前视相机
            rgb_array = np.array(Image.open(rgb_paths[0]).convert('RGB'))
            depth_img = Image.open(depth_paths[0])
            if depth_img.mode != 'RGB':
                depth_array = np.array(depth_img.convert('L'))
                depth_array_rgb = np.stack([depth_array] * 3, axis=-1)
            else:
                depth_array_rgb = np.array(depth_img)
            rgb_panorama = Image.fromarray(rgb_array)
        
        # 2. 编码视觉特征 (三模态)
        with torch.no_grad():
            # RGB特征
            image_embeds = clip_model(
                **clip_processor(images=rgb_array, return_tensors="pt").to(device)
            ).last_hidden_state  # [1, 197, 1024]
            
            # 深度特征
            depth_embeds = siglip_model(
                **siglip_processor(images=depth_array_rgb, return_tensors="pt").to(device)
            ).last_hidden_state  # [1, 197, 1152]
            
            # 语义特征 (三模态新增)
            semantic_outputs = segformer_model(
                **segformer_processor(images=semantic_array, return_tensors="pt").to(device)
            )
            semantic_features = semantic_outputs.last_hidden_state  # [1, 512, H, W]
            B, C, H, W = semantic_features.shape
            # 重塑为序列格式: [1, 512, H, W] -> [1, H*W, 512]
            semantic_embeds = semantic_features.squeeze(0).permute(1, 2, 0).reshape(H * W, C).unsqueeze(0)  # [1, H*W, 512]
        
        # 3. 准备文本输入
        messages = sample.get("messages", [])
        # 提取用户问题
        question = ""
        for msg in messages:
            if msg['role'] == 'user':
                content = msg.get('content', '')
                if isinstance(content, str):
                    # 清理文本，去除<image>标记
                    question = content.replace('<image>', '').strip()
                    break
        
        # 改进的文本处理：保留更多空间语义信息，专门为MIDI的空间理解设计
        if "predict" in question.lower() and "trajectory" in question.lower():
            import re
            
            # 1. 明确告诉MIDI这是一个全景图的空间布局
            panorama_description = (
                "Analyzing panoramic sensor fusion with 6 camera views arranged in a 2x3 grid: "
                "Top row shows front, front-left, front-right views. "
                "Bottom row shows rear, rear-left, rear-right views. "
            )
            
            # 2. 提取并保留所有历史轨迹信息（MIDI需要完整的时序信息）
            trajectory_pattern = r'\(t-[\d.]+s\)[^)]*\)[^,\n]*(?:,\s*[^,\n]+){3,4}'
            trajectory_matches = re.findall(trajectory_pattern, question)
            
            # 保留最近的6个时间点（如果有的话）
            recent_trajectories = trajectory_matches[-6:] if trajectory_matches else []
            
            # 格式化历史轨迹，保留完整信息
            history_info = "Historical trajectory sequence: "
            for traj in recent_trajectories:
                # 保留完整的轨迹信息，包括位置、速度、加速度、转向
                history_info += traj + " "
            
            # 3. 提取任务要求
            task_info = "Task: Predict vehicle's future trajectory for next 3 seconds at 12Hz based on multi-modal spatial analysis. "
            
            # 4. 组合问题文本，保留更多空间语义
            question = panorama_description + history_info + task_info
        
        # 改进的TOR token插入策略：使用空间锚定
        if USE_MIDI_MODE:
            # MIDI模式：总是使用空间锚定策略
            text_with_tor = insert_tor_intelligent(question, 10)
            print(f"[TOR策略] 使用空间锚定插入（优化MIDI空间理解）")
        else:
            # TMI模式使用智能插入
            text_with_tor = insert_tor_intelligent(question, 10)
            print(f"[TOR策略] 使用智能插入")
        
        # 验证TOR tokens数量
        tor_count = text_with_tor.count('<TOR>') + text_with_tor.count('<tor>')
        if tor_count < 10:
            # 如果插入不足，补充插入
            remaining = 10 - tor_count
            text_with_tor = supplement_tor_tokens(text_with_tor, remaining)
            print(f"[TOR策略] 补充了{remaining}个TOR tokens")
        
        # 最终验证TOR数量
        final_tor_count = text_with_tor.count('<TOR>') + text_with_tor.count('<tor>')
        print(f"[TOR验证] 最终TOR数量: {final_tor_count}/10")
        if final_tor_count != 10:
            print(f"[警告] TOR数量不正确！实际: {final_tor_count}, 期望: 10")
        
        print(f"[TOR改进] 文本: {text_with_tor[:100]}...")
        
        # Tokenize
        # 注意：由于文本包含大量轨迹历史数据，需要足够的长度来容纳所有10个TOR tokens
        # 经过分析，完整的文本+10个TOR需要约700-800个tokens
        mamba_encoding = mamba_tokenizer(
            text_with_tor,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=True,
            max_length=1024  # 增加到1024以确保容纳完整的文本和所有10个TOR tokens
        )
        mamba_input_ids = mamba_encoding.input_ids.to(device)
        
        # 4. 创建attention mask (三模态)
        visual_seq_len = image_embeds.size(1) + depth_embeds.size(1) + semantic_embeds.size(1)
        text_seq_len = mamba_input_ids.size(1)
        mamba_attention_mask = torch.ones(1, visual_seq_len + text_seq_len).to(device)
        
        print(f"[三模态] 视觉序列长度: RGB={image_embeds.size(1)}, Depth={depth_embeds.size(1)}, Semantic={semantic_embeds.size(1)}, 总计={visual_seq_len}")
        
        # 5. MIDI推理 (三模态)
        with torch.no_grad():
            outputs = midi_model(
                mamba_input_ids=mamba_input_ids,
                mamba_attention_mask=mamba_attention_mask,
                image_embeds=image_embeds,
                depth_embeds=depth_embeds,
                semantic_embeds=semantic_embeds,  # ✓ 三模态
                tor_token_id=(tor_token_id, tor_token_id),
                alignment=False
            )
            
            # 获取TOR嵌入
            tor_embeds = outputs.tor_embeds  # 期望: [batch, n_tor, hidden_size]
            
            # 保持所有10个独立的TOR tokens，不进行平均
            if len(tor_embeds.shape) == 3:
                # [1, 10, 3584] -> [10, 3584]
                # 只去掉batch维度，保持10个独立tokens
                tor_embeds = tor_embeds.squeeze(0)
                print(f"[MIDI] TOR embeds shape after squeeze: {tor_embeds.shape}")
            elif len(tor_embeds.shape) == 2:
                # 已经是 [10, 3584] 形状，保持不变
                print(f"[MIDI] TOR embeds shape (already 2D): {tor_embeds.shape}")
            else:
                print(f"[MIDI] Unexpected TOR embeds shape: {tor_embeds.shape}")
            
            # 验证token之间的相似度（调试用）
            if tor_embeds.shape[0] > 1:
                with torch.no_grad():
                    sims = []
                    for i in range(min(tor_embeds.shape[0], 10)):
                        for j in range(i+1, min(tor_embeds.shape[0], 10)):
                            sim = torch.cosine_similarity(tor_embeds[i], tor_embeds[j], dim=0)
                            sims.append(sim.item())
                    if sims:
                        avg_sim = sum(sims) / len(sims)
                        print(f"[MIDI] Average similarity between TOR tokens: {avg_sim:.4f}")
        
        # 6. 保存特征（与TMI格式兼容）
        sample_id = sample.get('id', 'unknown')
        
        # 转换为numpy并保存为.npy（与TMI一致）
        features_to_save = tor_embeds.cpu().float().numpy()
        
        if output_dir:
            feature_path = os.path.join(output_dir, f"{sample_id}_features.npy")
            np.save(feature_path, features_to_save)
            
            # 保存全景图
            image_path = os.path.join(output_dir, f"{sample_id}_panorama.png")
            rgb_panorama.save(image_path)
        else:
            feature_path = f"/tmp/{sample_id}_features.npy"
            np.save(feature_path, features_to_save)
            image_path = f"/tmp/{sample_id}_panorama.png"
            rgb_panorama.save(image_path)
        
        # 7. 创建输出格式（与TMI完全一致）
        processed_sample = {
            "id": sample.get("id", ""),
            "messages": messages,  # 保持原始的对话格式
            "tmi_features": feature_path,  # 使用相同的字段名
            "panorama_image": image_path,  # 全景图供可视化
            "feature_shape": list(features_to_save.shape),  # 实际保存的特征形状
            "num_tor_tokens": features_to_save.shape[0] if len(features_to_save.shape) > 1 else 1  # TOR tokens数量
        }
        
        return processed_sample
        
    except Exception as e:
        print(f"MIDI处理样本失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="提取TMI或MIDI融合特征")
    parser.add_argument("--input_file", type=str, required=True,
                       help="输入数据文件（ShareGPT格式）")
    parser.add_argument("--output_file", type=str, required=True,
                       help="输出文件路径（包含融合特征）")
    parser.add_argument("--tmi_checkpoint", type=str, required=True,
                       help="Stage 1训练的TMI checkpoint路径或MIDI模型路径")
    # 移除qwen_model_path参数，因为我们直接加载训练好的完整模型
    parser.add_argument("--data_root", type=str,
                       default="/code/VLA/datasets/sharegpt_data",
                       help="数据根目录")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="输出目录（保存特征文件）")
    parser.add_argument("--device", type=str, default="cuda",
                       help="计算设备")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="批处理大小")
    # 多GPU并行参数
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="使用的GPU数量")
    parser.add_argument("--gpu_id", type=int, default=None,
                       help="指定使用的GPU ID（用于多进程并行）")
    parser.add_argument("--start_idx", type=int, default=None,
                       help="处理数据的起始索引")
    parser.add_argument("--end_idx", type=int, default=None,
                       help="处理数据的结束索引")
    
    args = parser.parse_args()
    
    # 多GPU并行处理逻辑
    if args.num_gpus > 1 and args.gpu_id is None:
        # 主进程：启动多个子进程
        import subprocess
        import math
        
        print(f"启动 {args.num_gpus} 个GPU并行处理...")
        
        # 首先获取数据总数
        with open(args.input_file, 'r') as f:
            total_samples = len(json.load(f))
        
        samples_per_gpu = math.ceil(total_samples / args.num_gpus)
        processes = []
        
        for gpu_id in range(args.num_gpus):
            start_idx = gpu_id * samples_per_gpu
            end_idx = min((gpu_id + 1) * samples_per_gpu, total_samples)
            
            # 为每个GPU创建单独的输出文件
            gpu_output_file = args.output_file.replace('.json', f'_gpu{gpu_id}.json')
            
            cmd = [
                "python", __file__,
                "--input_file", args.input_file,
                "--output_file", gpu_output_file,
                "--tmi_checkpoint", args.tmi_checkpoint,
                "--output_dir", args.output_dir,
                "--gpu_id", str(gpu_id),
                "--start_idx", str(start_idx),
                "--end_idx", str(end_idx),
                "--num_gpus", "1"  # 子进程不再启动更多进程
            ]
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            print(f"GPU {gpu_id}: 处理样本 {start_idx}-{end_idx}")
            process = subprocess.Popen(cmd, env=env)
            processes.append(process)
        
        # 等待所有进程完成
        print("等待所有GPU完成处理...")
        for i, p in enumerate(processes):
            p.wait()
            print(f"GPU {i} 完成")
        
        # 合并所有结果
        print("合并结果...")
        all_data = []
        for gpu_id in range(args.num_gpus):
            gpu_output_file = args.output_file.replace('.json', f'_gpu{gpu_id}.json')
            if os.path.exists(gpu_output_file):
                with open(gpu_output_file, 'r') as f:
                    all_data.extend(json.load(f))
                os.remove(gpu_output_file)  # 清理临时文件
        
        # 保存最终结果
        with open(args.output_file, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"✓ 完成！处理了 {len(all_data)} 个样本")
        print(f"输出文件: {args.output_file}")
        return
    
    # 设置设备
    if args.gpu_id is not None:
        # 子进程已经通过CUDA_VISIBLE_DEVICES设置了GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用GPU {args.gpu_id} (映射到cuda:0)")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 检测是否使用MIDI模式
    use_midi = USE_MIDI_MODE or 'SSR-MIDI' in args.tmi_checkpoint or 'midi' in args.tmi_checkpoint.lower()
    
    if use_midi:
        print("========== 使用MIDI模式 ==========")
        # 加载MIDI相关模型
        print("加载MIDI模型和编码器...")
        
        # 确保MIDI相关模块已导入
        try:
            # 尝试使用已导入的模块
            quiet
        except NameError:
            # 如果没有导入，动态导入
            sys.path.append(str(Path("/Users/pengbaorui/Desktop/VLA/SSR-main")))
            sys.path.append(str(Path("/code/VLA/SSR-main")))
            from ssr.models.midi import MIDI
            from ssr.utils.prompt import SSRSpecialToken, insert_tor
            from ssr.utils.misc import quiet, freeze_module
            from transformers import (
                CLIPProcessor, CLIPVisionModel,
                SiglipProcessor, SiglipVisionModel
            )
        
        quiet()  # 减少日志输出
        
        # 加载编码器
        clip_path = "/code/VLA/models/clip-vit-large-patch14-336"
        siglip_path = "/code/VLA/models/siglip-so400m-patch14-384"
        mamba_path = "/code/VLA/models/state-spaces/mamba-130m-hf"
        
        print(f"  - 加载CLIP (RGB编码器): {clip_path}", flush=True)
        clip_processor = CLIPProcessor.from_pretrained(clip_path)
        clip_model = CLIPVisionModel.from_pretrained(clip_path).to(device)
        freeze_module(clip_model)
        
        print(f"  - 加载SigLIP (Depth编码器): {siglip_path}", flush=True)
        siglip_processor = SiglipProcessor.from_pretrained(siglip_path)
        siglip_model = SiglipVisionModel.from_pretrained(siglip_path).to(device)
        freeze_module(siglip_model)
        
        print(f"  - 加载Mamba tokenizer: {mamba_path}", flush=True)
        mamba_tokenizer = AutoTokenizer.from_pretrained(mamba_path)
        mamba_tokenizer.add_tokens(SSRSpecialToken.TOR_TOKEN, special_tokens=True)
        tor_token_id = mamba_tokenizer.convert_tokens_to_ids(SSRSpecialToken.TOR_TOKEN)
        
        # 加载SegFormer用于语义编码 (三模态)
        segformer_path = "/code/VLA/models/segmentation_models/segformer-b5-finetuned-ade-640-640"
        print(f"  - 加载SegFormer (三模态语义编码器): {segformer_path}", flush=True)
        from transformers import SegformerImageProcessor, SegformerModel
        segformer_processor = SegformerImageProcessor.from_pretrained(segformer_path)
        segformer_model = SegformerModel.from_pretrained(segformer_path).to(device)
        freeze_module(segformer_model)
        segformer_model.eval()
        print(f"  ✓ SegFormer加载完成 (输出维度: 512)", flush=True)
        
        print(f"  - 加载MIDI三模态融合模型: {args.tmi_checkpoint}", flush=True)
        midi_model = MIDI.from_pretrained(args.tmi_checkpoint).to(device)
        freeze_module(midi_model)
        midi_model.eval()
        
        print("✓ 所有模型加载完成 (三模态架构: RGB + Depth + Semantic → MIDI融合)", flush=True)
        
        # 创建一个包装器使接口统一
        model = None  # 不使用TMI模型
        
    else:
        print("========== 使用TMI模式 ==========")
        # 1. 加载训练好的完整模型 - 直接在目标设备上加载
        model = load_trained_model(args.tmi_checkpoint, device="cuda" if device.type == "cuda" else "cpu")
        
        # 重要：不要手动转换dtype！让模型保持训练时的dtype设置
        # Stage 1训练使用BF16（如果支持）或FP32，而不是FP16
        # model内部的_setup_dtype()已经正确设置了所有模块的dtype
        # 强制转换为half (float16)会破坏dtype一致性，导致位置编码错误
        
        # 获取模型的实际dtype用于日志
        model_dtype = next(model.parameters()).dtype
        print(f"模型dtype: {model_dtype}")
        
        # MIDI相关变量设为None
        midi_model = None
        clip_model = None
        clip_processor = None
        siglip_model = None
        siglip_processor = None
        mamba_tokenizer = None
        tor_token_id = None
    
    # 4. 设置输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. 加载输入数据
    print(f"加载数据: {args.input_file}")
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    # 如果指定了起始和结束索引，只处理部分数据
    if args.start_idx is not None and args.end_idx is not None:
        print(f"处理数据片段: {args.start_idx}-{args.end_idx}")
        data = data[args.start_idx:args.end_idx]
    
    # 6. 处理所有样本
    processed_data = []
    
    if use_midi:
        desc = f"提取MIDI三模态特征 (GPU {args.gpu_id})" if args.gpu_id is not None else "提取MIDI三模态特征"
        for sample in tqdm(data, desc=desc):
            processed_sample = process_single_sample_midi(
                sample=sample,
                midi_model=midi_model,
                clip_model=clip_model,
                clip_processor=clip_processor,
                siglip_model=siglip_model,
                siglip_processor=siglip_processor,
                segformer_model=segformer_model,
                segformer_processor=segformer_processor,
                mamba_tokenizer=mamba_tokenizer,
                tor_token_id=tor_token_id,
                output_dir=output_dir,
                device=device,
                create_panorama=True  # 创建全景图，与TMI保持一致
            )
            
            if processed_sample:
                processed_data.append(processed_sample)
    else:
        desc = f"提取TMI特征 (GPU {args.gpu_id})" if args.gpu_id is not None else "提取TMI特征"
        for sample in tqdm(data, desc=desc):
            processed_sample = process_sample(
                sample=sample,
                model=model,
                device=device,
                output_dir=output_dir
            )
            
            if processed_sample:
                processed_data.append(processed_sample)
    
    # 7. 保存处理后的数据
    print(f"保存融合特征数据: {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    print(f"✓ 完成！处理了 {len(processed_data)}/{len(data)} 个样本")
    print(f"输出文件: {args.output_file}")


if __name__ == "__main__":
    main()