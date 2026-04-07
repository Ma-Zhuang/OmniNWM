#!/usr/bin/env python3
"""
推理脚本

支持两种模式：
1. 原始模式：直接使用三模态图像输入
2. TMI特征模式：使用预提取的TMI特征（LLaMA Factory训练后的模型）

功能：
- 单样本和批量推理
- 6摄像头全景融合
- 交互式推理模式
- 轨迹可视化
- ADE/FDE评估指标
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import time

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError as e:
    raise ImportError(f"缺少必要的PyTorch依赖: {e}")

try:
    from transformers import AutoTokenizer, AutoImageProcessor
except ImportError as e:
    raise ImportError(f"缺少transformers依赖: {e}")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    import cv2
except ImportError:
    warnings.warn("可视化功能需要matplotlib、seaborn、PIL和opencv")
    plt = None
    sns = None
    Image = None
    cv2 = None

# 导入项目模块
try:
    from src.tri_modal_qwen.modeling.configuration_tri_modal_qwen import TriModalQwenConfig
    from src.tri_modal_qwen.modeling.modeling_tri_modal_qwen import TriModalQwenForCausalLM
    from src.tri_modal_qwen.data.processor import TriModalProcessor
    from src.tri_modal_qwen.utils.visualization import TriModalVisualizer
except ImportError as e:
    raise ImportError(f"无法导入项目模块: {e}")


@dataclass
class InferenceConfig:
    """
    推理配置
    """
    # 模型配置
    model_path: str = "./checkpoints/best_model"
    model_config_path: Optional[str] = None
    use_tmi_features: bool = False  # 是否使用TMI特征模式
    tmi_feature_path: Optional[str] = None  # TMI特征文件路径
    tmi_checkpoint: Optional[str] = None  # TMI模块checkpoint（用于在线提取特征）
    
    # 输入配置
    rgb_image_path: Optional[str] = None
    depth_image_path: Optional[str] = None
    semantic_image_path: Optional[str] = None
    text_prompt: str = "基于三模态感知信息，预测车辆的未来轨迹。"
    
    # 批量推理配置
    batch_input_dir: Optional[str] = None
    output_dir: str = "./inference_results"
    
    # 生成配置
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    do_sample: bool = False
    num_beams: int = 1
    repetition_penalty: float = 1.0
    
    # 处理配置
    max_length: int = 2048
    image_size: int = 392
    
    # 可视化配置
    save_visualization: bool = True
    show_attention: bool = False
    interactive_mode: bool = False
    
    # 性能配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: str = "float16"  # float16, float32
    use_compile: bool = False  # PyTorch 2.0 compile
    
    # 其他配置
    seed: int = 42
    verbose: bool = True


class TriModalInference:
    """
    三模态VLM推理器
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.visualizer = None
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # 设置输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        if config.verbose:
            print(f"推理配置:")
            print(f"  模型路径: {config.model_path}")
            print(f"  输出目录: {config.output_dir}")
            print(f"  设备: {config.device}")
            print(f"  数据类型: {config.torch_dtype}")
    
    def setup_model(self):
        """设置模型和相关组件"""
        
        if self.config.verbose:
            print("加载模型...")
        
        # 确定数据类型
        torch_dtype = getattr(torch, self.config.torch_dtype)
        
        # 根据模式选择不同的模型加载方式
        if self.config.use_tmi_features:
            # TMI特征模式：加载标准Qwen模型 + TMI支持
            if self.config.verbose:
                print("使用TMI特征模式，加载标准Qwen2.5-VL模型...")
            
            try:
                # 尝试加载LLaMA Factory训练的模型
                from transformers import Qwen2VLForConditionalGeneration
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.config.model_path,
                    torch_dtype=torch_dtype,
                    device_map="auto" if self.config.device == "cuda" else None,
                    trust_remote_code=True
                )
                
                # 动态注入TMI支持
                sys.path.append(str(Path(__file__).parent.parent / "llama_factory_configs"))
                from inject_tmi_to_qwen import inject_tmi_support
                self.model = inject_tmi_support(self.model, tmi_hidden_size=4096)
                
                if self.config.verbose:
                    print("✓ TMI支持已注入到标准Qwen模型")
                    
            except Exception as e:
                if self.config.verbose:
                    print(f"警告: 无法加载Qwen2.5-VL，回退到TriModalQwen: {e}")
                # 回退到原始三模态模型
                self.config.use_tmi_features = False
        
        if not self.config.use_tmi_features:
            # 原始模式：加载完整的三模态模型
            # 加载配置
            if self.config.model_config_path:
                with open(self.config.model_config_path, 'r') as f:
                    model_config_dict = json.load(f)
                model_config = TriModalQwenConfig.from_dict(model_config_dict)
            else:
                config_path = Path(self.config.model_path) / "config.json"
                if config_path.exists():
                    model_config = TriModalQwenConfig.from_pretrained(self.config.model_path)
                else:
                    model_config = TriModalQwenConfig()
            
            # 加载模型
            self.model = TriModalQwenForCausalLM.from_pretrained(
                self.config.model_path,
                config=model_config,
                torch_dtype=torch_dtype,
                device_map="auto" if self.config.device == "cuda" else None
            )
        
        if self.config.device != "cuda":
            self.model.to(self.config.device)
        
        self.model.eval()
        
        # 编译模型（可选）
        if self.config.use_compile and hasattr(torch, 'compile'):
            if self.config.verbose:
                print("编译模型...")
            self.model = torch.compile(self.model)
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            padding_side='left'  # Flash Attention要求
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 创建处理器
        self.processor = TriModalProcessor(
            tokenizer=self.tokenizer,
            image_processor=AutoImageProcessor.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            ),
            max_length=self.config.max_length
        )
        
        # 创建可视化器
        if self.config.save_visualization:
            self.visualizer = TriModalVisualizer(
                model=self.model,
                processor=self.processor
            )
        
        if self.config.verbose:
            param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"模型参数量: {param_count:.1f}M")
        
        return self.model, self.tokenizer, self.processor
    
    def load_images(
        self, 
        rgb_paths: Union[str, List[str]], 
        depth_paths: Optional[Union[str, List[str]]] = None, 
        semantic_paths: Optional[Union[str, List[str]]] = None
    ) -> Dict[str, Union[np.ndarray, List[np.ndarray]]]:
        """加载三模态图像（支持单图像或6摄像头）"""
        
        if not Image:
            raise ImportError("需要安装PIL库用于图像处理")
        
        images = {}
        
        # 转换为列表格式
        if isinstance(rgb_paths, str):
            rgb_paths = [rgb_paths]
        if isinstance(depth_paths, str):
            depth_paths = [depth_paths]
        if isinstance(semantic_paths, str):
            semantic_paths = [semantic_paths]
        
        # 加载RGB图像
        rgb_images = []
        for rgb_path in rgb_paths:
            if rgb_path and Path(rgb_path).exists():
                rgb_image = Image.open(rgb_path).convert('RGB')
                rgb_image = rgb_image.resize((self.config.image_size, self.config.image_size))
                rgb_images.append(np.array(rgb_image))
            else:
                raise FileNotFoundError(f"RGB图像文件不存在: {rgb_path}")
        images['rgb'] = rgb_images if len(rgb_images) > 1 else rgb_images[0]
        
        # 加载深度图像
        if depth_paths:
            depth_images = []
            for depth_path in depth_paths:
                if depth_path and Path(depth_path).exists():
                    depth_image = Image.open(depth_path).convert('L')  # 灰度图
                    depth_image = depth_image.resize((self.config.image_size, self.config.image_size))
                    depth_images.append(np.array(depth_image))
                else:
                    if self.config.verbose:
                        print(f"警告: 深度图像不存在: {depth_path}")
                    return None  # 如果缺少深度图，返回None表示无法处理
            images['depth'] = depth_images if len(depth_images) > 1 else depth_images[0]
        
        # 加载语义分割图像
        if semantic_paths:
            semantic_images = []
            for semantic_path in semantic_paths:
                if semantic_path and Path(semantic_path).exists():
                    semantic_image = Image.open(semantic_path).convert('L')  # 灰度图
                    semantic_image = semantic_image.resize((self.config.image_size, self.config.image_size))
                    semantic_images.append(np.array(semantic_image))
                else:
                    if self.config.verbose:
                        print(f"警告: 语义图像不存在: {semantic_path}")
                    return None  # 如果缺少语义图，返回None表示无法处理
            images['semantic'] = semantic_images if len(semantic_images) > 1 else semantic_images[0]
        
        return images
    
    
    def preprocess_inputs(
        self, 
        images: Dict[str, Union[np.ndarray, List[np.ndarray]]], 
        text_prompt: str
    ) -> Dict[str, torch.Tensor]:
        """预处理输入数据（支持6摄像头全景）"""
        
        # 检查是否是6摄像头模式
        if isinstance(images['rgb'], list) and len(images['rgb']) == 6:
            # 6摄像头全景模式
            if self.processor and hasattr(self.processor, 'create_panorama'):
                # 创建全景图
                rgb_panorama = self.processor.create_panorama(images['rgb'], modality="rgb")
                depth_panorama = None
                semantic_panorama = None
                
                if 'depth' in images and images['depth'] is not None:
                    depth_panorama = self.processor.create_panorama(images['depth'], modality="depth")
                if 'semantic' in images and images['semantic'] is not None:
                    semantic_panorama = self.processor.create_panorama(images['semantic'], modality="semantic")
                
                # 处理全景图
                processed_data = self.processor.process(
                    text=text_prompt,
                    rgb_image=rgb_panorama,
                    depth_image=depth_panorama,
                    semantic_image=semantic_panorama,
                    mode="inference"
                )
            else:
                raise ValueError("处理器不支持6摄像头全景模式")
        else:
            # 单摄像头模式
            processed_data = self.processor.process(
                text=text_prompt,
                rgb_image=images['rgb'],
                depth_image=images.get('depth'),
                semantic_image=images.get('semantic'),
                mode="inference"
            )
        
        # 转换为张量并移动到设备
        for key, value in processed_data.items():
            if isinstance(value, torch.Tensor):
                processed_data[key] = value.unsqueeze(0).to(self.config.device)
        
        return processed_data
    
    def load_tmi_features(self, feature_path: str) -> torch.Tensor:
        """加载预提取的TMI特征"""
        
        if not Path(feature_path).exists():
            raise FileNotFoundError(f"TMI特征文件不存在: {feature_path}")
        
        features = np.load(feature_path)
        return torch.from_numpy(features).float()
    
    def generate_trajectory(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """生成轨迹预测"""
        
        start_time = time.time()
        
        with torch.no_grad():
            # 准备生成参数
            generation_kwargs = {
                'max_new_tokens': self.config.max_new_tokens,
                'temperature': self.config.temperature if self.config.do_sample else None,
                'top_p': self.config.top_p if self.config.do_sample else None,
                'do_sample': self.config.do_sample,
                'num_beams': self.config.num_beams,
                'repetition_penalty': self.config.repetition_penalty,
                'pad_token_id': self.tokenizer.pad_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
            }
            
            # 如果使用TMI特征，添加到输入中
            if self.config.use_tmi_features and self.config.tmi_feature_path:
                tmi_features = self.load_tmi_features(self.config.tmi_feature_path)
                inputs['tmi_features'] = tmi_features.unsqueeze(0).to(self.config.device)
            
            # 获取输入长度
            input_length = inputs['input_ids'].shape[1]
            
            # 生成
            with torch.autocast(device_type='cuda' if 'cuda' in self.config.device else 'cpu', enabled=self.config.torch_dtype=='float16'):
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # 解码生成的部分
            generated_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            
            # 提取轨迹
            trajectory = self._extract_trajectory_from_text(generated_text)
            
            return {
                'generated_text': generated_text,
                'trajectory': trajectory,
                'generation_time': generation_time,
                'input_length': input_length,
                'output_length': len(generated_tokens)
            }
    
    def _extract_trajectory_from_text(self, text: str) -> Optional[np.ndarray]:
        """从生成文本中提取轨迹坐标"""
        
        try:
            import re
            
            # 查找PLANNING标签
            if "<PLANNING>" in text and "</PLANNING>" in text:
                planning_content = text.split("<PLANNING>")[1].split("</PLANNING>")[0]
                
                # 匹配多种坐标格式
                # 格式1: [x, y, heading]
                pattern1 = r'\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]'
                matches = re.findall(pattern1, planning_content)
                
                if matches:
                    trajectory = np.array([[float(x), float(y), float(h)] for x, y, h in matches])
                    return trajectory
                
                # 格式2: [x: 1.23, y: 4.56]
                pattern2 = r'\[x:\s*([-+]?\d*\.?\d+),\s*y:\s*([-+]?\d*\.?\d+)\]'
                matches = re.findall(pattern2, planning_content)
                
                if matches:
                    trajectory = np.array([[float(x), float(y)] for x, y in matches])
                    return trajectory
            
            return None
            
        except Exception as e:
            if self.config.verbose:
                print(f"轨迹提取失败: {e}")
            return None
    
    def _validate_trajectory(self, trajectory: np.ndarray) -> bool:
        """验证轨迹合理性"""
        
        if trajectory is None or len(trajectory) == 0:
            return False
        
        # 检查形状
        if len(trajectory.shape) != 2 or trajectory.shape[1] < 2:
            return False
        
        # 检查数值范围（假设单位是米）
        if np.any(np.abs(trajectory) > 1000):  # 超过1km认为不合理
            return False
        
        # 检查连续性（相邻点距离不应过大）
        if len(trajectory) > 1:
            diffs = np.diff(trajectory, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            if np.any(distances > 50):  # 相邻点距离超过50m认为不合理
                return False
        
        return True
    
    def single_inference(
        self,
        rgb_paths: Union[str, List[str]],
        depth_paths: Optional[Union[str, List[str]]] = None,
        semantic_paths: Optional[Union[str, List[str]]] = None,
        text_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """单样本推理（支持6摄像头）"""
        
        if not self.model:
            self.setup_model()
        
        if text_prompt is None:
            text_prompt = self.config.text_prompt
        
        # 加载图像
        images = self.load_images(rgb_paths, depth_paths, semantic_paths)
        
        if images is None:
            if self.config.verbose:
                print("警告: 缺少必要的模态数据，跳过该样本")
            return None
        
        # 预处理
        inputs = self.preprocess_inputs(images, text_prompt)
        
        # 生成
        results = self.generate_trajectory(inputs)
        
        # 添加输入信息
        results.update({
            'rgb_paths': rgb_paths if isinstance(rgb_paths, list) else [rgb_paths],
            'depth_paths': depth_paths if isinstance(depth_paths, list) else [depth_paths] if depth_paths else None,
            'semantic_paths': semantic_paths if isinstance(semantic_paths, list) else [semantic_paths] if semantic_paths else None,
            'text_prompt': text_prompt,
            'images': images
        })
        
        return results
    
    def batch_inference(self, input_dir: str) -> List[Dict[str, Any]]:
        """批量推理"""
        
        if not self.model:
            self.setup_model()
        
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")
        
        # 查找RGB图像文件
        rgb_files = list(input_path.glob("*_rgb.*")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
        
        if not rgb_files:
            raise FileNotFoundError(f"在 {input_dir} 中未找到图像文件")
        
        results = []
        
        for rgb_file in rgb_files:
            try:
                # 构建对应的深度和语义文件路径
                base_name = rgb_file.stem.replace("_rgb", "")
                depth_file = input_path / f"{base_name}_depth{rgb_file.suffix}"
                semantic_file = input_path / f"{base_name}_semantic{rgb_file.suffix}"
                
                # 推理
                result = self.single_inference(
                    rgb_path=str(rgb_file),
                    depth_path=str(depth_file) if depth_file.exists() else None,
                    semantic_path=str(semantic_file) if semantic_file.exists() else None
                )
                
                results.append(result)
                
                if self.config.verbose:
                    print(f"处理完成: {rgb_file.name}")
                
            except Exception as e:
                if self.config.verbose:
                    print(f"处理失败 {rgb_file.name}: {e}")
                continue
        
        return results
    
    def save_results(self, results: Union[Dict, List[Dict]], output_name: str = "inference_results"):
        """保存推理结果"""
        
        if isinstance(results, dict):
            results = [results]
        
        # 保存JSON结果
        json_results = []
        for i, result in enumerate(results):
            json_result = {
                'index': i,
                'rgb_path': result.get('rgb_path'),
                'depth_path': result.get('depth_path'),
                'semantic_path': result.get('semantic_path'),
                'text_prompt': result.get('text_prompt'),
                'generated_text': result.get('generated_text'),
                'trajectory': result.get('trajectory').tolist() if result.get('trajectory') is not None else None,
                'generation_time': result.get('generation_time'),
                'input_length': result.get('input_length'),
                'output_length': result.get('output_length')
            }
            json_results.append(json_result)
        
        json_file = Path(self.config.output_dir) / f"{output_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        if self.config.verbose:
            print(f"结果已保存到: {json_file}")
        
        # 保存可视化结果
        if self.config.save_visualization and self.visualizer:
            self.create_visualizations(results, output_name)
        
        return json_file
    
    def create_visualizations(self, results: List[Dict], output_name: str):
        """创建可视化结果"""
        
        if not plt:
            if self.config.verbose:
                print("matplotlib不可用，跳过可视化")
            return
        
        try:
            # 为每个结果创建可视化
            for i, result in enumerate(results):
                if result.get('trajectory') is None:
                    continue
                
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # 1. RGB图像
                if 'images' in result and 'rgb' in result['images']:
                    axes[0].imshow(result['images']['rgb'])
                    axes[0].set_title('RGB Image')
                    axes[0].axis('off')
                
                # 2. 深度图像
                if 'images' in result and 'depth' in result['images']:
                    axes[1].imshow(result['images']['depth'], cmap='viridis')
                    axes[1].set_title('Depth Image')
                    axes[1].axis('off')
                
                # 3. 语义分割图像
                if 'images' in result and 'semantic' in result['images']:
                    axes[2].imshow(result['images']['semantic'], cmap='tab20')
                    axes[2].set_title('Semantic Image')
                    axes[2].axis('off')
                
                # 4. 预测轨迹
                trajectory = result['trajectory']
                axes[3].plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, marker='o')
                axes[3].plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=8, label='起点')
                axes[3].plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=8, label='终点')
                axes[3].set_xlabel('X (meters)')
                axes[3].set_ylabel('Y (meters)')
                axes[3].set_title('Predicted Trajectory')
                axes[3].legend()
                axes[3].grid(True, alpha=0.3)
                axes[3].axis('equal')
                
                plt.tight_layout()
                
                # 保存图像
                output_file = Path(self.config.output_dir) / f"{output_name}_sample_{i}.png"
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
                if self.config.verbose:
                    print(f"可视化已保存: {output_file}")
        
        except Exception as e:
            if self.config.verbose:
                print(f"创建可视化时出错: {e}")
    
    def interactive_mode(self):
        """交互式推理模式"""
        
        if not self.model:
            self.setup_model()
        
        print("\n=== 三模态VLM交互式推理 ===")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'help' 查看帮助信息")
        
        while True:
            try:
                print("\n" + "="*50)
                
                # 获取用户输入
                rgb_path = input("RGB图像路径: ").strip()
                
                if rgb_path.lower() in ['quit', 'exit']:
                    print("退出交互模式")
                    break
                
                if rgb_path.lower() == 'help':
                    self._print_help()
                    continue
                
                if not rgb_path or not Path(rgb_path).exists():
                    print("❌ 无效的RGB图像路径")
                    continue
                
                # 可选的深度和语义图像
                depth_path = input("深度图像路径 (可选): ").strip()
                if not depth_path or not Path(depth_path).exists():
                    depth_path = None
                
                semantic_path = input("语义图像路径 (可选): ").strip()
                if not semantic_path or not Path(semantic_path).exists():
                    semantic_path = None
                
                # 自定义提示词
                custom_prompt = input(f"自定义提示词 (默认: {self.config.text_prompt}): ").strip()
                text_prompt = custom_prompt if custom_prompt else self.config.text_prompt
                
                # 执行推理
                print("\n🚀 开始推理...")
                result = self.single_inference(rgb_path, depth_path, semantic_path, text_prompt)
                
                # 显示结果
                self._display_result(result)
                
                # 保存结果
                save_result = input("\n是否保存结果? (y/n): ").strip().lower()
                if save_result == 'y':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.save_results(result, f"interactive_{timestamp}")
                
            except KeyboardInterrupt:
                print("\n\n用户中断，退出交互模式")
                break
            except Exception as e:
                print(f"\n❌ 推理失败: {e}")
                continue
    
    def _print_help(self):
        """打印帮助信息"""
        print("\n=== 帮助信息 ===")
        print("1. RGB图像路径: 必需，支持jpg、png等格式")
        print("2. 深度图像路径: 可选，如不提供将自动生成模拟深度图")
        print("3. 语义图像路径: 可选，如不提供将自动生成模拟语义图")
        print("4. 自定义提示词: 可选，用于指导模型生成特定类型的轨迹")
        print("5. 支持的命令:")
        print("   - 'help': 显示此帮助信息")
        print("   - 'quit' 或 'exit': 退出交互模式")
        print("================")
    
    def _display_result(self, result: Dict[str, Any]):
        """显示推理结果"""
        print("\n📊 推理结果:")
        print(f"  生成时间: {result['generation_time']:.3f} 秒")
        print(f"  输入长度: {result['input_length']} tokens")
        print(f"  输出长度: {result['output_length']} tokens")
        
        print(f"\n💬 生成文本:")
        print(f"  {result['generated_text']}")
        
        if result['trajectory'] is not None:
            trajectory = result['trajectory']
            print(f"\n🛣️  预测轨迹 ({len(trajectory)} 个点):")
            for i, point in enumerate(trajectory[:5]):  # 只显示前5个点
                print(f"    点{i+1}: x={point[0]:.2f}, y={point[1]:.2f}")
            if len(trajectory) > 5:
                print(f"    ... (还有 {len(trajectory)-5} 个点)")
        else:
            print("\n❌ 未找到有效轨迹")


def parse_args():
    """解析命令行参数"""
    
    parser = argparse.ArgumentParser(description="三模态VLM推理脚本")
    
    # 模型配置
    parser.add_argument("--model_path", type=str, required=True,
                       help="训练好的模型路径")
    parser.add_argument("--model_config_path", type=str, default=None,
                       help="模型配置文件路径")
    
    # 推理模式
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--rgb_image", type=str,
                      help="单张RGB图像路径")
    group.add_argument("--batch_input_dir", type=str,
                      help="批量输入目录")
    group.add_argument("--interactive", action="store_true",
                      help="交互式推理模式")
    
    # TMI特征模式
    parser.add_argument("--use_tmi_features", action="store_true",
                       help="使用TMI特征模式（用于LLaMA Factory训练的模型）")
    parser.add_argument("--tmi_feature_path", type=str, default=None,
                       help="TMI特征文件路径（.npy格式）")
    parser.add_argument("--tmi_checkpoint", type=str, default=None,
                       help="TMI模块checkpoint路径（用于在线提取特征）")
    
    # 输入文件（单样本模式）
    parser.add_argument("--depth_image", type=str, default=None,
                       help="深度图像路径（可选）")
    parser.add_argument("--semantic_image", type=str, default=None,
                       help="语义图像路径（可选）")
    parser.add_argument("--text_prompt", type=str,
                       default="基于三模态感知信息，预测车辆的未来轨迹。",
                       help="文本提示词")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, default="./inference_results",
                       help="输出目录")
    
    # 生成配置
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="生成温度")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="nucleus采样参数")
    parser.add_argument("--do_sample", action="store_true",
                       help="是否使用采样")
    parser.add_argument("--num_beams", type=int, default=1,
                       help="束搜索beam数量")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                       help="重复惩罚")
    
    # 处理配置
    parser.add_argument("--max_length", type=int, default=2048,
                       help="最大序列长度")
    parser.add_argument("--image_size", type=int, default=392,
                       help="图像尺寸")
    
    # 可视化配置
    parser.add_argument("--save_visualization", action="store_true",
                       help="是否保存可视化结果")
    parser.add_argument("--show_attention", action="store_true",
                       help="是否显示注意力权重")
    
    # 性能配置
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="计算设备")
    parser.add_argument("--torch_dtype", type=str, default="float16",
                       choices=["float16", "float32"],
                       help="模型数据类型")
    parser.add_argument("--use_compile", action="store_true",
                       help="是否使用PyTorch 2.0编译")
    
    # 其他配置
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    
    return parser.parse_args()


def main():
    """主函数"""
    
    args = parse_args()
    
    # 设置设备
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # 创建推理配置
    inference_config = InferenceConfig(
        model_path=args.model_path,
        model_config_path=args.model_config_path,
        use_tmi_features=args.use_tmi_features,
        tmi_feature_path=args.tmi_feature_path,
        tmi_checkpoint=args.tmi_checkpoint,
        rgb_image_path=args.rgb_image,
        depth_image_path=args.depth_image,
        semantic_image_path=args.semantic_image,
        text_prompt=args.text_prompt,
        batch_input_dir=args.batch_input_dir,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        max_length=args.max_length,
        image_size=args.image_size,
        save_visualization=args.save_visualization,
        show_attention=args.show_attention,
        interactive_mode=args.interactive,
        device=device,
        torch_dtype=args.torch_dtype,
        use_compile=args.use_compile,
        use_flash_attention=getattr(args, 'use_flash_attention', True),
        use_better_transformer=getattr(args, 'use_better_transformer', True),
        compute_metrics=getattr(args, 'compute_metrics', False),
        ground_truth_path=getattr(args, 'ground_truth_path', None),
        profile=getattr(args, 'profile', False),
        seed=args.seed,
        verbose=args.verbose
    )
    
    # 打印配置
    if args.verbose:
        print("=== 推理配置 ===")
        for key, value in inference_config.__dict__.items():
            print(f"{key}: {value}")
        print("===============")
    
    try:
        # 创建推理器
        inference_engine = TriModalInference(inference_config)
        
        if args.interactive:
            # 交互式模式
            inference_engine.interactive_mode()
        elif args.batch_input_dir:
            # 批量推理
            print("开始批量推理...")
            results = inference_engine.batch_inference(args.batch_input_dir)
            inference_engine.save_results(results, "batch_inference")
            print(f"批量推理完成，处理了 {len(results)} 个样本")
        else:
            # 单样本推理
            print("开始单样本推理...")
            result = inference_engine.single_inference(
                rgb_path=args.rgb_image,
                depth_path=args.depth_image,
                semantic_path=args.semantic_image,
                text_prompt=args.text_prompt
            )
            inference_engine.save_results(result, "single_inference")
            print("单样本推理完成")
        
        print("\n✓ 推理完成!")
        return 0
        
    except KeyboardInterrupt:
        print("\n推理被用户中断")
        return 1
    except Exception as e:
        print(f"\n✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())