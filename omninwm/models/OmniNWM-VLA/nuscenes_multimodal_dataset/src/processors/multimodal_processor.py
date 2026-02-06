#!/usr/bin/env python3
"""
多模态处理器

该模块负责处理多种模态的数据，包括：
- RGB图像处理
- 深度图像处理
- 语义分割数据处理
- 雷达点云处理
- 多模态数据融合
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from PIL import Image
import json


@dataclass
class ProcessingConfig:
    """处理配置"""
    image_size: Tuple[int, int] = (800, 600)
    normalize: bool = True
    mean: List[float] = None
    std: List[float] = None
    device: str = "cpu"
    
    def __post_init__(self):
        if self.mean is None:
            self.mean = [0.485, 0.456, 0.406]  # ImageNet标准
        if self.std is None:
            self.std = [0.229, 0.224, 0.225]   # ImageNet标准


class BaseModalityProcessor(ABC):
    """基础模态处理器抽象类"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.device = torch.device(config.device)
    
    @abstractmethod
    def process(self, data: Any) -> torch.Tensor:
        """处理数据"""
        pass
    
    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        """转换为tensor"""
        if len(data.shape) == 3:  # HWC -> CHW
            data = data.transpose(2, 0, 1)
        return torch.from_numpy(data).float().to(self.device)


class RGBImageProcessor(BaseModalityProcessor):
    """RGB图像处理器"""
    
    def process(self, image_path: str) -> torch.Tensor:
        """
        处理RGB图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            处理后的tensor (C, H, W)
        """
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # 调整大小
            if image_array.shape[:2] != self.config.image_size:
                image_array = cv2.resize(image_array, self.config.image_size)
            
            # 归一化到[0, 1]
            image_array = image_array.astype(np.float32) / 255.0
            
            # 标准化
            if self.config.normalize:
                mean = np.array(self.config.mean).reshape(1, 1, 3)
                std = np.array(self.config.std).reshape(1, 1, 3)
                image_array = (image_array - mean) / std
            
            # 转换为tensor
            image_tensor = self._to_tensor(image_array)
            
            return image_tensor
            
        except Exception as e:
            logging.error(f"RGB图像处理失败 {image_path}: {e}")
            # 返回零tensor作为备选
            return torch.zeros(3, self.config.image_size[1], self.config.image_size[0], 
                             device=self.device)
    
    def process_batch(self, image_paths: List[str]) -> torch.Tensor:
        """批量处理RGB图像"""
        batch_images = []
        for path in image_paths:
            image_tensor = self.process(path)
            batch_images.append(image_tensor)
        
        return torch.stack(batch_images, dim=0)


class DepthImageProcessor(BaseModalityProcessor):
    """深度图像处理器"""
    
    def __init__(self, config: ProcessingConfig, max_depth: float = 100.0):
        super().__init__(config)
        self.max_depth = max_depth
    
    def process(self, depth_path: str) -> torch.Tensor:
        """
        处理深度图像
        
        Args:
            depth_path: 深度图像文件路径
            
        Returns:
            处理后的tensor (1, H, W)
        """
        try:
            # 加载深度图像
            if depth_path.endswith('.png'):
                depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            else:
                depth_image = np.load(depth_path)
            
            # 调整大小
            if depth_image.shape != self.config.image_size:
                depth_image = cv2.resize(depth_image, self.config.image_size)
            
            # 归一化深度值
            depth_image = depth_image.astype(np.float32)
            depth_image = np.clip(depth_image, 0, self.max_depth) / self.max_depth
            
            # 添加通道维度
            if len(depth_image.shape) == 2:
                depth_image = depth_image[:, :, np.newaxis]
            
            # 转换为tensor
            depth_tensor = self._to_tensor(depth_image)
            
            return depth_tensor
            
        except Exception as e:
            logging.error(f"深度图像处理失败 {depth_path}: {e}")
            return torch.zeros(1, self.config.image_size[1], self.config.image_size[0], 
                             device=self.device)


class SemanticSegmentationProcessor(BaseModalityProcessor):
    """语义分割处理器"""
    
    def __init__(self, config: ProcessingConfig, num_classes: int = 32):
        super().__init__(config)
        self.num_classes = num_classes
    
    def process(self, semantic_path: str) -> torch.Tensor:
        """
        处理语义分割数据
        
        Args:
            semantic_path: 语义分割文件路径
            
        Returns:
            处理后的tensor (num_classes, H, W)
        """
        try:
            # 加载语义分割掩码
            if semantic_path.endswith('.png'):
                semantic_mask = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)
            else:
                semantic_mask = np.load(semantic_path)
            
            # 调整大小
            if semantic_mask.shape != self.config.image_size:
                semantic_mask = cv2.resize(semantic_mask, self.config.image_size, 
                                         interpolation=cv2.INTER_NEAREST)
            
            # 转换为one-hot编码
            semantic_onehot = np.zeros((self.num_classes, *self.config.image_size))
            for class_id in range(self.num_classes):
                semantic_onehot[class_id] = (semantic_mask == class_id).astype(np.float32)
            
            # 转换为tensor
            semantic_tensor = torch.from_numpy(semantic_onehot).float().to(self.device)
            
            return semantic_tensor
            
        except Exception as e:
            logging.error(f"语义分割处理失败 {semantic_path}: {e}")
            return torch.zeros(self.num_classes, self.config.image_size[1], 
                             self.config.image_size[0], device=self.device)


class TrajectoryProcessor(BaseModalityProcessor):
    """轨迹数据处理器"""
    
    def __init__(self, config: ProcessingConfig, sequence_length: int = 36):
        super().__init__(config)
        self.sequence_length = sequence_length  # 12Hz × 3秒 = 36
    
    def process(self, trajectory_data: List[Dict[str, float]]) -> torch.Tensor:
        """
        处理轨迹数据
        
        Args:
            trajectory_data: 轨迹点列表，每个点包含x, y, heading
            
        Returns:
            处理后的tensor (sequence_length, 3)
        """
        try:
            # 确保有正确的长度
            if len(trajectory_data) != self.sequence_length:
                logging.warning(f"轨迹长度不匹配: 期望{self.sequence_length}, 实际{len(trajectory_data)}")
                
                # 填充或截断
                if len(trajectory_data) < self.sequence_length:
                    # 填充最后一个点
                    last_point = trajectory_data[-1] if trajectory_data else {"x": 0, "y": 0, "heading": 0}
                    while len(trajectory_data) < self.sequence_length:
                        trajectory_data.append(last_point.copy())
                else:
                    # 截断
                    trajectory_data = trajectory_data[:self.sequence_length]
            
            # 提取坐标
            coordinates = []
            for point in trajectory_data:
                x = point.get('x', 0.0)
                y = point.get('y', 0.0)
                heading = point.get('heading', 0.0)
                coordinates.append([x, y, heading])
            
            # 转换为tensor
            trajectory_tensor = torch.tensor(coordinates, dtype=torch.float32, device=self.device)
            
            return trajectory_tensor
            
        except Exception as e:
            logging.error(f"轨迹处理失败: {e}")
            return torch.zeros(self.sequence_length, 3, device=self.device)
    
    def normalize_trajectory(self, trajectory_tensor: torch.Tensor, 
                           position_scale: float = 100.0) -> torch.Tensor:
        """
        归一化轨迹数据
        
        Args:
            trajectory_tensor: 轨迹tensor (sequence_length, 3)
            position_scale: 位置缩放因子
            
        Returns:
            归一化后的轨迹
        """
        normalized = trajectory_tensor.clone()
        
        # 归一化位置 (x, y)
        normalized[:, :2] = normalized[:, :2] / position_scale
        
        # 归一化角度到[-1, 1]范围
        normalized[:, 2] = normalized[:, 2] / np.pi
        
        return normalized


class MultiModalProcessor:
    """多模态处理器主类"""
    
    def __init__(self, config: Dict[str, Any]):
        # 解析配置
        if isinstance(config, dict):
            self.config = ProcessingConfig(
                image_size=tuple(config.get('processing', {}).get('image_size', [384, 384])),
                normalize=config.get('processing', {}).get('normalize', True),
                device=config.get('processing', {}).get('device', 'cpu')
            )
        else:
            self.config = config
        
        # 初始化各个模态处理器
        self.rgb_processor = RGBImageProcessor(self.config)
        self.depth_processor = DepthImageProcessor(self.config)
        self.semantic_processor = SemanticSegmentationProcessor(self.config)
        self.trajectory_processor = TrajectoryProcessor(self.config)
        
        # 摄像头通道
        self.camera_channels = [
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
        ]
        
        # 处理器配置
        self.depth_processor_instance = None
        self.semantic_processor_instance = None
        self._initialize_processors(config)
    
    def process_sample(self, sample_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        处理完整样本
        
        Args:
            sample_data: 样本数据字典
            
        Returns:
            处理后的多模态tensor字典
        """
        processed_data = {}
        
        try:
            # 处理视觉输入
            if "visual_inputs" in sample_data:
                visual_data = self._process_visual_inputs(sample_data["visual_inputs"])
                processed_data.update(visual_data)
            
            # 处理轨迹数据
            if "ground_truth" in sample_data and "future_trajectory" in sample_data["ground_truth"]:
                trajectory = sample_data["ground_truth"]["future_trajectory"]
                trajectory_tensor = self.trajectory_processor.process(trajectory)
                processed_data["trajectory"] = trajectory_tensor
            
            # 处理其他模态数据
            for modality in ["depth", "semantic", "radar"]:
                if modality in sample_data:
                    processed_data[modality] = self._process_modality(modality, sample_data[modality])
            
            return processed_data
            
        except Exception as e:
            logging.error(f"样本处理失败: {e}")
            return {}
    
    def _process_visual_inputs(self, visual_inputs: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """处理视觉输入"""
        visual_data = {}
        
        # 处理每个摄像头的图像
        for camera in self.camera_channels:
            if camera in visual_inputs:
                image_path = visual_inputs[camera]
                if os.path.exists(image_path):
                    rgb_tensor = self.rgb_processor.process(image_path)
                    visual_data[f"{camera.lower()}_rgb"] = rgb_tensor
                else:
                    logging.warning(f"图像文件不存在: {image_path}")
        
        # 如果有多个摄像头，创建拼接视图
        if len(visual_data) > 1:
            all_images = list(visual_data.values())
            concatenated = torch.cat(all_images, dim=0)  # 在通道维度拼接
            visual_data["multi_view"] = concatenated
        
        return visual_data
    
    def _initialize_processors(self, config: Dict[str, Any]):
        """初始化深度和语义处理器"""
        try:
            # 初始化深度处理器
            from .depth_processor import DepthProcessor
            depth_config = config.get('depth', {})
            depth_model = depth_config.get('model_name', 'ZoeDepth')
            depth_model_path = depth_config.get('model_path', None)
            use_local_depth = depth_config.get('use_local_weights', False)
            device = depth_config.get('device', 'auto')
            
            self.depth_processor_instance = DepthProcessor(
                model_name=depth_model, 
                device=device,
                model_path=depth_model_path,
                use_local_weights=use_local_depth
            )
            
            # 初始化语义处理器
            from .semantic_processor import SemanticProcessor
            semantic_config = config.get('semantic', {})
            semantic_model = semantic_config.get('model_name', 'SegFormer')
            semantic_model_path = semantic_config.get('model_path', None)
            use_local_semantic = semantic_config.get('use_local_weights', False)
            device = semantic_config.get('device', 'auto')
            
            self.semantic_processor_instance = SemanticProcessor(
                model_name=semantic_model, 
                device=device,
                model_path=semantic_model_path,
                use_local_weights=use_local_semantic
            )
            
            # 初始化降级处理器
            from .modality_fallback import ModalityFallbackHandler
            self.fallback_handler = ModalityFallbackHandler(config)
            
            logging.info("深度和语义处理器初始化成功")
            
        except Exception as e:
            logging.warning(f"初始化深度/语义处理器失败: {e}")
            # 创建降级处理器作为备用
            try:
                from .modality_fallback import ModalityFallbackHandler
                self.fallback_handler = ModalityFallbackHandler(config)
                logging.info("降级处理器创建成功")
            except Exception as fallback_error:
                logging.error(f"降级处理器创建也失败: {fallback_error}")
                self.fallback_handler = None
    
    def process_all_cameras(self, sample_token: str, camera_data: Dict[str, str], 
                          output_dir: str) -> Dict[str, Dict[str, str]]:
        """
        处理所有摄像头数据，生成RGB、深度图和语义分割图
        
        Args:
            sample_token: 样本令牌
            camera_data: 摄像头数据路径字典
            output_dir: 输出目录
            
        Returns:
            包含RGB、深度图和语义分割图相对路径的字典
        """
        try:
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 创建图像输出子目录
            rgb_dir = os.path.join(output_dir, 'images')
            depth_dir = os.path.join(output_dir, 'depth')
            semantic_dir = os.path.join(output_dir, 'semantic')
            
            os.makedirs(rgb_dir, exist_ok=True)
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(semantic_dir, exist_ok=True)
            
            visual_inputs = {
                'images': [],
                'depth_maps': [],
                'semantic_maps': []
            }
            
            # 处理每个摄像头
            for camera_channel in self.camera_channels:
                if camera_channel in camera_data:
                    rgb_path = camera_data[camera_channel]
                    
                    # 生成输出文件名
                    camera_name = camera_channel.lower()
                    rgb_filename = f"{sample_token}_{camera_name}.jpg"
                    depth_filename = f"{sample_token}_{camera_name}_depth.png"
                    semantic_filename = f"{sample_token}_{camera_name}_semantic.png"
                    
                    # 完整输出路径
                    rgb_output_path = os.path.join(rgb_dir, rgb_filename)
                    depth_output_path = os.path.join(depth_dir, depth_filename)
                    semantic_output_path = os.path.join(semantic_dir, semantic_filename)
                    
                    try:
                        # 复制RGB图像到输出目录
                        import shutil
                        if os.path.exists(rgb_path):
                            shutil.copy2(rgb_path, rgb_output_path)
                            
                            # 生成深度图
                            if self.depth_processor_instance:
                                try:
                                    depth_array = self.depth_processor_instance.generate_depth_map(rgb_path)
                                    self.depth_processor_instance.save_depth_map(depth_array, depth_output_path)
                                except Exception as e:
                                    logging.warning(f"深度图生成失败 {camera_channel}: {e}")
                                    # 使用降级处理器
                                    if self.fallback_handler:
                                        success = self.fallback_handler.handle_depth_failure(
                                            rgb_path, depth_output_path, strategy='auto'
                                        )
                                        if not success:
                                            self._create_placeholder_depth(depth_output_path)
                                    else:
                                        self._create_placeholder_depth(depth_output_path)
                            else:
                                # 没有深度处理器，直接使用降级处理器
                                if self.fallback_handler:
                                    success = self.fallback_handler.handle_depth_failure(
                                        rgb_path, depth_output_path, strategy='rgb_estimation'
                                    )
                                    if not success:
                                        self._create_placeholder_depth(depth_output_path)
                                else:
                                    self._create_placeholder_depth(depth_output_path)
                            
                            # 生成语义分割图
                            if self.semantic_processor_instance:
                                try:
                                    semantic_array = self.semantic_processor_instance.generate_semantic_map(rgb_path)
                                    self.semantic_processor_instance.save_semantic_map(semantic_array, semantic_output_path)
                                except Exception as e:
                                    logging.warning(f"语义分割图生成失败 {camera_channel}: {e}")
                                    # 使用降级处理器
                                    if self.fallback_handler:
                                        success = self.fallback_handler.handle_semantic_failure(
                                            rgb_path, semantic_output_path, strategy='auto'
                                        )
                                        if not success:
                                            self._create_placeholder_semantic(semantic_output_path)
                                    else:
                                        self._create_placeholder_semantic(semantic_output_path)
                            else:
                                # 没有语义处理器，直接使用降级处理器
                                if self.fallback_handler:
                                    success = self.fallback_handler.handle_semantic_failure(
                                        rgb_path, semantic_output_path, strategy='color_clustering'
                                    )
                                    if not success:
                                        self._create_placeholder_semantic(semantic_output_path)
                                else:
                                    self._create_placeholder_semantic(semantic_output_path)
                            
                            # 生成相对路径（相对于输出目录的父目录）
                            base_output_dir = os.path.dirname(output_dir)
                            rgb_relative = os.path.relpath(rgb_output_path, base_output_dir)
                            depth_relative = os.path.relpath(depth_output_path, base_output_dir)
                            semantic_relative = os.path.relpath(semantic_output_path, base_output_dir)
                            
                            # 统一路径分隔符为Unix风格
                            rgb_relative = rgb_relative.replace('\\', '/')
                            depth_relative = depth_relative.replace('\\', '/')
                            semantic_relative = semantic_relative.replace('\\', '/')
                            
                            visual_inputs['images'].append(rgb_relative)
                            visual_inputs['depth_maps'].append(depth_relative)
                            visual_inputs['semantic_maps'].append(semantic_relative)
                            
                        else:
                            logging.warning(f"RGB图像文件不存在: {rgb_path}")
                            
                    except Exception as e:
                        logging.error(f"处理摄像头{camera_channel}失败: {e}")
                        continue
            
            # 验证是否成功处理了足够的摄像头
            if len(visual_inputs['images']) < len(self.camera_channels):
                logging.warning(f"只成功处理了{len(visual_inputs['images'])}个摄像头，期望{len(self.camera_channels)}个")
            
            return visual_inputs
            
        except Exception as e:
            logging.error(f"处理所有摄像头失败: {e}")
            return {}
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = {
            'cameras_processed': 0,
            'cameras_successful': 0,
            'fallback_statistics': {}
        }
        
        # 获取降级处理统计
        if hasattr(self, 'fallback_handler') and self.fallback_handler:
            stats['fallback_statistics'] = self.fallback_handler.get_fallback_statistics()
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        if hasattr(self, 'fallback_handler') and self.fallback_handler:
            self.fallback_handler.reset_statistics()
    
    def _create_placeholder_depth(self, output_path: str):
        """创建占位深度图"""
        try:
            import cv2
            placeholder = np.zeros((384, 384), dtype=np.float32)
            # 保存为16位PNG
            depth_normalized = (placeholder * 65535).astype(np.uint16)
            cv2.imwrite(output_path, depth_normalized)
        except Exception as e:
            logging.error(f"创建占位深度图失败: {e}")
    
    def _create_placeholder_semantic(self, output_path: str):
        """创建占位语义分割图"""
        try:
            from PIL import Image
            placeholder = np.zeros((384, 384), dtype=np.uint8)
            semantic_image = Image.fromarray(placeholder, mode='L')
            semantic_image.save(output_path)
        except Exception as e:
            logging.error(f"创建占位语义图失败: {e}")
    
    def _process_modality(self, modality: str, data: Any) -> torch.Tensor:
        """处理指定模态数据"""
        if modality == "depth":
            return self.depth_processor.process(data)
        elif modality == "semantic":
            return self.semantic_processor.process(data)
        else:
            logging.warning(f"未知模态类型: {modality}")
            return torch.tensor([])
    
    def process_batch(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        批量处理样本
        
        Args:
            batch_samples: 样本列表
            
        Returns:
            批量处理后的tensor字典
        """
        batch_data = {}
        
        # 处理每个样本
        processed_samples = []
        for sample in batch_samples:
            processed_sample = self.process_sample(sample)
            processed_samples.append(processed_sample)
        
        # 合并为批量tensor
        if processed_samples:
            # 获取所有可能的键
            all_keys = set()
            for sample in processed_samples:
                all_keys.update(sample.keys())
            
            # 为每个键创建批量tensor
            for key in all_keys:
                tensors = []
                for sample in processed_samples:
                    if key in sample:
                        tensors.append(sample[key])
                
                if tensors:
                    try:
                        batch_data[key] = torch.stack(tensors, dim=0)
                    except RuntimeError as e:
                        logging.warning(f"无法堆叠键 {key}: {e}")
        
        return batch_data
    
    def create_data_collator(self):
        """创建数据整理器"""
        def collate_fn(batch):
            """数据整理函数"""
            return self.process_batch(batch)
        
        return collate_fn
    
    def get_feature_dimensions(self) -> Dict[str, Tuple[int, ...]]:
        """获取特征维度信息"""
        dimensions = {}
        
        # RGB图像维度
        for camera in self.camera_channels:
            key = f"{camera.lower()}_rgb"
            dimensions[key] = (3, self.config.image_size[1], self.config.image_size[0])
        
        # 多视图维度
        dimensions["multi_view"] = (18, self.config.image_size[1], self.config.image_size[0])  # 6*3通道
        
        # 轨迹维度
        dimensions["trajectory"] = (36, 3)  # 12Hz × 3秒, (x, y, heading)
        
        # 深度维度
        dimensions["depth"] = (1, self.config.image_size[1], self.config.image_size[0])
        
        # 语义分割维度
        dimensions["semantic"] = (32, self.config.image_size[1], self.config.image_size[0])
        
        return dimensions
    
    def validate_sample(self, sample_data: Dict[str, Any], 
                       comprehensive: bool = False,
                       base_path: Optional[str] = None) -> Dict[str, Any]:
        """
        验证样本数据完整性
        
        Args:
            sample_data: 样本数据
            comprehensive: 是否使用全面验证
            base_path: 图像文件基础路径
            
        Returns:
            验证结果字典
        """
        if comprehensive:
            # 使用数据验证器进行全面验证
            try:
                from ..utils.data_validator import create_data_validator
                validator = create_data_validator({
                    'check_file_existence': True,
                    'check_image_integrity': True,
                    'check_trajectory_format': True,
                    'check_sharegpt_format': True
                })
                return validator.validate_sample(sample_data, base_path)
            except ImportError:
                logging.warning("无法导入数据验证器，使用基础验证")
        
        # 基础验证逻辑（向后兼容）
        validation_results = {}
        
        # 验证视觉输入
        if "visual_inputs" in sample_data:
            visual_inputs = sample_data["visual_inputs"]
            validation_results["has_visual"] = len(visual_inputs) > 0
            validation_results["all_cameras"] = len(visual_inputs) == len(self.camera_channels)
            
            # 验证文件存在性
            missing_files = []
            for camera, path in visual_inputs.items():
                # 解析完整路径
                if base_path and not os.path.isabs(path):
                    full_path = os.path.join(base_path, path)
                else:
                    full_path = path
                
                if not os.path.exists(full_path):
                    missing_files.append(path)
            validation_results["missing_visual_files"] = missing_files
            validation_results["visual_files_exist"] = len(missing_files) == 0
        else:
            validation_results["has_visual"] = False
        
        # 验证轨迹数据
        if "ground_truth" in sample_data and "future_trajectory" in sample_data["ground_truth"]:
            trajectory = sample_data["ground_truth"]["future_trajectory"]
            validation_results["has_trajectory"] = True
            validation_results["trajectory_length_correct"] = len(trajectory) == 36
            
            # 验证轨迹数据格式
            valid_format = True
            for point in trajectory:
                if not all(key in point for key in ["x", "y", "heading"]):
                    valid_format = False
                    break
            validation_results["trajectory_format_valid"] = valid_format
        else:
            validation_results["has_trajectory"] = False
        
        # 整体验证
        validation_results["sample_valid"] = (
            validation_results.get("has_visual", False) and
            validation_results.get("has_trajectory", False) and
            validation_results.get("visual_files_exist", False) and
            validation_results.get("trajectory_format_valid", False)
        )
        
        return validation_results
    
    def validate_sharegpt_dataset(self, dataset_path: str, 
                                 base_path: Optional[str] = None,
                                 output_report: Optional[str] = None) -> Dict[str, Any]:
        """
        验证ShareGPT格式数据集
        
        Args:
            dataset_path: 数据集JSON文件路径
            base_path: 图像文件基础路径
            output_report: 验证报告输出路径
            
        Returns:
            验证结果统计
        """
        try:
            from ..utils.data_validator import create_data_validator
            
            validator = create_data_validator({
                'check_file_existence': True,
                'check_image_integrity': True,
                'check_trajectory_format': True,
                'check_sharegpt_format': True
            })
            
            # 执行完整验证
            validation_result = validator.validate_dataset(dataset_path, base_path)
            
            # 生成验证报告
            if output_report:
                validator.generate_validation_report(validation_result, output_report)
            
            return validation_result
            
        except ImportError:
            logging.error("无法导入数据验证器，请确保utils/data_validator.py存在")
            return {
                'error': '数据验证器不可用',
                'is_valid': False
            }
        except Exception as e:
            logging.error(f"数据集验证失败: {e}")
            return {
                'error': str(e),
                'is_valid': False
            }


def create_multimodal_processor(config_dict: Dict[str, Any] = None) -> MultiModalProcessor:
    """
    创建多模态处理器
    
    Args:
        config_dict: 配置字典
        
    Returns:
        多模态处理器实例
    """
    if config_dict is None:
        config_dict = {}
    
    config = ProcessingConfig(**config_dict)
    return MultiModalProcessor(config)


def process_nuscenes_sample(sample_data: Dict[str, Any], 
                           processor: MultiModalProcessor = None) -> Dict[str, torch.Tensor]:
    """
    处理nuScenes样本的便利函数
    
    Args:
        sample_data: nuScenes样本数据
        processor: 多模态处理器（可选）
        
    Returns:
        处理后的tensor字典
    """
    if processor is None:
        processor = create_multimodal_processor()
    
    return processor.process_sample(sample_data)