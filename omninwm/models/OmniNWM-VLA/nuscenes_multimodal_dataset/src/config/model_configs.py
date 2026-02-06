#!/usr/bin/env python3
"""
模型配置文件

该模块定义了各种模型的配置参数：
- 模型架构配置
- 训练超参数
- 推理配置
- 特定模型优化设置
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """模型配置基类"""
    name: str
    architecture: str
    input_resolution: tuple
    max_sequence_length: int
    batch_size: int
    learning_rate: float
    num_epochs: int


@dataclass
class QwenVLConfig(ModelConfig):
    """Qwen 2.5 VL模型配置"""
    name: str = "qwen2.5-vl"
    architecture: str = "qwen-vl"
    input_resolution: tuple = (1024, 1024)
    max_sequence_length: int = 8192
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    
    # Qwen VL特定配置
    vision_encoder: str = "clip-vit-large"
    text_encoder: str = "qwen2.5-7b"
    fusion_method: str = "cross_attention"
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # 轨迹预测特定配置
    trajectory_embedding_dim: int = 256
    trajectory_sequence_length: int = 36  # 12Hz × 3秒
    coordinate_normalization: str = "minmax"
    heading_representation: str = "sincos"


@dataclass
class TrajectoryPredictionConfig:
    """轨迹预测任务配置"""
    prediction_horizon: float = 3.0  # 3秒预测时域
    sampling_rate: float = 12.0      # 12Hz采样频率
    num_waypoints: int = 36          # 路径点数量
    coordinate_system: str = "ego"   # 坐标系统
    
    # 数据增强
    use_data_augmentation: bool = True
    rotation_range: float = 0.1      # 弧度
    translation_range: float = 2.0   # 米
    noise_std: float = 0.05
    
    # 损失函数配置
    position_loss_weight: float = 1.0
    heading_loss_weight: float = 0.5
    velocity_loss_weight: float = 0.3
    acceleration_loss_weight: float = 0.2


@dataclass
class MultiModalConfig:
    """多模态配置"""
    # 视觉输入配置
    image_channels: List[str] = None
    image_resize: tuple = (800, 600)
    image_normalization: Dict[str, List[float]] = None
    
    # 文本输入配置
    max_text_length: int = 512
    text_tokenizer: str = "qwen2.5-tokenizer"
    
    # 轨迹配置
    trajectory_config: TrajectoryPredictionConfig = None
    
    def __post_init__(self):
        if self.image_channels is None:
            self.image_channels = [
                "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
            ]
        
        if self.image_normalization is None:
            self.image_normalization = {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        
        if self.trajectory_config is None:
            self.trajectory_config = TrajectoryPredictionConfig()


class ModelConfigManager:
    """模型配置管理器"""
    
    def __init__(self):
        """初始化模型配置管理器"""
        self.configs = {
            "qwen2.5-vl": QwenVLConfig(),
            "trajectory_prediction": TrajectoryPredictionConfig(),
            "multimodal": MultiModalConfig()
        }
    
    def get_config(self, model_name: str) -> Optional[ModelConfig]:
        """
        获取模型配置
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型配置对象
        """
        return self.configs.get(model_name)
    
    def register_config(self, model_name: str, config: ModelConfig) -> None:
        """
        注册新的模型配置
        
        Args:
            model_name: 模型名称
            config: 模型配置对象
        """
        self.configs[model_name] = config
    
    def get_all_configs(self) -> Dict[str, ModelConfig]:
        """获取所有配置"""
        return self.configs.copy()
    
    def create_training_config(self, model_name: str, custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        创建训练配置
        
        Args:
            model_name: 模型名称
            custom_params: 自定义参数
            
        Returns:
            训练配置字典
        """
        base_config = self.get_config(model_name)
        if base_config is None:
            raise ValueError(f"未找到模型配置: {model_name}")
        
        # 转换为字典
        config_dict = asdict(base_config) if hasattr(base_config, '__dataclass_fields__') else vars(base_config)
        
        # 应用自定义参数
        if custom_params:
            config_dict.update(custom_params)
        
        return config_dict
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        验证配置有效性
        
        Args:
            config: 配置字典
            
        Returns:
            错误信息列表
        """
        errors = []
        
        # 验证必需字段
        required_fields = ['name', 'architecture', 'batch_size', 'learning_rate']
        for field in required_fields:
            if field not in config:
                errors.append(f"缺少必需字段: {field}")
        
        # 验证数值范围
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                errors.append(f"batch_size必须是正整数: {batch_size}")
        
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append(f"learning_rate必须是正数: {lr}")
        
        # 验证轨迹配置
        if 'trajectory_sequence_length' in config:
            seq_len = config['trajectory_sequence_length']
            if seq_len != 36:
                errors.append(f"轨迹序列长度应为36 (12Hz×3秒): {seq_len}")
        
        return errors


def get_default_qwen_config() -> QwenVLConfig:
    """获取默认的Qwen VL配置"""
    return QwenVLConfig()


def get_12hz_trajectory_config() -> TrajectoryPredictionConfig:
    """获取12Hz轨迹预测配置"""
    return TrajectoryPredictionConfig(
        prediction_horizon=3.0,
        sampling_rate=12.0,
        num_waypoints=36,
        coordinate_system="ego"
    )


def get_production_config() -> Dict[str, Any]:
    """获取生产环境配置"""
    config_manager = ModelConfigManager()
    qwen_config = get_default_qwen_config()
    trajectory_config = get_12hz_trajectory_config()
    
    return {
        "model": asdict(qwen_config),
        "trajectory": asdict(trajectory_config),
        "training": {
            "gradient_checkpointing": True,
            "fp16": True,
            "dataloader_num_workers": 4,
            "save_steps": 500,
            "eval_steps": 100,
            "logging_steps": 10,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0
        },
        "inference": {
            "batch_size": 1,
            "temperature": 0.1,
            "top_p": 0.9,
            "max_new_tokens": 512,
            "do_sample": False
        }
    }


def get_development_config() -> Dict[str, Any]:
    """获取开发环境配置"""
    config = get_production_config()
    
    # 开发环境的调整
    config["model"]["batch_size"] = 2
    config["model"]["num_epochs"] = 1
    config["training"]["save_steps"] = 50
    config["training"]["eval_steps"] = 25
    config["training"]["logging_steps"] = 5
    
    return config


def get_config_for_hardware(gpu_memory_gb: int) -> Dict[str, Any]:
    """
    根据硬件配置获取合适的模型配置
    
    Args:
        gpu_memory_gb: GPU内存大小（GB）
        
    Returns:
        适配的配置字典
    """
    base_config = get_production_config()
    
    if gpu_memory_gb < 8:
        # 低内存配置
        base_config["model"]["batch_size"] = 1
        base_config["model"]["input_resolution"] = (512, 512)
        base_config["training"]["gradient_checkpointing"] = True
        base_config["training"]["fp16"] = True
    elif gpu_memory_gb < 16:
        # 中等内存配置
        base_config["model"]["batch_size"] = 4
        base_config["model"]["input_resolution"] = (800, 600)
    else:
        # 高内存配置
        base_config["model"]["batch_size"] = 8
        base_config["model"]["input_resolution"] = (1024, 1024)
        base_config["training"]["fp16"] = False  # 使用fp32获得更好精度
    
    return base_config


# 预定义配置常量
SUPPORTED_MODELS = {
    "qwen2.5-vl-2b": {"size": "2B", "recommended_memory": 8},
    "qwen2.5-vl-7b": {"size": "7B", "recommended_memory": 16},
    "qwen2.5-vl-14b": {"size": "14B", "recommended_memory": 32},
    "qwen2.5-vl-72b": {"size": "72B", "recommended_memory": 80}
}

TRAJECTORY_FORMATS = {
    "12hz_3s": {
        "sampling_rate": 12.0,
        "prediction_horizon": 3.0,
        "num_waypoints": 36,
        "description": "12Hz采样，3秒预测时域，36个路径点"
    },
    "6hz_3s": {
        "sampling_rate": 6.0,
        "prediction_horizon": 3.0,
        "num_waypoints": 18,
        "description": "6Hz采样，3秒预测时域，18个路径点"
    },
    "2hz_6s": {
        "sampling_rate": 2.0,
        "prediction_horizon": 6.0,
        "num_waypoints": 12,
        "description": "2Hz采样，6秒预测时域，12个路径点（传统格式）"
    }
}

# 默认使用12Hz配置
DEFAULT_TRAJECTORY_FORMAT = "12hz_3s"