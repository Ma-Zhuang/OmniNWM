"""
nuScenes多模态数据集构建器

基于nuScenes数据集构建用于Qwen 2.5 VL模型微调的多模态指令数据集的完整工具包。

主要功能：
- nuScenes数据集读取和处理
- 多模态数据生成（RGB、深度、语义分割）
- 未来轨迹计算和标注
- VLM微调格式的JSON数据生成
- 数据验证和质量控制
- 统计分析和可视化

核心模块：
- core: nuScenes数据读取、坐标变换、轨迹计算
- processors: 深度图生成、语义分割、多模态处理
- generators: 提示生成、JSON格式化、数据集生成
- utils: 工具函数、验证、数学计算、日志
- config: 配置管理、模型配置
- pipeline: 数据处理流水线、并行处理

使用示例：
    from src.pipeline.data_pipeline import DataProcessingPipeline
    
    pipeline = DataProcessingPipeline("configs/dataset_config.yaml")
    pipeline.run_pipeline("/path/to/output")
"""

__version__ = "1.0.0"
__author__ = "nuScenes Multimodal Dataset Builder Team"

# 导入主要的公共接口
from .pipeline.data_pipeline import DataProcessingPipeline
from .config.config_manager import ConfigManager
from .utils.logging_utils import setup_logging

# 核心组件
from .core.nuscenes_reader import NuScenesReader
from .core.coordinate_transform import CoordinateTransformer
from .core.trajectory_calculator import TrajectoryCalculator

# 处理器
from .processors.depth_processor import DepthProcessor
from .processors.semantic_processor import SemanticProcessor
from .processors.multimodal_processor import MultiModalProcessor

# 生成器
from .generators.prompt_generator import PromptGenerator
from .generators.json_formatter import JSONFormatter
from .generators.conversation_prompt_generator import ConversationPromptGenerator
from .generators.sharegpt_formatter import ShareGPTFormatter

# 工具
from .utils.file_utils import FileManager
from .utils.math_utils import AngleUtils, GeometryUtils, TrajectoryUtils
from .utils.validation_utils import DatasetValidator

__all__ = [
    # 主要接口
    'DataProcessingPipeline',
    'ConfigManager',
    'setup_logging',
    
    # 核心组件
    'NuScenesReader',
    'CoordinateTransformer', 
    'TrajectoryCalculator',
    
    # 处理器
    'DepthProcessor',
    'SemanticProcessor',
    'MultiModalProcessor',
    
    # 生成器
    'PromptGenerator',
    'JSONFormatter',
    'ConversationPromptGenerator',
    'ShareGPTFormatter',
    
    # 工具
    'FileManager',
    'AngleUtils',
    'GeometryUtils',
    'TrajectoryUtils',
    'DatasetValidator'
]