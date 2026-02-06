"""
nuScenes多模态数据集构建器 - 核心模块

该模块包含处理nuScenes数据集的核心功能：
- nuScenes数据读取器
- 坐标系变换工具  
- 轨迹计算器
"""

from .nuscenes_reader import NuScenesReader
from .coordinate_transform import CoordinateTransformer
from .trajectory_calculator import TrajectoryCalculator

__all__ = [
    'NuScenesReader',
    'CoordinateTransformer', 
    'TrajectoryCalculator'
]