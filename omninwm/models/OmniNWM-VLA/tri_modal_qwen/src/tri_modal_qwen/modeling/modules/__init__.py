"""
模型子模块

包含TMI的各个组件：
- TMI核心模块
- 编码器模块
- 融合模块
"""

from .tmi_module import TriModalInterpreter
from .encoders import DepthEncoder, SemanticEncoder
from .fusion import MambaFusionCore

__all__ = [
    "TriModalInterpreter",
    "DepthEncoder", 
    "SemanticEncoder",
    "MambaFusionCore"
]