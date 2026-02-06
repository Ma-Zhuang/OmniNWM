"""
模型定义模块

包含三模态VLM的核心模型实现：
- 配置类
- 主模型类  
- TMI模块及其子组件
"""

from .configuration_tri_modal_qwen import TriModalQwenConfig
from .modeling_tri_modal_qwen import TriModalQwenForCausalLM
from .modules.tmi_module import TriModalInterpreter
from .modules.encoders import DepthEncoder, SemanticEncoder
from .modules.fusion import MambaFusionCore

__all__ = [
    "TriModalQwenConfig",
    "TriModalQwenForCausalLM",
    "TriModalInterpreter", 
    "DepthEncoder",
    "SemanticEncoder",
    "MambaFusionCore"
]