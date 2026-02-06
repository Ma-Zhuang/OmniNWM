"""
三模态视觉语言模型 (Tri-Modal VLM)

基于LLaMA-Factory与Qwen2.5-VL的三模态（RGB、深度、语义）视觉语言模型实现。
"""

from .modeling.configuration_tri_modal_qwen import TriModalQwenConfig
from .modeling.modeling_tri_modal_qwen import TriModalQwenForCausalLM
from .modeling.modules.tmi_module import TriModalInterpreter
from .data.dataset import TriModalDataset
from .data.collator import TriModalCollator
from .utils.registry import register_tri_modal_qwen

# 版本信息
__version__ = "0.1.0"
__author__ = "VLA Project Team"

# 导出的主要类
__all__ = [
    "TriModalQwenConfig",
    "TriModalQwenForCausalLM", 
    "TriModalInterpreter",
    "TriModalDataset",
    "TriModalCollator",
    "register_tri_modal_qwen"
]

# 自动注册模型
try:
    register_tri_modal_qwen()
except Exception as e:
    # 注册失败不影响正常使用，只是无法通过AutoModel加载
    pass