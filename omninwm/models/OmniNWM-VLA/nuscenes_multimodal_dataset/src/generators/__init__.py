"""
生成器模块

负责生成各种数据格式和内容：
- 提示文本生成器（支持传统和ShareGPT格式）
- JSON格式化器（支持传统和ShareGPT格式）
- 对话式提示生成器
- ShareGPT格式化器
"""

# 原有模块（保持向后兼容）
from .prompt_generator import PromptManager as LegacyPromptManager, PromptGenerator as LegacyPromptGenerator
from .json_formatter import JSONFormatter as LegacyJSONFormatter

# 新模块（推荐使用）
from .conversation_prompt_generator import ConversationPromptGenerator
from .sharegpt_formatter import ShareGPTFormatter

# 向后兼容的别名
PromptGenerator = LegacyPromptGenerator
JSONFormatter = LegacyJSONFormatter

__all__ = [
    # 推荐使用的新模块
    'ConversationPromptGenerator',
    'ShareGPTFormatter',
    
    # 向后兼容的接口
    'PromptGenerator',
    'JSONFormatter',
    'LegacyPromptManager',
    'LegacyPromptGenerator',
    'LegacyJSONFormatter'
]