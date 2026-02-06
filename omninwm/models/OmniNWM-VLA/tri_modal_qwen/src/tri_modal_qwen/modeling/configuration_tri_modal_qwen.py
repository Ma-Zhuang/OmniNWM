"""
三模态Qwen配置类

定义三模态VLM的所有配置参数，包括：
- 基座模型配置
- TMI模块配置  
- 编码器配置
- 训练配置
"""

from transformers import PretrainedConfig
from typing import Dict, Any, Optional, Union, List
import json
import os


class TriModalQwenConfig(PretrainedConfig):
    """
    三模态Qwen模型的配置类
    
    继承自PretrainedConfig，支持从Hugging Face Hub加载和保存配置
    包含所有Qwen2.5-VL所需的配置参数
    """
    
    model_type = "tri_modal_qwen"
    
    def __init__(
        self,
        # 基座模型配置
        base_model_name_or_path: str = "/code/VLA/models/Qwen2.5-VL-7B-Instruct",
        base_model_config: Optional[Dict[str, Any]] = None,
        
        # Qwen2.5-VL必需的配置参数
        vocab_size: int = 152064,  # Qwen2.5-VL的词表大小
        hidden_size: int = 3584,  # Qwen2.5-VL-7B的隐藏层大小
        intermediate_size: int = 18944,  # FFN中间层大小
        num_hidden_layers: int = 28,  # Transformer层数
        num_attention_heads: int = 28,  # 注意力头数
        num_key_value_heads: int = 4,  # GQA的KV头数
        hidden_act: str = "silu",  # 激活函数
        rope_theta: float = 10000.0,  # RoPE参数
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE缩放配置
        rms_norm_eps: float = 1e-6,  # RMSNorm的epsilon
        initializer_range: float = 0.02,  # 权重初始化范围
        use_cache: bool = True,  # 是否使用KV缓存
        tie_word_embeddings: bool = False,  # 是否共享词嵌入
        bos_token_id: int = 151643,  # BOS token ID
        eos_token_id: int = 151645,  # EOS token ID
        
        # TMI模块配置
        tmi_config: Optional[Dict[str, Any]] = None,
        
        # 编码器配置
        depth_encoder_config: Optional[Dict[str, Any]] = None,
        semantic_encoder_config: Optional[Dict[str, Any]] = None,
        
        # 融合配置（基于SSR优化）
        fusion_hidden_size: int = 2048,
        fusion_num_layers: int = 18,  # 增加到18层，接近SSR的24层，提升容量
        fusion_type: str = "mamba",  # ["mamba", "attention", "linear"]
        fusion_dropout: float = 0.05,  # 减小dropout，SSR证明不需要太强的正则化
        use_pretrained_mamba: bool = True,  # 使用预训练的Mamba模型
        pretrained_mamba_model: str = "/code/VLA/models/state-spaces/mamba-130m-hf",  # 本地预训练Mamba模型路径
        
        # 架构配置
        vision_hidden_size: int = 3584,  # Qwen2.5-VL vision encoder的输出维度(out_hidden_size)
        llm_hidden_size: int = 3584,     # Qwen2.5-VL LLM隐藏维度 - 匹配实际模型
        max_position_embeddings: int = 32768,
        image_size: int = 392,  # 保留392，能被14整除
        patch_size: int = 14,
        
        # 训练配置
        freeze_base_model: bool = False,
        freeze_vision_tower: bool = False,
        freeze_language_model: bool = False,
        gradient_checkpointing: bool = False,  # 梯度检查点
        
        # 特殊token配置
        planning_token: str = "<PLANNING>",
        planning_end_token: str = "</PLANNING>",
        image_token: str = "<image>",
        
        # 优化配置
        use_flash_attention: bool = True,
        enable_flash_attention: bool = True,  # 兼容性别名
        enable_xformers: bool = True,  # xFormers支持
        enable_gradient_checkpointing: bool = False,  # 兼容性别名
        torch_dtype: str = "float16",
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # 基座模型配置
        self.base_model_name_or_path = base_model_name_or_path
        self.base_model_config = base_model_config or {}
        
        # Qwen2.5-VL模型参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # TMI模块配置
        self.tmi_config = tmi_config or {
            "fusion_hidden_size": fusion_hidden_size,
            "fusion_num_layers": fusion_num_layers,
            "fusion_type": fusion_type,
            "fusion_dropout": fusion_dropout,
            "use_pretrained_mamba": use_pretrained_mamba,
            "pretrained_mamba_model": pretrained_mamba_model
        }
        
        # 深度编码器配置
        self.depth_encoder_config = depth_encoder_config or {
            "type": "cnn",               # ["cnn", "vit", "resnet"]
            "input_channels": 1,         # 深度图单通道
            "hidden_size": 1024,         # 输出特征维度
            "num_layers": 3,             # 层数
            "kernel_size": 3,            # 卷积核大小
            "stride": 2,                 # 步长
            "padding": 1,                # 填充
            "activation": "gelu",        # 激活函数
            "use_batch_norm": True,      # 是否使用BatchNorm
            "dropout": 0.1               # Dropout率
        }
        
        # 语义编码器配置
        self.semantic_encoder_config = semantic_encoder_config or {
            "type": "cnn",               # ["cnn", "vit", "resnet"]
            "input_channels": 150,       # 语义分割类别数
            "hidden_size": 1024,         # 输出特征维度
            "num_layers": 3,             # 层数
            "kernel_size": 3,            # 卷积核大小
            "stride": 2,                 # 步长
            "padding": 1,                # 填充
            "activation": "gelu",        # 激活函数
            "use_batch_norm": True,      # 是否使用BatchNorm
            "dropout": 0.1               # Dropout率
        }
        
        # 融合配置
        self.fusion_hidden_size = fusion_hidden_size
        self.fusion_num_layers = fusion_num_layers
        self.fusion_type = fusion_type
        self.fusion_dropout = fusion_dropout
        self.use_pretrained_mamba = use_pretrained_mamba
        self.pretrained_mamba_model = pretrained_mamba_model
        
        # 架构配置
        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size if llm_hidden_size else hidden_size  # 使用传入的hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.image_size = image_size
        self.patch_size = patch_size
        
        # 训练配置
        self.freeze_base_model = freeze_base_model
        self.freeze_vision_tower = freeze_vision_tower
        self.freeze_language_model = freeze_language_model
        
        # 特殊token配置
        self.planning_token = planning_token
        self.planning_end_token = planning_end_token
        self.image_token = image_token
        
        # 其他配置
        self.use_flash_attention = use_flash_attention
        self.torch_dtype = torch_dtype
        
        # Flash Attention优化配置
        self.enable_flash_attention = kwargs.pop("enable_flash_attention", True)
        self.enable_xformers = kwargs.pop("enable_xformers", True)
        self.enable_gradient_checkpointing = kwargs.pop("enable_gradient_checkpointing", True)  # 恢复默认为True
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        self.flash_attention_block_size = kwargs.pop("flash_attention_block_size", 512)
        
        # 性能优化配置
        self.use_cache = kwargs.pop("use_cache", True)
        self.low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", True)
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数的有效性"""
        
        # 验证融合类型
        valid_fusion_types = ["mamba", "attention", "linear"]
        if self.fusion_type not in valid_fusion_types:
            raise ValueError(f"fusion_type必须是{valid_fusion_types}之一，得到: {self.fusion_type}")
        
        # 验证编码器类型
        valid_encoder_types = ["cnn", "vit", "resnet"]
        if self.depth_encoder_config["type"] not in valid_encoder_types:
            raise ValueError(f"depth_encoder_config.type必须是{valid_encoder_types}之一")
        if self.semantic_encoder_config["type"] not in valid_encoder_types:
            raise ValueError(f"semantic_encoder_config.type必须是{valid_encoder_types}之一")
        
        # 验证维度配置
        if self.fusion_hidden_size <= 0:
            raise ValueError("fusion_hidden_size必须大于0")
        if self.fusion_num_layers <= 0:
            raise ValueError("fusion_num_layers必须大于0")
        if self.vision_hidden_size <= 0:
            raise ValueError("vision_hidden_size必须大于0")
        if self.llm_hidden_size <= 0:
            raise ValueError("llm_hidden_size必须大于0")
        
        # 验证图像配置
        if self.image_size <= 0:
            raise ValueError("image_size必须大于0")
        if self.patch_size <= 0:
            raise ValueError("patch_size必须大于0")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size必须能被patch_size整除")
    
    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: Union[str, os.PathLike], 
        **kwargs
    ) -> "TriModalQwenConfig":
        """从预训练模型或路径加载配置"""
        
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "TriModalQwenConfig":
        """从字典创建配置"""
        
        # 移除transformers相关的特殊键
        config_dict = {key: value for key, value in config_dict.items() 
                      if not key.startswith('_')}
        
        # 合并kwargs
        config_dict.update(kwargs)
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        
        output = super().to_dict()
        
        # 添加自定义配置
        output.update({
            "model_type": self.model_type,
            "base_model_name_or_path": self.base_model_name_or_path,
            "base_model_config": self.base_model_config,
            "tmi_config": self.tmi_config,
            "depth_encoder_config": self.depth_encoder_config,
            "semantic_encoder_config": self.semantic_encoder_config,
            "fusion_hidden_size": self.fusion_hidden_size,
            "fusion_num_layers": self.fusion_num_layers,
            "fusion_type": self.fusion_type,
            "fusion_dropout": self.fusion_dropout,
            "use_pretrained_mamba": self.use_pretrained_mamba,
            "pretrained_mamba_model": self.pretrained_mamba_model,
            "vision_hidden_size": self.vision_hidden_size,
            "llm_hidden_size": self.llm_hidden_size,
            "max_position_embeddings": self.max_position_embeddings,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "freeze_base_model": self.freeze_base_model,
            "freeze_vision_tower": self.freeze_vision_tower,
            "freeze_language_model": self.freeze_language_model,
            "planning_token": self.planning_token,
            "planning_end_token": self.planning_end_token,
            "image_token": self.image_token,
            "use_flash_attention": self.use_flash_attention,
            "torch_dtype": self.torch_dtype
        })
        
        return output
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """保存配置到指定目录"""
        
        if os.path.isfile(save_directory):
            raise AssertionError(f"提供的路径({save_directory})应该是一个目录，而不是文件")
        
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存配置文件
        output_config_file = os.path.join(save_directory, "config.json")
        
        with open(output_config_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def get_num_patches(self) -> int:
        """计算图像patch数量"""
        return (self.image_size // self.patch_size) ** 2
    
    def get_sequence_length(self) -> int:
        """计算三模态拼接后的总序列长度"""
        num_patches = self.get_num_patches()
        return num_patches * 3  # RGB + Depth + Semantic
    
    def update_from_string(self, update_str: str):
        """从字符串更新配置（用于命令行参数）"""
        
        if not update_str:
            return
        
        try:
            update_dict = json.loads(update_str)
            for key, value in update_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"警告: 未知的配置键 '{key}'")
        except json.JSONDecodeError as e:
            raise ValueError(f"配置字符串格式错误: {e}")
    
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"