"""
三模态Qwen主模型类

实现继承自PreTrainedModel的主模型，集成:
- Qwen2.5-VL基座模型
- TMI三模态解释器
- 轨迹预测头
- 分阶段训练支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union, List
import warnings
import os
import math

from transformers import (
    PreTrainedModel, 
    AutoModelForCausalLM, 
    AutoConfig,
    GenerationConfig,
    GenerationMixin
)

# 尝试导入Qwen2VL特定的类
try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2VLModel
    QWEN2VL_AVAILABLE = True
except ImportError:
    QWEN2VL_AVAILABLE = False
    try:
        # 尝试直接导入
        from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLModel
        QWEN2VL_AVAILABLE = True
    except ImportError:
        QWEN2VL_AVAILABLE = False

# 注意：Qwen2.5-VL和Qwen2-VL是不同的模型架构，不能混用
# Qwen2.5-VL需要使用AutoModel加载，并设置trust_remote_code=True
QWEN25VL_AVAILABLE = False  # 不使用Qwen2VL的类加载Qwen2.5-VL
    
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging

from .configuration_tri_modal_qwen import TriModalQwenConfig
from .modules.tmi_module import TriModalInterpreter
from .modules.encoders import DepthEncoder, SemanticEncoder
from .modules.flash_attention import (
    FlashAttentionConfig, 
    OptimizedMultiHeadAttention,
    apply_flash_attention_optimization,
    get_memory_usage
)

logger = logging.get_logger(__name__)


class TriModalQwenForCausalLM(PreTrainedModel, GenerationMixin):
    """
    三模态Qwen因果语言模型
    
    基于Qwen2.5-VL构建，支持RGB、深度、语义三种模态输入
    主要用于轨迹预测任务
    """
    
    config_class = TriModalQwenConfig
    base_model_prefix = "tri_modal_qwen"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TriModalInterpreter", "DepthEncoder", "SemanticEncoder"]
    
    def __init__(self, config: TriModalQwenConfig):
        super().__init__(config)
        
        self.config = config
        
        # 1. 创建Flash Attention配置
        self.flash_config = FlashAttentionConfig(
            enable_flash_attention=getattr(config, 'enable_flash_attention', True),
            enable_xformers=getattr(config, 'enable_xformers', True),
            enable_gradient_checkpointing=getattr(config, 'enable_gradient_checkpointing', True),
            attention_dropout=getattr(config, 'attention_dropout', 0.0),
            block_size=getattr(config, 'flash_attention_block_size', 512)
        )
        
        # 2. 加载基座模型组件 (避免创建完整模型实例)
        self.model = None  # 使用model而不是base_model，避免与父类property冲突
        self._load_base_model_components(config)
        
        # 3. 创建TMI模块
        self.tmi_module = TriModalInterpreter(config)
        
        # 4. 创建任务特定的头部
        self.task_head = TaskHead(config)
        
        # 4.5 Stage 1训练：创建TMI辅助预测头
        # 这个头直接从TMI特征预测轨迹，帮助TMI学习融合
        # 与TaskHead不同，这个头更简单，专门用于监督TMI学习
        if config.freeze_base_model:  # Stage 1训练模式
            self.tmi_auxiliary_head = nn.Sequential(
                nn.LayerNorm(config.llm_hidden_size),
                nn.Linear(config.llm_hidden_size, config.llm_hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(config.llm_hidden_size // 2, 108)  # 36个waypoints × 3坐标
            )
            # FP16友好的初始化策略
            for m in self.tmi_auxiliary_head.modules():
                if isinstance(m, nn.Linear):
                    # 使用Kaiming初始化，更适合GELU激活函数
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(m.bias, -bound, bound)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
            
            # 对最后一层使用更小的初始化，确保初始输出在合理范围
            last_linear = None
            for m in self.tmi_auxiliary_head.modules():
                if isinstance(m, nn.Linear):
                    last_linear = m
            if last_linear is not None:
                # 使用更小的范围初始化最后一层
                nn.init.uniform_(last_linear.weight, -0.01, 0.01)
                if last_linear.bias is not None:
                    nn.init.zeros_(last_linear.bias)
            
            logger.info("创建TMI辅助预测头用于Stage 1训练")
        
        # 5. 创建语言模型头 (用于生成)
        # 只有在模型加载成功且没有lm_head时才创建
        if self.model is not None and not hasattr(self.model, 'lm_head'):
            if not hasattr(self, 'lm_head'):
                self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
                logger.info("创建了新的lm_head")
        
        # 6. 应用Flash Attention优化
        if config.enable_flash_attention and self.model is not None:
            try:
                self.model = apply_flash_attention_optimization(
                    self.model, 
                    self.flash_config
                )
                logger.info("✅ Flash Attention优化已应用到基座模型")
            except Exception as e:
                logger.warning(f"Flash Attention优化失败: {e}")
        
        # 7. 梯度检查点配置
        if getattr(config, 'gradient_checkpointing', False) and self.model is not None:
            self._enable_gradient_checkpointing()
            logger.info("✅ 梯度检查点已启用，将减少显存占用")
        
        # 8. 检查模型配置兼容性
        self._validate_model_compatibility()
        
        # 9. 手动初始化TMI和TaskHead的权重
        self._init_custom_weights()
        
        # 10. 统一设置所有模块的dtype
        self._setup_dtype()
        
        # 11. 模型冻结配置 - 在初始化后设置参数状态
        if config.freeze_base_model:
            self._setup_stage1_training()
        
        # 12. 打印可训练参数统计
        self._print_trainable_parameters()
    
    def _load_base_model_components(self, config: TriModalQwenConfig):
        """
        加载基座模型的组件，而不是完整模型
        避免创建嵌套的模型实例导致递归
        """
        try:
            # 尝试从预训练模型加载组件
            from transformers import AutoModel, AutoModelForCausalLM
            
            # 临时加载模型以提取组件
            logger.info(f"加载基座模型组件: {config.base_model_name_or_path}")
            
            # 使用上下文管理器临时加载
            import torch
            with torch.no_grad():
                # 先尝试加载配置
                from transformers import AutoConfig
                base_config = AutoConfig.from_pretrained(
                    config.base_model_name_or_path,
                    trust_remote_code=True
                )
                
                # 更新我们的配置以匹配实际模型
                if hasattr(base_config, 'hidden_size'):
                    if base_config.hidden_size != config.hidden_size:
                        logger.warning(f"配置中的hidden_size ({config.hidden_size}) 与模型实际值 ({base_config.hidden_size}) 不匹配")
                        logger.info(f"自动调整为: {base_config.hidden_size}")
                        config.hidden_size = base_config.hidden_size
                        config.llm_hidden_size = base_config.hidden_size
                
                if hasattr(base_config, 'vocab_size'):
                    config.vocab_size = base_config.vocab_size
                
                if hasattr(base_config, 'num_hidden_layers'):
                    config.num_hidden_layers = base_config.num_hidden_layers
                
                # 根据模型类型加载
                if "qwen2" in base_config.model_type.lower() or "qwen2.5" in str(config.base_model_name_or_path).lower():
                    # 对于Qwen2.5-VL，优先尝试加载视觉语言模型
                    logger.info("检测到Qwen2.5-VL模型，加载核心组件...")
                    
                    # 对于Qwen2.5-VL，直接使用AutoModel（会自动选择正确的类）
                    temp_model = None
                    
                    # 对于Qwen2.5-VL，需要确保加载正确的模型类
                    # Qwen2.5-VL是视觉-语言模型，需要ForConditionalGeneration版本
                    try:
                        logger.info("加载Qwen2.5-VL模型（启用Flash Attention）...")
                        
                        # 首先尝试获取配置来确定正确的模型类
                        from transformers import AutoConfig
                        model_config = AutoConfig.from_pretrained(
                            config.base_model_name_or_path,
                            trust_remote_code=True
                        )
                        
                        # 根据配置的architectures字段选择正确的加载方式
                        if hasattr(model_config, 'architectures') and model_config.architectures:
                            if 'Qwen2_5_VLForConditionalGeneration' in model_config.architectures:
                                logger.info("检测到Qwen2_5_VLForConditionalGeneration架构")
                        
                        # 直接导入并使用Qwen2_5_VLForConditionalGeneration类
                        try:
                            from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
                            logger.info("使用Qwen2_5_VLForConditionalGeneration从transformers")
                        except ImportError:
                            # 如果无法直接导入，使用动态导入
                            logger.info("使用动态导入获取Qwen2_5_VLForConditionalGeneration")
                            from transformers.dynamic_module_utils import get_class_from_dynamic_module
                            Qwen2_5_VLForConditionalGeneration = get_class_from_dynamic_module(
                                class_reference="Qwen2_5_VLForConditionalGeneration",
                                pretrained_model_name_or_path=config.base_model_name_or_path,
                                trust_remote_code=True,
                                code_revision=None,
                            )
                        
                        # 使用正确的模型类加载 - 使用BF16或FP32避免FP16问题
                        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                            load_dtype = torch.bfloat16
                            logger.info("使用BF16加载模型（比FP16更稳定）")
                        else:
                            load_dtype = torch.float32
                            logger.info("使用FP32加载模型（最稳定）")
                        
                        temp_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            config.base_model_name_or_path,
                            torch_dtype=load_dtype,
                            trust_remote_code=True,  # 关键：允许加载远程代码
                            low_cpu_mem_usage=True,
                            device_map=None,
                            attn_implementation="flash_attention_2"  # 启用Flash Attention 2
                        )
                        
                        logger.info(f"✅ 成功加载模型: {type(temp_model).__name__}")
                    except Exception as e:
                        logger.error(f"Flash Attention加载失败: {e}")
                        # 如果Flash Attention失败，仍然尝试使用eager模式
                        # 但会警告用户性能会下降
                        try:
                            logger.warning("⚠️ Flash Attention不可用，使用标准attention（性能会下降）")
                            
                            # 确保使用相同的模型类
                            if 'Qwen2_5_VLForConditionalGeneration' not in locals():
                                try:
                                    from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
                                except ImportError:
                                    from transformers.dynamic_module_utils import get_class_from_dynamic_module
                                    Qwen2_5_VLForConditionalGeneration = get_class_from_dynamic_module(
                                        class_reference="Qwen2_5_VLForConditionalGeneration",
                                        pretrained_model_name_or_path=config.base_model_name_or_path,
                                        trust_remote_code=True,
                                        code_revision=None,
                                    )
                            
                            temp_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                                config.base_model_name_or_path,
                                torch_dtype=torch.float16,
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                                device_map=None,
                                attn_implementation="eager"  # 标准attention
                            )
                            logger.info("✅ 成功加载（标准attention模式）")
                            logger.warning("⚠️ 建议安装flash-attn以获得更好的性能：pip install flash-attn")
                        except Exception as e2:
                            logger.error(f"所有加载方式都失败: {e2}")
                            logger.info("提示：确保transformers版本 >= 4.40.0")
                            raise
                    
                    # 提取需要的组件 - Qwen2.5-VL的结构
                    # AutoModelForCausalLM.from_pretrained返回的是完整模型
                    # 对于Qwen2.5-VL，我们需要整个模型来处理视觉和语言
                    
                    # 保存完整的Qwen2_5_VLForConditionalGeneration模型
                    # 重要：只保存整个模型，不保存子组件引用以避免循环引用
                    self.model = temp_model
                    logger.info(f"✅ 成功加载完整的Qwen2.5-VL模型: {type(temp_model).__name__}")
                    
                    # 不再单独保存visual和lm_head的引用，通过self.model访问
                    # 这样避免了循环引用导致的递归错误
                    if hasattr(temp_model, 'visual'):
                        logger.info("✅ 检测到visual组件")
                    if hasattr(temp_model, 'lm_head'):
                        logger.info("✅ 检测到lm_head组件")
                    
                    # 清理临时变量（temp_model现在是self.model，不需要删除）
                    torch.cuda.empty_cache()
                    
                else:
                    # 其他模型类型
                    logger.warning(f"未知的模型类型，使用默认加载策略")
                    self.model = AutoModel.from_pretrained(
                        config.base_model_name_or_path,
                        torch_dtype=torch.float16,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
            
            logger.info("✅ 基座模型组件加载成功")
            
        except Exception as e:
            logger.warning(f"无法加载基座模型组件: {e}")
            logger.info("将使用随机初始化的模型")
            self._initialize_empty_model(config)
    
    def _initialize_empty_model(self, config: TriModalQwenConfig):
        """
        当无法加载预训练模型时，初始化一个空模型
        注意：这种情况下RGB编码器将无法正常工作，因为缺少预训练的视觉模块
        """
        logger.warning("⚠️ 警告：将创建空模型，这会导致RGB编码器无法使用预训练的ViT")
        logger.warning("⚠️ 建议：请确保Qwen2.5-VL模型权重已正确下载到: " + config.base_model_name_or_path)
        
        # 尝试使用AutoModel加载，让transformers自动选择正确的模型类
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 检查是否有视觉模块
            if hasattr(self.model, 'visual') or hasattr(self.model, 'vision_tower'):
                logger.info("✅ 成功加载带有视觉模块的模型")
                if hasattr(self.model, 'visual'):
                    self.visual = self.model.visual
                elif hasattr(self.model, 'vision_tower'): 
                    self.visual = self.model.vision_tower
            else:
                logger.warning("⚠️ 加载的模型没有视觉模块")
                
        except Exception as e:
            logger.error(f"无法加载模型，将创建空的文本模型: {e}")
            # 最后的降级方案：创建纯文本模型
            from transformers.models.qwen2 import Qwen2Model, Qwen2Config
            
            # 创建配置
            model_config = Qwen2Config(
                vocab_size=config.vocab_size,
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                hidden_act=config.hidden_act,
                max_position_embeddings=config.max_position_embeddings,
                initializer_range=config.initializer_range,
                rms_norm_eps=config.rms_norm_eps,
                use_cache=config.use_cache,
                tie_word_embeddings=config.tie_word_embeddings,
                rope_theta=config.rope_theta,
                rope_scaling=config.rope_scaling,
                bos_token_id=config.bos_token_id,
                eos_token_id=config.eos_token_id,
            )
            
            # 创建模型
            self.model = Qwen2Model(model_config)
            logger.warning("⚠️ 创建了空的Qwen2文本模型（没有视觉模块）")
    
    def _init_custom_weights(self):
        """
        初始化自定义模块的权重
        
        注意：TMI模块有自己的初始化逻辑，不应该被覆盖
        """
        # TMI模块已经在__init__中正确初始化，不要覆盖
        # 只初始化TaskHead和lm_head
        
        # 初始化TaskHead
        if hasattr(self, 'task_head'):
            # TaskHead有自己的_init_weights方法，已经在__init__中调用
            # 这里不需要重复初始化
            pass
        
        # 初始化TMI辅助头（如果存在）
        if hasattr(self, 'tmi_auxiliary_head'):
            # 这个已经在__init__中初始化了
            pass
        
        # 初始化lm_head（如果是新创建的）
        if hasattr(self, 'lm_head') and not self.config.tie_word_embeddings:
            self.lm_head.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        
        logger.info("✅ 自定义权重初始化完成")
    
    def _load_base_model(self, config: TriModalQwenConfig):
        """
        健壮的基座模型加载方法
        
        实现多种加载策略和错误处理
        """
        
        loading_strategies = [
            self._load_strategy_full_model,
            self._load_strategy_components,
            self._load_strategy_compatible_fallback
        ]
        
        base_model = None
        last_error = None
        
        for strategy in loading_strategies:
            try:
                logger.info(f"尝试加载策略: {strategy.__name__}")
                base_model = strategy(config)
                if base_model is not None:
                    logger.info(f"✅ 基座模型加载成功: {config.base_model_name_or_path}")
                    return base_model
            except Exception as e:
                logger.warning(f"加载策略 {strategy.__name__} 失败: {e}")
                last_error = e
                continue
        
        # 如果所有策略都失败了，抛出详细错误
        error_msg = f"""
基座模型加载完全失败。已尝试所有加载策略：
1. 完整模型加载
2. 组件式加载
3. 兼容性降级加载

最后一个错误: {last_error}

请检查：
1. 网络连接是否正常（需要从Hugging Face下载模型）
2. transformers版本是否兼容 (>= 4.36.0)
3. 模型路径是否正确: {config.base_model_name_or_path}
4. 是否有足够的磁盘空间和内存
        """
        raise RuntimeError(error_msg)
    
    def _load_strategy_full_model(self, config: TriModalQwenConfig):
        """策略1: 直接加载完整的Qwen2.5-VL模型"""
        
        # 检查是否是Qwen2.5-VL模型
        if "Qwen2.5-VL" in config.base_model_name_or_path or "qwen2_5_vl" in config.base_model_name_or_path.lower():
            # 对于Qwen2.5-VL，必须使用AutoModel + trust_remote_code=True
            # 不能使用Qwen2VL的类，因为架构不同
            from transformers import AutoModel
            return AutoModel.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=getattr(torch, config.torch_dtype),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None,
                attn_implementation="flash_attention_2",  # 启用Flash Attention
                **config.base_model_config
            )
        else:
            # 对于其他模型，使用AutoModelForCausalLM
            return AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=getattr(torch, config.torch_dtype),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto" if torch.cuda.is_available() else None,
                **config.base_model_config
            )
    
    def _load_strategy_components(self, config: TriModalQwenConfig):
        """策略2: 分组件加载（针对内存不足的情况）"""
        
        # 首先尝试只加载配置
        model_config = AutoConfig.from_pretrained(
            config.base_model_name_or_path,
            trust_remote_code=True
        )
        
        # 检查是否是Qwen2.5-VL模型
        if "Qwen2.5-VL" in config.base_model_name_or_path or "qwen2_5_vl" in model_config.model_type:
            # 对于Qwen2.5-VL，使用AutoModel + trust_remote_code
            from transformers import AutoModel
            base_model = AutoModel.from_pretrained(
                config.base_model_name_or_path,
                config=model_config,
                torch_dtype=torch.float16,  # 使用更小的精度
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",  # 启用Flash Attention
                load_in_8bit=True if "load_in_8bit" not in config.base_model_config else config.base_model_config["load_in_8bit"]
            )
        else:
            # 对于其他模型
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                config=model_config,
                torch_dtype=torch.float16,  # 使用更小的精度
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_8bit=True if "load_in_8bit" not in config.base_model_config else config.base_model_config["load_in_8bit"]
            )
        
        return base_model
    
    def _load_strategy_compatible_fallback(self, config: TriModalQwenConfig):
        """策略3: 兼容性降级加载"""
        
        logger.warning("使用兼容性降级策略，某些功能可能受限")
        
        # 尝试加载一个较小或更通用的模型作为基础
        fallback_models = [
            "Qwen/Qwen2-VL-7B-Instruct",  # 备选Qwen模型
            "microsoft/DialoGPT-medium",   # 小型备选模型
        ]
        
        for fallback_model in fallback_models:
            try:
                logger.info(f"尝试加载备选模型: {fallback_model}")
                return AutoModelForCausalLM.from_pretrained(
                    fallback_model,
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            except Exception as e:
                logger.warning(f"备选模型 {fallback_model} 加载失败: {e}")
                continue
        
        return None
    
    def _validate_model_compatibility(self):
        """验证模型配置兼容性"""
        
        if self.model is None:
            raise RuntimeError("基座模型未正确加载")
        
        # 检查必要的模型属性
        required_attributes = ['config', 'get_input_embeddings']
        missing_attributes = []
        
        for attr in required_attributes:
            if not hasattr(self.model, attr):
                missing_attributes.append(attr)
        
        if missing_attributes:
            raise RuntimeError(f"基座模型缺少必要属性: {missing_attributes}")
        
        # 检查视觉模块 - 现在检查self.model.visual而不是self.visual
        if hasattr(self.model, 'visual') and self.model.visual is not None:
            logger.info("✅ 检测到视觉模块")
            self.has_vision_module = True
        else:
            logger.warning("⚠️ 未检测到视觉模块，将使用TMI模块处理视觉输入")
            self.has_vision_module = False
        
        # 检查模型维度兼容性
        try:
            vocab_size = self.model.config.vocab_size
            hidden_size = self.model.config.hidden_size
            
            logger.info(f"基座模型信息: vocab_size={vocab_size}, hidden_size={hidden_size}")
            
            # 更新配置以匹配实际模型
            if hasattr(self.config, 'llm_hidden_size'):
                if self.config.llm_hidden_size != hidden_size:
                    logger.warning(f"调整LLM隐藏维度: {self.config.llm_hidden_size} -> {hidden_size}")
                    self.config.llm_hidden_size = hidden_size
                    
                    # 重新初始化TMI模块以匹配新的维度
                    self.tmi_module = TriModalInterpreter(self.config)
                    self.task_head = TaskHead(self.config)
        
        except Exception as e:
            logger.warning(f"模型兼容性检查出现问题: {e}")
        
        logger.info("✅ 模型兼容性验证完成")

    def _extract_rgb_features(
        self, 
        pixel_values: torch.FloatTensor, 
        image_grid_thw: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """
        提取RGB特征，支持多种策略
        """
        
        # 保存原始形状信息
        original_shape = pixel_values.shape
        batch_size = pixel_values.shape[0]
        
        # 策略1: 使用完整模型的前向传播来提取视觉特征
        # 这是更可靠的方法，因为它处理了所有必要的维度转换
        if hasattr(self.model, 'visual') and self.model.visual is not None:
            try:
                # Qwen2.5-VL的visual期望5D输入: [batch, num_frames/images, channels, height, width]
                if pixel_values.dim() == 4:
                    # 添加num_frames维度
                    pixel_values = pixel_values.unsqueeze(1)  # [batch, 1, channels, height, width]
                elif pixel_values.dim() == 5:
                    # 已经是正确格式
                    pass
                else:
                    raise ValueError(f"Unexpected pixel_values dimensions: {pixel_values.dim()}")
                
                # 创建grid_thw - 修正为匹配patch_embed输出的392个patches
                if image_grid_thw is None:
                    num_frames = pixel_values.shape[1]
                    height = pixel_values.shape[3]
                    width = pixel_values.shape[4]
                    
                    # grid_thw应该是patch_embed阶段的输出维度
                    # 根据正确的理解：
                    # - patch_size = 14
                    # - temporal_patch_size = 2 (宽度方向)
                    # - grid_thw = [1, height//14, width//14//2]
                    
                    patch_size = 14
                    temporal_patch_size = 2
                    
                    # 计算patch_embed输出的grid尺寸
                    grid_h = height // patch_size
                    grid_w = width // patch_size // temporal_patch_size
                    
                    # 创建grid_thw
                    image_grid_thw = [[num_frames, grid_h, grid_w] for _ in range(batch_size)]
                    image_grid_thw = torch.tensor(image_grid_thw, dtype=torch.long, device=pixel_values.device)
                    
                    # 记录信息（用于调试）
                    logger.debug(f"自动计算grid_thw: {image_grid_thw[0].tolist()} for {height}×{width} images")
                    logger.debug(f"  Patch embed输出: {grid_h}×{grid_w}={grid_h*grid_w} patches")
                    logger.debug(f"  Merger后输出: {(grid_h*grid_w)//4} patches")
                
                # 方法1: 尝试通过模型的完整forward获取视觉嵌入
                # 这避免了直接调用visual时的维度不匹配问题
                try:
                    # 不使用虚拟input_ids
                    # 如果需要单独处理视觉输入，应该使用专门的视觉编码器
                    # 这里暂时跳过这个分支
                    raise NotImplementedError(
                        "单独的视觉特征提取需要重新实现，"
                        "不能使用虚拟input_ids。"
                        "请提供真实的文本指令与视觉输入配对。"
                    )
                    # 原代码已禁用:
                    # dummy_ids = torch.zeros(...)
                    # with torch.no_grad():
                    #     outputs = self.model(
                    #         input_ids=dummy_ids,
                    #         pixel_values=pixel_values,
                    #         image_grid_thw=image_grid_thw,
                    #         output_hidden_states=True,
                    #         return_dict=True
                    #     )
                    
                    # # 从hidden_states中提取视觉特征
                    # # 通常视觉特征会被添加到序列的开头
                    # # if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    #     # 获取最后一层的hidden states
                    #     last_hidden = outputs.hidden_states[-1]
                    #     
                    #     # 计算视觉token数量
                    #     num_visual_tokens = image_grid_thw[0, 1].item() * image_grid_thw[0, 2].item() * image_grid_thw[0, 0].item()
                    #     
                    #     # 检查实际序列长度
                    #     actual_seq_len = last_hidden.shape[1]
                    #     
                    #     # 如果实际序列长度小于期望的视觉tokens，说明可能有问题
                    #     if actual_seq_len < num_visual_tokens:
                    #         # 可能是98个视觉tokens + 1个文本token = 99
                    #         # 提取除了最后一个文本token外的所有tokens
                    #         rgb_features = last_hidden[:, :actual_seq_len-1, :]
                    #         logger.warning(f"[方法1] 序列长度{actual_seq_len}小于期望{num_visual_tokens}，提取前{actual_seq_len-1}个tokens")
                    #     else:
                    #         rgb_features = last_hidden[:, :num_visual_tokens, :]
                    #         logger.debug(f"通过完整模型提取RGB特征: {rgb_features.shape}")
                    #     
                    #     return rgb_features
                
                except Exception as e:
                    logger.debug(f"完整模型方法失败: {e}")
                
                # 方法2: 直接调用visual（可能会有维度问题）
                # 但我们尝试处理已知的维度不匹配
                try:
                    # 尝试直接调用，但准备处理维度错误
                    rgb_features = self.model.visual(
                        pixel_values,
                        grid_thw=image_grid_thw
                    )
                    
                    # 重要：visual encoder可能返回没有batch维度的特征
                    # 根据visual_encoder_issue_analysis.md，输出是[num_patches, hidden_dim]
                    if rgb_features.dim() == 2 and batch_size > 1:
                        # Visual encoder 返回2D时，通常是单个样本的输出
                        total_elements = rgb_features.shape[0]
                        hidden_dim = rgb_features.shape[1]
                        
                        # 计算期望的patches数（最终输出，merger层后）
                        # 基于实际测试结果和正确理解：
                        # 784×1176: patch_embed输出2352(56×42)→merger降采样4倍→588 patches
                        # 392×392: patch_embed输出392(28×14)→merger降采样4倍→98 patches
                        # 注意：grid_thw传递的是patch_embed的维度，不是最终输出维度
                        if pixel_values.shape[-2] == 784 and pixel_values.shape[-1] == 1176:
                            # 784×1176全景图
                            # grid_thw=[1, 56, 42] (patch_embed输出维度)
                            # 最终输出: 2352÷4=588 patches
                            expected_patches = 588
                        elif pixel_values.shape[-2] == 392 and pixel_values.shape[-1] == 392:
                            # 392×392单图
                            # grid_thw=[1, 28, 14] (patch_embed输出维度)
                            # 最终输出: 392÷4=98 patches
                            expected_patches = 98
                        else:
                            # 其他尺寸，基于相同的处理流程
                            # 步骤1: patch_embed (patch_size=14, temporal_patch_size=2)
                            h_patches = pixel_values.shape[-2] // 14  # 高度patches
                            w_patches = pixel_values.shape[-1] // 14 // 2  # 宽度patches (temporal)
                            # 步骤2: merger层4倍降采样
                            expected_patches = (h_patches * w_patches) // 4
                        
                        if total_elements == expected_patches:
                            # Visual encoder返回了单个样本的输出
                            # 需要为每个batch样本单独处理
                            features_list = []
                            for i in range(batch_size):
                                single_pixel = pixel_values[i:i+1]
                                single_grid = image_grid_thw[i:i+1] if image_grid_thw is not None else None
                                single_feat = self.model.visual(single_pixel, grid_thw=single_grid)
                                if single_feat.dim() == 2:
                                    single_feat = single_feat.unsqueeze(0)
                                features_list.append(single_feat)
                            rgb_features = torch.cat(features_list, dim=0)
                            
                        elif total_elements == expected_patches * batch_size:
                            # Visual encoder将batch展平到序列维度
                            try:
                                rgb_features = rgb_features.view(batch_size, expected_patches, hidden_dim)
                                logger.debug(f"成功将展平的RGB特征reshape: [{total_elements}, {hidden_dim}] -> [{batch_size}, {expected_patches}, {hidden_dim}]")
                            except Exception as reshape_error:
                                logger.warning(f"无法reshape RGB特征: {reshape_error}")
                                # 降级到单样本处理
                                features_list = []
                                for i in range(batch_size):
                                    single_pixel = pixel_values[i:i+1]
                                    single_grid = image_grid_thw[i:i+1] if image_grid_thw is not None else None
                                    single_feat = self.model.visual(single_pixel, grid_thw=single_grid)
                                    if single_feat.dim() == 2:
                                        single_feat = single_feat.unsqueeze(0)
                                    features_list.append(single_feat)
                                rgb_features = torch.cat(features_list, dim=0)
                        else:
                            # 意外的输出大小
                            logger.warning(f"Visual encoder输出意外大小: {total_elements} patches")
                            logger.warning(f"期望: 单样本{expected_patches}或批次{expected_patches * batch_size}")
                            logger.warning(f"图像尺寸: {pixel_values.shape[-2]}×{pixel_values.shape[-1]}, batch_size: {batch_size}")
                            logger.warning("降级到单样本处理模式")
                            features_list = []
                            for i in range(batch_size):
                                single_pixel = pixel_values[i:i+1]
                                single_grid = image_grid_thw[i:i+1] if image_grid_thw is not None else None
                                single_feat = self.model.visual(single_pixel, grid_thw=single_grid)
                                if single_feat.dim() == 2:
                                    single_feat = single_feat.unsqueeze(0)
                                features_list.append(single_feat)
                            rgb_features = torch.cat(features_list, dim=0)
                    elif rgb_features.dim() == 2 and batch_size == 1:
                        # batch_size = 1，只需添加维度
                        rgb_features = rgb_features.unsqueeze(0)
                    elif rgb_features.dim() == 3:
                        # 已经是正确的3D格式
                        pass
                    else:
                        logger.warning(f"意外的RGB特征维度: {rgb_features.shape}")
                    
                    return rgb_features
                except RuntimeError as e:
                    if "size of tensor" in str(e) and "must match" in str(e):
                        # 这是已知的维度不匹配问题
                        logger.warning(f"Visual encoder维度不匹配: {e}")
                        # 继续尝试其他方法
                    else:
                        raise
                    
            except Exception as e:
                logger.warning(f"模型视觉特征提取失败: {e}")
        
        
        # 策略2: 使用兼容的视觉处理
        try:
            # 尝试通过基座模型的其他接口处理图像
            if hasattr(self.model, 'encode_images'):
                rgb_features = self.model.encode_images(pixel_values)
                logger.debug(f"使用基座模型encode_images接口，输出shape: {rgb_features.shape}")
                
                # 确保有batch维度
                if rgb_features.dim() == 2:
                    rgb_features = rgb_features.unsqueeze(0)
                
                return rgb_features
        except Exception as e:
            logger.warning(f"encode_images接口失败: {e}")
        
        # 策略3: 使用简化的ViT处理
        try:
            # 创建一个临时的简化ViT编码器
            rgb_features = self._create_simple_rgb_features(pixel_values)
            logger.warning("使用简化RGB特征提取")
            return rgb_features
        except Exception as e:
            logger.error(f"简化RGB特征提取失败: {e}")
        
        # 如果所有策略都失败，抛出错误而不是使用虚拟数据
        error_msg = "无法提取RGB特征：所有策略都失败了。请检查模型配置和输入数据。"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def _create_simple_rgb_features(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        创建简化的RGB特征（当基座模型视觉模块不可用时）
        """
        
        # 处理可能的5维输入 [batch_size, num_images, channels, height, width]
        if pixel_values.dim() == 5:
            batch_size, num_images, channels, height, width = pixel_values.shape
            # 合并batch和num_images维度
            pixel_values = pixel_values.view(batch_size * num_images, channels, height, width)
        elif pixel_values.dim() == 4:
            batch_size, channels, height, width = pixel_values.shape
            num_images = 1
        else:
            raise ValueError(f"Unexpected pixel_values dimensions: {pixel_values.dim()}")
        
        # 简单的平均池化和线性变换
        # 将图像分割成patches
        patch_size = self.config.patch_size
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        
        # Reshape为patches
        patches = pixel_values.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # patches shape: [batch*num_images, channels, num_patches_h, num_patches_w, patch_size, patch_size]
        
        # 重新排列维度
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # shape: [batch*num_images, num_patches_h, num_patches_w, channels, patch_size, patch_size]
        
        patches = patches.view(
            batch_size * num_images, num_patches_h * num_patches_w, channels * patch_size * patch_size
        )
        # shape: [batch*num_images, num_patches, channels * patch_size * patch_size]
        
        # 平均池化每个patch（如果需要）或直接投影
        # 由于patches已经展平，我们使用线性层来处理
        
        # 线性投影到期望的维度
        if not hasattr(self, '_simple_rgb_projector'):
            self._simple_rgb_projector = nn.Linear(
                channels * patch_size * patch_size, self.config.vision_hidden_size
            ).to(pixel_values.device)
        
        rgb_features = self._simple_rgb_projector(patches)
        
        # 确保输出是3维的 [batch_size, seq_len, hidden_dim]
        # 这里rgb_features已经是 [batch_size * num_images, num_patches, hidden_dim]
        # 需要合并为 [batch_size, total_patches, hidden_dim]
        total_patches = num_patches_h * num_patches_w
        if num_images > 1:
            # 重新整形为 [batch_size, num_images * num_patches, hidden_dim]
            rgb_features = rgb_features.view(batch_size, num_images * total_patches, -1)
        else:
            # 保持原样，只需要确保是3维
            if rgb_features.dim() == 2:
                rgb_features = rgb_features.unsqueeze(0)
        
        return rgb_features

    def _enable_gradient_checkpointing(self):
        """启用梯度检查点以减少显存占用"""
        if self.model is not None:
            # 检查基座模型是否支持梯度检查点
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("基座模型梯度检查点已启用")
            elif hasattr(self.model, 'enable_gradient_checkpointing'):
                self.model.enable_gradient_checkpointing()
                logger.info("基座模型梯度检查点已启用（使用enable_gradient_checkpointing）")
            else:
                # 手动为transformer层启用梯度检查点
                try:
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                        # 对于Qwen类模型
                        for layer in self.model.model.layers:
                            layer.gradient_checkpointing = True
                        logger.info("手动为transformer层启用梯度检查点")
                    else:
                        logger.warning("基座模型不支持梯度检查点")
                except Exception as e:
                    logger.warning(f"启用梯度检查点失败: {e}")
        
        # 为TMI模块启用梯度检查点
        if hasattr(self.tmi_module, 'enable_gradient_checkpointing'):
            self.tmi_module.enable_gradient_checkpointing()
        
        # 设置梯度检查点标志
        self.gradient_checkpointing = True
    
    def _freeze_base_model(self):
        """冻结基座模型参数"""
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("基座模型参数已冻结")
    
    def _setup_stage1_training(self):
        """Stage 1训练设置: 冻结基座模型，只训练TMI和TaskHead"""
        # 首先冻结所有参数
        for param in self.parameters():
            param.requires_grad = False
        
        # 然后解冻TMI模块和TaskHead
        if hasattr(self, 'tmi_module'):
            for param in self.tmi_module.parameters():
                param.requires_grad = True
            logger.info("TMI模块参数设置为可训练")
        
        if hasattr(self, 'task_head'):
            for param in self.task_head.parameters():
                param.requires_grad = True
            logger.info("TaskHead参数设置为可训练")
        
        # TMI辅助头也需要可训练
        if hasattr(self, 'tmi_auxiliary_head'):
            for param in self.tmi_auxiliary_head.parameters():
                param.requires_grad = True
            logger.info("TMI辅助预测头设置为可训练")
        
        logger.info("Stage 1训练设置完成: 只训练TMI、TaskHead和辅助头")
    
    def _setup_dtype(self):
        """统一设置所有模块的dtype - 优化FP16稳定性"""
        # 确定目标dtype - 优先使用基座模型的dtype
        target_dtype = None
        
        # 尝试从基座模型获取dtype
        if self.model is not None:
            # 获取模型的第一个参数的dtype
            for param in self.model.parameters():
                target_dtype = param.dtype
                break
        
        # 如果没有获取到，根据配置决定 - 优先使用BF16
        if target_dtype is None:
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                target_dtype = torch.bfloat16
                logger.info("使用BF16作为默认dtype（比FP16更稳定）")
            elif self.config.fp16 or getattr(self.config, 'torch_dtype', None) == 'float16':
                target_dtype = torch.float16
                logger.warning("使用FP16 - 可能导致数值不稳定，建议使用BF16或FP32")
            else:
                target_dtype = torch.float32
        
        # 应用dtype到所有自定义模块
        if hasattr(self, 'tmi_module'):
            self.tmi_module = self.tmi_module.to(target_dtype)
        
        if hasattr(self, 'task_head'):
            self.task_head = self.task_head.to(target_dtype)
        
        if hasattr(self, 'tmi_auxiliary_head'):
            self.tmi_auxiliary_head = self.tmi_auxiliary_head.to(target_dtype)
        
        if hasattr(self, 'lm_head'):
            self.lm_head = self.lm_head.to(target_dtype)
        
        logger.info(f"所有模块已设置为 dtype: {target_dtype}")
    
    def _print_trainable_parameters(self):
        """打印可训练参数统计"""
        trainable_params = 0
        all_param = 0
        trainable_modules = []
        
        for name, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                # 记录可训练模块
                module_name = name.split('.')[0]
                if module_name not in trainable_modules:
                    trainable_modules.append(module_name)
        
        print(f"\n=== 可训练参数统计 ===")
        print(f"总参数量: {all_param:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"可训练比例: {100 * trainable_params / all_param:.2f}%")
        print(f"可训练模块: {', '.join(trainable_modules)}")
        print(f"===================\n")
    
    def get_input_embeddings(self):
        """获取输入嵌入层"""
        if self.model is not None:
            return self.model.get_input_embeddings()
        return None
    
    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        if self.model is not None:
            self.model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        """获取输出嵌入层"""
        if self.model is not None:
            return self.model.get_output_embeddings()
        return None
    
    def set_output_embeddings(self, new_embeddings):
        """设置输出嵌入层"""
        if self.model is not None:
            self.model.set_output_embeddings(new_embeddings)
    
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None):
        """调整token嵌入大小"""
        if self.model is not None:
            return self.model.resize_token_embeddings(new_num_tokens)
        return None
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 三模态输入
        pixel_values: Optional[torch.FloatTensor] = None,  # RGB图像
        depth_pixel_values: Optional[torch.FloatTensor] = None,  # 深度图像
        semantic_pixel_values: Optional[torch.FloatTensor] = None,  # 语义图像
        image_grid_thw: Optional[torch.LongTensor] = None,
        # 任务特定参数
        task_type: str = "generation",  # ["generation", "trajectory_prediction"]
        trajectory_labels: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        前向传播
        
        Args:
            input_ids: 文本token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            pixel_values: RGB图像 [batch_size, num_images, channels, height, width]
            depth_pixel_values: 深度图像 [batch_size, num_images, 1, height, width]
            semantic_pixel_values: 语义图像 [batch_size, num_images, channels, height, width]
            task_type: 任务类型
            trajectory_labels: 轨迹标签 [batch_size, seq_len, coord_dim]
            ...
        
        Returns:
            CausalLMOutputWithPast或元组
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 对于轨迹预测任务，必须输出hidden states
        if task_type == "trajectory_prediction" and not output_hidden_states:
            output_hidden_states = True
            logger.info("轨迹预测任务自动开启output_hidden_states")
        
        # 1. 处理图像输入
        vision_outputs = None
        if pixel_values is not None:
            # 尝试从kwargs中获取深度和语义图（兼容性处理）
            if depth_pixel_values is None:
                depth_pixel_values = kwargs.get('depth_maps', None)
            if semantic_pixel_values is None:
                semantic_pixel_values = kwargs.get('semantic_maps', None)
            
            vision_outputs = self._process_vision_inputs(
                pixel_values=pixel_values,
                depth_pixel_values=depth_pixel_values,
                semantic_pixel_values=semantic_pixel_values,
                image_grid_thw=image_grid_thw
            )
            
            # 保存vision_outputs供BDD训练使用（不影响返回结构）
            self._last_vision_outputs = vision_outputs
        
        # 2. 处理文本和视觉的融合
        if vision_outputs is not None and inputs_embeds is None:
            # 检查input_ids是否存在
            if input_ids is not None:
                inputs_embeds = self._merge_vision_text_inputs(
                    input_ids=input_ids,
                    vision_outputs=vision_outputs,
                    attention_mask=attention_mask
                )
            else:
                # 如果没有input_ids，直接使用vision_outputs作为inputs_embeds
                inputs_embeds = vision_outputs
                logger.warning("input_ids为None，仅使用视觉特征作为输入")
            # 更新attention_mask以匹配新的序列长度
            if attention_mask is not None:
                vision_seq_len = vision_outputs.size(1)
                vision_mask = torch.ones(
                    attention_mask.size(0), vision_seq_len,
                    dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        
        # 3. 基座模型前向传播（支持梯度检查点）
        # 对于Qwen2_5_VLForConditionalGeneration，直接调用整个模型
        # 注意：数据已经在collator中使用左填充格式，满足Flash Attention的要求
        
        if self.flash_config.enable_gradient_checkpointing and self.training:
            # 使用梯度检查点
            from torch.utils.checkpoint import checkpoint
            
            # 定义一个包装函数来避免input_ids=None的问题
            def model_forward_wrapper(attention_mask, position_ids, past_key_values, 
                                     inputs_embeds, use_cache, output_attentions,
                                     output_hidden_states, return_dict):
                # 不再创建虚拟input_ids - 要求真实的文本输入
                # 如果没有input_ids但有inputs_embeds，这可能是合法的
                # （例如已经将文本转换为embeddings）
                # 但我们仍然需要确保这不是用虚拟数据
                if input_ids is None and inputs_embeds is None:
                    raise ValueError(
                        "必须提供input_ids或inputs_embeds之一。"
                        "不能使用完全空的输入进行训练。"
                    )
                
                # 如果只有inputs_embeds，直接传递None作为input_ids
                # 让模型自己处理
                return self.model(
                    input_ids=input_ids,  # 可能是None，但不是虚拟数据
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
            
            outputs = checkpoint(
                model_forward_wrapper,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                use_cache,
                output_attentions,
                output_hidden_states,
                True,  # return_dict
                use_reentrant=False  # 使用新的checkpoint API，性能更好
            )
        else:
            # 检查模型类型，决定如何调用
            if hasattr(self.model, '__class__') and 'ForConditionalGeneration' in self.model.__class__.__name__:
                # 对于Qwen2_5_VLForConditionalGeneration，直接调用
                # 不再创建虚拟input_ids
                if inputs_embeds is not None and input_ids is None:
                    # 如果只有embeddings没有input_ids，传递None
                    actual_input_ids = None
                else:
                    actual_input_ids = input_ids
                    
                outputs = self.model(
                    input_ids=actual_input_ids,  # 使用真实的input_ids或None
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True
                )
            else:
                # 对于其他模型类型，尝试访问内部model
                # 不再创建虚拟input_ids
                if inputs_embeds is not None and input_ids is None:
                    # 如果只有embeddings没有input_ids，这可能是合法的
                    # 但需要记录警告
                    logger.warning(
                        "没有提供input_ids，只使用inputs_embeds。"
                        "请确保这些embeddings来自真实的文本数据。"
                    )
                    actual_input_ids = None
                else:
                    actual_input_ids = input_ids
                    
                if hasattr(self.model, 'model'):
                    outputs = self.model.model(
                        input_ids=actual_input_ids,  # 使用真实的input_ids
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=True
                    )
                else:
                    outputs = self.model(
                        input_ids=actual_input_ids,  # 使用真实的input_ids
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=True
                    )
        
        # 4. 任务特定处理
        task_loss = None
        task_logits = None
        trajectory_pred = None
        tmi_auxiliary_loss = None
        
        # Stage 1训练: 添加TMI辅助损失以确保梯度流回
        if self.config.freeze_base_model and vision_outputs is not None and hasattr(self, 'tmi_auxiliary_head'):
            # 使用vision_outputs直接计算一个辅助损失
            # 这确保TMI模块参与梯度计算
            if trajectory_labels is not None:
                # 使用融合特征的平均值预测轨迹
                pooled_vision = vision_outputs.mean(dim=1)  # [batch_size, hidden_size]
                
                # 确保输入特征的dtype与辅助头一致
                # 获取辅助头的第一个模块（LayerNorm）的dtype
                if hasattr(self.tmi_auxiliary_head[0], 'weight'):
                    target_dtype = self.tmi_auxiliary_head[0].weight.dtype
                    if pooled_vision.dtype != target_dtype:
                        pooled_vision = pooled_vision.to(target_dtype)
                
                # 使用独立的TMI辅助头预测轨迹
                # 这个头专门用于监督TMI特征的质量
                
                aux_trajectory_logits = self.tmi_auxiliary_head(pooled_vision)
                aux_trajectory_pred = aux_trajectory_logits.view(-1, 36, 3)
                
                # 确保trajectory_labels也是正确的dtype
                if trajectory_labels.dtype != aux_trajectory_pred.dtype:
                    trajectory_labels = trajectory_labels.to(aux_trajectory_pred.dtype)
                    
                # 直接计算损失，不进行人工标准化
                # 使用Huber Loss，对异常值更稳健
                loss_fct = nn.SmoothL1Loss()
                tmi_auxiliary_loss = loss_fct(aux_trajectory_pred, trajectory_labels)
                
                # 使用较小的权重作为辅助损失
                tmi_auxiliary_loss = tmi_auxiliary_loss * 0.1  # 使用0.1的权重作为辅助损失
                
                # 确保损失是float32用于backward - 这对FP16训练非常重要
                if tmi_auxiliary_loss.dtype != torch.float32:
                    tmi_auxiliary_loss = tmi_auxiliary_loss.float()
                
                # 添加损失裁剪，防止梯度爆炸
                tmi_auxiliary_loss = torch.clamp(tmi_auxiliary_loss, max=10.0)
                
        
        if task_type == "trajectory_prediction":
            # 为了获取hidden states，需要确保output_hidden_states=True
            # 但这应该在调用时设置，这里我们尝试从outputs中获取
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1]  # 使用最后一层的hidden states
            elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                hidden_states = outputs.last_hidden_state
            else:
                # 如果没有hidden states，说明配置有问题
                raise ValueError(
                    "无法获取hidden states用于轨迹预测。"
                    "请确保在调用模型时设置output_hidden_states=True"
                )
            
            # 检查hidden states是否包含NaN
            if torch.isnan(hidden_states).any():
                logger.error("LLM hidden states contain NaN!")
                logger.error(f"  hidden_states shape: {hidden_states.shape}")
                logger.error(f"  hidden_states stats: min={hidden_states.min():.4f}, max={hidden_states.max():.4f}, mean={hidden_states.mean():.4f}")
            
            task_outputs = self.task_head(
                hidden_states=hidden_states,
                task_type=task_type,
                labels=trajectory_labels
            )
            
            # 从task_outputs字典中提取结果
            task_loss = task_outputs.get("loss", None)
            task_logits = task_outputs.get("logits", None)
            trajectory_pred = task_outputs.get("trajectory_pred", None)
        
        # 5. 计算总损失
        total_loss = None
        if labels is not None:
            # 语言建模损失
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss = lm_loss
            
            # 添加任务损失
            if task_loss is not None:
                total_loss = total_loss + task_loss
        elif trajectory_labels is not None and task_loss is not None:
            # 如果没有文本标签但有轨迹标签，使用轨迹预测损失
            total_loss = task_loss
            
            # 添加TMI辅助损失（Stage 1训练）
            if tmi_auxiliary_loss is not None:
                total_loss = total_loss + tmi_auxiliary_loss
            
            # 确保loss是Float32用于反向传播
            if total_loss.dtype == torch.float16:
                total_loss = total_loss.float()
        
        # 如果既没有文本标签也没有轨迹标签
        if total_loss is None:
            # 在生成模式下这是正常的，不需要标签
            if use_cache or (past_key_values is not None):
                # 生成模式，不需要标签
                pass  
            else:
                # 训练/评估模式，应该有标签
                logger.debug("没有提供任何标签（labels或trajectory_labels），无法计算损失。")
            # 打印调试信息
        
        # 最终确保loss是float32用于backward
        if total_loss is not None and total_loss.dtype != torch.float32:
            total_loss = total_loss.float()
        
        if return_dict:
            # 创建一个新的输出对象，包含所有必要的字段
            return CausalLMOutputWithPast(
                loss=total_loss,
                logits=outputs.logits if hasattr(outputs, 'logits') else None,
                past_key_values=outputs.past_key_values if hasattr(outputs, 'past_key_values') else None,
                hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            )
        else:
            output_tuple = (outputs.logits,) + outputs[1:]
            return (total_loss,) + output_tuple if total_loss is not None else output_tuple
    
    def _process_vision_inputs(
        self,
        pixel_values: torch.FloatTensor,
        depth_pixel_values: Optional[torch.FloatTensor] = None,
        semantic_pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """处理视觉输入"""
        
        batch_size = pixel_values.size(0)
        
        # 调试信息
        
        # 处理RGB图像 - 使用基座模型的视觉编码器
        rgb_features = self._extract_rgb_features(pixel_values, image_grid_thw)
        
        # 打印RGB特征形状和统计信息
        
        # 检查RGB特征是否包含NaN
        if torch.isnan(rgb_features).any():
            logger.error("RGB features contain NaN after extraction!")
            logger.error(f"  rgb_features shape: {rgb_features.shape}")
            logger.error(f"  rgb_features stats: min={rgb_features.min():.4f}, max={rgb_features.max():.4f}, mean={rgb_features.mean():.4f}")
            logger.error(f"  pixel_values had NaN: {torch.isnan(pixel_values).any()}")
        
        # 检查是否有深度和语义输入
        if depth_pixel_values is None or semantic_pixel_values is None:
            # 如果缺少模态，创建零填充
            if depth_pixel_values is None:
                depth_pixel_values = torch.zeros(
                    batch_size, 1, self.config.image_size, self.config.image_size,
                    device=pixel_values.device, dtype=pixel_values.dtype
                )
            if semantic_pixel_values is None:
                semantic_pixel_values = torch.zeros(
                    batch_size, 150, self.config.image_size, self.config.image_size,
                    device=pixel_values.device, dtype=pixel_values.dtype
                )
        
        # 打印各模态输入形状以调试
        
        # TMI模块会内部处理dtype转换
        # 输入可以是float16，TMI内部会转换为float32计算，然后转回原始dtype
        
        # 使用TMI模块融合三模态特征
        fused_features = self.tmi_module(
            rgb_features=rgb_features,
            depth_pixel_values=depth_pixel_values,
            semantic_pixel_values=semantic_pixel_values
        )
        
        # FP16稳定性检查
        if torch.isnan(fused_features).any():
            print("TMI module output contains NaN!")
            print(f"  fused_features stats: min={fused_features.min():.4f}, max={fused_features.max():.4f}, mean={fused_features.mean():.4f}")
            print(f"  rgb_features had NaN: {torch.isnan(rgb_features).any()}")
            print(f"  depth_pixel_values had NaN: {torch.isnan(depth_pixel_values).any() if depth_pixel_values is not None else 'N/A'}")
            print(f"  semantic_pixel_values had NaN: {torch.isnan(semantic_pixel_values).any() if semantic_pixel_values is not None else 'N/A'}")
        
        # 检查TMI输出是否有NaN
        if torch.isnan(fused_features).any():
            logger.error("TMI module output contains NaN!")
            logger.error(f"  fused_features stats: min={fused_features.min():.4f}, max={fused_features.max():.4f}, mean={fused_features.mean():.4f}")
            logger.error(f"  rgb_features had NaN: {torch.isnan(rgb_features).any()}")
            logger.error(f"  depth_pixel_values had NaN: {torch.isnan(depth_pixel_values).any()}")
            logger.error(f"  semantic_pixel_values had NaN: {torch.isnan(semantic_pixel_values).any()}")
        
        return fused_features
    
    def _merge_vision_text_inputs(
        self,
        input_ids: torch.LongTensor,
        vision_outputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """合并视觉和文本输入"""
        
        # 获取文本嵌入
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # 拼接视觉特征和文本嵌入
        # 视觉特征在前，文本在后
        merged_embeds = torch.cat([vision_outputs, inputs_embeds], dim=1)
        
        return merged_embeds
    
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor = None,
        stopping_criteria = None,
        prefix_allowed_tokens_fn = None,
        synced_gpus: Optional[bool] = None,
        assistant_model = None,
        streamer = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """生成方法，支持三模态输入"""
        
        # 调用基座模型的生成方法
        return super().generate(
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            **kwargs
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        """为生成准备输入"""
        
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            
            # 如果input_ids长度超过past_length，只保留最后一个token
            if input_ids is not None and input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]
        
        # 构建模型输入
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
        }
        
        # 添加三模态输入
        for key in ["pixel_values", "depth_pixel_values", "semantic_pixel_values", "image_grid_thw"]:
            if key in kwargs:
                model_inputs[key] = kwargs[key]
        
        return model_inputs
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """重新排序缓存（用于beam search）"""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
    
    def enable_input_require_grads(self):
        """启用输入梯度（用于LoRA等参数高效微调）"""
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        
        self.get_input_embeddings().register_forward_hook(make_inputs_require_grads)
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """获取模型内存占用情况"""
        
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        for param in self.parameters():
            param_size = param.numel()
            total_params += param_size
            
            if param.requires_grad:
                trainable_params += param_size
            else:
                frozen_params += param_size
        
        # 转换为GB
        param_size_gb = total_params * 4 / (1024 ** 3)  # 假设float32
        
        memory_usage = get_memory_usage()
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'param_size_gb': param_size_gb,
            'gpu_memory_allocated_gb': memory_usage['allocated'],
            'gpu_memory_cached_gb': memory_usage['cached'],
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0
        }
    
    def print_memory_summary(self):
        """打印内存使用摘要"""
        
        memory_info = self.get_memory_footprint()
        
        print("=" * 60)
        print("模型内存使用摘要")
        print("=" * 60)
        print(f"总参数量: {memory_info['total_params']:,}")
        print(f"可训练参数: {memory_info['trainable_params']:,}")
        print(f"冻结参数: {memory_info['frozen_params']:,}")
        print(f"可训练比例: {memory_info['trainable_ratio']:.2%}")
        print(f"参数内存占用: {memory_info['param_size_gb']:.2f} GB")
        
        if torch.cuda.is_available():
            print(f"GPU内存已分配: {memory_info['gpu_memory_allocated_gb']:.2f} GB")
            print(f"GPU内存缓存: {memory_info['gpu_memory_cached_gb']:.2f} GB")
        
        print("=" * 60)
    
    def optimize_for_inference(self):
        """为推理优化模型"""
        
        self.eval()
        
        # 启用推理优化
        if hasattr(self, 'flash_config'):
            self.flash_config.enable_gradient_checkpointing = False
        
        # 融合BatchNorm和Conv（如果适用）
        try:
            from torch.jit import optimize_for_inference
            self = optimize_for_inference(self)
            logger.info("✅ 推理优化已应用")
        except:
            logger.warning("推理优化不可用")
        
        return self
    
    def enable_flash_attention_optimization(self):
        """启用Flash Attention优化"""
        
        if not hasattr(self, 'flash_config'):
            self.flash_config = FlashAttentionConfig()
        
        self.flash_config.enable_flash_attention = True
        
        # 重新应用优化
        if self.model is not None:
            self.model = apply_flash_attention_optimization(
                self.model, 
                self.flash_config
            )
        
        logger.info("✅ Flash Attention优化已启用")


class TaskHead(nn.Module):
    """任务特定的输出头"""
    
    def __init__(self, config: TriModalQwenConfig):
        super().__init__()
        
        self.config = config
        
        # LayerNorm用于稳定特征
        self.trajectory_norm = nn.LayerNorm(config.llm_hidden_size)
        
        # 轨迹预测头
        self.trajectory_head = nn.Sequential(
            nn.Linear(config.llm_hidden_size, config.llm_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.llm_hidden_size // 2, config.llm_hidden_size // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.llm_hidden_size // 4, 108)  # 36个未来waypoints，每个3D坐标(x,y,heading)
        )
        
        # 初始化权重以防止梯度爆炸
        self._init_weights()
        
        # 分类头（用于场景理解等）
        self.classification_head = nn.Sequential(
            nn.Linear(config.llm_hidden_size, config.llm_hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.llm_hidden_size // 2, 10)  # 10个场景类别
        )
    
    def _init_weights(self):
        """保守的初始化权重，避免NaN"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用更保守的正态分布初始化
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # 对最后一层使用更小的初始化，确保初始输出在合理范围
        if hasattr(self.trajectory_head[-1], 'weight'):
            nn.init.normal_(self.trajectory_head[-1].weight, mean=0.0, std=0.01)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        task_type: str = "generation",
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """任务头前向传播"""
        
        outputs = {}
        
        if task_type == "trajectory_prediction":
            # 使用全局平均池化而不是只用最后一个token，获取全局信息
            # hidden_states: [batch_size, seq_len, hidden_size]
            pooled_hidden = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
            
            # 确保LayerNorm的dtype与输入一致
            if self.trajectory_norm.weight.dtype != pooled_hidden.dtype:
                self.trajectory_norm = self.trajectory_norm.to(pooled_hidden.dtype)
            
            # 检查输入是否有NaN
            if torch.isnan(pooled_hidden).any():
                logger.error(f"pooled_hidden contains NaN before LayerNorm!")
                logger.error(f"  pooled_hidden stats: min={pooled_hidden.min():.4f}, max={pooled_hidden.max():.4f}, mean={pooled_hidden.mean():.4f}")
            
            # 添加LayerNorm来稳定特征
            pooled_hidden = self.trajectory_norm(pooled_hidden)
            
            # 检查LayerNorm后是否有NaN
            if torch.isnan(pooled_hidden).any():
                logger.error(f"pooled_hidden contains NaN after LayerNorm!")
                logger.error(f"  LayerNorm weight mean: {self.trajectory_norm.weight.mean():.4f}")
                logger.error(f"  LayerNorm bias mean: {self.trajectory_norm.bias.mean():.4f}")
            
            # 确保dtype一致性 - 将输入转换为与trajectory_head权重相同的dtype
            if hasattr(self.trajectory_head[0], 'weight'):
                target_dtype = self.trajectory_head[0].weight.dtype
                if pooled_hidden.dtype != target_dtype:
                    pooled_hidden = pooled_hidden.to(target_dtype)
            
            trajectory_logits = self.trajectory_head(pooled_hidden)  # [batch_size, 108]
            
            # 检查输出是否有NaN
            if torch.isnan(trajectory_logits).any():
                logger.error(f"trajectory_logits contains NaN!")
                # 检查每一层的权重
                for i, layer in enumerate(self.trajectory_head):
                    if hasattr(layer, 'weight'):
                        logger.error(f"  Layer {i} weight stats: min={layer.weight.min():.4f}, max={layer.weight.max():.4f}, mean={layer.weight.mean():.4f}")
                        if hasattr(layer, 'bias') and layer.bias is not None:
                            logger.error(f"  Layer {i} bias stats: min={layer.bias.min():.4f}, max={layer.bias.max():.4f}, mean={layer.bias.mean():.4f}")
            
            # reshape为轨迹格式 [batch_size, num_waypoints, coord_dim]
            trajectory_pred = trajectory_logits.view(-1, 36, 3)  # x, y, heading
            
            outputs["trajectory_pred"] = trajectory_pred
            outputs["logits"] = trajectory_logits
            
            # 计算轨迹预测损失
            if labels is not None:
                # 直接计算损失，不进行人工标准化
                # 使用Huber Loss，对异常值更稳健
                loss_fct = nn.SmoothL1Loss()  # Huber Loss
                trajectory_loss = loss_fct(trajectory_pred, labels)
                
                # 只检查NaN/Inf，不做人工干预
                if torch.isnan(trajectory_loss) or torch.isinf(trajectory_loss):
                    logger.error(f"TaskHead loss is NaN/Inf!")
                    logger.error(f"  trajectory_pred stats: min={trajectory_pred.min():.4f}, max={trajectory_pred.max():.4f}, mean={trajectory_pred.mean():.4f}")
                    logger.error(f"  labels stats: min={labels.min():.4f}, max={labels.max():.4f}, mean={labels.mean():.4f}")
                    raise ValueError("Loss is NaN or Inf, stopping training")
                
                # 确保loss是Float32用于反向传播（即使模型使用混合精度）
                if trajectory_loss.dtype != torch.float32:
                    trajectory_loss = trajectory_loss.float()
                
                outputs["loss"] = trajectory_loss
        
        elif task_type == "classification":
            # 场景分类
            pooled_hidden = hidden_states.mean(dim=1)  # 平均池化
            classification_logits = self.classification_head(pooled_hidden)
            
            outputs["logits"] = classification_logits
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                classification_loss = loss_fct(classification_logits, labels)
                outputs["loss"] = classification_loss
        
        return outputs


# 注册模型（兼容不同版本的transformers）
try:
    # 尝试使用新版本的注册方式
    if hasattr(AutoConfig, 'register'):
        AutoConfig.register("tri_modal_qwen", TriModalQwenConfig)
    if hasattr(AutoModelForCausalLM, 'register'):
        AutoModelForCausalLM.register("tri_modal_qwen", TriModalQwenConfig, TriModalQwenForCausalLM)
    logger.info("✅ 模型注册成功")
except AttributeError as e:
    # 兼容旧版本或其他注册方式
    logger.warning(f"模型注册失败: {e}")
    # 尝试其他注册方式
    try:
        from transformers import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING
        if hasattr(CONFIG_MAPPING, '_extra_content'):
            CONFIG_MAPPING._extra_content["tri_modal_qwen"] = TriModalQwenConfig
        if hasattr(MODEL_FOR_CAUSAL_LM_MAPPING, '_extra_content'):
            MODEL_FOR_CAUSAL_LM_MAPPING._extra_content[TriModalQwenConfig] = TriModalQwenForCausalLM
        logger.info("✅ 使用备用方式注册模型成功")
    except Exception as e2:
        logger.warning(f"备用注册方式也失败: {e2}")
        pass


# 测试函数
def test_tri_modal_qwen_model():
    """测试三模态Qwen模型"""
    
    # 创建配置
    config = TriModalQwenConfig(
        base_model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
        fusion_hidden_size=1024,
        fusion_num_layers=2,
        fusion_type="mamba"
    )
    
    try:
        # 创建模型
        model = TriModalQwenForCausalLM(config)
        print(f"模型创建成功")
        
        # 创建模拟输入
        batch_size = 2
        seq_len = 128
        img_size = 392
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        pixel_values = torch.randn(batch_size, 3, img_size, img_size)
        depth_pixel_values = torch.randn(batch_size, 1, img_size, img_size)
        semantic_pixel_values = torch.randn(batch_size, 150, img_size, img_size)
        
        # 前向传播测试
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                depth_pixel_values=depth_pixel_values,
                semantic_pixel_values=semantic_pixel_values,
                task_type="generation"
            )
        
        print(f"模型输出logits形状: {outputs.logits.shape}")
        
        # 测试轨迹预测任务
        trajectory_labels = torch.randn(batch_size, 36, 3)  # 36个waypoints，每个(x,y,heading)
        
        outputs_traj = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            depth_pixel_values=depth_pixel_values,
            semantic_pixel_values=semantic_pixel_values,
            task_type="trajectory_prediction",
            trajectory_labels=trajectory_labels
        )
        
        print(f"轨迹预测输出形状: {outputs_traj.trajectory_pred.shape}")
        print(f"轨迹预测损失: {outputs_traj.task_loss.item() if outputs_traj.task_loss is not None else 'None'}")
        
        print("三模态Qwen模型测试成功！")
        
    except Exception as e:
        print(f"模型测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_tri_modal_qwen_model()