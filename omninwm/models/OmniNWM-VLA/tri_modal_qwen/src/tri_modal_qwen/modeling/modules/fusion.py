"""
Mamba融合核心模块

实现基于Mamba的多模态序列融合：
- MambaFusionCore: 核心融合模块，O(N)复杂度处理长序列
- AttentionFusionCore: 基于注意力的融合模块（备选）
- LinearFusionCore: 简单线性融合模块（基线）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("警告: mamba_ssm未安装，将使用Attention替代Mamba")

try:
    from transformers import MambaForCausalLM, MambaConfig
    PRETRAINED_MAMBA_AVAILABLE = True
except ImportError:
    PRETRAINED_MAMBA_AVAILABLE = False
    print("警告: transformers版本过低或未安装MambaForCausalLM支持")


class MambaFusionCore(nn.Module):
    """基于Mamba的融合核心
    
    使用状态空间模型进行高效的序列融合，复杂度为O(N)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        if not MAMBA_AVAILABLE:
            # 如果Mamba不可用，使用Attention作为后备
            print("Mamba不可用，使用AttentionFusionCore作为后备")
            self.fusion_layers = AttentionFusionCore(
                hidden_size, num_layers, dropout, use_flash_attention
            ).fusion_layers
        else:
            # 构建Mamba层
            self.fusion_layers = nn.ModuleList([
                MambaBlock(hidden_size, dropout) for _ in range(num_layers)
            ])
        
        # 输入层归一化
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: 输入序列 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len] (可选)
        
        Returns:
            融合后的特征序列 [batch_size, seq_len, hidden_size]
        """
        
        # 输入归一化
        x = self.input_norm(hidden_states)
        
        # 通过融合层
        for layer in self.fusion_layers:
            if isinstance(layer, MambaBlock):
                x = layer(x)
            else:
                # Attention层需要attention_mask
                x = layer(x, attention_mask=attention_mask)
        
        # 输出投影
        output = self.output_proj(x)
        
        # 残差连接
        output = output + hidden_states
        
        return output


class PretrainedMambaFusionCore(nn.Module):
    """使用预训练Mamba模型的融合核心（类似SSR的MIDI）
    
    使用预训练的MambaForCausalLM来处理多模态融合
    优化版本：如果mamba-ssm可用，使用其CUDA加速
    
    基于SSR改进：
    1. 增加Mamba层数到12-16层
    2. 采用两步投影策略减少信息损失
    3. 添加可学习的中间表示token
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 12,  # SSR风格：使用更多层（默认12层）
        dropout: float = 0.1,
        use_flash_attention: bool = True,
        pretrained_model_name: str = "state-spaces/mamba-130m-hf"
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        
        # 检查是否可以使用优化路径
        use_optimized = MAMBA_AVAILABLE and PRETRAINED_MAMBA_AVAILABLE
        
        if not PRETRAINED_MAMBA_AVAILABLE:
            print("警告: 预训练Mamba不可用，降级到标准MambaFusionCore")
            # 降级到原始实现
            self.use_pretrained = False
            self.fusion_core = MambaFusionCore(
                hidden_size, num_layers, dropout, use_flash_attention
            )
        elif use_optimized and MAMBA_AVAILABLE:  # 启用优化路径，权重兼容性问题已解决
            # 优化路径：加载预训练权重但使用mamba-ssm的CUDA加速
            self.use_pretrained = True
            self.use_optimized = True
            
            print(f"加载预训练Mamba模型（优化版）: {pretrained_model_name}")
            try:
                # 先加载预训练模型获取权重
                from transformers import MambaForCausalLM
                pretrained_model = MambaForCausalLM.from_pretrained(
                    pretrained_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                self.mamba_hidden_size = pretrained_model.config.hidden_size  # 通常是768
                total_layers = pretrained_model.config.num_hidden_layers  # 24层
                
                # SSR极简投影设计：无LayerNorm，保持特征分布
                self.input_proj = nn.Sequential(
                    nn.Linear(hidden_size, self.mamba_hidden_size),  # 直接投影
                    nn.GELU()  # 只用激活函数
                    # 去掉LayerNorm和Dropout，学习SSR的设计
                )
                
                # 输出投影也采用极简设计
                self.output_proj = nn.Sequential(
                    nn.Linear(self.mamba_hidden_size, hidden_size),  # 直接投影回
                    nn.GELU()
                    # 极简设计：无LayerNorm，无Dropout
                )
                
                # SSR风格：添加可学习的中间表示token（类似TOR token）
                self.num_intermediate_tokens = 10  # 10个中间表示
                self.intermediate_tokens = nn.Parameter(
                    torch.randn(self.num_intermediate_tokens, self.mamba_hidden_size) * 0.02
                )
                
                # SSR改进：为串联的模态添加位置编码，帮助区分不同模态
                # 当输入是串联的3个模态时，需要知道每个部分来自哪个模态
                self.modality_position_embeddings = nn.Parameter(
                    torch.randn(3, self.mamba_hidden_size) * 0.02  # 3个模态的位置编码
                )
                
                # 使用mamba-ssm的Mamba层（有CUDA加速）
                from mamba_ssm import Mamba
                self.mamba_layers = nn.ModuleList()
                self.layer_norms = nn.ModuleList()  # 添加层归一化列表
                
                # SSR策略：使用中后部的层（第8层到第20层，共12层）
                # 这些层已经学习了较好的序列建模能力
                if num_layers <= 16:
                    # 使用中后部的连续层
                    start_layer = min(8, total_layers - num_layers)  # 从第8层开始
                    end_layer = min(start_layer + num_layers, total_layers)
                else:
                    # 如果需要更多层，从更早开始
                    start_layer = max(0, total_layers - num_layers)
                    end_layer = total_layers
                
                layer_indices = list(range(start_layer, end_layer))
                print(f"使用Mamba层 {start_layer} 到 {end_layer-1}（共{len(layer_indices)}层）")
                
                for i in layer_indices:
                    # 创建mamba-ssm的Mamba层
                    mamba_layer = Mamba(
                        d_model=self.mamba_hidden_size,
                        d_state=16,  # Mamba-130m的配置
                        d_conv=4,
                        expand=2,
                    )
                    
                    # 从预训练模型复制权重
                    if hasattr(pretrained_model, 'backbone') and hasattr(pretrained_model.backbone, 'layers'):
                        pretrained_layer = pretrained_model.backbone.layers[i]
                        # 复制权重（如果结构兼容）
                        if hasattr(pretrained_layer, 'mixer'):
                            try:
                                pretrained_mixer = pretrained_layer.mixer
                                # 完整复制所有7个参数，避免NaN问题
                                
                                # 1. in_proj: 输入投影
                                if hasattr(pretrained_mixer, 'in_proj') and hasattr(mamba_layer, 'in_proj'):
                                    mamba_layer.in_proj.weight.data = pretrained_mixer.in_proj.weight.data.clone()
                                
                                # 2. conv1d: 因果卷积（包括权重和偏置）
                                if hasattr(pretrained_mixer, 'conv1d') and hasattr(mamba_layer, 'conv1d'):
                                    mamba_layer.conv1d.weight.data = pretrained_mixer.conv1d.weight.data.clone()
                                    if hasattr(pretrained_mixer.conv1d, 'bias') and pretrained_mixer.conv1d.bias is not None:
                                        mamba_layer.conv1d.bias.data = pretrained_mixer.conv1d.bias.data.clone()
                                
                                # 3. x_proj: 选择性扫描投影
                                if hasattr(pretrained_mixer, 'x_proj') and hasattr(mamba_layer, 'x_proj'):
                                    mamba_layer.x_proj.weight.data = pretrained_mixer.x_proj.weight.data.clone()
                                
                                # 4. dt_proj: 时间步投影（包括权重和偏置）
                                if hasattr(pretrained_mixer, 'dt_proj') and hasattr(mamba_layer, 'dt_proj'):
                                    mamba_layer.dt_proj.weight.data = pretrained_mixer.dt_proj.weight.data.clone()
                                    if hasattr(pretrained_mixer.dt_proj, 'bias') and pretrained_mixer.dt_proj.bias is not None:
                                        mamba_layer.dt_proj.bias.data = pretrained_mixer.dt_proj.bias.data.clone()
                                
                                # 5. A_log: 状态转换矩阵（对数形式）
                                if hasattr(pretrained_mixer, 'A_log') and hasattr(mamba_layer, 'A_log'):
                                    mamba_layer.A_log.data = pretrained_mixer.A_log.data.clone()
                                
                                # 6. D: 跳跃连接参数
                                if hasattr(pretrained_mixer, 'D') and hasattr(mamba_layer, 'D'):
                                    mamba_layer.D.data = pretrained_mixer.D.data.clone()
                                
                                # 7. out_proj: 输出投影
                                if hasattr(pretrained_mixer, 'out_proj') and hasattr(mamba_layer, 'out_proj'):
                                    mamba_layer.out_proj.weight.data = pretrained_mixer.out_proj.weight.data.clone()
                                
                                print(f"成功复制层{i}的全部7个核心参数")
                            except Exception as e:
                                print(f"层{i}权重复制失败（将使用随机初始化）: {e}")
                    
                    self.mamba_layers.append(mamba_layer)
                
                # SSR优化设计：创建3个LayerNorm（每6层一个）
                self.layer_norms = nn.ModuleList()
                # 创建3个LayerNorm，每6层使用一个
                for j in range(3):
                    layer_norm = nn.LayerNorm(self.mamba_hidden_size)
                    # 使用对应层的norm权重初始化
                    layer_idx = start_layer + j * 6  # 每6层一个
                    if layer_idx < end_layer and layer_idx < len(pretrained_model.backbone.layers):
                        pretrained_layer = pretrained_model.backbone.layers[layer_idx]
                        if hasattr(pretrained_layer, 'norm'):
                            try:
                                layer_norm.weight.data = pretrained_layer.norm.weight.data.clone()
                                print(f"创建第{j}个LayerNorm（对应层{layer_idx}）")
                            except Exception as e:
                                print(f"norm权重复制失败: {e}")
                    self.layer_norms.append(layer_norm)
                
                # 最终层归一化
                self.norm = nn.LayerNorm(self.mamba_hidden_size)
                
                # 添加动态中间表示token生成器（更灵活的设计）
                self.num_intermediate_tokens = 10
                # 使用可学习的查询向量，根据输入动态生成中间token
                self.intermediate_query = nn.Parameter(
                    torch.randn(self.num_intermediate_tokens, self.mamba_hidden_size) * 0.02
                )
                # 简单的注意力机制来生成动态token
                self.intermediate_generator = nn.Sequential(
                    nn.Linear(self.mamba_hidden_size, self.mamba_hidden_size),
                    nn.SiLU(),
                    nn.Dropout(dropout * 0.3)
                )
                
                # 删除原始的慢速模型以节省内存
                del pretrained_model
                
                print(f"成功加载预训练权重并使用mamba-ssm加速（使用{len(self.mamba_layers)}层）")
                
            except Exception as e:
                print(f"优化加载失败，降级到标准版本: {e}")
                self.use_optimized = False
                # 降级到慢速版本
                self.mamba = MambaForCausalLM.from_pretrained(
                    pretrained_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                self.mamba_hidden_size = self.mamba.config.hidden_size
                
                # SSR极简投影设计
                self.input_proj = nn.Sequential(
                    nn.Linear(hidden_size, self.mamba_hidden_size),
                    nn.GELU()
                )
                
                self.output_proj = nn.Sequential(
                    nn.Linear(self.mamba_hidden_size, hidden_size),
                    nn.GELU()
                )
                
                # 添加动态中间表示token生成器（更灵活的设计）
                self.num_intermediate_tokens = 10
                # 使用可学习的查询向量，根据输入动态生成中间token
                self.intermediate_query = nn.Parameter(
                    torch.randn(self.num_intermediate_tokens, self.mamba_hidden_size) * 0.02
                )
                # 简单的注意力机制来生成动态token
                self.intermediate_generator = nn.Sequential(
                    nn.Linear(self.mamba_hidden_size, self.mamba_hidden_size),
                    nn.SiLU(),
                    nn.Dropout(dropout * 0.3)
                )
                
                # 冻结Mamba的大部分参数，只微调最后几层
                self._setup_mamba_freezing()
                
            except Exception as e:
                print(f"加载预训练Mamba失败: {e}")
                print("降级到标准MambaFusionCore")
                self.use_pretrained = False
                self.fusion_core = MambaFusionCore(
                    hidden_size, num_layers, dropout, use_flash_attention
                )
        else:
            # 标准路径：使用慢速但稳定的预训练MambaForCausalLM
            self.use_pretrained = True
            self.use_optimized = False
            
            print(f"加载预训练Mamba模型: {pretrained_model_name}")
            try:
                self.mamba = MambaForCausalLM.from_pretrained(
                    pretrained_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32  # 保持FP32避免梯度爆炸
                )
                self.mamba_hidden_size = self.mamba.config.hidden_size  # 通常是768
                
                # SSR极简投影设计
                self.input_proj = nn.Sequential(
                    nn.Linear(hidden_size, self.mamba_hidden_size),
                    nn.GELU()
                )
                
                self.output_proj = nn.Sequential(
                    nn.Linear(self.mamba_hidden_size, hidden_size),
                    nn.GELU()
                )
                
                # 添加动态中间表示token生成器（更灵活的设计）
                self.num_intermediate_tokens = 10
                # 使用可学习的查询向量，根据输入动态生成中间token
                self.intermediate_query = nn.Parameter(
                    torch.randn(self.num_intermediate_tokens, self.mamba_hidden_size) * 0.02
                )
                # 简单的注意力机制来生成动态token
                self.intermediate_generator = nn.Sequential(
                    nn.Linear(self.mamba_hidden_size, self.mamba_hidden_size),
                    nn.SiLU(),
                    nn.Dropout(dropout * 0.3)
                )
                
                # 冻结Mamba的大部分参数，只微调最后几层
                self._setup_mamba_freezing()
                
            except Exception as e:
                print(f"加载预训练Mamba失败: {e}")
                print("降级到标准MambaFusionCore")
                self.use_pretrained = False
                self.fusion_core = MambaFusionCore(
                    hidden_size, num_layers, dropout, use_flash_attention
                )
    
    def _setup_mamba_freezing(self):
        """冻结Mamba的前面层，只训练最后几层
        
        基于SSR策略：分阶段训练
        - 第一阶段：只训练投影层
        - 第二阶段：微调后面的Mamba层
        """
        if hasattr(self.mamba, 'backbone') and hasattr(self.mamba.backbone, 'layers'):
            layers = self.mamba.backbone.layers
            total_layers = len(layers)
            
            # SSR策略：冻结前60%的层，微调后40%
            # 这允许模型适应新的任务同时保留预训练知识
            freeze_until = int(total_layers * 0.6)
            for i, layer in enumerate(layers[:freeze_until]):
                for param in layer.parameters():
                    param.requires_grad = False
            
            print(f"冻结了Mamba的前{freeze_until}/{total_layers}层，将微调后{total_layers-freeze_until}层")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: 输入序列 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len] (可选)
        
        Returns:
            融合后的特征序列 [batch_size, seq_len, hidden_size]
        """
        
        if not self.use_pretrained:
            # 使用标准实现
            return self.fusion_core(hidden_states, attention_mask)
        
        # 保存原始dtype、设备和输入用于残差连接
        original_dtype = hidden_states.dtype
        original_device = hidden_states.device
        original_input = hidden_states  # 保存原始输入用于残差连接
        
        # 投影到Mamba的维度
        projected_states = self.input_proj(hidden_states)
        
        # SSR改进：处理串联的模态特征
        # 假设输入是3个模态串联的：[batch, 3*seq_len, hidden_size]
        batch_size, total_seq_len, _ = projected_states.shape
        
        # 检测是否是串联的多模态输入（序列长度是3的倍数）
        if total_seq_len % 3 == 0 and hasattr(self, 'modality_position_embeddings'):
            seq_len_per_modality = total_seq_len // 3
            
            # 分割成3个模态
            modality_features = projected_states.reshape(batch_size, 3, seq_len_per_modality, -1)
            
            # 添加模态位置编码（帮助Mamba区分不同模态）
            modality_pos_emb = self.modality_position_embeddings.unsqueeze(1).expand(-1, seq_len_per_modality, -1)
            modality_pos_emb = modality_pos_emb.unsqueeze(0).expand(batch_size, -1, -1, -1)
            
            # 加权添加位置编码（避免破坏特征）
            modality_features = modality_features + 0.1 * modality_pos_emb.to(modality_features.dtype)
            
            # 重新reshape回串联形式
            projected_states = modality_features.reshape(batch_size, total_seq_len, -1)
        
        # SSR风格：动态生成中间表示token
        if hasattr(self, 'intermediate_query'):
            # 从输入序列生成上下文向量（平均池化）
            context_vector = projected_states.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
            
            # 使用查询向量和上下文生成动态的中间token
            intermediate_query_expanded = self.intermediate_query.unsqueeze(0).expand(
                batch_size, -1, -1
            ).to(projected_states.dtype)
            
            # 通过生成器网络产生动态token
            intermediate_tokens = self.intermediate_generator(intermediate_query_expanded)
            # 添加上下文信息（简单的加性组合）
            intermediate_tokens = intermediate_tokens + 0.1 * context_vector
            
            # 将中间token插入到序列中（在序列末尾添加）
            projected_states = torch.cat([
                projected_states, 
                intermediate_tokens
            ], dim=1)
        
        # 检查是否使用优化版本
        if hasattr(self, 'use_optimized') and self.use_optimized:
            # 优化路径：使用mamba-ssm的CUDA加速层
            x = projected_states
            
            # SSR风格：最小化LayerNorm使用，但确保数值稳定
            for i, mamba_layer in enumerate(self.mamba_layers):
                # 每6层应用一次LayerNorm（总共3次）
                if i % 6 == 0:
                    ln_idx = min(i // 6, len(self.layer_norms) - 1)
                    if ln_idx < len(self.layer_norms):
                        x = self.layer_norms[ln_idx](x)
                
                # 保存残差
                residual = x
                
                # Mamba层前向传播
                mamba_out = mamba_layer(x)
                
                # 改进的残差连接策略：
                # 1. 对Mamba输出进行裁剪，防止极值
                mamba_out = torch.clamp(mamba_out, min=-10, max=10)
                
                # 2. 使用逐层递减的残差权重
                residual_weight = 1.0 / (1.0 + i * 0.1)  # 从1.0逐渐减少到约0.35
                
                # 3. 应用残差连接
                x = residual + mamba_out * residual_weight
            
            # 最终归一化（保留一个防止数值问题）
            x = self.norm(x)
            
            # 移除中间token（如果添加了的话）
            if hasattr(self, 'intermediate_query'):
                # 只保留原始序列长度的部分
                original_seq_len = original_input.size(1)
                x = x[:, :original_seq_len, :]
            
            # 确保dtype一致
            if x.dtype != original_dtype:
                x = x.to(original_dtype)
            
            # 投影回原始维度
            output = self.output_proj(x)
            
        else:
            # 慢速路径：使用transformers的MambaForCausalLM
            outputs = self.mamba(
                inputs_embeds=projected_states,
                output_hidden_states=True,
                return_dict=True
            )
            
            # 获取最后一层的隐藏状态
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                mamba_output = outputs.hidden_states[-1]
            else:
                raise ValueError("预训练Mamba模型必须返回hidden_states，请检查模型配置")
            
            # 移除中间token（如果添加了的话）
            if hasattr(self, 'intermediate_tokens'):
                # 只保留原始序列长度的部分
                original_seq_len = original_input.size(1)
                mamba_output = mamba_output[:, :original_seq_len, :]
            
            # 确保dtype一致
            if mamba_output.dtype != original_dtype:
                mamba_output = mamba_output.to(original_dtype)
            
            # 投影回原始维度
            output = self.output_proj(mamba_output)
        
        # 残差连接（与原始输入）
        output = output + original_input
        
        # 确保输出在正确的设备上
        if output.device != original_device:
            output = output.to(original_device)
        
        return output


class MambaBlock(nn.Module):
    """Mamba块实现"""
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        if MAMBA_AVAILABLE:
            # 使用官方Mamba实现
            self.mamba = Mamba(
                d_model=hidden_size,
                d_state=16,  # 状态维度
                d_conv=4,    # 卷积核大小
                expand=2,    # 扩展因子
            )
        else:
            # 简化的替代实现
            self.mamba = SimpleMambaAlternative(hidden_size)
        
        # 层归一化
        self.norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入序列 [batch_size, seq_len, hidden_size]
        
        Returns:
            输出序列 [batch_size, seq_len, hidden_size]
        """
        
        # 残差连接 + Mamba
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        x = x + residual
        
        return x


class SimpleMambaAlternative(nn.Module):
    """增强版Mamba替代实现（当mamba_ssm不可用时）
    
    使用更接近原始Mamba的状态空间模型设计：
    - 多层深度卷积模拟长程依赖
    - 选择性门控机制
    - 状态传递模拟
    """
    
    def __init__(self, hidden_size: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = hidden_size * expand
        
        # 输入投影（扩展维度）
        self.in_proj = nn.Linear(hidden_size, self.d_inner * 2, bias=False)
        
        # 深度可分离卷积（模拟因果卷积）
        self.conv1d = nn.ModuleList([
            nn.Conv1d(
                self.d_inner, 
                self.d_inner, 
                kernel_size=d_conv,
                padding=d_conv - 1,  # 因果padding
                groups=self.d_inner  # 深度卷积
            ) for _ in range(2)  # 多层以增加感受野
        ])
        
        # 选择性机制参数
        self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_state + 1, bias=False)  # dt + B + C
        self.dt_proj = nn.Linear(self.d_state, self.d_inner, bias=True)
        
        # 状态空间参数
        A = torch.arange(1, self.d_state + 1).unsqueeze(0).repeat(self.d_inner, 1)
        self.register_buffer("A", torch.log(A.float()))  # log(A) for stability
        self.D = nn.Parameter(torch.ones(self.d_inner))  # 跳跃连接参数
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, hidden_size, bias=False)
        
        # 激活函数
        self.activation = nn.SiLU()
        
        # 层归一化
        self.norm = nn.LayerNorm(self.d_inner)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 输入投影并分割
        x_and_res = self.in_proj(x)  # [batch, seq, 2 * d_inner]
        x_inner, res = x_and_res.chunk(2, dim=-1)  # 各 [batch, seq, d_inner]
        
        # 2. 应用激活函数
        x_inner = self.activation(x_inner)
        
        # 3. 因果卷积（模拟序列传播）
        x_conv = x_inner.transpose(1, 2)  # [batch, d_inner, seq]
        for conv in self.conv1d:
            x_conv = conv(x_conv)
            # 因果截断（只保留有效的部分）
            if self.d_conv > 1:
                x_conv = x_conv[:, :, :seq_len]
            x_conv = self.activation(x_conv)
        x_inner = x_conv.transpose(1, 2)  # [batch, seq, d_inner]
        
        # 4. 选择性状态空间机制
        x_proj = self.x_proj(x_inner)  # [batch, seq, d_state + d_state + 1]
        
        # 分解投影：dt, B, C
        dt, B, C = x_proj.split([1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt.squeeze(-1)))  # [batch, seq, d_inner]
        
        # 5. 状态空间计算（简化版）
        # 计算离散化的A
        A = -torch.exp(self.A)  # [d_inner, d_state]
        
        # 初始化状态
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device)
        outputs = []
        
        for t in range(seq_len):
            # 状态更新（简化的状态空间方程）
            h = h + dt[:, t:t+1].unsqueeze(-1) * (
                torch.einsum('bdn,dn->bdn', h, A) + 
                x_inner[:, t:t+1].unsqueeze(-1) * B[:, t:t+1].unsqueeze(1)
            )
            # 输出计算
            y = torch.einsum('bdn,bn->bd', h, C[:, t])
            outputs.append(y)
        
        # 堆叠输出
        y = torch.stack(outputs, dim=1)  # [batch, seq, d_inner]
        
        # 6. 加入跳跃连接（D参数）
        y = y + x_inner * self.D
        
        # 7. 门控和输出投影
        y = self.norm(y)
        y = y * torch.sigmoid(res)  # 门控
        output = self.out_proj(y)
        
        return output


class AttentionFusionCore(nn.Module):
    """基于多头注意力的融合核心（备选方案）"""
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
        num_heads: int = 16
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # 构建Transformer层
        self.fusion_layers = nn.ModuleList([
            AttentionBlock(
                hidden_size, num_heads, dropout, use_flash_attention
            ) for _ in range(num_layers)
        ])
        
        # 输入层归一化
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # 输入归一化
        x = self.input_norm(hidden_states)
        
        # 通过注意力层
        for layer in self.fusion_layers:
            x = layer(x, attention_mask=attention_mask)
        
        # 输出投影
        output = self.output_proj(x)
        
        # 残差连接
        output = output + hidden_states
        
        return output


class AttentionBlock(nn.Module):
    """多头注意力块"""
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size
        
        # 多头注意力
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # 自注意力 + 残差连接
        residual = x
        x = self.norm1(x)
        
        # 处理attention_mask
        if attention_mask is not None:
            # 转换mask格式：0表示有效位置，-inf表示无效位置
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)
        
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        x = residual + self.dropout(attn_output)
        
        # Feed Forward + 残差连接
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + x
        
        return x


class LinearFusionCore(nn.Module):
    """简单线性融合核心（基线方法）"""
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # 简单的MLP层
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_size)
            ])
        
        self.fusion_layers = nn.Sequential(*layers)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        # 简单的前向传播
        x = self.fusion_layers(hidden_states)
        output = self.output_proj(x)
        
        # 残差连接
        output = output + hidden_states
        
        return output


def create_fusion_core(
    fusion_type: str,
    hidden_size: int,
    num_layers: int = 4,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """创建融合核心的工厂函数"""
    
    if fusion_type == "mamba":
        # 检查是否使用预训练Mamba
        use_pretrained_mamba = kwargs.get("use_pretrained_mamba", False)
        if use_pretrained_mamba:
            return PretrainedMambaFusionCore(
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                use_flash_attention=kwargs.get("use_flash_attention", True),
                pretrained_model_name=kwargs.get("pretrained_mamba_model", "state-spaces/mamba-130m-hf")
            )
        else:
            return MambaFusionCore(
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                use_flash_attention=kwargs.get("use_flash_attention", True)
            )
    elif fusion_type == "attention":
        return AttentionFusionCore(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_flash_attention=kwargs.get("use_flash_attention", True),
            num_heads=kwargs.get("num_heads", 16)
        )
    elif fusion_type == "linear":
        return LinearFusionCore(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"不支持的融合类型: {fusion_type}")


class AdaptiveFusionCore(nn.Module):
    """自适应融合核心
    
    根据输入序列长度自动选择最优的融合策略
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        attention_threshold: int = 512  # 超过此长度使用Mamba
    ):
        super().__init__()
        
        self.attention_threshold = attention_threshold
        
        # 创建两种融合核心
        self.mamba_core = MambaFusionCore(hidden_size, num_layers, dropout)
        self.attention_core = AttentionFusionCore(hidden_size, num_layers, dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        seq_len = hidden_states.size(1)
        
        # 根据序列长度选择融合策略
        if seq_len > self.attention_threshold:
            # 长序列使用Mamba（O(N)复杂度）
            return self.mamba_core(hidden_states, attention_mask)
        else:
            # 短序列使用Attention（更好的建模能力）
            return self.attention_core(hidden_states, attention_mask)


# 测试函数
def test_fusion_cores():
    """测试各种融合核心"""
    
    batch_size, seq_len, hidden_size = 2, 768, 1024
    x = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    print("测试融合核心...")
    
    # 测试Mamba融合核心
    try:
        mamba_core = MambaFusionCore(hidden_size, num_layers=2)
        mamba_output = mamba_core(x, attention_mask)
        print(f"Mamba融合核心输出形状: {mamba_output.shape}")
    except Exception as e:
        print(f"Mamba融合核心测试失败: {e}")
    
    # 测试Attention融合核心
    try:
        attention_core = AttentionFusionCore(hidden_size, num_layers=2)
        attention_output = attention_core(x, attention_mask)
        print(f"Attention融合核心输出形状: {attention_output.shape}")
    except Exception as e:
        print(f"Attention融合核心测试失败: {e}")
    
    # 测试Linear融合核心
    try:
        linear_core = LinearFusionCore(hidden_size, num_layers=2)
        linear_output = linear_core(x, attention_mask)
        print(f"Linear融合核心输出形状: {linear_output.shape}")
    except Exception as e:
        print(f"Linear融合核心测试失败: {e}")
    
    # 测试工厂函数
    try:
        fusion_core = create_fusion_core("mamba", hidden_size, num_layers=2)
        factory_output = fusion_core(x, attention_mask)
        print(f"工厂函数创建的融合核心输出形状: {factory_output.shape}")
    except Exception as e:
        print(f"工厂函数测试失败: {e}")


if __name__ == "__main__":
    test_fusion_cores()