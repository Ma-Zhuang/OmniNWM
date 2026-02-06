"""
三模态解释器（TMI）核心模块

TMI是本项目的核心创新，集成了：
- 三种模态的独立处理
- 特征投影与对齐
- Mamba融合核心
- 最终输出投影
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
import math

from .encoders import DepthEncoder, SemanticEncoder
from .fusion import create_fusion_core
from .flash_attention import (
    FlashAttentionConfig,
    OptimizedMultiHeadAttention,
    MemoryEfficientCrossAttention
)


class TriModalInterpreter(nn.Module):
    """三模态解释器（TMI）核心模块
    
    受SSR论文的MIDI模块启发，扩展为支持RGB、深度、语义三种模态的融合
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # 从配置中获取参数
        self.fusion_hidden_size = config.fusion_hidden_size
        self.fusion_num_layers = config.fusion_num_layers
        self.fusion_type = config.fusion_type
        # 保持原有的dropout设置，避免破坏现有训练
        self.fusion_dropout = getattr(config, 'fusion_dropout', 0.1)  # 保持原值
        # 只在训练时动态增强dropout，不改变模型结构
        self.training_dropout_scale = getattr(config, 'training_dropout_scale', 1.5)  # 训练时放大倍数
        self.vision_hidden_size = config.vision_hidden_size
        self.llm_hidden_size = config.llm_hidden_size
        
        # 1. 深度和语义编码器（RGB编码器复用Qwen2.5-VL的ViT）
        self.depth_encoder = DepthEncoder(config.depth_encoder_config)
        self.semantic_encoder = SemanticEncoder(config.semantic_encoder_config)
        
        # 2. 特征投影层 - 差异化结构避免相同投影
        # RGB: 3584 -> 2048 -> 2048 (两步投影，使用GELU)
        self.rgb_projector = nn.Sequential(
            nn.Linear(self.vision_hidden_size, self.fusion_hidden_size),
            nn.GELU(),
            nn.Linear(self.fusion_hidden_size, self.fusion_hidden_size)
        )
        
        # Depth: 1024 -> 1536 -> 2048 (不同中间维度，使用SiLU)
        self.depth_projector = nn.Sequential(
            nn.Linear(config.depth_encoder_config.get("hidden_size", 1024), 1536),
            nn.SiLU(),  # 使用不同的激活函数
            nn.Dropout(0.05),  # 添加轻微dropout
            nn.Linear(1536, self.fusion_hidden_size)
        )
        
        # Semantic: 1024 -> 1280 -> 2048 (另一个中间维度，使用ReLU)
        self.semantic_projector = nn.Sequential(
            nn.Linear(config.semantic_encoder_config.get("hidden_size", 1024), 1280),
            nn.ReLU(),  # 又一个不同的激活函数
            nn.Linear(1280, self.fusion_hidden_size)
        )
        
        # 使用不同的初始化来区分模态（而不是模态嵌入）
        self._init_projectors()
        
        # 3. 可学习的模态缩放因子（轻量级多样性机制）
        # 不是模态嵌入，而是简单的缩放，避免强制相似性
        # 使用float32初始化，运行时会自动转换
        self.modality_scales = nn.Parameter(torch.tensor([1.0, 0.85, 0.7], dtype=torch.float32))
        
        # 4. 轻量级多样性约束（可选）
        self.use_diversity_regularization = getattr(config, 'use_diversity_regularization', True)
        
        # 添加特征正则化机制
        self.feature_regularization = nn.ModuleDict({
            'rgb': nn.Sequential(
                nn.LayerNorm(self.fusion_hidden_size),
                nn.Dropout(0.1)
            ),
            'depth': nn.Sequential(
                nn.LayerNorm(self.fusion_hidden_size),
                nn.Dropout(0.1)
            ),
            'semantic': nn.Sequential(
                nn.LayerNorm(self.fusion_hidden_size),
                nn.Dropout(0.1)
            )
        })
        
        # 4. 融合核心
        self.fusion_core = create_fusion_core(
            fusion_type=self.fusion_type,
            hidden_size=self.fusion_hidden_size,
            num_layers=self.fusion_num_layers,
            dropout=self.fusion_dropout,
            use_flash_attention=config.use_flash_attention,
            use_pretrained_mamba=getattr(config, 'use_pretrained_mamba', True),  # 默认使用预训练Mamba
            pretrained_mamba_model=getattr(config, 'pretrained_mamba_model', '/code/VLA/models/state-spaces/mamba-130m-hf')
        )
        
        # 6. 最终投影到LLM维度
        self.final_projector = self._create_projector(
            self.fusion_hidden_size, self.llm_hidden_size
        )
        
        # 7. 输出层归一化
        self.output_norm = nn.LayerNorm(self.llm_hidden_size)
        
        # 8. Flash Attention配置
        self.flash_config = FlashAttentionConfig(
            enable_flash_attention=getattr(config, 'enable_flash_attention', True),
            enable_xformers=getattr(config, 'enable_xformers', True),
            enable_gradient_checkpointing=getattr(config, 'enable_gradient_checkpointing', True),
            attention_dropout=getattr(config, 'attention_dropout', 0.0)
        )
        
        # 9. 优化的跨模态注意力
        self.use_cross_attention = getattr(config, 'use_cross_attention', False)
        if self.use_cross_attention:
            self.cross_attention = MemoryEfficientCrossAttention(
                query_dim=self.fusion_hidden_size,
                key_dim=self.fusion_hidden_size,
                value_dim=self.fusion_hidden_size,
                hidden_dim=self.fusion_hidden_size,
                num_heads=16,
                dropout=self.fusion_dropout,
                flash_config=self.flash_config
            )
            self.cross_attention_norm = nn.LayerNorm(self.fusion_hidden_size)
    
    def _init_projectors(self):
        """初始化投影器，使用稍微不同的初始化来避免相同投影"""
        # RGB投影器：使用标准初始化
        for m in self.rgb_projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Depth投影器：稍微不同的gain
        for m in self.depth_projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.9)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Semantic投影器：更不同的gain
        for m in self.semantic_projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _create_projector(self, input_dim: int, output_dim: int) -> nn.Module:
        """创建特征投影器 - 基于SSR但增强容量防止坍塌"""
        # 参考SSR的MIDI设计：两步投影但保持简洁
        if input_dim == output_dim:
            # 维度相同时，使用两层MLP增加表达能力（参考SSR的image_proj）
            projector = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim)  # 额外的变换层，增加容量
                # 不使用LayerNorm，保持特征分布多样性
            )
        else:
            # 维度不同时，使用两层MLP，中间维度适当增大
            # 参考Qwen的intermediate_size设计（hidden_size的5.3倍）
            hidden_dim = min(max(input_dim, output_dim) * 2, 3584)  # 限制在Qwen的hidden_size内
            
            projector = nn.Sequential(
                # 第一层：维度变换
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),  # Qwen使用silu，但GELU在这里效果相似
                # 第二层：到目标维度
                nn.Linear(hidden_dim, output_dim)
                # 关键：不使用LayerNorm避免过度归一化
                # 但容量要足够大防止信息瓶颈
            )
        
        # FP16/BF16友好的初始化策略 - 更保守的初始化防止数值问题
        for module in projector.modules():
            if isinstance(module, nn.Linear):
                # 使用Xavier初始化，标准差调小一些以防止FP16溢出
                # Xavier初始化考虑了前后层的维度，更适合深层网络
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
                std = math.sqrt(2.0 / (fan_in + fan_out)) * 0.8  # 缩小0.8倍更保守
                nn.init.normal_(module.weight, mean=0.0, std=std)
                
                if module.bias is not None:
                    # bias初始化为0
                    nn.init.zeros_(module.bias)
                    
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm的标准初始化
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # 注释掉权重缩放，让模型正常初始化
        # 权重缩放会导致梯度过小和模式坍塌
        # with torch.no_grad():
        #     # 获取最后一个Linear层
        #     for module in reversed(list(projector.modules())):
        #         if isinstance(module, nn.Linear):
        #             # 将最后一层的权重缩小
        #             module.weight.data *= 0.5
        #             break
        
        return projector
    
    def compute_diversity_loss(self, features: torch.Tensor) -> torch.Tensor:
        """计算轻量级多样性损失，鼓励不同样本产生不同特征"""
        if not self.training or not self.use_diversity_regularization:
            return torch.tensor(0.0, device=features.device)
        
        # 使用第一个token作为代表
        if features.dim() == 3:
            features = features[:, 0, :]  # [batch, hidden]
        
        # 归一化
        features_norm = F.normalize(features, dim=-1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features_norm, features_norm.t())
        
        # 去除对角线，只看不同样本间的相似度
        batch_size = sim_matrix.size(0)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)
        off_diagonal = sim_matrix[mask]
        
        # 惩罚过高的相似度（soft penalty）
        # 使用平滑的惩罚函数，避免硬阈值
        threshold = 0.7  # 相似度阈值
        penalty = F.relu(off_diagonal - threshold).mean()
        
        return penalty * 0.1  # 轻量级，权重0.1
    
    def forward(
        self,
        rgb_features: torch.Tensor,
        depth_pixel_values: torch.Tensor,
        semantic_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            rgb_features: RGB特征 [batch_size, num_patches, vision_hidden_size]
            depth_pixel_values: 深度图像 [batch_size, 1, height, width] 
            semantic_pixel_values: 语义图像 [batch_size, channels, height, width]
            attention_mask: 注意力掩码 [batch_size, total_seq_len] (可选)
        
        Returns:
            融合后的特征表示 [batch_size, total_seq_len, llm_hidden_size]
        """
        
        batch_size = rgb_features.size(0)
        
        
        # 保存原始输入的dtype，最后需要转换回去
        original_dtype = rgb_features.dtype
        
        # depth和semantic保持原样，不做转换
        
        # 1. 处理深度和语义图像
        try:
            depth_features = self.depth_encoder(depth_pixel_values)    # [batch, num_patches, hidden]
            semantic_features = self.semantic_encoder(semantic_pixel_values)  # [batch, num_patches, hidden]
            
            
        except Exception as e:
            raise e
        
        # 2. 特征投影与对齐 - 增加FP16稳定性措施
        # 在投影前进行梯度裁剪，防止梯度爆炸
        rgb_features = torch.clamp(rgb_features, min=-100, max=100)
        depth_features = torch.clamp(depth_features, min=-100, max=100)
        semantic_features = torch.clamp(semantic_features, min=-100, max=100)
        
        rgb_projected = self.rgb_projector(rgb_features)           # [batch, num_patches, fusion_hidden]
        depth_projected = self.depth_projector(depth_features)     # [batch, num_patches, fusion_hidden]  
        semantic_projected = self.semantic_projector(semantic_features)  # [batch, num_patches, fusion_hidden]
        
        # 应用特征正则化（新增）
        if hasattr(self, 'feature_regularization'):
            rgb_projected = self.feature_regularization['rgb'](rgb_projected)
            depth_projected = self.feature_regularization['depth'](depth_projected)
            semantic_projected = self.feature_regularization['semantic'](semantic_projected)
        
        # 确保投影输出dtype与输入一致
        if rgb_projected.dtype != original_dtype:
            rgb_projected = rgb_projected.to(original_dtype)
        if depth_projected.dtype != original_dtype:
            depth_projected = depth_projected.to(original_dtype)
        if semantic_projected.dtype != original_dtype:
            semantic_projected = semantic_projected.to(original_dtype)
        
        # FP16稳定性：投影后立即裁剪，防止数值溢出
        rgb_projected = torch.clamp(rgb_projected, min=-65504, max=65504)  # FP16最大值
        depth_projected = torch.clamp(depth_projected, min=-65504, max=65504)
        semantic_projected = torch.clamp(semantic_projected, min=-65504, max=65504)
        
        
        
        
        # 基于SSR：完全去掉特征白化，让特征保持自然分布
        # 特征白化会强制归一化，反而增加相似性
        
        # 3.5 添加模态类型嵌入（使用更稳定的缩放）
        # 模态嵌入已移除 - 让Mamba自己学习区分不同模态
        
        # 基于SSR：大幅减少噪声，让模型更稳定地学习
        if self.training:
            # 极小的噪声，仅用于数值稳定性
            noise_std = 0.001  # 从0.03降到0.001
            rgb_projected = rgb_projected + torch.randn_like(rgb_projected) * noise_std
            depth_projected = depth_projected + torch.randn_like(depth_projected) * noise_std
            semantic_projected = semantic_projected + torch.randn_like(semantic_projected) * noise_std
        
        # 使用可学习的缩放因子（替代固定缩放）
        if hasattr(self, 'modality_scales'):
            # 确保缩放因子的dtype与特征一致
            scales = self.modality_scales.to(rgb_projected.dtype)
            rgb_projected = rgb_projected * scales[0]
            depth_projected = depth_projected * scales[1]
            semantic_projected = semantic_projected * scales[2]
        else:
            # 后备方案：固定缩放
            rgb_projected = rgb_projected * 1.0
            depth_projected = depth_projected * 0.9
            semantic_projected = semantic_projected * 0.8
        
        # 训练时的额外正则化（简化版，避免引入未定义的变量）
        if self.training and hasattr(self, 'training_dropout_scale'):
            # 轻微的特征扰动，增强鲁棒性
            dropout_rate = min(0.1, self.fusion_dropout * 0.5)  # 限制最大扰动
            if dropout_rate > 0 and torch.rand(1).item() < 0.3:  # 30%概率应用
                # 对某个模态应用轻微dropout
                modal_choice = torch.randint(0, 3, (1,)).item()
                if modal_choice == 1:
                    depth_projected = F.dropout(depth_projected, p=dropout_rate, training=True)
                elif modal_choice == 2:
                    semantic_projected = F.dropout(semantic_projected, p=dropout_rate, training=True)
        
        # 4. 优化的跨模态注意力
        if self.use_cross_attention:
            # RGB作为query，深度和语义作为key/value
            depth_semantic = torch.cat([depth_projected, semantic_projected], dim=1)
            
            rgb_enhanced = self.cross_attention(
                query=rgb_projected,
                key=depth_semantic,
                value=depth_semantic
            )
            rgb_projected = self.cross_attention_norm(rgb_enhanced + rgb_projected)
        
        # 5. 序列拼接
        fused_features = torch.cat([
            rgb_projected,
            depth_projected,
            semantic_projected
        ], dim=1)  # [batch, 3*num_patches, fusion_hidden_size]
        
        # 6. 扩展attention_mask以匹配拼接后的序列长度
        if attention_mask is not None:
            num_patches = rgb_projected.size(1)
            extended_mask = torch.cat([attention_mask] * 3, dim=1)
        else:
            extended_mask = None
        
        # 7. Mamba/Attention融合
        fused_output = self.fusion_core(
            hidden_states=fused_features,
            attention_mask=extended_mask
        )
        
        
        # 8. 最终投影到LLM维度
        final_features = self.final_projector(fused_output)
        
        
        # 9. SSR设计：不使用最终的LayerNorm，保持特征的自然分布
        # 避免过度归一化导致特征坍塌
        # final_features = self.output_norm(final_features)  # 注释掉，参考SSR的MIDI设计
        
        # 10. 确保输出dtype与输入一致
        # 这非常重要，因为后续的Transformer期望BF16输入
        if final_features.dtype != original_dtype:
            final_features = final_features.to(original_dtype)
        
        return final_features
    
    def get_attention_weights(
        self,
        rgb_features: torch.Tensor,
        depth_pixel_values: torch.Tensor,
        semantic_pixel_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """获取注意力权重用于可视化分析"""
        
        # 这个方法用于分析模型对不同模态的关注程度
        with torch.no_grad():
            # 处理输入
            depth_features = self.depth_encoder(depth_pixel_values)
            semantic_features = self.semantic_encoder(semantic_pixel_values)
            
            # 投影
            rgb_proj = self.rgb_projector(rgb_features)
            depth_proj = self.depth_projector(depth_features)
            semantic_proj = self.semantic_projector(semantic_features)
            
            # 计算模态间的相似度
            rgb_norm = F.normalize(rgb_proj, dim=-1)
            depth_norm = F.normalize(depth_proj, dim=-1)
            semantic_norm = F.normalize(semantic_proj, dim=-1)
            
            # 计算注意力权重
            rgb_depth_sim = torch.einsum('bnh,bmh->bnm', rgb_norm, depth_norm)
            rgb_semantic_sim = torch.einsum('bnh,bmh->bnm', rgb_norm, semantic_norm)
            depth_semantic_sim = torch.einsum('bnh,bmh->bnm', depth_norm, semantic_norm)
            
            return {
                'rgb_depth_similarity': rgb_depth_sim,
                'rgb_semantic_similarity': rgb_semantic_sim,
                'depth_semantic_similarity': depth_semantic_sim,
                'modality_embeddings': self.modality_embeddings.detach()
            }
    
    def compute_modality_importance(
        self,
        rgb_features: torch.Tensor,
        depth_pixel_values: torch.Tensor,
        semantic_pixel_values: torch.Tensor
    ) -> Dict[str, float]:
        """计算各模态的重要性分数"""
        
        with torch.no_grad():
            # 分别计算每个模态的特征
            depth_features = self.depth_encoder(depth_pixel_values)
            semantic_features = self.semantic_encoder(semantic_pixel_values)
            
            # 投影后计算特征的信息熵作为重要性指标
            rgb_proj = self.rgb_projector(rgb_features)
            depth_proj = self.depth_projector(depth_features)
            semantic_proj = self.semantic_projector(semantic_features)
            
            # 计算每个模态的方差（作为信息量的代理指标）
            rgb_var = torch.var(rgb_proj, dim=(0, 1)).mean().item()
            depth_var = torch.var(depth_proj, dim=(0, 1)).mean().item()
            semantic_var = torch.var(semantic_proj, dim=(0, 1)).mean().item()
            
            total_var = rgb_var + depth_var + semantic_var
            
            return {
                'rgb_importance': rgb_var / total_var,
                'depth_importance': depth_var / total_var,
                'semantic_importance': semantic_var / total_var
            }
    
    def ablation_forward(
        self,
        rgb_features: torch.Tensor,
        depth_pixel_values: Optional[torch.Tensor] = None,
        semantic_pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """消融实验：可以选择性地禁用某些模态"""
        
        batch_size = rgb_features.size(0)
        features_list = []
        
        # RGB模态（始终包含）
        rgb_projected = self.rgb_projector(rgb_features)
        rgb_projected = rgb_projected + self.modality_embeddings[0].unsqueeze(0).unsqueeze(0)
        features_list.append(rgb_projected)
        
        # 深度模态（可选）
        if depth_pixel_values is not None:
            depth_features = self.depth_encoder(depth_pixel_values)
            depth_projected = self.depth_projector(depth_features)
            depth_projected = depth_projected + self.modality_embeddings[1].unsqueeze(0).unsqueeze(0)
            features_list.append(depth_projected)
        
        # 语义模态（可选）
        if semantic_pixel_values is not None:
            semantic_features = self.semantic_encoder(semantic_pixel_values)
            semantic_projected = self.semantic_projector(semantic_features)
            semantic_projected = semantic_projected + self.modality_embeddings[2].unsqueeze(0).unsqueeze(0)
            features_list.append(semantic_projected)
        
        # 拼接可用的模态
        fused_features = torch.cat(features_list, dim=1)
        
        # 调整attention_mask
        if attention_mask is not None:
            num_modalities = len(features_list)
            extended_mask = torch.cat([attention_mask] * num_modalities, dim=1)
        else:
            extended_mask = None
        
        # 融合和投影
        fused_output = self.fusion_core(fused_features, extended_mask)
        final_features = self.final_projector(fused_output)
        final_features = self.output_norm(final_features)
        
        return final_features


class TMIWithMemory(TriModalInterpreter):
    """带记忆机制的TMI模块
    
    在原TMI基础上添加记忆机制，用于序列建模任务
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 记忆模块
        self.memory_size = getattr(config, 'memory_size', 256)
        self.memory_bank = nn.Parameter(
            torch.randn(self.memory_size, self.fusion_hidden_size) * 0.02
        )
        
        # 记忆更新机制
        self.memory_gate = nn.Sequential(
            nn.Linear(self.fusion_hidden_size * 2, self.fusion_hidden_size),
            nn.Sigmoid()
        )
        
        # 记忆读取注意力
        self.memory_attention = OptimizedMultiHeadAttention(
            hidden_size=self.fusion_hidden_size,
            num_attention_heads=8,
            attention_dropout=self.fusion_dropout,
            flash_config=FlashAttentionConfig(
                enable_flash_attention=getattr(config, 'enable_flash_attention', True),
                enable_gradient_checkpointing=False  # 记忆模块不使用梯度检查点
            )
        )
    
    def forward(
        self,
        rgb_features: torch.Tensor,
        depth_pixel_values: torch.Tensor,
        semantic_pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        memory_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回: (输出特征, 更新后的记忆状态)
        """
        
        # 调用父类的forward方法
        fused_features = super().forward(
            rgb_features, depth_pixel_values, semantic_pixel_values, attention_mask
        )
        
        # 记忆机制
        if memory_state is None:
            memory_state = self.memory_bank.unsqueeze(0).repeat(fused_features.size(0), 1, 1)
        
        # 从记忆中读取信息
        memory_output, _, _ = self.memory_attention(
            hidden_states=fused_features,
            attention_mask=None,
            past_key_value=(memory_state, memory_state),
            use_cache=False
        )
        
        # 更新记忆
        memory_input = torch.cat([fused_features.mean(dim=1, keepdim=True), 
                                 memory_state.mean(dim=1, keepdim=True)], dim=-1)
        memory_gate_weights = self.memory_gate(memory_input)
        
        updated_memory = memory_gate_weights * memory_state + \
                        (1 - memory_gate_weights) * fused_features.mean(dim=1, keepdim=True)
        
        # 合并记忆信息
        enhanced_features = fused_features + memory_output
        
        return enhanced_features, updated_memory


def test_tmi_module():
    """测试TMI模块"""
    
    # 创建模拟配置
    class MockConfig:
        def __init__(self):
            self.fusion_hidden_size = 1024
            self.fusion_num_layers = 4
            self.fusion_type = "mamba"
            self.fusion_dropout = 0.1
            self.vision_hidden_size = 1152
            self.llm_hidden_size = 2048
            self.use_flash_attention = True
            
            self.depth_encoder_config = {
                "type": "cnn",
                "input_channels": 1,
                "hidden_size": 1024,
                "num_layers": 3
            }
            
            self.semantic_encoder_config = {
                "type": "cnn", 
                "input_channels": 150,
                "hidden_size": 1024,
                "num_layers": 3
            }
    
    config = MockConfig()
    
    # 创建TMI模块
    tmi = TriModalInterpreter(config)
    
    # 创建模拟输入
    batch_size = 2
    num_patches = 576  # 24x24 patches
    
    rgb_features = torch.randn(batch_size, num_patches, config.vision_hidden_size)
    depth_images = torch.randn(batch_size, 1, 392, 392)
    semantic_images = torch.randn(batch_size, 150, 392, 392)
    
    # 前向传播
    try:
        output = tmi(rgb_features, depth_images, semantic_images)
        # print(f"TMI模块输出形状: {output.shape}")
        # print(f"期望形状: [batch_size={batch_size}, seq_len={num_patches*3}, hidden_size={config.llm_hidden_size}]")
        
        # 测试注意力权重
        attention_weights = tmi.get_attention_weights(rgb_features, depth_images, semantic_images)
        # print(f"注意力权重keys: {list(attention_weights.keys())}")
        
        # 测试模态重要性
        importance = tmi.compute_modality_importance(rgb_features, depth_images, semantic_images)
        # print(f"模态重要性: {importance}")
        
        # print("TMI模块测试成功！")
        
    except Exception as e:
        # print(f"TMI模块测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_tmi_module()