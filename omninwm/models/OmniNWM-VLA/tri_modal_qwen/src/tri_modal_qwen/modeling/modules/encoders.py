"""
深度和语义编码器模块

实现专门处理深度图和语义分割图的编码器：
- DepthEncoder: 处理单通道深度图像
- SemanticEncoder: 处理多通道语义分割图像
- 支持CNN、ResNet、ViT等多种架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math
from .batchnorm_fp32 import BatchNorm2dFP32


class CNNEncoder(nn.Module): 
    """基于CNN的编码器基类"""
    
    def __init__(
        self,
        input_channels: int,
        hidden_size: int,
        num_layers: int = 3,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        activation: str = "gelu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        output_size: int = 392  # 输出特征图大小，用于计算patches数量
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # 激活函数
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # CNNEncoder不再处理投影，让调用方（如SemanticEncoder）处理
        # 这样保持了单一职责原则
        actual_input_channels = input_channels
        
        # 构建卷积层
        layers = []
        in_channels = actual_input_channels
        
        for i in range(num_layers):
            # 计算输出通道数
            if i == num_layers - 1:
                out_channels = hidden_size
            else:
                out_channels = min(64 * (2 ** i), hidden_size // 2)
            
            # 卷积层
            layers.append(nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ))
            
            # BatchNorm - 使用FP32版本
            if use_batch_norm:
                bn = BatchNorm2dFP32(out_channels)
                layers.append(bn)
            
            # 激活函数
            layers.append(self.activation)
            
            # Dropout
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 自适应池化到固定大小
        # 修改为输出98个patches以匹配RGB
        # 98 ≈ 10×10，但为了更好的空间布局，使用7×14
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 14))
        
        # 投影到序列格式 (patch_embed)
        self.patch_size = 16
        self.num_patches = 98  # 固定为98个patches
        self.patch_embed = nn.Linear(hidden_size, hidden_size)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_size) * 0.02
        )
        
        # Layer Norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 [batch_size, channels, height, width]
        
        Returns:
            特征序列 [batch_size, num_patches, hidden_size]
        """
        batch_size = x.size(0)
        
        # 确保输入dtype与模型一致
        # 通过检查第一个卷积层的权重dtype
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                model_dtype = layer.weight.dtype
                if x.dtype != model_dtype:
                    x = x.to(model_dtype)
                break
        
        # 处理包含BatchNorm的Sequential模块
        features = x
        for layer in self.conv_layers:
            features = layer(features)
        
        # 自适应池化
        features = self.adaptive_pool(features)  # [batch, hidden_size, H', W']
        
        # 转换为patch序列
        H, W = features.size(2), features.size(3)
        # 直接使用H*W作为patches数量（应该是7*14=98）
        
        # Reshape为patch序列
        features = features.view(
            batch_size, self.hidden_size, H * W
        ).transpose(1, 2)  # [batch, num_patches, hidden_size]
        
        # 投影
        features = self.patch_embed(features)
        
        # 添加位置编码
        features = features + self.pos_embedding
        
        # Layer Norm
        features = self.layer_norm(features)
        
        return features


class ResNetBlock(nn.Module):
    """ResNet残差块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_batch_norm: bool = True,
        activation: str = "gelu"
    ):
        super().__init__()
        
        # 激活函数
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.SiLU()
        
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = BatchNorm2dFP32(out_channels) if use_batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = BatchNorm2dFP32(out_channels) if use_batch_norm else nn.Identity()
        
        # 跳跃连接
        if stride != 1 or in_channels != out_channels:
            shortcut_layers = [nn.Conv2d(in_channels, out_channels, 1, stride)]
            if use_batch_norm:
                bn = BatchNorm2dFP32(out_channels)
                shortcut_layers.append(bn)
            self.shortcut = nn.Sequential(*shortcut_layers)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle shortcut with potential BatchNorm
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.activation(out)
        
        return out


class ResNetEncoder(nn.Module):
    """基于ResNet的编码器"""
    
    def __init__(
        self,
        input_channels: int,
        hidden_size: int,
        num_layers: int = 3,
        activation: str = "gelu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        output_size: int = 392
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始卷积
        self.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3)
        self.bn1 = BatchNorm2dFP32(64) if use_batch_norm else nn.Identity()
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.SiLU()
        
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # ResNet块
        layers = []
        in_channels = 64
        for i in range(num_layers):
            out_channels = min(128 * (2 ** i), hidden_size)
            stride = 2 if i > 0 else 1
            
            layers.append(ResNetBlock(
                in_channels, out_channels, stride, 
                use_batch_norm, activation
            ))
            in_channels = out_channels
        
        self.res_layers = nn.Sequential(*layers)
        
        # 最终投影
        self.final_conv = nn.Conv2d(in_channels, hidden_size, 1)
        
        # 自适应池化
        # 修改为输出98个patches以匹配RGB
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 14))
        
        # Patch embedding
        self.patch_size = 16
        self.num_patches = 98  # 固定为98个patches
        self.patch_embed = nn.Linear(hidden_size, hidden_size)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_size) * 0.02
        )
        
        # Layer Norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 确保输入dtype与模型一致
        # 通过检查第一个卷积层的权重dtype
        if hasattr(self.conv1, 'weight'):
            model_dtype = self.conv1.weight.dtype
            if x.dtype != model_dtype:
                x = x.to(model_dtype)
        
        # 初始处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)
        
        # ResNet层
        x = self.res_layers(x)
        
        # 最终投影
        x = self.final_conv(x)
        
        # 自适应池化
        x = self.adaptive_pool(x)
        
        # 转换为patch序列
        H, W = x.size(2), x.size(3)
        x = x.view(batch_size, self.hidden_size, H * W).transpose(1, 2)
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # 位置编码
        x = x + self.pos_embedding
        
        # Layer Norm和Dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x


class DepthEncoder(nn.Module):
    """深度图编码器
    
    专门处理单通道深度图像，提取空间和几何信息
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        encoder_type = config.get("type", "cnn")
        
        # 深度图预处理 - 使用FP32 BatchNorm
        self.depth_norm = BatchNorm2dFP32(1)
        
        if encoder_type == "cnn":
            self.encoder = CNNEncoder(
                input_channels=config.get("input_channels", 1),
                hidden_size=config.get("hidden_size", 1024),
                num_layers=config.get("num_layers", 3),
                kernel_size=config.get("kernel_size", 3),
                stride=config.get("stride", 2),
                padding=config.get("padding", 1),
                activation=config.get("activation", "gelu"),
                use_batch_norm=config.get("use_batch_norm", True),
                dropout=config.get("dropout", 0.1)
            )
        elif encoder_type == "resnet":
            self.encoder = ResNetEncoder(
                input_channels=config.get("input_channels", 1),
                hidden_size=config.get("hidden_size", 1024),
                num_layers=config.get("num_layers", 3),
                activation=config.get("activation", "gelu"),
                use_batch_norm=config.get("use_batch_norm", True),
                dropout=config.get("dropout", 0.1)
            )
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
    
    def forward(self, depth_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth_images: 深度图像 [batch_size, 1, height, width] (单通道深度图)
        
        Returns:
            深度特征序列 [batch_size, num_patches, hidden_size]
        """
        # 处理可能的5维输入 [batch, num_images, channels, height, width]
        if depth_images.dim() == 5:
            batch_size, num_images, channels, height, width = depth_images.shape
            # 展平batch和num_images维度
            depth_images = depth_images.view(batch_size * num_images, channels, height, width)
        elif depth_images.dim() != 4:
            raise ValueError(f"Expected 4D or 5D tensor, got {depth_images.dim()}D")
        
        if depth_images.size(1) != 1:
            raise ValueError(f"Expected single channel depth image, got {depth_images.size(1)} channels")
        
        # 深度值归一化和验证
        # 确保深度值在合理范围内 (0-100米)
        depth_images = torch.clamp(depth_images, 0.0, 100.0)
        
        # 归一化到0-1范围
        depth_images = depth_images / 100.0
        
        # 使用FP32 BatchNorm（内部自动处理）
        depth_images = self.depth_norm(depth_images)
        
        # 特征提取
        features = self.encoder(depth_images)
        
        # 确保输出dtype与期望一致
        # 如果输入是BF16/FP16，输出也应该是相同dtype
        if hasattr(self.encoder, 'conv_layers'):
            # 检查第一个卷积层的dtype作为参考
            for layer in self.encoder.conv_layers:
                if isinstance(layer, nn.Conv2d):
                    target_dtype = layer.weight.dtype
                    if features.dtype != target_dtype:
                        features = features.to(target_dtype)
                    break
        
        return features


class SemanticEncoder(nn.Module):
    """语义分割编码器
    
    专门处理多通道语义分割图像，提取语义信息
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        encoder_type = config.get("type", "cnn")
        
        # 语义图预处理（nuScenes有17个类别）
        num_classes = config.get("num_classes", 17)
        input_channels = config.get("input_channels", 17)
        
        # 判断输入类型：如果input_channels > num_classes，说明是多通道特征而非类别索引
        self.use_embedding = config.get("use_embedding", True) and input_channels <= num_classes
        
        if self.use_embedding:
            # 输入是类别索引，使用embedding
            embed_dim = config.get("embed_dim", 64)
            self.semantic_embed = nn.Embedding(num_classes, embed_dim)
            encoder_input_channels = embed_dim
        else:
            # 输入是多通道特征，不使用embedding
            # 如果通道数过多，添加投影层
            if input_channels > 64:
                self.projection = nn.Conv2d(input_channels, 64, kernel_size=1)
                encoder_input_channels = 64
            else:
                self.projection = None
                encoder_input_channels = input_channels
        
        if encoder_type == "cnn":
            self.encoder = CNNEncoder(
                input_channels=encoder_input_channels,  # 使用处理后的通道数
                hidden_size=config.get("hidden_size", 1024),
                num_layers=config.get("num_layers", 3),
                kernel_size=config.get("kernel_size", 3),
                stride=config.get("stride", 2),
                padding=config.get("padding", 1),
                activation=config.get("activation", "gelu"),
                use_batch_norm=config.get("use_batch_norm", True),
                dropout=config.get("dropout", 0.1)
            )
        elif encoder_type == "resnet":
            self.encoder = ResNetEncoder(
                input_channels=encoder_input_channels,  # 使用处理后的通道数
                hidden_size=config.get("hidden_size", 1024),
                num_layers=config.get("num_layers", 3),
                activation=config.get("activation", "gelu"),
                use_batch_norm=config.get("use_batch_norm", True),
                dropout=config.get("dropout", 0.1)
            )
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
    
    def forward(self, semantic_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            semantic_images: 语义图像 
                           - 如果use_embedding=True: [batch_size, height, width] (类别ID格式)
                           - 如果use_embedding=False: [batch_size, channels, height, width] (one-hot格式)
        
        Returns:
            语义特征序列 [batch_size, num_patches, hidden_size]
        """
        # 获取目标dtype（从encoder获取）
        target_dtype = torch.float32
        if hasattr(self.encoder, 'conv_layers') and len(self.encoder.conv_layers) > 0:
            for module in self.encoder.conv_layers:
                if hasattr(module, 'weight'):
                    target_dtype = module.weight.dtype
                    break
        
        if self.use_embedding:
            # 处理分类标签格式的语义图 (nuScenes标准格式)
            if semantic_images.dim() == 3:
                # [batch, height, width] -> [batch, 1, height, width]
                semantic_images = semantic_images.unsqueeze(1)
            
            # 确保输入为整数类型且在有效范围内
            semantic_images = semantic_images.long()
            semantic_images = torch.clamp(semantic_images, 0, self.config.get("num_classes", 17) - 1)
            
            # 嵌入到特征空间
            # 处理可能的5维输入 [batch, num_images, channels, height, width]
            if semantic_images.dim() == 5:
                batch_size, num_images, channels, height, width = semantic_images.shape
                # 展平batch和num_images维度
                semantic_images = semantic_images.view(batch_size * num_images, channels, height, width)
                batch_size = batch_size * num_images
            else:
                batch_size, _, height, width = semantic_images.shape
                
            semantic_images = self.semantic_embed(semantic_images)  # [batch, channels, H, W, embed_dim]
            semantic_images = semantic_images.to(target_dtype)  # 确保dtype一致
            
            # 处理embedding输出
            if semantic_images.dim() == 5:
                # [batch, channels, H, W, embed_dim]
                # 需要移除channels维度（应该是1）并调整顺序
                if semantic_images.shape[1] == 1:
                    semantic_images = semantic_images.squeeze(1)  # [batch, H, W, embed_dim]
                    semantic_images = semantic_images.permute(0, 3, 1, 2)  # [batch, embed_dim, H, W]
                else:
                    # 如果channels维度不是1，需要不同的处理
                    # 这种情况下输入可能是多通道的，我们需要处理成合理的维度
                    # 简单的方案：对每个位置的所有通道的embedding求平均
                    batch, channels, h, w, embed_dim = semantic_images.shape
                    # 对channels维度求平均，得到 [batch, H, W, embed_dim]
                    semantic_images = semantic_images.mean(dim=1)  # [batch, H, W, embed_dim]
                    semantic_images = semantic_images.permute(0, 3, 1, 2)  # [batch, embed_dim, H, W]
            elif semantic_images.dim() == 4:
                # [batch, H, W, embed_dim] -> [batch, embed_dim, H, W]
                semantic_images = semantic_images.permute(0, 3, 1, 2)
            else:
                # 处理其他可能的输出格式
                raise ValueError(f"Unexpected semantic_embed output shape: {semantic_images.shape}")
        else:
            # 处理多通道特征格式的语义图
            # 首先处理5维输入
            if semantic_images.dim() == 5:
                batch_size, num_images, channels, height, width = semantic_images.shape
                semantic_images = semantic_images.view(batch_size * num_images, channels, height, width)
            elif semantic_images.dim() == 3:
                # 扩展维度
                semantic_images = semantic_images.unsqueeze(1)
            
            # 确保输入使用正确的dtype
            semantic_images = semantic_images.to(target_dtype)
            
            # 如果有投影层，使用它来降维
            if hasattr(self, 'projection') and self.projection is not None:
                # 确保投影层也使用正确的dtype
                if self.projection.weight.dtype != target_dtype:
                    self.projection = self.projection.to(target_dtype)
                semantic_images = self.projection(semantic_images)  # [batch, 64, H, W]
        
        # 特征提取
        features = self.encoder(semantic_images)
        
        # 确保输出dtype与期望一致
        # 检查encoder的目标dtype
        if hasattr(self.encoder, 'conv_layers'):
            for layer in self.encoder.conv_layers:
                if isinstance(layer, nn.Conv2d):
                    target_dtype = layer.weight.dtype
                    if features.dtype != target_dtype:
                        features = features.to(target_dtype)
                    break
        elif hasattr(self.encoder, 'conv1'):
            # ResNetEncoder的情况
            target_dtype = self.encoder.conv1.weight.dtype
            if features.dtype != target_dtype:
                features = features.to(target_dtype)
        
        return features


# 工厂函数
def create_encoder(encoder_type: str, config: Dict[str, Any]) -> nn.Module:
    """创建编码器的工厂函数"""
    
    if encoder_type == "depth":
        return DepthEncoder(config)
    elif encoder_type == "semantic":
        return SemanticEncoder(config)
    else:
        raise ValueError(f"未知的编码器类型: {encoder_type}")


# 测试函数
def test_encoders():
    """测试编码器功能"""
    
    # 测试深度编码器
    depth_config = {
        "type": "cnn",
        "input_channels": 1,
        "hidden_size": 1024,
        "num_layers": 3
    }
    
    depth_encoder = DepthEncoder(depth_config)
    depth_input = torch.randn(2, 1, 392, 392)
    depth_output = depth_encoder(depth_input)
    # print(f"深度编码器输出形状: {depth_output.shape}")
    
    # 测试语义编码器
    semantic_config = {
        "type": "resnet",
        "input_channels": 150,
        "hidden_size": 1024,
        "num_layers": 3
    }
    
    semantic_encoder = SemanticEncoder(semantic_config)
    semantic_input = torch.randn(2, 150, 392, 392)
    semantic_output = semantic_encoder(semantic_input)
    # print(f"语义编码器输出形状: {semantic_output.shape}")


if __name__ == "__main__":
    test_encoders()