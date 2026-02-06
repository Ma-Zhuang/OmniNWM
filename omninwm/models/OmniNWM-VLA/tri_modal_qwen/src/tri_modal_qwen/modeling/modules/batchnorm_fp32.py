"""
FP32 BatchNorm包装器 - 确保BatchNorm始终在FP32中运行
即使在混合精度训练中也保持FP32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm2dFP32(nn.Module):
    """始终在FP32中运行的BatchNorm2d包装器"""
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            # 权重和偏置始终保持FP32
            self.register_buffer('weight', torch.ones(num_features, dtype=torch.float32))
            self.register_buffer('bias', torch.zeros(num_features, dtype=torch.float32))
            # 使用register_buffer而不是Parameter，防止被优化器更新
            self.weight_param = nn.Parameter(torch.ones(num_features))
            self.bias_param = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight_param', None)
            self.register_parameter('bias_param', None)
            self.register_buffer('weight', None)
            self.register_buffer('bias', None)
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype=torch.float32))
            self.register_buffer('running_var', torch.ones(num_features, dtype=torch.float32))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
    
    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
    
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight_param.data.fill_(1)
            self.bias_param.data.zero_()
            self.weight.fill_(1)
            self.bias.zero_()
    
    def forward(self, input):
        # 确保权重参数是FP32
        if self.affine:
            self.weight.data = self.weight_param.data.float()
            self.bias.data = self.bias_param.data.float()
        
        # 保存原始dtype和设备
        original_dtype = input.dtype
        original_device = input.device
        
        # 检测模型的目标dtype（通过检查weight_param的dtype）
        # 这告诉我们模型整体被转换成了什么dtype
        model_dtype = self.weight_param.dtype if self.affine else original_dtype
        
        # 转换输入为FP32
        input_fp32 = input.float()
        
        # 确保所有buffer都在正确的设备上且是FP32
        if self.track_running_stats:
            if self.running_mean.device != original_device:
                self.running_mean = self.running_mean.to(original_device)
                self.running_var = self.running_var.to(original_device)
            if self.running_mean.dtype != torch.float32:
                self.running_mean = self.running_mean.float()
                self.running_var = self.running_var.float()
        
        if self.affine:
            if self.weight.device != original_device:
                self.weight = self.weight.to(original_device)
                self.bias = self.bias.to(original_device)
            if self.weight.dtype != torch.float32:
                self.weight = self.weight.float()
                self.bias = self.bias.float()
        
        # 在FP32中执行BatchNorm
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # 使用累积移动平均
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
            else:
                exponential_average_factor = self.momentum
        else:
            exponential_average_factor = 0.0
        
        # 执行BatchNorm
        output = F.batch_norm(
            input_fp32,
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps
        )
        
        # 转换到适当的dtype
        # 如果模型被转换为BF16/FP16，输出也应该是相同的dtype
        # 这样可以匹配下一层（Conv2d等）的期望
        return output.to(model_dtype)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """加载状态字典时确保FP32"""
        # 转换所有加载的参数为FP32
        for key in list(state_dict.keys()):
            if key.startswith(prefix):
                param = state_dict[key]
                if param is not None and param.dtype != torch.float32:
                    state_dict[key] = param.float()
        
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
    
    def to(self, *args, **kwargs):
        """重写to方法，防止BatchNorm被转换为其他dtype"""
        # 获取目标device
        device = None
        dtype = None
        
        for arg in args:
            if isinstance(arg, torch.device):
                device = arg
            elif isinstance(arg, str):
                device = torch.device(arg)
            elif isinstance(arg, torch.dtype):
                dtype = arg
        
        if 'device' in kwargs:
            device = kwargs['device']
        if 'dtype' in kwargs:
            dtype = kwargs['dtype']
        
        # 只移动到设备，不改变dtype
        if device is not None:
            self.running_mean = self.running_mean.to(device) if self.running_mean is not None else None
            self.running_var = self.running_var.to(device) if self.running_var is not None else None
            if self.affine:
                self.weight = self.weight.to(device)
                self.bias = self.bias.to(device)
                self.weight_param = self.weight_param.to(device)
                self.bias_param = self.bias_param.to(device)
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked.to(device)
        
        # 始终保持FP32，忽略dtype参数
        return self
    
    def half(self):
        """防止被转换为半精度"""
        return self
    
    def bfloat16(self):
        """防止被转换为BF16"""
        return self
    
    def float(self):
        """确保是FP32"""
        return self
    
    def double(self):
        """防止被转换为FP64"""
        return self