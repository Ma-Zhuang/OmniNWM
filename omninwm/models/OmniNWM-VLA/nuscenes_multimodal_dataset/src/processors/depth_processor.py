"""
深度图生成处理器

使用预训练的单目深度估计模型生成深度图：
- 支持多种深度估计模型（ZoeDepth, MiDaS等）
- 批量处理优化
- 深度图保存和加载
- GPU加速支持
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from PIL import Image
import logging
import os
from pathlib import Path

try:
    import cv2
except ImportError:
    cv2 = None
    logging.warning("OpenCV (cv2) not installed. Some features may be limited.")


class DepthProcessor:
    """深度图生成处理器"""
    
    def __init__(self, model_name: str = 'ZoeDepth', device: str = 'auto', 
                 model_path: Optional[str] = None, use_local_weights: bool = False):
        """
        初始化深度处理器
        
        Args:
            model_name: 深度估计模型名称
            device: 计算设备 ('cpu', 'cuda', 'auto')
            model_path: 本地模型权重路径（可选）
            use_local_weights: 是否使用本地权重
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model_path = model_path
        self.use_local_weights = use_local_weights
        self.model = None
        self.processor = None  # 用于Hugging Face格式的processor
        self._load_depth_model()
        
    def _setup_device(self, device: str) -> str:
        """设置计算设备"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_depth_model(self):
        """加载深度估计模型"""
        try:
            if self.model_name.lower() == 'dpt':
                self._load_dpt()
            elif self.model_name.lower() == 'zoedepth':
                self._load_zoedepth()
            elif self.model_name.lower() == 'midas':
                self._load_midas()
            else:
                raise ValueError(f"Unsupported depth model: {self.model_name}")
                
            logging.info(f"Successfully loaded {self.model_name} on {self.device}")
            
        except Exception as e:
            logging.error(f"Failed to load depth model {self.model_name}: {e}")
            raise
    
    def _load_dpt(self):
        """加载DPT模型"""
        try:
            from transformers import DPTForDepthEstimation, DPTImageProcessor
            
            if self.use_local_weights and self.model_path and os.path.exists(self.model_path):
                # 从本地加载
                logging.info(f"Loading DPT model from local path: {self.model_path}")
                self.processor = DPTImageProcessor.from_pretrained(self.model_path)
                self.model = DPTForDepthEstimation.from_pretrained(self.model_path)
            else:
                # 从网络加载
                logging.info("Loading DPT model from Hugging Face hub...")
                model_id = "Intel/dpt-large"
                self.processor = DPTImageProcessor.from_pretrained(model_id)
                self.model = DPTForDepthEstimation.from_pretrained(model_id)
            
            self.model.to(self.device)
            self.model.eval()
            logging.info("DPT model loaded successfully")
            
        except ImportError as e:
            logging.error(f"DPT dependencies missing: {e}")
            logging.error("Please install: pip install transformers>=4.36.0")
            raise
        except Exception as e:
            logging.error(f"Failed to load DPT model: {e}")
            raise
    
    def _load_zoedepth(self):
        """加载ZoeDepth模型"""
        try:
            if self.use_local_weights and self.model_path:
                # 从本地加载模型权重
                logging.info(f"Loading ZoeDepth model from local weights: {self.model_path}")
                self._load_zoedepth_local()
            else:
                # 从网络加载ZoeDepth
                try:
                    import torch.hub
                    logging.info("Loading ZoeDepth model from torch.hub...")
                    self.model = torch.hub.load('isl-org/ZoeDepth', 'ZoeD_N', pretrained=True)
                    self.model.to(self.device)
                    self.model.eval()
                    logging.info("ZoeDepth model loaded successfully from hub")
                except Exception as e:
                    logging.warning(f"Failed to load ZoeDepth from hub: {e}")
                    # 回退到MiDaS
                    logging.info("Falling back to MiDaS model...")
                    self._load_midas()
                    
        except ImportError as e:
            logging.error(f"Dependencies missing: {e}")
            logging.error("Please install: pip install torch torchvision")
            raise
        except Exception as e:
            logging.error(f"Failed to load ZoeDepth model: {e}")
            raise
    
    def _load_zoedepth_local(self):
        """从本地文件夹加载ZoeDepth模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        logging.info(f"Loading ZoeDepth model from local folder: {self.model_path}")
        
        # 先清理torch.hub缓存中损坏的文件
        self._clean_torch_hub_cache()
        
        # 确定模型文件路径
        if os.path.isdir(self.model_path):
            # 文件夹路径 - 查找ZoeDepth文件
            model_file = os.path.join(self.model_path, "ZoeD_M12_NK.pt")
        else:
            # 直接的.pt文件路径
            model_file = self.model_path
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # 方法1: 尝试加载本地checkpoint
        try:
            logging.info(f"Loading local ZoeDepth checkpoint: {model_file}")
            
            # 首先构建模型结构（不加载预训练权重）
            # 检查文件名决定使用哪个变体
            if 'NK' in os.path.basename(model_file):
                logging.info("Building ZoeDepth NK model structure...")
                self.model = torch.hub.load('isl-org/ZoeDepth', 'ZoeD_NK', pretrained=False)
            else:
                logging.info("Building ZoeDepth N model structure...")
                self.model = torch.hub.load('isl-org/ZoeDepth', 'ZoeD_N', pretrained=False)
            
            # 然后加载本地权重
            logging.info("Loading local checkpoint...")
            checkpoint = torch.load(model_file, map_location=self.device)
            
            # 根据checkpoint格式加载权重
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    # 检查'model'键的值是完整模型还是state_dict
                    model_data = checkpoint['model']
                    if isinstance(model_data, dict):
                        # 'model'键包含的是state_dict，不是完整模型
                        self.model.load_state_dict(model_data, strict=False)
                        logging.info("Loaded state_dict from checkpoint['model']")
                    elif hasattr(model_data, 'eval'):
                        # 'model'键包含的是完整模型
                        self.model = model_data
                        logging.info("Loaded complete model from checkpoint['model']")
                    else:
                        raise ValueError(f"Invalid model data type in checkpoint: {type(model_data)}")
                elif 'state_dict' in checkpoint:
                    # 如果有state_dict，加载到模型中
                    self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                    logging.info("Loaded state_dict from checkpoint['state_dict']")
                else:
                    # 尝试将整个checkpoint作为state_dict
                    self.model.load_state_dict(checkpoint, strict=False)
                    logging.info("Loaded checkpoint as state_dict")
            else:
                # checkpoint可能直接就是模型
                if hasattr(checkpoint, 'eval'):
                    self.model = checkpoint
                    logging.info("Checkpoint is a complete model")
                else:
                    raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
            
            logging.info("Successfully loaded ZoeDepth with local weights")
            
        except Exception as e1:
            logging.warning(f"Local checkpoint loading failed: {e1}")
            
            # 方法2: 回退到torch.hub预训练模型
            try:
                logging.info("Falling back to torch.hub pretrained ZoeDepth...")
                self.model = torch.hub.load('isl-org/ZoeDepth', 'ZoeD_N', pretrained=True)
                logging.info("Successfully loaded ZoeDepth from torch.hub")
                
            except Exception as e2:
                logging.warning(f"Torch.hub ZoeDepth failed: {e2}")
                
                # 方法3: 最后回退到MiDaS
                logging.info("Final fallback to MiDaS...")
                try:
                    self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
                    self.model_name = 'MiDaS'  # 更新模型名称
                    # 加载MiDaS变换
                    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
                    self.transform = midas_transforms.default_transform
                    logging.info("Successfully loaded MiDaS as final fallback")
                except Exception as e3:
                    raise RuntimeError(f"All loading methods failed. Local: {e1}, Hub: {e2}, MiDaS: {e3}")
        
        self.processor = None  # 本地加载不使用processor
        
        # 验证加载的模型类型
        if isinstance(self.model, dict):
            raise RuntimeError(f"Loaded object is a dictionary, not a model: {type(self.model)}")
        
        # 确保模型在正确的设备上并设为评估模式
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        if hasattr(self.model, 'eval'):
            self.model.eval()
            
        # 验证模型可以进行推理
        if not (hasattr(self.model, 'infer_pil') or hasattr(self.model, 'infer') or 
                (hasattr(self.model, '__call__') and hasattr(self.model, 'forward'))):
            raise RuntimeError(f"Model doesn't have inference capability: {type(self.model)}")
            
        logging.info(f"Model {self.model_name} loaded and validated successfully")
    
    def _clean_torch_hub_cache(self):
        """清理损坏的torch.hub缓存文件"""
        try:
            import shutil
            hub_dir = torch.hub.get_dir()
            cache_file = os.path.join(hub_dir, "checkpoints", "ZoeD_M12_N.pt")
            if os.path.exists(cache_file):
                # 检查文件是否损坏
                try:
                    torch.load(cache_file, map_location='cpu')
                except:
                    logging.info("Removing corrupted cache file...")
                    os.remove(cache_file)
        except Exception as e:
            logging.debug(f"Cache cleanup failed: {e}")
    
    def _load_midas(self):
        """加载MiDaS模型"""
        try:
            import torch.hub
            logging.info("Loading MiDaS model...")
            # 加载MiDaS模型
            self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
            self.model.to(self.device)
            self.model.eval()
            
            # 加载变换
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            self.transform = midas_transforms.default_transform
            logging.info("MiDaS model loaded successfully")
        except ImportError as e:
            logging.error(f"MiDaS dependencies missing: {e}")
            logging.error("Please install: pip install timm")
            raise
        except Exception as e:
            logging.error(f"Failed to load MiDaS model: {e}")
            logging.error("This might be due to network issues. Please check your internet connection.")
            raise
    
    def process(self, image: Image.Image) -> np.ndarray:
        """
        处理图像生成深度图
        
        Args:
            image: PIL图像对象
            
        Returns:
            深度图数组 (H, W)，单位：米
        """
        if self.model is None:
            raise RuntimeError("Depth model not loaded")
        
        try:
            # 转换为numpy数组
            img_array = np.array(image)
            original_size = image.size  # (width, height)
            
            if self.model_name.lower() == 'dpt':
                # DPT处理（来自transformers）
                with torch.no_grad():
                    # 使用processor预处理
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # 推理
                    outputs = self.model(**inputs)
                    predicted_depth = outputs.predicted_depth
                    
                    # 插值到原始大小
                    prediction = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=(original_size[1], original_size[0]),
                        mode="bicubic",
                        align_corners=False,
                    )
                    
                    # 转换为numpy并归一化到米制单位
                    depth = prediction.squeeze().cpu().numpy()
                    
                    # DPT输出是相对深度，需要缩放到合理的米制范围（0-100米）
                    depth_min = depth.min()
                    depth_max = depth.max()
                    if depth_max > depth_min:
                        depth = (depth - depth_min) / (depth_max - depth_min)
                        depth = depth * 80.0 + 0.5  # 缩放到0.5-80.5米范围
                    
            elif self.model_name.lower() == 'zoedepth':
                # ZoeDepth处理
                with torch.no_grad():
                    # 预处理
                    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0).to(self.device)
                    
                    # 推理
                    depth = self.model(img_tensor)
                    
                    # 后处理
                    if isinstance(depth, dict):
                        depth = depth.get('metric_depth', depth.get('depth', depth))
                    
                    depth = depth.squeeze().cpu().numpy()
                    
            elif self.model_name.lower() == 'midas':
                # MiDaS处理
                with torch.no_grad():
                    # 预处理
                    input_batch = self.transform(image).to(self.device)
                    
                    # 推理
                    prediction = self.model(input_batch)
                    
                    # 后处理
                    depth = prediction.squeeze().cpu().numpy()
                    
                    # MiDaS输出的是反向深度，需要转换
                    depth = 1.0 / (depth + 1e-6)
                    
                    # 缩放到合理的米制范围
                    depth = depth * 10.0  # 经验值缩放
                    
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            # 确保深度图为正值并限制在合理范围内（0-100米）
            depth = np.clip(depth, 0, 100)
            
            return depth
            
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            raise
    
    def generate_depth_map(self, rgb_image_path: str) -> np.ndarray:
        """
        生成深度图
        
        Args:
            rgb_image_path: RGB图像路径
            
        Returns:
            深度图数组
        """
        try:
            # 加载RGB图像
            if not os.path.exists(rgb_image_path):
                raise FileNotFoundError(f"RGB image not found: {rgb_image_path}")
                
            image = Image.open(rgb_image_path).convert('RGB')
            image_np = np.array(image)
            
            # 生成深度图
            if self.model_name.lower() == 'zoedepth':
                depth = self._generate_depth_zoedepth(image_np)
            elif self.model_name.lower() == 'midas':
                depth = self._generate_depth_midas(image_np)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            return depth
            
        except Exception as e:
            logging.error(f"Error generating depth map for {rgb_image_path}: {e}")
            raise
    
    def _generate_depth_zoedepth(self, image_np: np.ndarray) -> np.ndarray:
        """使用ZoeDepth生成深度图"""
        try:
            with torch.no_grad():
                # ZoeDepth期望PIL图像
                image_pil = Image.fromarray(image_np)
                original_size = image_pil.size
                
                # 尝试不同的推理方法
                if self.processor is not None:
                    # 使用Hugging Face processor
                    inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    # 提取深度图
                    if hasattr(outputs, 'depth'):
                        depth = outputs.depth
                    elif hasattr(outputs, 'prediction'):
                        depth = outputs.prediction
                    elif hasattr(outputs, 'last_hidden_state'):
                        depth = outputs.last_hidden_state
                    else:
                        depth = outputs
                elif hasattr(self.model, 'infer_pil'):
                    # 标准ZoeDepth API
                    depth = self.model.infer_pil(image_pil)
                elif hasattr(self.model, 'infer'):
                    # 可能的替代API
                    depth = self.model.infer(image_pil)
                elif hasattr(self.model, '__call__') and hasattr(self.model, 'forward'):
                    # 确保是真正的PyTorch模型而不是字典
                    # 将PIL图像转换为tensor
                    image_tensor = torch.from_numpy(image_np).float()
                    if len(image_tensor.shape) == 3:
                        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
                    image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度
                    image_tensor = image_tensor.to(self.device)
                    
                    # 归一化
                    image_tensor = image_tensor / 255.0
                    
                    # 推理
                    depth = self.model(image_tensor)
                    
                    # 如果输出是字典，提取深度图
                    if isinstance(depth, dict):
                        # 查找可能的深度键
                        for key in ['depth', 'pred', 'prediction', 'output']:
                            if key in depth:
                                depth = depth[key]
                                break
                    
                    # 移除batch维度
                    if isinstance(depth, torch.Tensor) and depth.dim() == 4:
                        depth = depth.squeeze(0)
                    if isinstance(depth, torch.Tensor) and depth.dim() == 3:
                        depth = depth.squeeze(0)
                else:
                    raise RuntimeError("Cannot find appropriate inference method for the model")
                
                # 转换为numpy数组
                if isinstance(depth, torch.Tensor):
                    depth_np = depth.cpu().numpy()
                else:
                    depth_np = depth
                
                # 如果输出形状与输入不匹配，进行调整
                if depth_np.shape != (image_np.shape[0], image_np.shape[1]):
                    import cv2
                    depth_np = cv2.resize(depth_np, (image_np.shape[1], image_np.shape[0]))
                
                logging.debug(f"Generated depth map with shape: {depth_np.shape}, range: [{depth_np.min():.2f}, {depth_np.max():.2f}]")
                return depth_np
                
        except Exception as e:
            logging.error(f"Error in ZoeDepth inference: {e}")
            import traceback
            traceback.print_exc()
            # 返回零填充深度图作为后备
            return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
    
    def _generate_depth_midas(self, image_np: np.ndarray) -> np.ndarray:
        """使用MiDaS生成深度图"""
        try:
            with torch.no_grad():
                # 预处理图像
                input_tensor = self.transform(image_np).to(self.device)
                
                # 推理
                prediction = self.model(input_tensor)
                
                # 后处理
                depth = prediction.squeeze().cpu().numpy()
                
                # MiDaS输出逆深度，需要转换为真实深度
                # 避免除零并应用合理的深度范围
                depth = np.clip(depth, 1e-6, None)  # 避免除零
                depth = 1.0 / depth  # 转换为真实深度
                
                # 调整输出尺寸以匹配输入
                if depth.shape != (image_np.shape[0], image_np.shape[1]):
                    import cv2
                    depth = cv2.resize(depth, (image_np.shape[1], image_np.shape[0]))
                
                # 应用合理的深度范围限制（0-100米）
                depth = np.clip(depth, 0.1, 100.0)
                
                logging.debug(f"Generated MiDaS depth map with shape: {depth.shape}, range: [{depth.min():.2f}, {depth.max():.2f}]")
                return depth
                
        except Exception as e:
            logging.error(f"Error in MiDaS inference: {e}")
            # 返回零填充深度图作为后备
            return np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)
    
    def save_depth_map(self, depth_array: np.ndarray, output_path: str, format: str = 'png16'):
        """
        保存深度图
        
        Args:
            depth_array: 深度数组（单位：米）
            output_path: 输出路径
            format: 保存格式 ('png16', 'png8', 'npy')
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if format == 'png16':
                # 保存为16位PNG（统一转换为毫米单位）
                depth_normalized = self._normalize_depth_for_png16(depth_array)
                if cv2 is not None:
                    cv2.imwrite(output_path, depth_normalized)
                else:
                    # 使用PIL作为后备
                    Image.fromarray(depth_normalized).save(output_path)
                
            elif format == 'png8':
                # 保存为8位PNG（用于可视化）
                depth_normalized = self._normalize_depth_for_png8(depth_array)
                if cv2 is not None:
                    cv2.imwrite(output_path, depth_normalized)
                else:
                    # 使用PIL作为后备
                    Image.fromarray(depth_normalized).save(output_path)
                
            elif format == 'npy':
                # 保存为numpy数组（保持原始米单位）
                np.save(output_path, depth_array)
                
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            logging.debug(f"Saved depth map to {output_path}")
            
        except Exception as e:
            logging.error(f"Error saving depth map to {output_path}: {e}")
            raise
    
    def _normalize_depth_for_png16(self, depth: np.ndarray) -> np.ndarray:
        """将深度图归一化为16位PNG格式（统一使用毫米为单位）"""
        # depth输入单位：米
        # 转换为毫米
        depth_mm = depth * 1000.0
        
        # 裁剪到16位整数范围（0-65535毫米，即0-65.535米）
        depth_clipped = np.clip(depth_mm, 0, 65535)
        
        # 转换为uint16
        depth_normalized = depth_clipped.astype(np.uint16)
        
        return depth_normalized
    
    def _normalize_depth_for_png8(self, depth: np.ndarray) -> np.ndarray:
        """将深度图归一化为8位PNG格式（用于可视化）"""
        # 归一化到0-255
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            depth_normalized = np.zeros_like(depth, dtype=np.uint8)
            
        return depth_normalized
    
    def batch_process_depths(self, rgb_paths: List[str], output_dir: str, batch_size: int = 8) -> List[str]:
        """
        批量处理深度图
        
        Args:
            rgb_paths: RGB图像路径列表
            output_dir: 输出目录
            batch_size: 批量大小
            
        Returns:
            生成的深度图路径列表
        """
        depth_paths = []
        
        try:
            for i in range(0, len(rgb_paths), batch_size):
                batch_paths = rgb_paths[i:i+batch_size]
                
                for rgb_path in batch_paths:
                    # 生成输出路径
                    rgb_filename = Path(rgb_path).stem
                    depth_filename = f"{rgb_filename}_depth.png"
                    depth_path = os.path.join(output_dir, depth_filename)
                    
                    # 生成深度图
                    depth_array = self.generate_depth_map(rgb_path)
                    
                    # 保存深度图
                    self.save_depth_map(depth_array, depth_path)
                    
                    depth_paths.append(depth_path)
                
                logging.info(f"Processed batch {i//batch_size + 1}/{(len(rgb_paths)-1)//batch_size + 1}")
            
            return depth_paths
            
        except Exception as e:
            logging.error(f"Error in batch processing: {e}")
            raise
    
    def validate_depth_map(self, depth_path: str) -> bool:
        """
        验证深度图文件
        
        Args:
            depth_path: 深度图路径
            
        Returns:
            是否有效
        """
        try:
            if not os.path.exists(depth_path):
                return False
                
            # 尝试加载深度图
            if depth_path.endswith('.npy'):
                depth = np.load(depth_path)
            else:
                if cv2 is not None:
                    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                else:
                    # 使用PIL作为后备
                    depth = np.array(Image.open(depth_path))
                
            # 检查维度和数据类型
            if depth is None or len(depth.shape) != 2:
                return False
                
            # 检查数值范围
            if np.any(np.isnan(depth)) or np.any(np.isinf(depth)):
                return False
                
            return True
            
        except Exception:
            return False
    
    def get_depth_statistics(self, depth_array: np.ndarray) -> Dict:
        """
        获取深度图统计信息
        
        Args:
            depth_array: 深度数组
            
        Returns:
            统计信息字典
        """
        stats = {
            'shape': depth_array.shape,
            'dtype': str(depth_array.dtype),
            'min': float(np.min(depth_array)),
            'max': float(np.max(depth_array)),
            'mean': float(np.mean(depth_array)),
            'std': float(np.std(depth_array)),
            'median': float(np.median(depth_array))
        }
        
        return stats