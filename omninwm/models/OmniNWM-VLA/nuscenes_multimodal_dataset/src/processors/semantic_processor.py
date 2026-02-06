"""
语义分割处理器

使用预训练的语义分割模型生成语义分割图：
- 支持多种分割模型（SegFormer, Mask2Former等）  
- 批量处理优化
- 类别映射管理
- GPU加速支持
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from PIL import Image
import logging
import os
from pathlib import Path
import json


class SemanticProcessor:
    """语义分割处理器"""
    
    def __init__(self, model_name: str = 'SegFormer', device: str = 'auto',
                 model_path: Optional[str] = None, use_local_weights: bool = False):
        """
        初始化语义分割处理器
        
        Args:
            model_name: 分割模型名称
            device: 计算设备
            model_path: 本地模型权重路径（可选）
            use_local_weights: 是否使用本地权重
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model_path = model_path
        self.use_local_weights = use_local_weights
        self.model = None
        self.processor = None
        self.class_mapping = {}
        self._load_semantic_model()
        
    def _setup_device(self, device: str) -> str:
        """设置计算设备"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_semantic_model(self):
        """加载语义分割模型"""
        try:
            if self.model_name.lower() == 'segformer':
                self._load_segformer()
            elif self.model_name.lower() == 'mask2former':
                self._load_mask2former()
            else:
                raise ValueError(f"Unsupported semantic model: {self.model_name}")
                
            logging.info(f"Successfully loaded {self.model_name} on {self.device}")
            
        except Exception as e:
            logging.error(f"Failed to load semantic model {self.model_name}: {e}")
            raise
    
    def _load_segformer(self):
        """加载SegFormer模型"""
        try:
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            
            if self.use_local_weights and self.model_path:
                # 从本地加载模型
                logging.info(f"Loading SegFormer model from local path: {self.model_path}")
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Model path not found: {self.model_path}")
                
                # 从本地目录加载模型和处理器
                self.processor = SegformerImageProcessor.from_pretrained(self.model_path)
                self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_path)
            else:
                # 从Hugging Face加载模型
                logging.info("Loading SegFormer model from Hugging Face...")
                model_id = "nvidia/segformer-b5-finetuned-ade-640-640"  # 使用ADE20K版本
                self.processor = SegformerImageProcessor.from_pretrained(model_id)
                self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
            
            self.model.to(self.device)
            self.model.eval()
            
            # 设置类别数量（ADE20K有150个类别）
            self.num_classes = 150
            # 设置nuScenes类别映射
            self.class_mapping = self._get_nuscenes_mapping()
            # 创建ADE20K到nuScenes的映射
            self.ade20k_to_nuscenes = self._create_ade20k_to_nuscenes_mapping()
            
            logging.info(f"SegFormer model loaded successfully with {self.num_classes} classes")
            logging.info("Created ADE20K to nuScenes mapping")
            
        except ImportError as e:
            logging.error(f"SegFormer dependencies missing: {e}")
            logging.error("Please install: pip install transformers>=4.36.0")
            raise
        except Exception as e:
            logging.error(f"Failed to load SegFormer model: {e}")
            logging.error("This might be due to network issues. Please check your internet connection.")
            raise
    
    def _load_mask2former(self):
        """加载Mask2Former模型"""
        try:
            from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
            
            logging.info("Loading Mask2Former model...")
            # 加载预训练模型
            model_id = "facebook/mask2former-swin-large-cityscapes-semantic"
            self.processor = Mask2FormerImageProcessor.from_pretrained(model_id)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
            self.model.to(self.device)
            self.model.eval()
            
            # 设置类别映射（使用nuScenes类别）
            self.class_mapping = self._get_nuscenes_mapping()
            
            # 创建Cityscapes到nuScenes的类别映射
            self.cityscapes_to_nuscenes = self._create_cityscapes_to_nuscenes_mapping()
            
            logging.info("Mask2Former model loaded successfully")
            
        except ImportError as e:
            logging.error(f"Mask2Former dependencies missing: {e}")
            logging.error("Please install: pip install transformers>=4.36.0")
            raise
        except Exception as e:
            logging.error(f"Failed to load Mask2Former model: {e}")
            logging.error("This might be due to network issues. Please check your internet connection.")
            raise
    
    def _get_nuscenes_mapping(self) -> Dict[int, str]:
        """获取nuScenes-lidarseg数据集的16个核心类别映射"""
        return {
            0: 'void',
            1: 'barrier',
            2: 'bicycle',
            3: 'bus',
            4: 'car',
            5: 'construction_vehicle',
            6: 'motorcycle',
            7: 'pedestrian',
            8: 'traffic_cone',
            9: 'trailer',
            10: 'truck',
            11: 'driveable_surface',
            12: 'other_flat',
            13: 'sidewalk',
            14: 'terrain',
            15: 'manmade',
            16: 'vegetation'
        }
    
    def _create_ade20k_to_nuscenes_mapping(self) -> Dict[int, int]:
        """创建ADE20K到nuScenes的映射"""
        # ADE20K类别索引参考：
        # https://github.com/CSAILVision/ADE20K
        return {
            # ADE20K ID -> nuScenes ID
            0: 0,     # void -> void
            1: 15,    # wall -> manmade
            2: 15,    # building -> manmade
            3: 0,     # sky -> void
            4: 11,    # floor -> driveable_surface
            5: 16,    # tree -> vegetation
            6: 0,     # ceiling -> void  
            7: 11,    # road/route -> driveable_surface
            8: 15,    # bed -> manmade
            9: 15,    # window -> manmade
            10: 16,   # grass -> vegetation
            11: 15,   # cabinet -> manmade
            12: 13,   # sidewalk/pavement -> sidewalk
            13: 7,    # person -> pedestrian
            14: 14,   # earth/ground -> terrain
            15: 15,   # door -> manmade
            16: 15,   # table -> manmade
            17: 14,   # mountain -> terrain
            18: 16,   # plant -> vegetation
            19: 15,   # curtain -> manmade
            20: 4,    # car -> car
            21: 15,   # chair -> manmade
            22: 4,    # automobile -> car
            23: 15,   # sofa -> manmade
            24: 15,   # shelf -> manmade
            25: 15,   # house -> manmade
            26: 0,    # sea -> void
            27: 15,   # mirror -> manmade
            28: 15,   # rug -> manmade
            29: 14,   # field -> terrain
            30: 15,   # armchair -> manmade
            31: 15,   # seat -> manmade
            32: 1,    # fence -> barrier
            33: 15,   # desk -> manmade
            34: 14,   # rock -> terrain
            35: 15,   # wardrobe -> manmade
            36: 15,   # lamp -> manmade
            37: 15,   # bathtub -> manmade
            38: 1,    # railing -> barrier
            39: 15,   # cushion -> manmade
            40: 15,   # base -> manmade
            41: 15,   # box -> manmade
            42: 15,   # column -> manmade
            43: 15,   # signboard/sign -> manmade
            44: 15,   # chest of drawers -> manmade
            45: 15,   # counter -> manmade
            46: 14,   # sand -> terrain
            47: 15,   # sink -> manmade
            48: 15,   # skyscraper -> manmade
            49: 15,   # fireplace -> manmade
            50: 15,   # refrigerator -> manmade
            51: 14,   # grandstand -> terrain
            52: 11,   # path -> driveable_surface
            53: 15,   # stairs -> manmade
            54: 11,   # runway -> driveable_surface
            55: 15,   # case -> manmade
            56: 15,   # pool table -> manmade
            57: 15,   # pillow -> manmade
            58: 15,   # screen door -> manmade
            59: 15,   # stairway -> manmade
            60: 0,    # river -> void
            61: 15,   # bridge -> manmade
            62: 15,   # bookcase -> manmade
            63: 15,   # blind -> manmade
            64: 15,   # coffee table -> manmade
            65: 15,   # toilet -> manmade
            66: 16,   # flower -> vegetation
            67: 15,   # book -> manmade
            68: 14,   # hill -> terrain
            69: 15,   # bench -> manmade
            70: 15,   # countertop -> manmade
            71: 15,   # stove -> manmade
            72: 16,   # palm tree -> vegetation
            73: 15,   # kitchen island -> manmade
            74: 15,   # computer -> manmade
            75: 15,   # swivel chair -> manmade
            76: 15,   # boat -> manmade
            77: 15,   # bar -> manmade
            78: 15,   # arcade machine -> manmade
            79: 15,   # hovel -> manmade
            80: 3,    # bus -> bus
            81: 15,   # towel -> manmade
            82: 15,   # light -> manmade
            83: 10,   # truck -> truck
            84: 15,   # tower -> manmade
            85: 15,   # chandelier -> manmade
            86: 15,   # awning -> manmade
            87: 15,   # streetlight -> manmade
            88: 15,   # booth -> manmade
            89: 15,   # television -> manmade
            90: 15,   # airplane -> manmade
            91: 14,   # dirt track -> terrain
            92: 15,   # apparel -> manmade
            93: 15,   # pole -> manmade
            94: 14,   # land -> terrain
            95: 15,   # bannister -> manmade
            96: 15,   # escalator -> manmade
            97: 15,   # ottoman -> manmade
            98: 15,   # bottle -> manmade
            99: 15,   # buffet -> manmade
            100: 15,  # poster -> manmade
            101: 15,  # stage -> manmade
            102: 5,   # van -> construction_vehicle
            103: 15,  # ship -> manmade
            104: 15,  # fountain -> manmade
            105: 15,  # conveyer belt -> manmade
            106: 15,  # canopy -> manmade
            107: 15,  # washer -> manmade
            108: 15,  # plaything -> manmade
            109: 15,  # swimming pool -> manmade
            110: 15,  # stool -> manmade
            111: 15,  # barrel -> manmade
            112: 15,  # basket -> manmade
            113: 0,   # waterfall -> void
            114: 15,  # tent -> manmade
            115: 15,  # bag -> manmade
            116: 6,   # minibike/motorbike -> motorcycle
            117: 15,  # cradle -> manmade
            118: 15,  # oven -> manmade
            119: 15,  # ball -> manmade
            120: 15,  # food -> manmade
            121: 15,  # step -> manmade
            122: 15,  # tank -> manmade
            123: 15,  # trade name -> manmade
            124: 15,  # microwave -> manmade
            125: 15,  # pot -> manmade
            126: 7,   # animal -> pedestrian (动物归为行人类)
            127: 2,   # bicycle -> bicycle
            128: 0,   # lake -> void
            129: 15,  # dishwasher -> manmade
            130: 15,  # screen -> manmade
            131: 15,  # blanket -> manmade
            132: 15,  # sculpture -> manmade
            133: 15,  # hood -> manmade
            134: 15,  # sconce -> manmade
            135: 15,  # vase -> manmade
            136: 8,   # traffic light -> traffic_cone (归为交通设施)
            137: 15,  # tray -> manmade
            138: 15,  # ashcan -> manmade
            139: 15,  # fan -> manmade
            140: 15,  # pier -> manmade
            141: 15,  # crt screen -> manmade
            142: 15,  # plate -> manmade
            143: 15,  # monitor -> manmade
            144: 15,  # bulletin board -> manmade
            145: 15,  # shower -> manmade
            146: 15,  # radiator -> manmade
            147: 15,  # glass -> manmade
            148: 15,  # clock -> manmade
            149: 15,  # flag -> manmade
            # 默认值
            255: 0    # unlabeled -> void
        }
    
    def process(self, image: Image.Image) -> np.ndarray:
        """
        处理图像生成语义分割图
        
        Args:
            image: PIL图像对象
            
        Returns:
            语义分割图数组 (H, W)，值为类别索引
        """
        if self.model is None:
            raise RuntimeError("Semantic model not loaded")
        
        try:
            if self.model_name.lower() == 'segformer':
                # SegFormer处理
                with torch.no_grad():
                    # 预处理
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # 推理
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # 后处理 - 上采样到原始尺寸并获取类别
                    upsampled_logits = torch.nn.functional.interpolate(
                        logits,
                        size=image.size[::-1],  # (height, width)
                        mode="bilinear",
                        align_corners=False
                    )
                    
                    # 获取每个像素的类别
                    semantic_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
                    
                    # 应用ADE20K到nuScenes的映射
                    if hasattr(self, 'ade20k_to_nuscenes'):
                        mapped = np.zeros_like(semantic_map, dtype=np.uint8)
                        for ade20k_id, nuscenes_id in self.ade20k_to_nuscenes.items():
                            mapped[semantic_map == ade20k_id] = nuscenes_id
                        semantic_map = mapped
                    
            elif self.model_name.lower() == 'mask2former':
                # Mask2Former处理
                with torch.no_grad():
                    # 预处理
                    inputs = self.processor(images=image, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # 推理
                    outputs = self.model(**inputs)
                    
                    # 后处理
                    semantic_segmentation = self.processor.post_process_semantic_segmentation(
                        outputs, target_sizes=[(image.height, image.width)]
                    )[0]
                    
                    semantic_map = semantic_segmentation.cpu().numpy()
                    
                    # 如果需要，映射到nuScenes类别
                    if hasattr(self, 'cityscapes_to_nuscenes'):
                        mapped = np.zeros_like(semantic_map)
                        for cityscapes_id, nuscenes_id in self.cityscapes_to_nuscenes.items():
                            mapped[semantic_map == cityscapes_id] = nuscenes_id
                        semantic_map = mapped
                    
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            return semantic_map.astype(np.uint8)
            
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            raise
    
    def _create_cityscapes_to_nuscenes_mapping(self) -> Dict[int, int]:
        """创建Cityscapes类别到nuScenes类别的映射"""
        return {
            # Cityscapes ID -> nuScenes ID
            0: 11,   # road -> driveable_surface
            1: 13,   # sidewalk -> sidewalk
            2: 15,   # building -> manmade
            3: 15,   # wall -> manmade
            4: 1,    # fence -> barrier
            5: 15,   # pole -> manmade
            6: 15,   # traffic_light -> manmade
            7: 15,   # traffic_sign -> manmade
            8: 16,   # vegetation -> vegetation
            9: 14,   # terrain -> terrain
            10: 0,   # sky -> void (背景)
            11: 7,   # person -> pedestrian
            12: 7,   # rider -> pedestrian (暂时映射到行人)
            13: 4,   # car -> car
            14: 10,  # truck -> truck
            15: 3,   # bus -> bus
            16: 3,   # train -> bus (映射到最接近的类别)
            17: 6,   # motorcycle -> motorcycle
            18: 2,   # bicycle -> bicycle
            255: 0   # void -> void
        }
    
    def _convert_cityscapes_to_nuscenes(self, cityscapes_map: np.ndarray) -> np.ndarray:
        """将Cityscapes格式的语义图转换为nuScenes格式"""
        nuscenes_map = np.zeros_like(cityscapes_map, dtype=np.uint8)
        
        for cityscapes_id, nuscenes_id in self.cityscapes_to_nuscenes.items():
            mask = cityscapes_map == cityscapes_id
            nuscenes_map[mask] = nuscenes_id
            
        return nuscenes_map
    
    def _convert_ade20k_to_nuscenes(self, ade20k_map: np.ndarray) -> np.ndarray:
        """将ADE20K格式的语义图转换为nuScenes格式"""
        nuscenes_map = np.zeros_like(ade20k_map, dtype=np.uint8)
        
        # 使用映射表转换
        if hasattr(self, 'ade20k_to_nuscenes'):
            for ade20k_id, nuscenes_id in self.ade20k_to_nuscenes.items():
                mask = ade20k_map == ade20k_id
                nuscenes_map[mask] = nuscenes_id
        else:
            # 如果没有映射表，使用默认映射（将所有未知类别映射为void）
            logging.warning("ADE20K to nuScenes mapping not found, using default mapping")
            nuscenes_map[:] = 0  # 全部设为void
            
        return nuscenes_map
    
    def generate_semantic_map(self, rgb_image_path: str) -> np.ndarray:
        """
        生成语义分割图
        
        Args:
            rgb_image_path: RGB图像路径
            
        Returns:
            语义分割图数组（nuScenes类别）
        """
        try:
            # 加载RGB图像
            if not os.path.exists(rgb_image_path):
                raise FileNotFoundError(f"RGB image not found: {rgb_image_path}")
                
            image = Image.open(rgb_image_path).convert('RGB')
            
            # 生成语义分割图
            if self.model_name.lower() == 'segformer':
                # SegFormer使用ADE20K，需要转换到nuScenes
                ade20k_map = self._generate_semantic_segformer(image)
                semantic_map = self._convert_ade20k_to_nuscenes(ade20k_map)
            elif self.model_name.lower() == 'mask2former':
                # Mask2Former使用Cityscapes，需要转换到nuScenes
                cityscapes_map = self._generate_semantic_mask2former(image)
                semantic_map = self._convert_cityscapes_to_nuscenes(cityscapes_map)
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")
            
            return semantic_map
            
        except Exception as e:
            logging.error(f"Error generating semantic map for {rgb_image_path}: {e}")
            raise
    
    def _generate_semantic_segformer(self, image: Image.Image) -> np.ndarray:
        """使用SegFormer生成语义分割图"""
        try:
            with torch.no_grad():
                # 预处理图像
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 推理
                outputs = self.model(**inputs)
                
                # 后处理
                logits = outputs.logits
                upsampled_logits = torch.nn.functional.interpolate(
                    logits,
                    size=image.size[::-1],  # (height, width)
                    mode="bilinear",
                    align_corners=False,
                )
                
                # 获取预测类别
                predicted = upsampled_logits.argmax(dim=1)
                semantic_map = predicted.squeeze().cpu().numpy().astype(np.uint8)
                
                logging.debug(f"Generated SegFormer semantic map with shape: {semantic_map.shape}, unique classes: {np.unique(semantic_map)}")
                return semantic_map
                
        except Exception as e:
            logging.error(f"Error in SegFormer inference: {e}")
            # 返回零填充语义图作为后备
            return np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
    
    def _generate_semantic_mask2former(self, image: Image.Image) -> np.ndarray:
        """使用Mask2Former生成语义分割图"""
        try:
            with torch.no_grad():
                # 预处理图像
                inputs = self.processor(images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 推理
                outputs = self.model(**inputs)
                
                # 后处理
                predicted_semantic_map = self.processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[image.size[::-1]]
                )[0]
                
                semantic_map = predicted_semantic_map.cpu().numpy().astype(np.uint8)
                
                logging.debug(f"Generated Mask2Former semantic map with shape: {semantic_map.shape}, unique classes: {np.unique(semantic_map)}")
                return semantic_map
                
        except Exception as e:
            logging.error(f"Error in Mask2Former inference: {e}")
            # 返回零填充语义图作为后备
            return np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
    
    def save_semantic_map(self, semantic_array: np.ndarray, output_path: str, save_colored: bool = False):
        """
        保存语义分割图
        
        Args:
            semantic_array: 语义分割数组
            output_path: 输出路径
            save_colored: 是否同时保存彩色版本
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存类别ID图（单通道）
            semantic_image = Image.fromarray(semantic_array, mode='L')
            semantic_image.save(output_path)
            
            # 可选：保存彩色版本
            if save_colored:
                colored_path = output_path.replace('.png', '_colored.png')
                colored_semantic = self._colorize_semantic_map(semantic_array)
                colored_image = Image.fromarray(colored_semantic)
                colored_image.save(colored_path)
            
            logging.debug(f"Saved semantic map to {output_path}")
            
        except Exception as e:
            logging.error(f"Error saving semantic map to {output_path}: {e}")
            raise
    
    def _colorize_semantic_map(self, semantic_array: np.ndarray) -> np.ndarray:
        """将语义分割图转换为彩色图像"""
        # nuScenes调色板（17个类别）
        palette = np.array([
            [0, 0, 0],        # 0: void
            [255, 120, 50],   # 1: barrier
            [255, 192, 203],  # 2: bicycle
            [255, 255, 0],    # 3: bus
            [0, 0, 142],      # 4: car
            [0, 0, 230],      # 5: construction_vehicle
            [119, 11, 32],    # 6: motorcycle
            [0, 0, 230],      # 7: pedestrian
            [255, 128, 0],    # 8: traffic_cone
            [255, 255, 255],  # 9: trailer
            [0, 0, 70],       # 10: truck
            [128, 64, 128],   # 11: driveable_surface
            [244, 35, 232],   # 12: other_flat
            [244, 35, 232],   # 13: sidewalk
            [152, 251, 152],  # 14: terrain
            [70, 70, 70],     # 15: manmade
            [107, 142, 35],   # 16: vegetation
        ], dtype=np.uint8)
        
        # 创建彩色图像
        h, w = semantic_array.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in enumerate(palette):
            if class_id < len(palette):
                colored[semantic_array == class_id] = color
            
        return colored
    
    def get_class_mapping(self) -> Dict[int, str]:
        """
        获取类别映射
        
        Returns:
            类别ID到类别名称的映射
        """
        return self.class_mapping.copy()
    
    def batch_process_semantics(self, rgb_paths: List[str], output_dir: str, batch_size: int = 4) -> List[str]:
        """
        批量处理语义分割图
        
        Args:
            rgb_paths: RGB图像路径列表
            output_dir: 输出目录
            batch_size: 批量大小
            
        Returns:
            生成的语义分割图路径列表
        """
        semantic_paths = []
        
        try:
            for i in range(0, len(rgb_paths), batch_size):
                batch_paths = rgb_paths[i:i+batch_size]
                
                for rgb_path in batch_paths:
                    # 生成输出路径
                    rgb_filename = Path(rgb_path).stem
                    semantic_filename = f"{rgb_filename}_semantic.png"
                    semantic_path = os.path.join(output_dir, semantic_filename)
                    
                    # 生成语义分割图
                    semantic_array = self.generate_semantic_map(rgb_path)
                    
                    # 保存语义分割图
                    self.save_semantic_map(semantic_array, semantic_path)
                    
                    semantic_paths.append(semantic_path)
                
                logging.info(f"Processed semantic batch {i//batch_size + 1}/{(len(rgb_paths)-1)//batch_size + 1}")
            
            return semantic_paths
            
        except Exception as e:
            logging.error(f"Error in semantic batch processing: {e}")
            raise
    
    def validate_semantic_map(self, semantic_path: str) -> bool:
        """
        验证语义分割图文件
        
        Args:
            semantic_path: 语义分割图路径
            
        Returns:
            是否有效
        """
        try:
            if not os.path.exists(semantic_path):
                return False
                
            # 加载语义分割图
            semantic_image = Image.open(semantic_path)
            semantic_array = np.array(semantic_image)
            
            # 检查维度（应该是单通道）
            if len(semantic_array.shape) != 2:
                return False
                
            # 检查类别ID范围
            unique_classes = np.unique(semantic_array)
            max_class_id = max(self.class_mapping.keys())
            
            if np.any(unique_classes > max_class_id):
                logging.warning(f"Found invalid class IDs in {semantic_path}")
                return False
                
            return True
            
        except Exception:
            return False
    
    def get_semantic_statistics(self, semantic_array: np.ndarray) -> Dict:
        """
        获取语义分割图统计信息
        
        Args:
            semantic_array: 语义分割数组
            
        Returns:
            统计信息字典
        """
        unique_classes, counts = np.unique(semantic_array, return_counts=True)
        total_pixels = semantic_array.size
        
        class_stats = {}
        for class_id, count in zip(unique_classes, counts):
            class_name = self.class_mapping.get(class_id, f"unknown_{class_id}")
            class_stats[class_name] = {
                'count': int(count),
                'percentage': float(count / total_pixels * 100)
            }
        
        stats = {
            'shape': semantic_array.shape,
            'total_pixels': total_pixels,
            'num_classes': len(unique_classes),
            'class_distribution': class_stats
        }
        
        return stats
    
    def save_class_mapping(self, output_path: str):
        """
        保存类别映射到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.class_mapping, f, indent=2)
            logging.info(f"Saved class mapping to {output_path}")
        except Exception as e:
            logging.error(f"Error saving class mapping: {e}")
            raise