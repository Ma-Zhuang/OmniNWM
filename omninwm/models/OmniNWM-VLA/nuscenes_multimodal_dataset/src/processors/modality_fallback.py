"""
模态缺失降级处理模块

当深度图或语义分割图生成失败时，提供多种备用方案：
1. 零填充策略
2. 从RGB图像估算深度图
3. 基于颜色聚类的简单语义分割
4. 历史数据回填
"""

import os
import cv2
import numpy as np
import logging
from PIL import Image
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ModalityFallbackHandler:
    """模态缺失降级处理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化降级处理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.fallback_strategies = {
            'depth': ['rgb_estimation', 'zero_fill', 'pattern_fill'],
            'semantic': ['color_clustering', 'zero_fill', 'pattern_fill']
        }
        
        # 统计信息
        self.stats = {
            'depth_failures': 0,
            'depth_fallbacks': 0,
            'semantic_failures': 0,
            'semantic_fallbacks': 0,
            'fallback_methods_used': {}
        }
        
        logger.info("模态缺失降级处理器初始化完成")
    
    def handle_depth_failure(self, rgb_image_path: str, output_path: str, 
                           strategy: str = 'auto') -> bool:
        """
        处理深度图生成失败
        
        Args:
            rgb_image_path: RGB图像路径
            output_path: 深度图输出路径
            strategy: 降级策略 ('auto', 'rgb_estimation', 'zero_fill', 'pattern_fill')
            
        Returns:
            是否成功创建备用深度图
        """
        try:
            self.stats['depth_failures'] += 1
            
            if strategy == 'auto':
                # 自动选择最优策略
                strategies = self.fallback_strategies['depth']
            else:
                strategies = [strategy]
            
            for method in strategies:
                if self._try_depth_method(rgb_image_path, output_path, method):
                    self.stats['depth_fallbacks'] += 1
                    self._track_fallback_method('depth', method)
                    logger.info(f"深度图降级处理成功: {method}")
                    return True
            
            logger.error("所有深度图降级策略都失败了")
            return False
            
        except Exception as e:
            logger.error(f"深度图降级处理异常: {e}")
            return False
    
    def handle_semantic_failure(self, rgb_image_path: str, output_path: str,
                              strategy: str = 'auto') -> bool:
        """
        处理语义分割图生成失败
        
        Args:
            rgb_image_path: RGB图像路径
            output_path: 语义分割图输出路径
            strategy: 降级策略 ('auto', 'color_clustering', 'zero_fill', 'pattern_fill')
            
        Returns:
            是否成功创建备用语义分割图
        """
        try:
            self.stats['semantic_failures'] += 1
            
            if strategy == 'auto':
                # 自动选择最优策略
                strategies = self.fallback_strategies['semantic']
            else:
                strategies = [strategy]
            
            for method in strategies:
                if self._try_semantic_method(rgb_image_path, output_path, method):
                    self.stats['semantic_fallbacks'] += 1
                    self._track_fallback_method('semantic', method)
                    logger.info(f"语义分割图降级处理成功: {method}")
                    return True
            
            logger.error("所有语义分割图降级策略都失败了")
            return False
            
        except Exception as e:
            logger.error(f"语义分割图降级处理异常: {e}")
            return False
    
    def _try_depth_method(self, rgb_path: str, output_path: str, method: str) -> bool:
        """尝试特定的深度图生成方法"""
        try:
            if method == 'rgb_estimation':
                return self._estimate_depth_from_rgb(rgb_path, output_path)
            elif method == 'zero_fill':
                return self._create_zero_depth(output_path)
            elif method == 'pattern_fill':
                return self._create_pattern_depth(output_path)
            else:
                logger.warning(f"未知的深度图降级方法: {method}")
                return False
                
        except Exception as e:
            logger.error(f"深度图方法 {method} 执行失败: {e}")
            return False
    
    def _try_semantic_method(self, rgb_path: str, output_path: str, method: str) -> bool:
        """尝试特定的语义分割图生成方法"""
        try:
            if method == 'color_clustering':
                return self._cluster_based_semantic(rgb_path, output_path)
            elif method == 'zero_fill':
                return self._create_zero_semantic(output_path)
            elif method == 'pattern_fill':
                return self._create_pattern_semantic(output_path)
            else:
                logger.warning(f"未知的语义分割图降级方法: {method}")
                return False
                
        except Exception as e:
            logger.error(f"语义分割图方法 {method} 执行失败: {e}")
            return False
    
    def _estimate_depth_from_rgb(self, rgb_path: str, output_path: str) -> bool:
        """
        从RGB图像估算深度图
        使用简单的梯度和颜色信息估算相对深度
        """
        try:
            # 加载RGB图像
            image = cv2.imread(rgb_path)
            if image is None:
                return False
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 调整大小到标准尺寸
            gray = cv2.resize(gray, (384, 384))
            
            # 计算梯度强度（边缘通常距离较近）
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # 归一化梯度
            gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            
            # 反转梯度（高梯度=近距离=小深度值）
            estimated_depth = 255 - gradient_magnitude
            
            # 添加一些随机变化和平滑
            noise = np.random.normal(0, 10, estimated_depth.shape)
            estimated_depth = np.clip(estimated_depth + noise, 0, 255)
            
            # 高斯平滑
            estimated_depth = cv2.GaussianBlur(estimated_depth, (5, 5), 1.0)
            
            # 保存为16位深度图
            depth_16bit = (estimated_depth / 255.0 * 65535).astype(np.uint16)
            cv2.imwrite(output_path, depth_16bit)
            
            return True
            
        except Exception as e:
            logger.error(f"RGB深度估算失败: {e}")
            return False
    
    def _cluster_based_semantic(self, rgb_path: str, output_path: str) -> bool:
        """
        基于颜色聚类的简单语义分割
        使用K-means聚类将图像分成几个区域
        """
        try:
            # 加载RGB图像
            image = cv2.imread(rgb_path)
            if image is None:
                return False
            
            # 调整大小
            image = cv2.resize(image, (384, 384))
            
            # 转换为RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 重塑数据用于聚类
            pixel_values = image_rgb.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
            
            # K-means聚类（聚成8个类别）
            k = 8
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, _ = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # 将聚类标签映射到nuScenes语义类别
            semantic_map = self._map_clusters_to_semantic(labels.reshape((384, 384)), image_rgb)
            
            # 保存语义分割图
            semantic_image = Image.fromarray(semantic_map.astype(np.uint8), mode='L')
            semantic_image.save(output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"颜色聚类语义分割失败: {e}")
            return False
    
    def _map_clusters_to_semantic(self, clusters: np.ndarray, rgb_image: np.ndarray) -> np.ndarray:
        """
        将聚类结果映射到nuScenes语义类别
        基于颜色特征和位置信息进行简单推理
        """
        try:
            semantic_map = np.zeros_like(clusters, dtype=np.uint8)
            
            height, width = clusters.shape
            
            # 为每个聚类分配语义标签
            unique_clusters = np.unique(clusters)
            
            for cluster_id in unique_clusters:
                mask = clusters == cluster_id
                
                # 计算聚类的平均颜色
                cluster_pixels = rgb_image[mask]
                mean_color = np.mean(cluster_pixels, axis=0)
                
                # 计算聚类的位置特征
                cluster_positions = np.where(mask)
                mean_y = np.mean(cluster_positions[0]) / height
                mean_x = np.mean(cluster_positions[1]) / width
                
                # 基于颜色和位置推断语义类别
                semantic_class = self._infer_semantic_class(mean_color, mean_y, mean_x)
                
                semantic_map[mask] = semantic_class
            
            return semantic_map
            
        except Exception as e:
            logger.error(f"聚类映射失败: {e}")
            # 返回默认的地面类别
            return np.full_like(clusters, 11, dtype=np.uint8)  # driveable_surface
    
    def _infer_semantic_class(self, color: np.ndarray, pos_y: float, pos_x: float) -> int:
        """
        基于颜色和位置推断语义类别
        
        Args:
            color: RGB颜色 [R, G, B]
            pos_y: 垂直位置比例 (0-1)
            pos_x: 水平位置比例 (0-1)
            
        Returns:
            nuScenes语义类别ID
        """
        try:
            r, g, b = color
            
            # 基于位置的粗略分类
            if pos_y > 0.7:  # 下半部分通常是路面
                if np.mean(color) < 100:  # 暗色
                    return 11  # driveable_surface
                else:
                    return 13  # sidewalk
            
            elif pos_y < 0.3:  # 上半部分通常是天空或建筑物
                if b > r and b > g and np.mean(color) > 150:  # 蓝色，可能是天空
                    return 0   # void (背景)
                else:
                    return 15  # manmade
            
            else:  # 中间部分
                # 基于颜色特征分类
                if g > r and g > b:  # 绿色倾向
                    return 16  # vegetation
                elif r > g and r > b:  # 红色倾向
                    return 4   # car (红色车辆)
                elif np.mean(color) > 200:  # 高亮度
                    return 15  # manmade (建筑物)
                else:
                    return 11  # driveable_surface (默认路面)
            
        except Exception:
            return 11  # 默认返回路面类别
    
    def _create_zero_depth(self, output_path: str) -> bool:
        """创建零填充深度图"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            placeholder = np.zeros((384, 384), dtype=np.uint16)
            cv2.imwrite(output_path, placeholder)
            return True
        except Exception as e:
            logger.error(f"创建零填充深度图失败: {e}")
            return False
    
    def _create_pattern_depth(self, output_path: str) -> bool:
        """创建模式填充深度图（渐变模式模拟距离）"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 创建从上到下的渐变（上远下近）
            depth_map = np.zeros((384, 384), dtype=np.float32)
            for y in range(384):
                # 上半部分距离较远，下半部分距离较近
                distance = 100.0 - (y / 384.0) * 80.0  # 100米到20米
                depth_map[y, :] = distance
            
            # 添加一些噪声使其更真实
            noise = np.random.normal(0, 5, depth_map.shape)
            depth_map = np.clip(depth_map + noise, 1.0, 100.0)
            
            # 转换为16位
            depth_16bit = (depth_map / 100.0 * 65535).astype(np.uint16)
            cv2.imwrite(output_path, depth_16bit)
            return True
        except Exception as e:
            logger.error(f"创建模式深度图失败: {e}")
            return False
    
    def _create_zero_semantic(self, output_path: str) -> bool:
        """创建零填充语义分割图"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            placeholder = np.zeros((384, 384), dtype=np.uint8)
            semantic_image = Image.fromarray(placeholder, mode='L')
            semantic_image.save(output_path)
            return True
        except Exception as e:
            logger.error(f"创建零填充语义图失败: {e}")
            return False
    
    def _create_pattern_semantic(self, output_path: str) -> bool:
        """创建模式填充语义分割图（简单的道路场景模式）"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 创建简单的道路场景模式
            semantic_map = np.zeros((384, 384), dtype=np.uint8)
            
            # 下半部分：道路表面 (class 11)
            semantic_map[240:, :] = 11  # driveable_surface
            
            # 中间部分：一些建筑物 (class 15)
            semantic_map[100:240, :] = 15  # manmade
            
            # 上半部分：天空/背景 (class 0)
            semantic_map[:100, :] = 0  # void
            
            # 添加一些车道线或其他细节
            # 左右两侧：人行道 (class 13)
            semantic_map[200:, :30] = 13   # sidewalk
            semantic_map[200:, -30:] = 13  # sidewalk
            
            # 中央：一些植被 (class 16)
            semantic_map[150:200, 150:234] = 16  # vegetation
            
            semantic_image = Image.fromarray(semantic_map, mode='L')
            semantic_image.save(output_path)
            return True
        except Exception as e:
            logger.error(f"创建模式语义图失败: {e}")
            return False
    
    def _track_fallback_method(self, modality: str, method: str):
        """记录降级方法使用统计"""
        key = f"{modality}_{method}"
        self.stats['fallback_methods_used'][key] = self.stats['fallback_methods_used'].get(key, 0) + 1
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """获取降级处理统计信息"""
        return {
            'depth_failure_rate': self.stats['depth_failures'],
            'depth_fallback_rate': self.stats['depth_fallbacks'],
            'semantic_failure_rate': self.stats['semantic_failures'], 
            'semantic_fallback_rate': self.stats['semantic_fallbacks'],
            'fallback_methods_used': self.stats['fallback_methods_used'].copy(),
            'depth_recovery_rate': (self.stats['depth_fallbacks'] / max(self.stats['depth_failures'], 1)) * 100,
            'semantic_recovery_rate': (self.stats['semantic_fallbacks'] / max(self.stats['semantic_failures'], 1)) * 100
        }
    
    def reset_statistics(self):
        """重置统计信息"""
        self.stats = {
            'depth_failures': 0,
            'depth_fallbacks': 0,
            'semantic_failures': 0,
            'semantic_fallbacks': 0,
            'fallback_methods_used': {}
        }


def create_fallback_handler(config: Dict[str, Any] = None) -> ModalityFallbackHandler:
    """
    创建模态缺失降级处理器的便利函数
    
    Args:
        config: 配置字典
        
    Returns:
        降级处理器实例
    """
    return ModalityFallbackHandler(config)