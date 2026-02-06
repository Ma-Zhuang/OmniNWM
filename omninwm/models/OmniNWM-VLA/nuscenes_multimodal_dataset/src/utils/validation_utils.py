"""
数据验证工具库

提供各种数据验证和检查功能：
- 图像文件完整性验证
- 轨迹数据格式检查
- JSON模式合规验证
- 数据集统计分析
- 配置文件验证
"""

import os
import json
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import cv2

from .math_utils import TrajectoryMetrics, GeometryUtils
from .file_utils import FileManager

logger = logging.getLogger(__name__)


class ImageValidator:
    """图像验证器"""
    
    SUPPORTED_FORMATS = {
        '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'
    }
    
    @staticmethod
    def validate_image_file(image_path: str, check_corruption: bool = True) -> Dict[str, Any]:
        """
        验证图像文件
        
        Args:
            image_path: 图像文件路径
            check_corruption: 是否检查文件损坏
            
        Returns:
            验证结果字典
        """
        result = {
            'valid': False,
            'exists': False,
            'format_supported': False,
            'readable': False,
            'corrupted': False,
            'dimensions': None,
            'file_size': 0,
            'error': None
        }
        
        try:
            # 检查文件是否存在
            if not os.path.exists(image_path):
                result['error'] = f"文件不存在: {image_path}"
                return result
            
            result['exists'] = True
            result['file_size'] = os.path.getsize(image_path)
            
            # 检查文件扩展名
            _, ext = os.path.splitext(image_path.lower())
            if ext not in ImageValidator.SUPPORTED_FORMATS:
                result['error'] = f"不支持的格式: {ext}"
                return result
            
            result['format_supported'] = True
            
            # 尝试读取图像
            try:
                with Image.open(image_path) as img:
                    result['dimensions'] = img.size  # (width, height)
                    result['mode'] = img.mode
                    result['readable'] = True
                    
                    # 检查文件损坏（如果启用）
                    if check_corruption:
                        try:
                            img.verify()  # 验证图像完整性
                        except Exception:
                            result['corrupted'] = True
                            result['error'] = "图像文件损坏"
                            return result
                        
                        # 重新打开进行进一步验证
                        with Image.open(image_path) as img2:
                            try:
                                img2.load()  # 强制加载所有像素数据
                            except Exception as e:
                                result['corrupted'] = True
                                result['error'] = f"加载像素数据损坏: {e}"
                                return result
                            
            except Exception as e:
                result['error'] = f"无法读取图像: {e}"
                return result
            
            # 检查图像尺寸合理性
            width, height = result['dimensions']
            if width <= 0 or height <= 0:
                result['error'] = f"无效的图像尺寸: {width}x{height}"
                return result
            
            if width > 10000 or height > 10000:
                logger.warning(f"图像尺寸异常: {width}x{height}")
            
            result['valid'] = True
            
        except Exception as e:
            result['error'] = f"验证过程异常: {e}"
        
        return result
    
    @staticmethod
    def validate_depth_image(depth_path: str) -> Dict[str, Any]:
        """
        验证深度图像
        
        Args:
            depth_path: 深度图路径
            
        Returns:
            验证结果
        """
        result = ImageValidator.validate_image_file(depth_path)
        
        if not result['valid']:
            return result
        
        try:
            # 使用opencv读取深度图
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            if depth_img is None:
                result['valid'] = False
                result['error'] = "无法读取深度图"
                return result
            
            # 检查数据类型
            if depth_img.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
                result['valid'] = False
                result['error'] = f"不支持的数据类型: {depth_img.dtype}"
                return result
            
            # 检查深度值范围
            if depth_img.dtype == np.uint16:
                max_depth = np.max(depth_img)
                if max_depth == 0:
                    logger.warning("深度图全为0值")
                elif max_depth > 65535:
                    result['valid'] = False
                    result['error'] = "深度值超出16位范围"
                    return result
            
            # 统计信息
            result['depth_stats'] = {
                'dtype': str(depth_img.dtype),
                'shape': depth_img.shape,
                'min_depth': float(np.min(depth_img)),
                'max_depth': float(np.max(depth_img)),
                'mean_depth': float(np.mean(depth_img)),
                'zero_pixels_ratio': float(np.sum(depth_img == 0) / depth_img.size)
            }
            
            # 检查零值像素比例
            if result['depth_stats']['zero_pixels_ratio'] > 0.5:
                logger.warning(f"零值像素比例过高: {result['depth_stats']['zero_pixels_ratio']:.2%}")
            
        except Exception as e:
            result['valid'] = False
            result['error'] = f"深度图验证失败: {e}"
        
        return result
    
    @staticmethod
    def validate_semantic_image(semantic_path: str, class_mapping: Dict[int, str] = None) -> Dict[str, Any]:
        """
        验证语义分割图像
        
        Args:
            semantic_path: 语义图路径
            class_mapping: 类别映射表
            
        Returns:
            验证结果
        """
        result = ImageValidator.validate_image_file(semantic_path)
        
        if not result['valid']:
            return result
        
        try:
            # 读取语义分割图像
            semantic_img = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)
            
            if semantic_img is None:
                result['valid'] = False
                result['error'] = "无法读取语义图"
                return result
            
            # 获取唯一类别
            unique_classes = np.unique(semantic_img)
            
            # 检查类别ID有效性
            max_class_id = np.max(unique_classes)
            if class_mapping and max_class_id not in class_mapping:
                max_valid_id = max(class_mapping.keys())
                if max_class_id > max_valid_id:
                    logger.warning(f"发现未知类别ID: {max_class_id} > {max_valid_id}")
            
            # 统计信息
            result['semantic_stats'] = {
                'unique_classes': unique_classes.tolist(),
                'num_classes': len(unique_classes),
                'shape': semantic_img.shape,
                'class_distribution': {}
            }
            
            # 计算类别分布
            total_pixels = semantic_img.size
            for class_id in unique_classes:
                pixel_count = np.sum(semantic_img == class_id)
                result['semantic_stats']['class_distribution'][int(class_id)] = {
                    'count': int(pixel_count),
                    'percentage': float(pixel_count / total_pixels * 100)
                }
            
        except Exception as e:
            result['valid'] = False
            result['error'] = f"语义图验证失败: {e}"
        
        return result


class TrajectoryValidator:
    """轨迹验证器"""
    
    @staticmethod
    def validate_trajectory_data(trajectory: List[Dict], 
                               expected_length: Optional[int] = None,
                               coordinate_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        验证轨迹数据
        
        Args:
            trajectory: 轨迹数据
            expected_length: 期望轨迹长度
            coordinate_bounds: 坐标边界 {'x': (min, max), 'y': (min, max)}
            
        Returns:
            验证结果
        """
        result = {
            'valid': False,
            'length_valid': False,
            'format_valid': False,
            'bounds_valid': False,
            'continuity_valid': False,
            'issues': [],
            'stats': {}
        }
        
        try:
            # 检查基本格式
            if not isinstance(trajectory, list):
                result['issues'].append("轨迹不是列表格式")
                return result
            
            if len(trajectory) == 0:
                result['issues'].append("轨迹为空")
                return result
            
            # 检查长度
            if expected_length and len(trajectory) != expected_length:
                result['issues'].append(f"轨迹长度不匹配: {len(trajectory)} != {expected_length}")
            else:
                result['length_valid'] = True
            
            # 检查每个轨迹点的格式
            required_fields = ['x', 'y', 'heading']
            format_valid = True
            
            for i, waypoint in enumerate(trajectory):
                if not isinstance(waypoint, dict):
                    result['issues'].append(f"轨迹点{i}不是字典格式")
                    format_valid = False
                    continue
                
                for field in required_fields:
                    if field not in waypoint:
                        result['issues'].append(f"轨迹点{i}缺少字段: {field}")
                        format_valid = False
                    elif not isinstance(waypoint[field], (int, float)):
                        result['issues'].append(f"轨迹点{i}字段{field}不是数值格式: {type(waypoint[field])}")
                        format_valid = False
                    elif not np.isfinite(waypoint[field]):
                        result['issues'].append(f"轨迹点{i}字段{field}包含无效值: {waypoint[field]}")
                        format_valid = False
            
            result['format_valid'] = format_valid
            
            if not format_valid:
                return result
            
            # 检查坐标边界
            bounds_valid = True
            if coordinate_bounds:
                for i, waypoint in enumerate(trajectory):
                    for coord in ['x', 'y']:
                        if coord in coordinate_bounds:
                            min_val, max_val = coordinate_bounds[coord]
                            if not (min_val <= waypoint[coord] <= max_val):
                                result['issues'].append(
                                    f"轨迹点{i}坐标{coord}超出边界: {waypoint[coord]} not in [{min_val}, {max_val}]"
                                )
                                bounds_valid = False
            
            result['bounds_valid'] = bounds_valid or coordinate_bounds is None
            
            # 检查轨迹连续性
            continuity_issues = TrajectoryValidator._check_trajectory_continuity(trajectory)
            if continuity_issues:
                result['issues'].extend(continuity_issues)
                result['continuity_valid'] = False
            else:
                result['continuity_valid'] = True
            
            # 计算统计信息
            result['stats'] = TrajectoryValidator._calculate_trajectory_stats(trajectory)
            
            # 综合验证结果
            result['valid'] = (result['length_valid'] and 
                             result['format_valid'] and 
                             result['bounds_valid'] and 
                             result['continuity_valid'])
            
        except Exception as e:
            result['issues'].append(f"验证过程异常: {e}")
        
        return result
    
    @staticmethod
    def _check_trajectory_continuity(trajectory: List[Dict]) -> List[str]:
        """检查轨迹连续性"""
        issues = []
        
        max_step_distance = 8.3  # 最大单步距离（米）适配12Hz采样
        max_heading_change = np.pi / 6  # 最大转向角（30度）适配12Hz采样
        
        for i in range(len(trajectory) - 1):
            current = trajectory[i]
            next_point = trajectory[i + 1]
            
            # 检查距离跳跃
            distance = GeometryUtils.euclidean_distance(
                (current['x'], current['y']),
                (next_point['x'], next_point['y'])
            )
            
            if distance > max_step_distance:
                issues.append(f"轨迹点{i}到{i+1}距离过大: {distance:.2f}m")
            
            # 检查转向角度
            from .math_utils import AngleUtils
            heading_diff = abs(AngleUtils.angle_difference(current['heading'], next_point['heading']))
            
            if heading_diff > max_heading_change:
                issues.append(f"轨迹点{i}到{i+1}转向过大: {heading_diff:.2f}rad")
        
        return issues
    
    @staticmethod
    def _calculate_trajectory_stats(trajectory: List[Dict]) -> Dict[str, Any]:
        """计算轨迹统计信息"""
        stats = {
            'length': len(trajectory),
            'total_distance': 0.0,
            'max_step_distance': 0.0,
            'avg_step_distance': 0.0,
            'total_heading_change': 0.0,
            'max_heading_change': 0.0
        }
        
        if len(trajectory) < 2:
            return stats
        
        distances = []
        heading_changes = []
        
        from .math_utils import AngleUtils
        
        for i in range(len(trajectory) - 1):
            current = trajectory[i]
            next_point = trajectory[i + 1]
            
            # 距离
            distance = GeometryUtils.euclidean_distance(
                (current['x'], current['y']),
                (next_point['x'], next_point['y'])
            )
            distances.append(distance)
            
            # 转向角
            heading_change = abs(AngleUtils.angle_difference(current['heading'], next_point['heading']))
            heading_changes.append(heading_change)
        
        stats['total_distance'] = sum(distances)
        stats['max_step_distance'] = max(distances) if distances else 0
        stats['avg_step_distance'] = np.mean(distances) if distances else 0
        stats['total_heading_change'] = sum(heading_changes)
        stats['max_heading_change'] = max(heading_changes) if heading_changes else 0
        
        # 坐标范围
        x_coords = [wp['x'] for wp in trajectory]
        y_coords = [wp['y'] for wp in trajectory]
        headings = [wp['heading'] for wp in trajectory]
        
        stats['coordinate_ranges'] = {
            'x': {'min': min(x_coords), 'max': max(x_coords)},
            'y': {'min': min(y_coords), 'max': max(y_coords)},
            'heading': {'min': min(headings), 'max': max(headings)}
        }
        
        return stats


class JSONValidator:
    """JSON数据验证器"""
    
    @staticmethod
    def validate_json_schema(json_data: Dict, schema: Dict) -> Dict[str, Any]:
        """
        验证JSON模式合规性
        
        Args:
            json_data: JSON数据
            schema: JSON模式定义
            
        Returns:
            验证结果
        """
        result = {
            'valid': False,
            'missing_fields': [],
            'type_errors': [],
            'validation_errors': []
        }
        
        try:
            # 检查必需字段
            required_fields = schema.get('required', [])
            for field in required_fields:
                if field not in json_data:
                    result['missing_fields'].append(field)
            
            # 检查字段类型
            properties = schema.get('properties', {})
            for field, field_schema in properties.items():
                if field in json_data:
                    expected_type = field_schema.get('type')
                    actual_value = json_data[field]
                    
                    if not JSONValidator._check_type(actual_value, expected_type):
                        result['type_errors'].append({
                            'field': field,
                            'expected': expected_type,
                            'actual': type(actual_value).__name__
                        })
            
            # 综合验证
            result['valid'] = (len(result['missing_fields']) == 0 and 
                             len(result['type_errors']) == 0)
            
        except Exception as e:
            result['validation_errors'].append(f"模式验证异常: {e}")
        
        return result
    
    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """检查数据类型"""
        if expected_type == 'string':
            return isinstance(value, str)
        elif expected_type == 'integer':
            return isinstance(value, int)
        elif expected_type == 'number':
            return isinstance(value, (int, float))
        elif expected_type == 'boolean':
            return isinstance(value, bool)
        elif expected_type == 'array':
            return isinstance(value, list)
        elif expected_type == 'object':
            return isinstance(value, dict)
        else:
            return True  # 未知类型，通过
    
    @staticmethod
    def validate_sample_json(json_data: Dict) -> Dict[str, Any]:
        """
        验证样本JSON完整性
        
        Args:
            json_data: 样本JSON数据
            
        Returns:
            验证结果
        """
        schema = {
            'required': ['id', 'scene_token', 'sample_token', 'visual_inputs', 'text_prompt', 'ground_truth'],
            'properties': {
                'id': {'type': 'string'},
                'scene_token': {'type': 'string'},
                'sample_token': {'type': 'string'},
                'visual_inputs': {'type': 'object'},
                'text_prompt': {'type': 'string'},
                'ground_truth': {'type': 'object'}
            }
        }
        
        result = JSONValidator.validate_json_schema(json_data, schema)
        
        # 验证visual_inputs结构
        if 'visual_inputs' in json_data:
            visual_inputs = json_data['visual_inputs']
            expected_cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                              'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            
            missing_cameras = []
            for camera in expected_cameras:
                if camera not in visual_inputs:
                    missing_cameras.append(camera)
                else:
                    camera_data = visual_inputs[camera]
                    required_paths = ['rgb_path', 'depth_path', 'semantic_path']
                    for path_key in required_paths:
                        if path_key not in camera_data:
                            result['missing_fields'].append(f'visual_inputs.{camera}.{path_key}')
            
            if missing_cameras:
                result['missing_fields'].extend([f'visual_inputs.{cam}' for cam in missing_cameras])
        
        # 验证ground_truth结构
        if 'ground_truth' in json_data:
            gt = json_data['ground_truth']
            if 'future_trajectory' not in gt:
                result['missing_fields'].append('ground_truth.future_trajectory')
            elif not isinstance(gt['future_trajectory'], list):
                result['type_errors'].append({
                    'field': 'ground_truth.future_trajectory',
                    'expected': 'array',
                    'actual': type(gt['future_trajectory']).__name__
                })
        
        # 重新检查有效性
        result['valid'] = (len(result['missing_fields']) == 0 and 
                         len(result['type_errors']) == 0 and
                         len(result['validation_errors']) == 0)
        
        return result


class DatasetValidator:
    """数据集验证器"""
    
    @staticmethod
    def validate_dataset_completeness(dataset_dir: str) -> Dict[str, Any]:
        """
        验证数据集完整性
        
        Args:
            dataset_dir: 数据集目录
            
        Returns:
            验证结果
        """
        result = {
            'valid': False,
            'directory_exists': False,
            'structure_valid': False,
            'samples_found': 0,
            'samples_valid': 0,
            'missing_files': [],
            'corrupted_files': [],
            'validation_errors': []
        }
        
        try:
            # 检查目录存在
            if not os.path.exists(dataset_dir):
                result['validation_errors'].append(f"数据集目录不存在: {dataset_dir}")
                return result
            
            result['directory_exists'] = True
            
            # 查找所有样本
            sample_dirs = []
            for root, dirs, files in os.walk(dataset_dir):
                if 'prompt.json' in files:
                    sample_dirs.append(root)
            
            result['samples_found'] = len(sample_dirs)
            
            if result['samples_found'] == 0:
                result['validation_errors'].append("未找到任何样本")
                return result
            
            # 验证每个样本
            valid_samples = 0
            for sample_dir in sample_dirs:
                sample_valid = DatasetValidator._validate_single_sample(sample_dir, result)
                if sample_valid:
                    valid_samples += 1
            
            result['samples_valid'] = valid_samples
            result['structure_valid'] = valid_samples == result['samples_found']
            result['valid'] = result['structure_valid'] and len(result['validation_errors']) == 0
            
        except Exception as e:
            result['validation_errors'].append(f"数据集验证异常: {e}")
        
        return result
    
    @staticmethod
    def _validate_single_sample(sample_dir: str, result: Dict) -> bool:
        """验证单个样本"""
        try:
            # 检查prompt.json
            json_path = os.path.join(sample_dir, 'prompt.json')
            if not os.path.exists(json_path):
                result['missing_files'].append(json_path)
                return False
            
            # 加载并验证JSON
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except Exception as e:
                result['corrupted_files'].append(f"{json_path}: {e}")
                return False
            
            # 验证JSON结构
            json_validation = JSONValidator.validate_sample_json(json_data)
            if not json_validation['valid']:
                result['validation_errors'].append(f"{json_path}: JSON结构无效")
                return False
            
            # 检查图像文件是否存在
            visual_inputs = json_data.get('visual_inputs', {})
            for camera, paths in visual_inputs.items():
                for path_type, rel_path in paths.items():
                    full_path = os.path.join(sample_dir, os.path.basename(rel_path))
                    if not os.path.exists(full_path):
                        result['missing_files'].append(full_path)
                        return False
            
            return True
            
        except Exception as e:
            result['validation_errors'].append(f"验证样本{sample_dir}失败: {e}")
            return False
    
    @staticmethod
    def calculate_dataset_statistics(dataset_dir: str) -> Dict[str, Any]:
        """
        计算数据集统计信息
        
        Args:
            dataset_dir: 数据集目录
            
        Returns:
            统计信息
        """
        stats = {
            'total_samples': 0,
            'total_size_bytes': 0,
            'file_counts': {},
            'image_stats': {},
            'trajectory_stats': {},
            'processing_time': 0
        }
        
        try:
            import time
            start_time = time.time()
            
            # 收集所有文件信息
            file_extensions = {}
            total_size = 0
            sample_count = 0
            
            trajectory_lengths = []
            image_dimensions = []
            
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    
                    # 统计文件类型
                    _, ext = os.path.splitext(file.lower())
                    file_extensions[ext] = file_extensions.get(ext, 0) + 1
                    
                    # 样本特定统计
                    if file == 'prompt.json':
                        sample_count += 1
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                            
                            # 轨迹长度统计
                            trajectory = json_data.get('ground_truth', {}).get('future_trajectory', [])
                            if trajectory:
                                trajectory_lengths.append(len(trajectory))
                        except Exception:
                            pass
                    
                    # 图像尺寸统计（采样前100个）
                    elif ext in ['.jpg', '.jpeg', '.png'] and len(image_dimensions) < 100:
                        try:
                            with Image.open(file_path) as img:
                                image_dimensions.append(img.size)
                        except Exception:
                            pass
            
            stats['total_samples'] = sample_count
            stats['total_size_bytes'] = total_size
            stats['file_counts'] = file_extensions
            
            # 图像统计
            if image_dimensions:
                widths, heights = zip(*image_dimensions)
                stats['image_stats'] = {
                    'samples_checked': len(image_dimensions),
                    'width_stats': {
                        'min': min(widths),
                        'max': max(widths),
                        'mean': sum(widths) / len(widths)
                    },
                    'height_stats': {
                        'min': min(heights),
                        'max': max(heights),
                        'mean': sum(heights) / len(heights)
                    }
                }
            
            # 轨迹统计
            if trajectory_lengths:
                from .math_utils import StatisticsUtils
                stats['trajectory_stats'] = StatisticsUtils.calculate_basic_stats(trajectory_lengths)
            
            stats['processing_time'] = time.time() - start_time
            
        except Exception as e:
            logger.error(f"计算数据集统计失败: {e}")
            stats['error'] = str(e)
        
        return stats


def validate_config_file(config_path: str) -> Dict[str, Any]:
    """
    验证配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        验证结果
    """
    result = {
        'valid': False,
        'exists': False,
        'parseable': False,
        'schema_valid': False,
        'issues': []
    }
    
    try:
        # 检查文件存在
        if not os.path.exists(config_path):
            result['issues'].append(f"配置文件不存在: {config_path}")
            return result
        
        result['exists'] = True
        
        # 解析YAML
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            result['issues'].append(f"配置文件解析失败: {e}")
            return result
        
        result['parseable'] = True
        
        # 检查基本结构
        required_sections = ['dataset', 'trajectory', 'processing']
        for section in required_sections:
            if section not in config:
                result['issues'].append(f"缺少配置节: {section}")
        
        # 检查具体配置项
        if 'dataset' in config:
            dataset_config = config['dataset']
            if 'nuscenes_dataroot' not in dataset_config:
                result['issues'].append("缺少nuScenes数据集配置")
            elif not os.path.exists(dataset_config['nuscenes_dataroot']):
                result['issues'].append(f"nuScenes数据集不存在: {dataset_config['nuscenes_dataroot']}")
        
        result['schema_valid'] = len(result['issues']) == 0
        result['valid'] = result['schema_valid']
        
    except Exception as e:
        result['issues'].append(f"配置文件验证异常: {e}")
    
    return result


def validate_image_file(image_path: str) -> bool:
    """
    验证图像文件（简化版）
    
    Args:
        image_path: 图像路径
        
    Returns:
        是否有效
    """
    return ImageValidator.validate_image_file(image_path)['valid']


def validate_trajectory_data(trajectory: List[Dict]) -> bool:
    """
    验证轨迹数据（简化版）
    
    Args:
        trajectory: 轨迹数据
        
    Returns:
        是否有效
    """
    return TrajectoryValidator.validate_trajectory_data(trajectory)['valid']


def validate_json_completeness(json_data: Dict) -> List[str]:
    """
    验证JSON完整性（简化版）
    
    Args:
        json_data: JSON数据
        
    Returns:
        缺失字段列表
    """
    validation_result = JSONValidator.validate_sample_json(json_data)
    return validation_result['missing_fields']


def calculate_dataset_statistics(dataset_dir: str) -> Dict:
    """
    计算数据集统计（简化版）
    
    Args:
        dataset_dir: 数据集目录
        
    Returns:
        统计信息
    """
    return DatasetValidator.calculate_dataset_statistics(dataset_dir)