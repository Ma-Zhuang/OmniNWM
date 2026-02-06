"""
ShareGPT格式JSON生成器

专门用于生成符合ShareGPT对话格式的数据集：
- 支持多模态输入（images, depth_maps, semantic_maps）
- 生成对话式prompt-response格式
- 兼容VLM训练标准
- 支持历史车辆状态数据
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ShareGPTFormatter:
    """ShareGPT格式化处理器"""
    
    # 标准ShareGPT模式版本
    SCHEMA_VERSION = "2.0"
    
    def __init__(self, schema_version: str = None, include_metadata: bool = True):
        """
        初始化ShareGPT格式化器
        
        Args:
            schema_version: ShareGPT模式版本
            include_metadata: 是否包含元数据（用于调试）
        """
        self.schema_version = schema_version or self.SCHEMA_VERSION
        self.include_metadata = include_metadata
        
        logger.debug(f"ShareGPTFormatter初始化，模式版本: {self.schema_version}")
    
    def create_sharegpt_sample(self,
                              sample_id: str,
                              scene_token: str,
                              sample_token: str,
                              images: List[str],
                              depth_maps: List[str],
                              semantic_maps: List[str],
                              user_prompt: str,
                              assistant_response: str,
                              metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        创建ShareGPT格式样本
        
        Args:
            sample_id: 样本唯一标识
            scene_token: 场景令牌
            sample_token: 样本令牌
            images: RGB图像路径列表（按摄像头顺序）
            depth_maps: 深度图路径列表
            semantic_maps: 语义分割图路径列表
            user_prompt: 用户提示文本
            assistant_response: 助手响应文本
            metadata: 额外元数据
            
        Returns:
            ShareGPT格式化数据
        """
        try:
            # 验证输入数据
            self._validate_inputs(images, depth_maps, semantic_maps)
            
            # 构建ShareGPT格式的基础结构
            sharegpt_data = {
                "id": sample_id,
                "images": self._format_image_paths(images),
                "depth_maps": self._format_image_paths(depth_maps),
                "semantic_maps": self._format_image_paths(semantic_maps),
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                    {
                        "role": "assistant", 
                        "content": assistant_response
                    }
                ]
            }
            
            # 添加调试元数据（可选）
            if self.include_metadata:
                sharegpt_data["metadata"] = self._create_metadata(
                    scene_token, sample_token, metadata
                )
            
            return sharegpt_data
            
        except Exception as e:
            logger.error(f"创建ShareGPT样本失败 {sample_id}: {e}")
            raise
    
    def _validate_inputs(self, images: List[str], depth_maps: List[str], 
                        semantic_maps: List[str]) -> None:
        """验证输入数据的完整性"""
        expected_cameras = 6
        
        if len(images) != expected_cameras:
            raise ValueError(f"期望{expected_cameras}个RGB图像，实际得到{len(images)}个")
        
        if len(depth_maps) != expected_cameras:
            raise ValueError(f"期望{expected_cameras}个深度图，实际得到{len(depth_maps)}个")
        
        if len(semantic_maps) != expected_cameras:
            raise ValueError(f"期望{expected_cameras}个语义分割图，实际得到{len(semantic_maps)}个")
        
        # 验证路径格式
        for paths, data_type in [(images, "RGB"), (depth_maps, "depth"), (semantic_maps, "semantic")]:
            for i, path in enumerate(paths):
                if not isinstance(path, str) or not path:
                    raise ValueError(f"{data_type}图像路径{i}无效: {path}")
    
    def _format_image_paths(self, paths: List[str]) -> List[str]:
        """
        格式化图像路径，确保使用相对路径
        
        Args:
            paths: 路径列表
            
        Returns:
            格式化后的相对路径列表
        """
        formatted_paths = []
        for path in paths:
            if path:
                # 标准化路径分隔符为Unix风格
                formatted_path = path.replace('\\', '/')
                
                # 确保路径是相对路径（不以/开头）
                if formatted_path.startswith('/'):
                    formatted_path = formatted_path[1:]
                
                formatted_paths.append(formatted_path)
            else:
                logger.warning("发现空路径，跳过")
        
        return formatted_paths
    
    def _create_metadata(self, scene_token: str, sample_token: str, 
                        additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """创建元数据"""
        metadata = {
            "schema_version": self.schema_version,
            "format_type": "sharegpt_multimodal_trajectory",
            "scene_token": scene_token,
            "sample_token": sample_token,
            "created_at": datetime.now().isoformat()
        }
        
        # 添加额外元数据
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata
    
    def format_historical_vehicle_states(self, historical_states: List[Dict[str, float]]) -> str:
        """
        格式化历史车辆状态为prompt文本
        
        Args:
            historical_states: 历史车辆状态列表
            
        Returns:
            格式化的历史状态文本
        """
        if not historical_states:
            return ""
        
        formatted_lines = []
        
        for state in historical_states:
            # 格式化时间戳
            time_delta = state.get('timestamp_delta', 0.0)
            if time_delta < 0:
                time_str = f"(t{time_delta:.3f}s)"
            else:
                time_str = f"(t-0.0s)"
            
            # 格式化位置和角度
            x = state.get('x', 0.0)
            y = state.get('y', 0.0) 
            heading = state.get('heading', 0.0)
            position_str = f"[{x:.2f}, {y:.2f}, {heading:.3f}]"
            
            # 格式化加速度
            acc_x = state.get('acceleration_x', 0.0)
            acc_y = state.get('acceleration_y', 0.0)
            acceleration_str = f"Acceleration: X {acc_x:.2f}, Y {acc_y:.2f} m/s²"
            
            # 格式化速度
            speed = state.get('speed', 0.0)
            velocity_str = f"Velocity: {speed:.2f} m/s"
            
            # 格式化转向角
            steering_angle = state.get('steering_angle', 0.0)
            steering_str = f"Steering angle: {steering_angle:.2f} (positive: left turn, negative: right turn)"
            
            # 组合成一行
            line = f"{time_str} {position_str}, {acceleration_str}, {velocity_str}, {steering_str}"
            formatted_lines.append(line)
        
        return "\\n".join(formatted_lines)
    
    def format_trajectory_response(self, trajectory: List[Dict[str, float]]) -> str:
        """
        格式化轨迹响应为助手回复格式
        
        Args:
            trajectory: 未来轨迹点列表
            
        Returns:
            格式化的轨迹响应文本
        """
        if not trajectory:
            return "<PLANNING>No valid trajectory predicted</PLANNING>"
        
        # 格式化轨迹点
        trajectory_points = []
        for point in trajectory:
            x = point.get('x', 0.0)
            y = point.get('y', 0.0)
            heading = point.get('heading', 0.0)
            trajectory_points.append(f"[{x:.2f}, {y:.2f}, {heading:.3f}]")
        
        # 构建响应文本
        trajectory_str = ", ".join(trajectory_points)
        
        response = (
            f"<PLANNING>Predicted future trajectory for the next 3 seconds "
            f"(36 waypoints sampled at 12Hz, 0.083-second intervals), including position and orientation "
            f"in ego-vehicle coordinate system. Positive x means forward direction, positive y means leftward direction, "
            f"heading angle in radians. The output is formatted as [x, y, heading]: \\n"
            f"{trajectory_str}</PLANNING>"
        )
        
        return response
    
    def parse_trajectory_from_text(self, text: str) -> Optional[List[Dict[str, float]]]:
        """
        从文本中解析轨迹数据，与tri_modal_qwen的解析器保持兼容
        
        Args:
            text: 包含轨迹信息的文本
            
        Returns:
            解析出的轨迹点列表，每个点包含 {x, y, heading}
        """
        import re
        
        try:
            # 寻找PLANNING标签中的轨迹数据
            planning_pattern = r'<PLANNING>(.*?)</PLANNING>'
            planning_match = re.search(planning_pattern, text, re.DOTALL)
            
            if not planning_match:
                logger.warning("未找到PLANNING标签")
                return None
            
            planning_content = planning_match.group(1).strip()
            
            # 解析轨迹点 [x, y, heading]格式
            point_pattern = r'\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]'
            matches = re.findall(point_pattern, planning_content)
            
            if not matches:
                logger.warning("未找到轨迹点数据")
                return None
            
            # 转换为标准字典格式
            trajectory_points = []
            for match in matches:
                x, y, heading = float(match[0]), float(match[1]), float(match[2])
                trajectory_points.append({
                    'x': x,
                    'y': y, 
                    'heading': heading
                })
            
            # 验证轨迹长度 (期望36个点)
            if len(trajectory_points) != 36:
                logger.warning(f"轨迹点数量不匹配: {len(trajectory_points)}, 期望: 36")
                
                # 如果点数不足，进行填充
                if len(trajectory_points) < 36 and len(trajectory_points) > 0:
                    padding_size = 36 - len(trajectory_points)
                    last_point = trajectory_points[-1].copy()
                    trajectory_points.extend([last_point for _ in range(padding_size)])
                elif len(trajectory_points) > 36:
                    # 截取前36个点
                    trajectory_points = trajectory_points[:36]
                elif len(trajectory_points) == 0:
                    logger.error("没有有效的轨迹点")
                    return None
            
            return trajectory_points
            
        except Exception as e:
            logger.error(f"解析轨迹失败: {e}")
            return None
    
    def save_sharegpt_file(self, sharegpt_data: Dict[str, Any], output_path: str, 
                          indent: int = 2) -> bool:
        """
        保存ShareGPT格式文件
        
        Args:
            sharegpt_data: ShareGPT数据
            output_path: 输出路径
            indent: JSON缩进
            
        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(sharegpt_data, f, indent=indent, ensure_ascii=False, sort_keys=False)
            
            logger.debug(f"ShareGPT文件保存成功: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存ShareGPT文件失败 {output_path}: {e}")
            return False
    
    def validate_sharegpt_format(self, sharegpt_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证ShareGPT格式
        
        Args:
            sharegpt_data: ShareGPT数据
            
        Returns:
            验证结果
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 检查必需字段
        required_fields = ['id', 'images', 'depth_maps', 'semantic_maps', 'messages']
        for field in required_fields:
            if field not in sharegpt_data:
                validation_result['errors'].append(f"缺少必需字段: {field}")
                validation_result['valid'] = False
        
        # 检查messages格式
        if 'messages' in sharegpt_data:
            messages = sharegpt_data['messages']
            if not isinstance(messages, list) or len(messages) != 2:
                validation_result['errors'].append("messages应该包含正好2条消息（user和assistant）")
                validation_result['valid'] = False
            else:
                # 检查消息角色
                if messages[0].get('role') != 'user':
                    validation_result['errors'].append("第一条消息应该是用户消息")
                    validation_result['valid'] = False
                
                if messages[1].get('role') != 'assistant':
                    validation_result['errors'].append("第二条消息应该是助手消息")
                    validation_result['valid'] = False
        
        # 检查图像数量
        for field in ['images', 'depth_maps', 'semantic_maps']:
            if field in sharegpt_data:
                if len(sharegpt_data[field]) != 6:
                    validation_result['warnings'].append(f"{field}应该包含6个文件路径")
        
        return validation_result
    
    def create_dataset_manifest(self, dataset_dir: str, output_path: str = None) -> Dict[str, Any]:
        """
        创建ShareGPT数据集清单
        
        Args:
            dataset_dir: 数据集目录
            output_path: 清单文件输出路径
            
        Returns:
            清单数据
        """
        try:
            output_path = output_path or os.path.join(dataset_dir, 'sharegpt_manifest.json')
            
            # 构建ShareGPT数据集信息
            manifest = {
                "dataset_info": {
                    "name": "nuScenes ShareGPT Multimodal Trajectory Dataset",
                    "version": self.schema_version,
                    "created_at": datetime.now().isoformat(),
                    "format": "ShareGPT",
                    "source": "nuScenes dataset",
                    "task": "multimodal_trajectory_prediction"
                },
                "data_format": {
                    "type": "conversation",
                    "modalities": ["RGB", "depth", "semantic"],
                    "cameras_per_sample": 6,
                    "trajectory_points": 36,
                    "historical_points": 12,
                    "sampling_frequency": "12Hz"
                },
                "statistics": {
                    "total_samples": 0,
                    "successful_samples": 0,
                    "failed_samples": 0,
                    "scenes_count": 0
                },
                "samples": []
            }
            
            # 扫描数据集目录获取统计信息
            if os.path.exists(dataset_dir):
                total_files = 0
                for root, dirs, files in os.walk(dataset_dir):
                    for file in files:
                        if file.endswith('.json') and file != 'sharegpt_manifest.json':
                            total_files += 1
                            
                            # 获取样本信息
                            sample_path = os.path.join(root, file)
                            try:
                                with open(sample_path, 'r', encoding='utf-8') as f:
                                    sample_data = json.load(f)
                                
                                sample_info = {
                                    "sample_id": sample_data.get('id', 'unknown'),
                                    "relative_path": os.path.relpath(sample_path, dataset_dir),
                                    "file_size": os.path.getsize(sample_path),
                                    "has_images": len(sample_data.get('images', [])) == 6,
                                    "has_depth": len(sample_data.get('depth_maps', [])) == 6,
                                    "has_semantic": len(sample_data.get('semantic_maps', [])) == 6,
                                    "has_conversation": len(sample_data.get('messages', [])) == 2
                                }
                                
                                manifest["samples"].append(sample_info)
                                manifest["statistics"]["successful_samples"] += 1
                                
                            except Exception as e:
                                logger.warning(f"Failed to process sample {sample_path}: {e}")
                                manifest["statistics"]["failed_samples"] += 1
                
                manifest["statistics"]["total_samples"] = total_files
            
            # 保存清单文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            logger.info(f"ShareGPT数据集清单已保存: {output_path}")
            return manifest
            
        except Exception as e:
            logger.error(f"创建ShareGPT数据集清单失败: {e}")
            raise


def create_sharegpt_formatter(include_metadata: bool = True) -> ShareGPTFormatter:
    """
    创建ShareGPT格式化器的便利函数
    
    Args:
        include_metadata: 是否包含元数据
        
    Returns:
        ShareGPT格式化器实例
    """
    return ShareGPTFormatter(include_metadata=include_metadata)