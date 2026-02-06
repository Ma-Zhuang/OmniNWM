#!/usr/bin/env python3
"""
配置管理器

该模块提供配置文件的加载、验证和管理功能：
- YAML配置文件加载
- 配置验证
- 配置合并
- 环境变量处理
"""

import os
import yaml
import copy
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        """初始化配置管理器"""
        self.config_cache = {}
        self.default_config = self._get_default_config()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 检查缓存
        abs_path = os.path.abspath(config_path)
        if abs_path in self.config_cache:
            return copy.deepcopy(self.config_cache[abs_path])
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                config = {}
            
            # 与默认配置合并
            merged_config = self._merge_configs(self.default_config, config)
            
            # 处理环境变量
            merged_config = self._process_environment_variables(merged_config)
            
            # 缓存配置
            self.config_cache[abs_path] = copy.deepcopy(merged_config)
            
            logging.info(f"配置文件加载成功: {config_path}")
            return merged_config
            
        except yaml.YAMLError as e:
            raise ValueError(f"YAML解析错误: {e}")
        except Exception as e:
            raise RuntimeError(f"配置加载失败: {e}")
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        验证配置有效性
        
        Args:
            config: 配置字典
            
        Returns:
            验证是否通过
        """
        try:
            # 验证必需的顶级字段
            required_sections = ['dataset', 'trajectory', 'processing']
            for section in required_sections:
                if section not in config:
                    logging.error(f"缺少必需的配置节: {section}")
                    return False
            
            # 验证数据集配置
            if not self._validate_dataset_config(config['dataset']):
                return False
            
            # 验证轨迹配置
            if not self._validate_trajectory_config(config['trajectory']):
                return False
            
            # 验证处理配置
            if not self._validate_processing_config(config['processing']):
                return False
            
            # 验证可选配置节
            if 'cameras' in config:
                if not self._validate_cameras_config(config['cameras']):
                    return False
            
            if 'quality_control' in config:
                if not self._validate_quality_control_config(config['quality_control']):
                    return False
            
            logging.info("配置验证通过")
            return True
            
        except Exception as e:
            logging.error(f"配置验证失败: {e}")
            return False
    
    def _validate_dataset_config(self, dataset_config: Dict[str, Any]) -> bool:
        """验证数据集配置"""
        required_fields = ['nuscenes_dataroot', 'nuscenes_version', 'output_directory']
        
        for field in required_fields:
            if field not in dataset_config:
                logging.error(f"数据集配置缺少必需字段: {field}")
                return False
        
        # 验证nuScenes版本
        valid_versions = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
        version = dataset_config['nuscenes_version']
        if version not in valid_versions:
            logging.error(f"无效的nuScenes版本: {version}")
            return False
        
        return True
    
    def _validate_trajectory_config(self, trajectory_config: Dict[str, Any]) -> bool:
        """验证轨迹配置"""
        required_fields = ['prediction_horizon', 'sampling_rate', 'num_waypoints']
        
        for field in required_fields:
            if field not in trajectory_config:
                logging.error(f"轨迹配置缺少必需字段: {field}")
                return False
        
        # 验证数值范围
        horizon = trajectory_config['prediction_horizon']
        sampling_rate = trajectory_config['sampling_rate']
        num_waypoints = trajectory_config['num_waypoints']
        
        if horizon <= 0:
            logging.error(f"预测时域必须大于0: {horizon}")
            return False
        
        if sampling_rate <= 0:
            logging.error(f"采样频率必须大于0: {sampling_rate}")
            return False
        
        if num_waypoints <= 0:
            logging.error(f"路径点数量必须大于0: {num_waypoints}")
            return False
        
        # 验证轨迹参数一致性
        expected_waypoints = int(horizon * sampling_rate)
        if num_waypoints != expected_waypoints:
            logging.error(f"路径点数量不一致: 期望{expected_waypoints}, 实际{num_waypoints}")
            return False
        
        return True
    
    def _validate_processing_config(self, processing_config: Dict[str, Any]) -> bool:
        """验证处理配置"""
        required_fields = ['num_workers', 'batch_size']
        
        for field in required_fields:
            if field not in processing_config:
                logging.error(f"处理配置缺少必需字段: {field}")
                return False
        
        # 验证数值范围
        num_workers = processing_config['num_workers']
        batch_size = processing_config['batch_size']
        
        if num_workers <= 0:
            logging.error(f"工作进程数必须大于0: {num_workers}")
            return False
        
        if batch_size <= 0:
            logging.error(f"批处理大小必须大于0: {batch_size}")
            return False
        
        return True
    
    def _validate_cameras_config(self, cameras_config: Dict[str, Any]) -> bool:
        """验证摄像头配置"""
        if 'channels' not in cameras_config:
            logging.error("摄像头配置缺少channels字段")
            return False
        
        valid_cameras = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        channels = cameras_config['channels']
        if not isinstance(channels, list):
            logging.error("摄像头通道必须是列表")
            return False
        
        for channel in channels:
            if channel not in valid_cameras:
                logging.error(f"无效的摄像头通道: {channel}")
                return False
        
        return True
    
    def _validate_quality_control_config(self, qc_config: Dict[str, Any]) -> bool:
        """验证质量控制配置"""
        numeric_fields = ['min_trajectory_length', 'max_trajectory_length']
        
        for field in numeric_fields:
            if field in qc_config:
                value = qc_config[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    logging.error(f"质量控制字段{field}必须是正数: {value}")
                    return False
        
        return True
    
    def _merge_configs(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置，override覆盖default"""
        merged = copy.deepcopy(default)
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        
        return merged
    
    def _process_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """处理配置中的环境变量"""
        processed = copy.deepcopy(config)
        
        # 处理nuScenes数据路径的环境变量
        if 'dataset' in processed and 'nuscenes_dataroot' in processed['dataset']:
            dataroot = processed['dataset']['nuscenes_dataroot']
            if isinstance(dataroot, str) and dataroot.startswith('$'):
                env_var = dataroot[1:]
                if env_var in os.environ:
                    processed['dataset']['nuscenes_dataroot'] = os.environ[env_var]
                    logging.info(f"使用环境变量 {env_var}: {os.environ[env_var]}")
        
        return processed
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'dataset': {
                'nuscenes_version': 'v1.0-mini',
                'output_directory': './output'
            },
            'trajectory': {
                'prediction_horizon': 3.0,
                'sampling_rate': 12.0,
                'num_waypoints': 36
            },
            'processing': {
                'num_workers': 4,
                'batch_size': 32,
                'enable_validation': True
            },
            'cameras': {
                'channels': [
                    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
                ]
            },
            'quality_control': {
                'min_trajectory_length': 36,
                'max_trajectory_length': 36
            }
        }
    
    def save_config(self, config: Dict[str, Any], output_path: str) -> None:
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        logging.info(f"配置已保存到: {output_path}")
    
    def get_config_summary(self, config: Dict[str, Any]) -> str:
        """
        获取配置摘要信息
        
        Args:
            config: 配置字典
            
        Returns:
            配置摘要字符串
        """
        summary_lines = []
        summary_lines.append("配置摘要:")
        summary_lines.append("=" * 40)
        
        # 数据集信息
        if 'dataset' in config:
            dataset = config['dataset']
            summary_lines.append(f"nuScenes版本: {dataset.get('nuscenes_version')}")
            summary_lines.append(f"输出目录: {dataset.get('output_directory')}")
        
        # 轨迹信息
        if 'trajectory' in config:
            traj = config['trajectory']
            summary_lines.append(f"预测时域: {traj.get('prediction_horizon')}秒")
            summary_lines.append(f"采样频率: {traj.get('sampling_rate')}Hz")
            summary_lines.append(f"路径点数: {traj.get('num_waypoints')}")
        
        # 处理信息
        if 'processing' in config:
            proc = config['processing']
            summary_lines.append(f"工作进程: {proc.get('num_workers')}")
            summary_lines.append(f"批大小: {proc.get('batch_size')}")
        
        return "\n".join(summary_lines)


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_file_paths(config: Dict[str, Any]) -> List[str]:
        """
        验证配置中的文件路径
        
        Args:
            config: 配置字典
            
        Returns:
            错误信息列表
        """
        errors = []
        
        # 检查nuScenes数据路径
        if 'dataset' in config and 'nuscenes_dataroot' in config['dataset']:
            dataroot = config['dataset']['nuscenes_dataroot']
            if dataroot and not os.path.exists(dataroot):
                errors.append(f"nuScenes数据路径不存在: {dataroot}")
        
        return errors
    
    @staticmethod
    def validate_parameter_ranges(config: Dict[str, Any]) -> List[str]:
        """
        验证参数范围
        
        Args:
            config: 配置字典
            
        Returns:
            错误信息列表
        """
        errors = []
        
        # 验证轨迹参数
        if 'trajectory' in config:
            traj = config['trajectory']
            
            horizon = traj.get('prediction_horizon', 0)
            if horizon < 1.0 or horizon > 10.0:
                errors.append(f"预测时域应在1-10秒范围内: {horizon}")
            
            sampling_rate = traj.get('sampling_rate', 0)
            if sampling_rate < 1.0 or sampling_rate > 100.0:
                errors.append(f"采样频率应在1-100Hz范围内: {sampling_rate}")
        
        return errors