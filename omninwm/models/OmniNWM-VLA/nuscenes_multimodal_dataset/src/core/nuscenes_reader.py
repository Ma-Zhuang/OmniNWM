"""
nuScenes数据集读取器

提供对nuScenes数据集的标准化访问接口，包括：
- 场景（scene）遍历
- 样本（sample）数据获取  
- 传感器数据（sample_data）访问
- 自车位姿（ego_pose）信息
"""

from typing import List, Dict, Optional, Tuple
import logging
import os
import json
import numpy as np
from nuscenes.nuscenes import NuScenes

# CAN总线支持（可选导入）
try:
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    CAN_BUS_AVAILABLE = True
except ImportError:
    logging.warning("CAN bus extension not available. Historical vehicle data will be limited.")
    CAN_BUS_AVAILABLE = False
    NuScenesCanBus = None


class NuScenesReader:
    """nuScenes数据集读取器"""
    
    def __init__(self, dataroot: str, version: str = 'v1.0-trainval', verbose: bool = True, enable_can_bus: bool = True):
        """
        初始化nuScenes读取器
        
        Args:
            dataroot: nuScenes数据集根目录路径
            version: 数据集版本，默认为'v1.0-trainval'
            verbose: 是否显示详细信息
            enable_can_bus: 是否启用CAN总线数据读取（需要can_bus扩展）
        """
        self.dataroot = dataroot
        self.version = version
        self.verbose = verbose
        self.enable_can_bus = enable_can_bus and CAN_BUS_AVAILABLE
        self.nusc = None
        self.nusc_can = None
        self._initialize_nuscenes()
        if self.enable_can_bus:
            self._initialize_can_bus()
        
    def _initialize_nuscenes(self):
        """初始化nuScenes实例"""
        try:
            self.nusc = NuScenes(
                version=self.version,
                dataroot=self.dataroot,
                verbose=self.verbose
            )
            logging.info(f"Successfully loaded nuScenes {self.version} from {self.dataroot}")
        except Exception as e:
            logging.error(f"Failed to initialize nuScenes: {e}")
            raise
    
    def _initialize_can_bus(self):
        """初始化CAN总线数据读取器"""
        try:
            if CAN_BUS_AVAILABLE and NuScenesCanBus:
                self.nusc_can = NuScenesCanBus(dataroot=self.dataroot)
                logging.info("CAN bus data reader initialized successfully")
            else:
                logging.warning("CAN bus data not available")
                self.enable_can_bus = False
        except Exception as e:
            logging.warning(f"Failed to initialize CAN bus reader: {e}")
            self.enable_can_bus = False
            self.nusc_can = None
    
    def get_scenes(self) -> List[Dict]:
        """
        获取所有场景信息
        
        Returns:
            包含所有场景信息的字典列表
        """
        return self.nusc.scene
    
    def get_samples_from_scene(self, scene_token: str) -> List[Dict]:
        """
        从指定场景获取所有样本
        
        Args:
            scene_token: 场景令牌
            
        Returns:
            该场景中所有样本的列表
        """
        scene = self.nusc.get('scene', scene_token)
        samples = []
        
        # 从第一个样本开始遍历
        current_sample_token = scene['first_sample_token']
        
        while current_sample_token:
            sample = self.nusc.get('sample', current_sample_token)
            samples.append(sample)
            current_sample_token = sample['next']
            
        return samples
    
    def get_sample_data(self, sample_token: str, sensor_channel: str) -> Dict:
        """
        获取指定样本和传感器通道的数据
        
        Args:
            sample_token: 样本令牌
            sensor_channel: 传感器通道名称（如'CAM_FRONT'）
            
        Returns:
            样本数据字典
        """
        sample = self.nusc.get('sample', sample_token)
        sample_data_token = sample['data'][sensor_channel]
        return self.nusc.get('sample_data', sample_data_token)
    
    def get_ego_pose(self, ego_pose_token: str) -> Dict:
        """
        获取自车位姿信息
        
        Args:
            ego_pose_token: 自车位姿令牌
            
        Returns:
            自车位姿字典，包含translation和rotation
        """
        return self.nusc.get('ego_pose', ego_pose_token)
    
    def validate_sample_chain_length(self, sample_token: str, required_length: int) -> bool:
        """
        验证从指定样本开始是否有足够长度的样本链
        
        Args:
            sample_token: 起始样本令牌
            required_length: 需要的样本数量
            
        Returns:
            是否有足够长度的样本链
        """
        current_token = sample_token
        count = 0
        
        while current_token and count < required_length:
            sample = self.nusc.get('sample', current_token)
            current_token = sample['next']
            count += 1
            
        return count >= required_length
    
    def get_camera_channels(self) -> List[str]:
        """
        获取所有摄像头通道名称
        
        Returns:
            摄像头通道名称列表
        """
        return [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT', 
            'CAM_BACK_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_FRONT_LEFT'
        ]
    
    def get_sample_by_token(self, sample_token: str) -> Dict:
        """
        通过令牌获取样本
        
        Args:
            sample_token: 样本令牌
            
        Returns:
            样本字典
        """
        return self.nusc.get('sample', sample_token)
    
    def get_next_samples(self, sample_token: str, num_samples: int) -> List[Dict]:
        """
        获取指定数量的后续样本
        
        Args:
            sample_token: 起始样本令牌
            num_samples: 需要获取的样本数量
            
        Returns:
            后续样本列表
        """
        samples = []
        current_token = sample_token
        
        for _ in range(num_samples):
            if not current_token:
                break
                
            sample = self.nusc.get('sample', current_token)
            samples.append(sample)
            current_token = sample['next']
            
        return samples
    
    def get_previous_samples(self, sample_token: str, num_samples: int) -> List[Dict]:
        """
        获取指定数量的前置样本（用于历史数据）
        
        Args:
            sample_token: 起始样本令牌
            num_samples: 需要获取的样本数量
            
        Returns:
            前置样本列表（按时间正序排列）
        """
        samples = []
        current_token = sample_token
        
        # 先向前搜集所有可用的前置样本
        all_previous = []
        while current_token:
            sample = self.nusc.get('sample', current_token)
            if sample['prev']:
                prev_sample = self.nusc.get('sample', sample['prev'])
                all_previous.append(prev_sample)
                current_token = sample['prev']
            else:
                break
        
        # 反转列表以保持时间正序，并只取所需数量
        all_previous.reverse()
        return all_previous[-num_samples:] if len(all_previous) >= num_samples else all_previous
    
    def get_can_bus_data(self, scene_token: str, sample_timestamp: int, 
                        message_names: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
        """
        获取CAN总线数据
        
        Args:
            scene_token: 场景令牌
            sample_timestamp: 样本时间戳
            message_names: 需要获取的消息类型列表
            
        Returns:
            CAN总线数据字典
        """
        if not self.enable_can_bus or not self.nusc_can:
            logging.warning("CAN bus data not available")
            return {}
        
        try:
            # 将场景token转换为场景名称（CAN总线API需要场景名称，如'scene-0001'）
            scene = self.nusc.get('scene', scene_token)
            scene_name = scene['name']
            
            # 默认获取车辆状态相关消息（使用nuScenes CAN文件的实际命名）
            if message_names is None:
                message_names = ['pose', 'steeranglefeedback', 'vehicle_monitor']
            
            can_data = {}
            for message_name in message_names:
                try:
                    # 使用场景名称而不是token来获取CAN数据
                    messages = self.nusc_can.get_messages(scene_name, message_name)
                    
                    # 过滤到时间戳附近的数据
                    relevant_messages = []
                    for msg in messages:
                        # 寻找在样本时间戳前后1秒内的数据
                        if abs(msg['utime'] - sample_timestamp) <= 1000000:  # 1秒 = 1,000,000 微秒
                            relevant_messages.append(msg)
                    
                    can_data[message_name] = relevant_messages
                except Exception as e:
                    # 只在调试模式下记录CAN消息错误
                    logging.debug(f"Failed to get CAN message {message_name}: {e}")
                    can_data[message_name] = []
            
            return can_data
            
        except Exception as e:
            logging.error(f"Failed to get CAN bus data: {e}")
            return {}
    
    def extract_vehicle_state_from_can(self, can_data: Dict[str, List[Dict]], 
                                     target_timestamp: int) -> Dict[str, float]:
        """
        从 CAN 总线数据中提取车辆状态
        
        Args:
            can_data: CAN总线数据字典
            target_timestamp: 目标时间戳
            
        Returns:
            车辆状态字典（包含速度、加速度、转向角等）
        """
        vehicle_state = {
            'velocity_x': 0.0,
            'velocity_y': 0.0,
            'acceleration_x': 0.0,
            'acceleration_y': 0.0,
            'steering_angle': 0.0,
            'speed': 0.0
        }
        
        try:
            # 从pose 消息中获取速度和加速度
            if 'pose' in can_data and can_data['pose']:
                pose_msg = self._find_closest_message(can_data['pose'], target_timestamp)
                if pose_msg:
                    vehicle_state['velocity_x'] = pose_msg.get('vel', [0, 0, 0])[0]
                    vehicle_state['velocity_y'] = pose_msg.get('vel', [0, 0, 0])[1]
                    vehicle_state['acceleration_x'] = pose_msg.get('accel', [0, 0, 0])[0]
                    vehicle_state['acceleration_y'] = pose_msg.get('accel', [0, 0, 0])[1]
                    
                    # 计算总速度
                    vel_x, vel_y = vehicle_state['velocity_x'], vehicle_state['velocity_y']
                    vehicle_state['speed'] = (vel_x**2 + vel_y**2)**0.5
            
            # 从转向角消息中获取转向角
            if 'steeranglefeedback' in can_data and can_data['steeranglefeedback']:
                steer_msg = self._find_closest_message(can_data['steeranglefeedback'], target_timestamp)
                if steer_msg:
                    vehicle_state['steering_angle'] = steer_msg.get('value', 0.0)
            
        except Exception as e:
            logging.warning(f"Failed to extract vehicle state from CAN data: {e}")
        
        return vehicle_state
    
    def _find_closest_message(self, messages: List[Dict], target_timestamp: int) -> Optional[Dict]:
        """
        在消息列表中找到最接近目标时间戳的消息
        
        Args:
            messages: 消息列表
            target_timestamp: 目标时间戳
            
        Returns:
            最接近的消息或None
        """
        if not messages:
            return None
        
        closest_msg = None
        min_diff = float('inf')
        
        for msg in messages:
            time_diff = abs(msg['utime'] - target_timestamp)
            if time_diff < min_diff:
                min_diff = time_diff
                closest_msg = msg
        
        return closest_msg