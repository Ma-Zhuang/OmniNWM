"""
坐标系变换核心逻辑

实现nuScenes数据集中所需的坐标系变换：
- 四元数到旋转矩阵转换
- 齐次变换矩阵构建和求逆  
- 全局坐标系到自车坐标系变换
- 航向角提取和归一化
"""

from typing import List, Tuple, Dict
import numpy as np
from pyquaternion import Quaternion
import logging


class CoordinateTransformer:
    """坐标系变换核心逻辑"""
    
    @staticmethod
    def quaternion_to_rotation_matrix(quaternion: List[float]) -> np.ndarray:
        """
        将四元数转换为旋转矩阵
        
        Args:
            quaternion: 四元数 [w, x, y, z]
            
        Returns:
            3x3旋转矩阵
        """
        q = Quaternion(quaternion)
        return q.rotation_matrix
    
    @staticmethod
    def build_transformation_matrix(translation: List[float], rotation: List[float]) -> np.ndarray:
        """
        构建4x4齐次变换矩阵
        
        Args:
            translation: 平移向量 [x, y, z]
            rotation: 四元数旋转 [w, x, y, z]
            
        Returns:
            4x4齐次变换矩阵
        """
        # 获取旋转矩阵
        R = CoordinateTransformer.quaternion_to_rotation_matrix(rotation)
        
        # 构建齐次变换矩阵
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation
        
        return T
    
    @staticmethod
    def inverse_transformation_matrix(transform_matrix: np.ndarray) -> np.ndarray:
        """
        计算变换矩阵的逆矩阵
        
        Args:
            transform_matrix: 4x4齐次变换矩阵
            
        Returns:
            逆变换矩阵
        """
        # 提取旋转矩阵和平移向量
        R = transform_matrix[:3, :3]
        t = transform_matrix[:3, 3]
        
        # 构建逆变换矩阵
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T  # 旋转矩阵的逆等于其转置
        T_inv[:3, 3] = -R.T @ t  # 逆平移
        
        return T_inv
    
    @staticmethod
    def transform_global_to_ego(global_pose: Dict, reference_pose: Dict) -> Tuple[float, float, float]:
        """
        将全局坐标系下的位姿转换到参考位姿的自车坐标系
        
        Args:
            global_pose: 全局坐标系下的位姿字典
            reference_pose: 参考位姿字典
            
        Returns:
            (x, y, heading) - 自车坐标系下的相对位置和航向角
        """
        # 构建变换矩阵
        T_global_to_reference = CoordinateTransformer.build_transformation_matrix(
            reference_pose['translation'], 
            reference_pose['rotation']
        )
        
        T_global_to_target = CoordinateTransformer.build_transformation_matrix(
            global_pose['translation'],
            global_pose['rotation'] 
        )
        
        # 计算相对变换
        T_reference_to_global = CoordinateTransformer.inverse_transformation_matrix(T_global_to_reference)
        T_relative = T_reference_to_global @ T_global_to_target
        
        # 提取相对位置
        x = T_relative[0, 3]
        y = T_relative[1, 3]
        
        # 提取相对航向角
        heading = CoordinateTransformer.extract_heading_from_rotation_matrix(T_relative[:3, :3])
        
        return x, y, heading
    
    @staticmethod
    def extract_heading_from_rotation_matrix(rotation_matrix: np.ndarray) -> float:
        """
        从旋转矩阵提取偏航角（heading）
        
        Args:
            rotation_matrix: 3x3旋转矩阵
            
        Returns:
            偏航角（弧度）
        """
        # 从旋转矩阵提取偏航角
        # 偏航角是绕z轴的旋转
        heading = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        
        return heading
    
    @staticmethod
    def interpolate_angle(angle1: float, angle2: float, alpha: float) -> float:
        """
        角度的球面线性插值，正确处理角度边界
        
        Args:
            angle1: 起始角度（弧度）
            angle2: 结束角度（弧度）
            alpha: 插值系数 [0, 1]
            
        Returns:
            插值后的角度
        """
        # 将角度归一化到[-π, π]
        angle1 = CoordinateTransformer.normalize_angle(angle1)
        angle2 = CoordinateTransformer.normalize_angle(angle2)
        
        # 计算角度差
        diff = angle2 - angle1
        
        # 处理角度跨越边界的情况
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
            
        # 线性插值
        result = angle1 + alpha * diff
        
        # 归一化结果
        return CoordinateTransformer.normalize_angle(result)
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        将角度归一化到[-π, π]区间
        
        Args:
            angle: 输入角度（弧度）
            
        Returns:
            归一化后的角度
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    @staticmethod
    def transform_point_global_to_ego(point_global: np.ndarray, ego_pose: Dict) -> np.ndarray:
        """
        将全局坐标系下的点转换到自车坐标系
        
        Args:
            point_global: 全局坐标系下的点 [x, y, z]
            ego_pose: 自车位姿字典
            
        Returns:
            自车坐标系下的点
        """
        # 构建变换矩阵
        T_global_to_ego = CoordinateTransformer.build_transformation_matrix(
            ego_pose['translation'],
            ego_pose['rotation']
        )
        
        # 计算逆变换矩阵
        T_ego_to_global = CoordinateTransformer.inverse_transformation_matrix(T_global_to_ego)
        
        # 转换点（添加齐次坐标）
        point_homo = np.append(point_global, 1.0)
        point_ego_homo = T_ego_to_global @ point_homo
        
        return point_ego_homo[:3]
    
    @staticmethod
    def calculate_relative_pose_chain(poses: List[Dict], reference_index: int = 0) -> List[Tuple[float, float, float]]:
        """
        计算相对于参考位姿的位姿链
        
        Args:
            poses: 位姿字典列表
            reference_index: 参考位姿的索引
            
        Returns:
            相对位姿列表 [(x, y, heading), ...]
        """
        if not poses or reference_index >= len(poses):
            raise ValueError("Invalid poses list or reference index")
        
        reference_pose = poses[reference_index]
        relative_poses = []
        
        for pose in poses:
            x, y, heading = CoordinateTransformer.transform_global_to_ego(pose, reference_pose)
            relative_poses.append((x, y, heading))
            
        return relative_poses
    
    @staticmethod
    def validate_transformation_matrix(T: np.ndarray) -> bool:
        """
        验证变换矩阵的有效性
        
        Args:
            T: 4x4变换矩阵
            
        Returns:
            矩阵是否有效
        """
        if T.shape != (4, 4):
            return False
            
        # 检查齐次坐标的最后一行
        if not np.allclose(T[3, :], [0, 0, 0, 1]):
            return False
            
        # 检查旋转矩阵的正交性
        R = T[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            return False
            
        # 检查行列式是否为1（右手坐标系）
        if not np.allclose(np.linalg.det(R), 1.0, atol=1e-6):
            return False
            
        return True