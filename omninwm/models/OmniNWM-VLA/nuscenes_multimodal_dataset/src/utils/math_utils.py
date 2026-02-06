"""
数学计算工具库

提供各种数学计算功能：
- 角度处理和归一化
- 几何距离计算
- 轨迹插值和处理
- 统计分析工具
- 轨迹评估指标
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class AngleUtils:
    """角度处理工具类"""
    
    @staticmethod
    def wrap_angle(angle: float, range_type: str = 'pi') -> float:
        """
        角度归一化
        
        Args:
            angle: 输入角度（弧度）
            range_type: 归一化范围 ('pi': [-π, π], '2pi': [0, 2π])
            
        Returns:
            归一化后的角度
        """
        if range_type == 'pi':
            # 归一化到 [-π, π]
            while angle > math.pi:
                angle -= 2 * math.pi
            while angle < -math.pi:
                angle += 2 * math.pi
        elif range_type == '2pi':
            # 归一化到 [0, 2π]
            while angle >= 2 * math.pi:
                angle -= 2 * math.pi
            while angle < 0:
                angle += 2 * math.pi
        else:
            raise ValueError(f"不支持的角度范围类型: {range_type}")
        
        return angle
    
    @staticmethod
    def angle_difference(angle1: float, angle2: float) -> float:
        """
        计算两个角度的最短差值
        
        Args:
            angle1: 第一个角度（弧度）
            angle2: 第二个角度（弧度）
            
        Returns:
            角度差值（-π到π之间）
        """
        diff = angle2 - angle1
        return AngleUtils.wrap_angle(diff, 'pi')
    
    @staticmethod
    def degrees_to_radians(degrees: float) -> float:
        """度数转弧度"""
        return degrees * math.pi / 180.0
    
    @staticmethod
    def radians_to_degrees(radians: float) -> float:
        """弧度转度数"""
        return radians * 180.0 / math.pi
    
    @staticmethod
    def angle_interpolation(angle1: float, angle2: float, t: float) -> float:
        """
        角度插值（考虑角度边界特性）
        
        Args:
            angle1: 起始角度
            angle2: 终止角度
            t: 插值参数 [0, 1]
            
        Returns:
            插值结果
        """
        diff = AngleUtils.angle_difference(angle1, angle2)
        return AngleUtils.wrap_angle(angle1 + t * diff, 'pi')


class GeometryUtils:
    """几何计算工具类"""
    
    @staticmethod
    def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        计算两点欧氏距离
        
        Args:
            point1: 第一个点 (x, y)
            point2: 第二个点 (x, y)
            
        Returns:
            欧氏距离
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        return math.sqrt(dx * dx + dy * dy)
    
    @staticmethod
    def manhattan_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        计算两点曼哈顿距离
        
        Args:
            point1: 第一个点 (x, y)
            point2: 第二个点 (x, y)
            
        Returns:
            曼哈顿距离
        """
        return abs(point2[0] - point1[0]) + abs(point2[1] - point1[1])
    
    @staticmethod
    def point_to_line_distance(point: Tuple[float, float], 
                              line_start: Tuple[float, float], 
                              line_end: Tuple[float, float]) -> float:
        """
        计算点到线段的距离
        
        Args:
            point: 目标点
            line_start: 线段起点
            line_end: 线段终点
            
        Returns:
            点到线段的最短距离
        """
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 线段长度的平方
        line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        
        if line_length_sq == 0:
            # 线段退化为点
            return GeometryUtils.euclidean_distance(point, line_start)
        
        # 计算投影参数
        t = max(0, min(1, ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / line_length_sq))
        
        # 投影点
        projection_x = x1 + t * (x2 - x1)
        projection_y = y1 + t * (y2 - y1)
        
        return GeometryUtils.euclidean_distance(point, (projection_x, projection_y))
    
    @staticmethod
    def calculate_heading(start_point: Tuple[float, float], 
                         end_point: Tuple[float, float]) -> float:
        """
        计算从起点到终点的航向角
        
        Args:
            start_point: 起点 (x, y)
            end_point: 终点 (x, y)
            
        Returns:
            航向角（弧度，相对于x轴正方向）
        """
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        return math.atan2(dy, dx)
    
    @staticmethod
    def rotate_point(point: Tuple[float, float], 
                    center: Tuple[float, float], 
                    angle: float) -> Tuple[float, float]:
        """
        绕中心点旋转点
        
        Args:
            point: 要旋转的点
            center: 旋转中心
            angle: 旋转角度（弧度，逆时针）
            
        Returns:
            旋转后的点
        """
        # 移动到原点
        x = point[0] - center[0]
        y = point[1] - center[1]
        
        # 旋转
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        
        new_x = x * cos_angle - y * sin_angle
        new_y = x * sin_angle + y * cos_angle
        
        # 移动回去
        return (new_x + center[0], new_y + center[1])
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """角度归一化到[-π, π]"""
        return AngleUtils.wrap_angle(angle, 'pi')
    
    @staticmethod
    def cos(angle: float) -> float:
        """余弦函数"""
        return math.cos(angle)
    
    @staticmethod
    def sin(angle: float) -> float:
        """正弦函数"""
        return math.sin(angle)


class TrajectoryUtils:
    """轨迹处理工具类"""
    
    @staticmethod
    def interpolate_trajectory(waypoints: List[Dict], target_length: int) -> List[Dict]:
        """
        轨迹插值
        
        Args:
            waypoints: 原始轨迹点
            target_length: 目标长度
            
        Returns:
            插值后的轨迹
        """
        if len(waypoints) < 2:
            return waypoints
        
        if len(waypoints) >= target_length:
            # 下采样
            indices = np.linspace(0, len(waypoints) - 1, target_length, dtype=int)
            return [waypoints[i] for i in indices]
        
        # 上采样插值
        interpolated = []
        
        # 计算累积距离
        distances = [0.0]
        for i in range(1, len(waypoints)):
            dist = GeometryUtils.euclidean_distance(
                (waypoints[i-1]['x'], waypoints[i-1]['y']),
                (waypoints[i]['x'], waypoints[i]['y'])
            )
            distances.append(distances[-1] + dist)
        
        total_distance = distances[-1]
        if total_distance == 0:
            return waypoints
        
        # 插值
        for i in range(target_length):
            t = i / (target_length - 1) if target_length > 1 else 0
            target_distance = t * total_distance
            
            # 找到对应的线段
            segment_idx = 0
            for j in range(len(distances) - 1):
                if distances[j] <= target_distance <= distances[j + 1]:
                    segment_idx = j
                    break
            
            # 在线段内插值
            if segment_idx < len(waypoints) - 1:
                segment_start_dist = distances[segment_idx]
                segment_end_dist = distances[segment_idx + 1]
                segment_length = segment_end_dist - segment_start_dist
                
                if segment_length > 0:
                    local_t = (target_distance - segment_start_dist) / segment_length
                else:
                    local_t = 0
                
                wp1 = waypoints[segment_idx]
                wp2 = waypoints[segment_idx + 1]
                
                # 线性插值位置
                x = wp1['x'] + local_t * (wp2['x'] - wp1['x'])
                y = wp1['y'] + local_t * (wp2['y'] - wp1['y'])
                
                # 角度插值
                heading = AngleUtils.angle_interpolation(wp1['heading'], wp2['heading'], local_t)
                
                interpolated.append({
                    'x': x,
                    'y': y,
                    'heading': heading
                })
            else:
                # 最后一个点
                interpolated.append(waypoints[-1].copy())
        
        return interpolated
    
    @staticmethod
    def resample_trajectory(trajectory: List[Dict], target_rate: float, horizon: float) -> List[Dict]:
        """
        重采样轨迹到目标频率
        
        Args:
            trajectory: 原始轨迹
            target_rate: 目标采样率 (Hz)
            horizon: 时间跨度 (秒)
            
        Returns:
            重采样后的轨迹
        """
        target_length = int(target_rate * horizon)
        return TrajectoryUtils.interpolate_trajectory(trajectory, target_length)
    
    @staticmethod
    def calculate_trajectory_curvature(waypoints: List[Dict]) -> List[float]:
        """
        计算轨迹曲率
        
        Args:
            waypoints: 轨迹点
            
        Returns:
            曲率列表
        """
        if len(waypoints) < 3:
            return [0.0] * len(waypoints)
        
        curvatures = [0.0]  # 第一个点曲率为0
        
        for i in range(1, len(waypoints) - 1):
            # 三个连续点
            p1 = (waypoints[i-1]['x'], waypoints[i-1]['y'])
            p2 = (waypoints[i]['x'], waypoints[i]['y'])
            p3 = (waypoints[i+1]['x'], waypoints[i+1]['y'])
            
            # 计算曲率
            curvature = TrajectoryUtils._calculate_curvature_at_point(p1, p2, p3)
            curvatures.append(curvature)
        
        curvatures.append(0.0)  # 最后一个点曲率为0
        
        return curvatures
    
    @staticmethod
    def _calculate_curvature_at_point(p1: Tuple[float, float], 
                                     p2: Tuple[float, float], 
                                     p3: Tuple[float, float]) -> float:
        """计算某点处的曲率"""
        # 向量
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # 长度
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # 单位向量
        u1 = (v1[0] / len1, v1[1] / len1)
        u2 = (v2[0] / len2, v2[1] / len2)
        
        # 角度变化
        cross_product = u1[0] * u2[1] - u1[1] * u2[0]
        dot_product = u1[0] * u2[0] + u1[1] * u2[1]
        
        angle_change = math.atan2(cross_product, dot_product)
        
        # 曲率 = 角度变化 / 弧长
        arc_length = (len1 + len2) / 2
        return abs(angle_change) / arc_length if arc_length > 0 else 0.0
    
    @staticmethod
    def smooth_trajectory(waypoints: List[Dict], window_size: int = 3) -> List[Dict]:
        """
        轨迹平滑
        
        Args:
            waypoints: 原始轨迹点
            window_size: 平滑窗口大小
            
        Returns:
            平滑后的轨迹点
        """
        if len(waypoints) < window_size:
            return waypoints
        
        smoothed = []
        half_window = window_size // 2
        
        for i in range(len(waypoints)):
            # 确定窗口范围
            start_idx = max(0, i - half_window)
            end_idx = min(len(waypoints), i + half_window + 1)
            
            # 计算平滑位置
            sum_x = sum(wp['x'] for wp in waypoints[start_idx:end_idx])
            sum_y = sum(wp['y'] for wp in waypoints[start_idx:end_idx])
            count = end_idx - start_idx
            
            # 平滑后的位置
            smooth_x = sum_x / count
            smooth_y = sum_y / count
            
            # 角度保持原始值（避免角度平滑的复杂性）
            smooth_heading = waypoints[i]['heading']
            
            smoothed.append({
                'x': smooth_x,
                'y': smooth_y,
                'heading': smooth_heading
            })
        
        return smoothed


class StatisticsUtils:
    """统计分析工具类"""
    
    @staticmethod
    def calculate_basic_stats(values: List[float]) -> Dict[str, float]:
        """
        计算基础统计量
        
        Args:
            values: 数值列表
            
        Returns:
            统计量字典
        """
        if not values:
            return {}
        
        values_array = np.array(values)
        
        return {
            'count': len(values),
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'median': float(np.median(values_array)),
            'q25': float(np.percentile(values_array, 25)),
            'q75': float(np.percentile(values_array, 75))
        }
    
    @staticmethod
    def calculate_correlation(x: List[float], y: List[float]) -> float:
        """
        计算相关系数
        
        Args:
            x: 第一组数据
            y: 第二组数据
            
        Returns:
            相关系数
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        return float(np.corrcoef(x, y)[0, 1])
    
    @staticmethod
    def moving_average(values: List[float], window_size: int) -> List[float]:
        """
        计算移动平均
        
        Args:
            values: 原始数据
            window_size: 窗口大小
            
        Returns:
            移动平均结果
        """
        if window_size <= 0 or len(values) < window_size:
            return values
        
        moving_avg = []
        for i in range(len(values) - window_size + 1):
            window_values = values[i:i + window_size]
            avg = sum(window_values) / window_size
            moving_avg.append(avg)
        
        return moving_avg


class TrajectoryMetrics:
    """轨迹评估指标类"""
    
    @staticmethod
    def calculate_ade(predicted: List[Dict], ground_truth: List[Dict]) -> float:
        """
        计算平均位移误差 (Average Displacement Error)
        
        Args:
            predicted: 预测轨迹
            ground_truth: 真实轨迹
            
        Returns:
            ADE值
        """
        if len(predicted) != len(ground_truth):
            logger.warning(f"轨迹长度不匹配: {len(predicted)} vs {len(ground_truth)}")
            min_len = min(len(predicted), len(ground_truth))
            predicted = predicted[:min_len]
            ground_truth = ground_truth[:min_len]
        
        if not predicted:
            return float('inf')
        
        total_error = 0.0
        for pred, gt in zip(predicted, ground_truth):
            error = GeometryUtils.euclidean_distance(
                (pred['x'], pred['y']),
                (gt['x'], gt['y'])
            )
            total_error += error
        
        return total_error / len(predicted)
    
    @staticmethod
    def calculate_fde(predicted: List[Dict], ground_truth: List[Dict]) -> float:
        """
        计算最终位移误差 (Final Displacement Error)
        
        Args:
            predicted: 预测轨迹
            ground_truth: 真实轨迹
            
        Returns:
            FDE值
        """
        if not predicted or not ground_truth:
            return float('inf')
        
        pred_final = predicted[-1]
        gt_final = ground_truth[-1]
        
        return GeometryUtils.euclidean_distance(
            (pred_final['x'], pred_final['y']),
            (gt_final['x'], gt_final['y'])
        )
    
    @staticmethod
    def calculate_heading_error(predicted: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        """
        计算航向角误差
        
        Args:
            predicted: 预测轨迹
            ground_truth: 真实轨迹
            
        Returns:
            航向角误差统计
        """
        if len(predicted) != len(ground_truth):
            min_len = min(len(predicted), len(ground_truth))
            predicted = predicted[:min_len]
            ground_truth = ground_truth[:min_len]
        
        if not predicted:
            return {}
        
        heading_errors = []
        for pred, gt in zip(predicted, ground_truth):
            error = abs(AngleUtils.angle_difference(pred['heading'], gt['heading']))
            heading_errors.append(error)
        
        return StatisticsUtils.calculate_basic_stats(heading_errors)


def interpolate_poses(pose1: Dict, pose2: Dict, alpha: float) -> Dict:
    """
    插值两个位姿
    
    Args:
        pose1: 第一个位姿
        pose2: 第二个位姿  
        alpha: 插值参数 [0, 1]
        
    Returns:
        插值结果
    """
    result = {}
    
    # 位置插值
    if 'translation' in pose1 and 'translation' in pose2:
        t1 = np.array(pose1['translation'])
        t2 = np.array(pose2['translation'])
        result['translation'] = (t1 * (1 - alpha) + t2 * alpha).tolist()
    
    # 旋转四元数插值
    if 'rotation' in pose1 and 'rotation' in pose2:
        try:
            from pyquaternion import Quaternion
            q1 = Quaternion(pose1['rotation'])
            q2 = Quaternion(pose2['rotation'])
            result['rotation'] = Quaternion.slerp(q1, q2, alpha).q.tolist()
        except ImportError:
            # 如果没有pyquaternion，使用简单线性插值
            r1 = np.array(pose1['rotation'])
            r2 = np.array(pose2['rotation'])
            result['rotation'] = (r1 * (1 - alpha) + r2 * alpha).tolist()
    
    # 插值其他字段
    for key in pose1:
        if key not in ['translation', 'rotation'] and key in pose2:
            if isinstance(pose1[key], (int, float)) and isinstance(pose2[key], (int, float)):
                result[key] = pose1[key] * (1 - alpha) + pose2[key] * alpha
            else:
                result[key] = pose1[key] if alpha < 0.5 else pose2[key]
    
    return result


def calculate_trajectory_metrics(pred_trajectory: List, gt_trajectory: List) -> Dict:
    """
    计算轨迹预测评估指标
    
    Args:
        pred_trajectory: 预测轨迹
        gt_trajectory: 真实轨迹
        
    Returns:
        评估指标字典
    """
    metrics = {}
    
    try:
        # ADE和FDE
        metrics['ade'] = TrajectoryMetrics.calculate_ade(pred_trajectory, gt_trajectory)
        metrics['fde'] = TrajectoryMetrics.calculate_fde(pred_trajectory, gt_trajectory)
        
        # 航向角误差
        heading_stats = TrajectoryMetrics.calculate_heading_error(pred_trajectory, gt_trajectory)
        metrics['heading_error'] = heading_stats
        
        # 轨迹长度
        metrics['pred_length'] = len(pred_trajectory)
        metrics['gt_length'] = len(gt_trajectory)
        
        # 总位移
        if pred_trajectory:
            pred_displacement = GeometryUtils.euclidean_distance(
                (pred_trajectory[0]['x'], pred_trajectory[0]['y']),
                (pred_trajectory[-1]['x'], pred_trajectory[-1]['y'])
            )
            metrics['pred_total_displacement'] = pred_displacement
        
        if gt_trajectory:
            gt_displacement = GeometryUtils.euclidean_distance(
                (gt_trajectory[0]['x'], gt_trajectory[0]['y']),
                (gt_trajectory[-1]['x'], gt_trajectory[-1]['y'])
            )
            metrics['gt_total_displacement'] = gt_displacement
            
    except Exception as e:
        logger.error(f"计算轨迹指标失败: {e}")
        metrics['error'] = str(e)
    
    return metrics