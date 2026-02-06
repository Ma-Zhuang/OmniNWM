"""
轨迹计算器

基于nuScenes数据计算未来轨迹：
- 提取未来样本的自车位姿
- 计算相对于当前位姿的轨迹点
- 验证轨迹完整性
- 角度归一化处理
"""

from typing import List, Dict, Optional, Tuple
import logging
import numpy as np
from .coordinate_transform import CoordinateTransformer
from .nuscenes_reader import NuScenesReader


class TrajectoryCalculator:
    """轨迹计算器"""
    
    def __init__(self, prediction_horizon: float = 3.0, sampling_rate: float = 12.0, 
                 interpolation_mode: bool = True, history_horizon: float = 1.0):
        """
        初始化轨迹计算器
        
        Args:
            prediction_horizon: 预测时域（秒）
            sampling_rate: 采样频率（Hz）
            interpolation_mode: 是否使用插值模式（从nuScenes 2Hz插值到目标频率）
            history_horizon: 历史轨迹时域（秒）
        """
        self.prediction_horizon = prediction_horizon
        self.sampling_rate = sampling_rate
        self.num_waypoints = int(prediction_horizon * sampling_rate)
        self.interpolation_mode = interpolation_mode
        self.history_horizon = history_horizon
        self.num_history_points = int(history_horizon * sampling_rate)
        
        # nuScenes原生采样频率
        self.nuscenes_native_rate = 2.0
        self.nuscenes_samples_needed = int(prediction_horizon * self.nuscenes_native_rate) + 1
        self.nuscenes_history_samples_needed = int(history_horizon * self.nuscenes_native_rate) + 1
        
        logging.info(f"TrajectoryCalculator initialized: {self.prediction_horizon}s horizon, "
                    f"{self.sampling_rate}Hz sampling, {self.num_waypoints} waypoints, "
                    f"interpolation_mode={self.interpolation_mode}, "
                    f"history: {self.history_horizon}s ({self.num_history_points} points)")
        
        if self.interpolation_mode and self.sampling_rate > self.nuscenes_native_rate:
            logging.info(f"Will interpolate from {self.nuscenes_samples_needed} nuScenes samples "
                        f"to {self.num_waypoints} target waypoints")
            logging.info(f"Will interpolate from {self.nuscenes_history_samples_needed} nuScenes history samples "
                        f"to {self.num_history_points} history points")
    
    def calculate_future_trajectory(self, current_sample_token: str, nusc_reader: NuScenesReader) -> Optional[List[Dict]]:
        """
        计算未来轨迹，支持插值模式，强制输出36个轨迹点
        
        Args:
            current_sample_token: 当前样本令牌
            nusc_reader: nuScenes读取器实例
            
        Returns:
            未来轨迹点列表，每个点包含 {x, y, heading}，始终返回36个点或None
        """
        try:
            if self.interpolation_mode and self.sampling_rate > self.nuscenes_native_rate:
                trajectory = self._calculate_trajectory_with_interpolation(current_sample_token, nusc_reader)
            else:
                trajectory = self._calculate_trajectory_direct(current_sample_token, nusc_reader)
            
            # 验证轨迹长度并进行必要的调整
            if trajectory is not None:
                trajectory = self._ensure_fixed_trajectory_length(trajectory)
                
                # 最终验证
                if len(trajectory) != self.num_waypoints:
                    logging.warning(f"Failed to generate exactly {self.num_waypoints} waypoints for {current_sample_token}")
                    return None
                    
            return trajectory
                
        except Exception as e:
            logging.error(f"Error calculating trajectory for {current_sample_token}: {e}")
            return None
    
    def _calculate_trajectory_direct(self, current_sample_token: str, nusc_reader: NuScenesReader) -> Optional[List[Dict]]:
        """
        直接从nuScenes样本计算轨迹（原始方法）
        """
        # 验证样本链长度
        if not nusc_reader.validate_sample_chain_length(current_sample_token, self.num_waypoints + 1):
            logging.warning(f"Insufficient sample chain length for token {current_sample_token}")
            return None
        
        # 获取当前样本的自车位姿
        current_sample = nusc_reader.get_sample_by_token(current_sample_token)
        current_sample_data = nusc_reader.get_sample_data(current_sample_token, 'CAM_FRONT')
        current_ego_pose = nusc_reader.get_ego_pose(current_sample_data['ego_pose_token'])
        
        # 获取未来样本
        future_samples = nusc_reader.get_next_samples(current_sample['next'], self.num_waypoints)
        
        if len(future_samples) < self.num_waypoints:
            logging.warning(f"Got {len(future_samples)} samples, expected {self.num_waypoints}")
            return None
        
        # 计算未来轨迹点
        trajectory = []
        
        for i, future_sample in enumerate(future_samples):
            # 获取未来样本的自车位姿
            future_sample_data = nusc_reader.get_sample_data(future_sample['token'], 'CAM_FRONT')
            future_ego_pose = nusc_reader.get_ego_pose(future_sample_data['ego_pose_token'])
            
            # 计算相对位置和航向角
            x, y, heading = CoordinateTransformer.transform_global_to_ego(
                future_ego_pose, current_ego_pose
            )
            
            # 归一化航向角
            heading = CoordinateTransformer.normalize_angle(heading)
            
            trajectory.append({
                'x': float(x),
                'y': float(y), 
                'heading': float(heading),
                'timestamp_delta': (i + 1) / self.sampling_rate  # 相对时间戳
            })
        
        return trajectory
    
    def _calculate_trajectory_with_interpolation(self, current_sample_token: str, nusc_reader: NuScenesReader) -> Optional[List[Dict]]:
        """
        从nuScenes 2Hz样本插值到目标频率
        """
        # 验证样本链长度（基于nuScenes原生频率）
        if not nusc_reader.validate_sample_chain_length(current_sample_token, self.nuscenes_samples_needed):
            logging.warning(f"Insufficient sample chain length for token {current_sample_token}")
            return None
        
        # 获取当前样本的自车位姿
        current_sample = nusc_reader.get_sample_by_token(current_sample_token)
        current_sample_data = nusc_reader.get_sample_data(current_sample_token, 'CAM_FRONT')
        current_ego_pose = nusc_reader.get_ego_pose(current_sample_data['ego_pose_token'])
        
        # 获取nuScenes原生频率的未来样本
        future_samples = nusc_reader.get_next_samples(current_sample['next'], self.nuscenes_samples_needed - 1)
        
        if len(future_samples) < self.nuscenes_samples_needed - 1:
            logging.warning(f"Got {len(future_samples)} samples, expected {self.nuscenes_samples_needed - 1}")
            return None
        
        # 计算原生频率的轨迹点
        native_trajectory = []
        
        for i, future_sample in enumerate(future_samples):
            # 获取未来样本的自车位姿
            future_sample_data = nusc_reader.get_sample_data(future_sample['token'], 'CAM_FRONT')
            future_ego_pose = nusc_reader.get_ego_pose(future_sample_data['ego_pose_token'])
            
            # 计算相对位置和航向角
            x, y, heading = CoordinateTransformer.transform_global_to_ego(
                future_ego_pose, current_ego_pose
            )
            
            # 归一化航向角
            heading = CoordinateTransformer.normalize_angle(heading)
            
            native_trajectory.append({
                'x': float(x),
                'y': float(y), 
                'heading': float(heading),
                'timestamp_delta': (i + 1) / self.nuscenes_native_rate  # nuScenes原生时间戳
            })
        
        # 插值到目标频率
        interpolated_trajectory = self.interpolate_trajectory(native_trajectory, self.sampling_rate)
        
        # 确保轨迹长度正确
        if len(interpolated_trajectory) > self.num_waypoints:
            interpolated_trajectory = interpolated_trajectory[:self.num_waypoints]
        elif len(interpolated_trajectory) < self.num_waypoints:
            logging.warning(f"Interpolated trajectory too short: {len(interpolated_trajectory)} < {self.num_waypoints}")
            return None
        
        return interpolated_trajectory
    
    def validate_trajectory_completeness(self, trajectory: List[Dict]) -> bool:
        """
        验证轨迹完整性
        
        Args:
            trajectory: 轨迹点列表
            
        Returns:
            轨迹是否完整
        """
        if not trajectory:
            return False
            
        if len(trajectory) != self.num_waypoints:
            logging.warning(f"Trajectory length {len(trajectory)} != expected {self.num_waypoints}")
            return False
        
        # 检查每个轨迹点的必需字段
        required_fields = ['x', 'y', 'heading']
        for i, waypoint in enumerate(trajectory):
            for field in required_fields:
                if field not in waypoint:
                    logging.error(f"Missing field '{field}' in waypoint {i}")
                    return False
                    
                if not isinstance(waypoint[field], (int, float)):
                    logging.error(f"Invalid type for field '{field}' in waypoint {i}")
                    return False
        
        return True
    
    def normalize_trajectory_angles(self, trajectory: List[Dict]) -> List[Dict]:
        """
        归一化轨迹中的所有航向角
        
        Args:
            trajectory: 输入轨迹
            
        Returns:
            归一化后的轨迹
        """
        normalized_trajectory = []
        
        for waypoint in trajectory:
            normalized_waypoint = waypoint.copy()
            normalized_waypoint['heading'] = CoordinateTransformer.normalize_angle(waypoint['heading'])
            normalized_trajectory.append(normalized_waypoint)
            
        return normalized_trajectory
    
    def calculate_trajectory_metrics(self, trajectory: List[Dict]) -> Dict:
        """
        计算轨迹统计指标
        
        Args:
            trajectory: 轨迹点列表
            
        Returns:
            轨迹统计指标字典
        """
        if not trajectory:
            return {}
        
        # 提取坐标
        x_coords = [wp['x'] for wp in trajectory]
        y_coords = [wp['y'] for wp in trajectory]
        headings = [wp['heading'] for wp in trajectory]
        
        # 计算距离
        distances = []
        for i in range(len(trajectory) - 1):
            dx = x_coords[i+1] - x_coords[i]
            dy = y_coords[i+1] - y_coords[i]
            dist = np.sqrt(dx**2 + dy**2)
            distances.append(dist)
        
        # 计算航向角变化
        heading_changes = []
        for i in range(len(trajectory) - 1):
            dh = CoordinateTransformer.normalize_angle(headings[i+1] - headings[i])
            heading_changes.append(abs(dh))
        
        metrics = {
            'total_distance': sum(distances),
            'avg_step_distance': np.mean(distances) if distances else 0,
            'max_step_distance': max(distances) if distances else 0,
            'total_heading_change': sum(heading_changes),
            'avg_heading_change': np.mean(heading_changes) if heading_changes else 0,
            'max_heading_change': max(heading_changes) if heading_changes else 0,
            'final_position': (x_coords[-1], y_coords[-1]),
            'final_heading': headings[-1]
        }
        
        return metrics
    
    def interpolate_trajectory(self, trajectory: List[Dict], target_frequency: float) -> List[Dict]:
        """
        插值轨迹到目标频率
        
        Args:
            trajectory: 输入轨迹（来自2Hz的nuScenes数据）
            target_frequency: 目标频率（Hz）
            
        Returns:
            插值后的轨迹
        """
        if len(trajectory) == 0:
            return trajectory
            
        # 使用nuScenes原生频率（2Hz）作为源频率
        source_frequency = self.nuscenes_native_rate  # 2Hz
        
        # 计算目标轨迹点数
        # trajectory有6个点（3秒的2Hz数据），需要插值到36个点（3秒的12Hz数据）
        duration = self.prediction_horizon  # 3秒
        target_length = int(duration * target_frequency)  # 3 * 12 = 36
        
        if target_length <= len(trajectory):
            return trajectory[:target_length]
        
        interpolated_trajectory = []
        
        # 创建时间序列
        # 原始轨迹的时间点（2Hz，6个点覆盖3秒）
        source_times = [i / source_frequency for i in range(len(trajectory))]
        
        # 目标轨迹的时间点（12Hz，36个点覆盖3秒）
        target_times = [i / target_frequency for i in range(target_length)]
        
        # 提取坐标用于插值
        x_coords = [wp['x'] for wp in trajectory]
        y_coords = [wp['y'] for wp in trajectory]
        headings = [wp['heading'] for wp in trajectory]
        
        # 对每个目标时间点进行插值
        for i, t in enumerate(target_times):
            if t <= source_times[-1]:
                # 在原始数据范围内，使用线性插值
                # 找到t所在的区间
                for j in range(len(source_times) - 1):
                    if source_times[j] <= t <= source_times[j + 1]:
                        # 计算插值权重
                        if source_times[j + 1] - source_times[j] > 0:
                            alpha = (t - source_times[j]) / (source_times[j + 1] - source_times[j])
                        else:
                            alpha = 0
                        
                        # 线性插值
                        interpolated_wp = {
                            'x': x_coords[j] * (1 - alpha) + x_coords[j + 1] * alpha,
                            'y': y_coords[j] * (1 - alpha) + y_coords[j + 1] * alpha,
                            'heading': CoordinateTransformer.interpolate_angle(
                                headings[j], headings[j + 1], alpha
                            ),
                            'timestamp_delta': t
                        }
                        interpolated_trajectory.append(interpolated_wp)
                        break
                else:
                    # 如果没找到合适的区间（t在第一个点之前），使用第一个点
                    if t < source_times[0]:
                        interpolated_trajectory.append({
                            'x': x_coords[0],
                            'y': y_coords[0],
                            'heading': headings[0],
                            'timestamp_delta': t
                        })
            else:
                # 超出原始数据范围，使用外推或最后一个点
                interpolated_trajectory.append({
                    'x': x_coords[-1],
                    'y': y_coords[-1],
                    'heading': headings[-1],
                    'timestamp_delta': t
                })
        
        return interpolated_trajectory
    
    def filter_trajectory_by_distance(self, trajectory: List[Dict], max_distance: float) -> List[Dict]:
        """
        根据距离阈值过滤轨迹点
        
        Args:
            trajectory: 输入轨迹
            max_distance: 最大允许距离
            
        Returns:
            过滤后的轨迹
        """
        filtered_trajectory = []
        
        for waypoint in trajectory:
            distance = np.sqrt(waypoint['x']**2 + waypoint['y']**2)
            if distance <= max_distance:
                filtered_trajectory.append(waypoint)
            else:
                break  # 一旦超过阈值就停止
                
        return filtered_trajectory
    
    def smooth_trajectory(self, trajectory: List[Dict], window_size: int = 3) -> List[Dict]:
        """
        使用移动平均法平滑轨迹
        
        Args:
            trajectory: 输入轨迹
            window_size: 平滑窗口大小（必须为奇数）
            
        Returns:
            平滑后的轨迹
        """
        if len(trajectory) < window_size or window_size < 3:
            return trajectory
            
        # 确保窗口大小为奇数
        if window_size % 2 == 0:
            window_size += 1
            
        half_window = window_size // 2
        smoothed_trajectory = []
        
        for i in range(len(trajectory)):
            # 计算窗口边界
            start_idx = max(0, i - half_window)
            end_idx = min(len(trajectory), i + half_window + 1)
            
            # 提取窗口内的数据
            window_points = trajectory[start_idx:end_idx]
            
            # 计算平均值
            avg_x = sum(wp['x'] for wp in window_points) / len(window_points)
            avg_y = sum(wp['y'] for wp in window_points) / len(window_points)
            
            # 对角度使用圆形平均
            avg_heading = self._circular_mean([wp['heading'] for wp in window_points])
            
            smoothed_waypoint = {
                'x': avg_x,
                'y': avg_y,
                'heading': avg_heading,
                'timestamp_delta': trajectory[i]['timestamp_delta']  # 保持原始时间戳
            }
            
            smoothed_trajectory.append(smoothed_waypoint)
            
        return smoothed_trajectory
    
    def _circular_mean(self, angles: List[float]) -> float:
        """
        计算角度的圆形平均值
        
        Args:
            angles: 角度列表（弧度）
            
        Returns:
            平均角度（弧度）
        """
        if not angles:
            return 0.0
            
        # 转换为复数
        complex_angles = [np.exp(1j * angle) for angle in angles]
        
        # 计算平均复数
        mean_complex = sum(complex_angles) / len(complex_angles)
        
        # 提取角度
        mean_angle = np.angle(mean_complex)
        
        return CoordinateTransformer.normalize_angle(mean_angle)
    
    def validate_trajectory_smoothness(self, trajectory: List[Dict], max_acceleration: float = 5.0) -> Dict:
        """
        验证轨迹平滑性
        
        Args:
            trajectory: 轨迹点列表
            max_acceleration: 最大允许加速度（m/s²）
            
        Returns:
            验证结果字典
        """
        if len(trajectory) < 3:
            return {'valid': True, 'issues': []}
        
        dt = 1.0 / self.sampling_rate
        issues = []
        
        # 计算速度和加速度
        for i in range(1, len(trajectory) - 1):
            prev_wp = trajectory[i - 1]
            curr_wp = trajectory[i]
            next_wp = trajectory[i + 1]
            
            # 计算速度
            v1_x = (curr_wp['x'] - prev_wp['x']) / dt
            v1_y = (curr_wp['y'] - prev_wp['y']) / dt
            v1_mag = np.sqrt(v1_x**2 + v1_y**2)
            
            v2_x = (next_wp['x'] - curr_wp['x']) / dt
            v2_y = (next_wp['y'] - curr_wp['y']) / dt
            v2_mag = np.sqrt(v2_x**2 + v2_y**2)
            
            # 计算加速度
            acc_x = (v2_x - v1_x) / dt
            acc_y = (v2_y - v1_y) / dt
            acc_mag = np.sqrt(acc_x**2 + acc_y**2)
            
            # 检查加速度是否超过阈值
            if acc_mag > max_acceleration:
                issues.append({
                    'waypoint_index': i,
                    'acceleration': acc_mag,
                    'max_allowed': max_acceleration,
                    'type': 'high_acceleration'
                })
            
            # 检查角速度
            heading_diff = CoordinateTransformer.normalize_angle(
                next_wp['heading'] - curr_wp['heading']
            )
            angular_velocity = abs(heading_diff) / dt
            
            # 检查角速度是否合理（例如 > 2 rad/s）
            if angular_velocity > 2.0:
                issues.append({
                    'waypoint_index': i,
                    'angular_velocity': angular_velocity,
                    'type': 'high_angular_velocity'
                })
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_issues': len(issues)
        }
    
    def calculate_historical_trajectory(self, current_sample_token: str, nusc_reader: NuScenesReader) -> Optional[List[Dict]]:
        """
        计算历史轨迹，用于生成历史车辆状态数据
        
        Args:
            current_sample_token: 当前样本令牌
            nusc_reader: nuScenes读取器实例
            
        Returns:
            历史轨迹点列表，每个点包含 {x, y, heading, timestamp_delta}，如果无法生成完整轨迹则返回None
        """
        try:
            # 获取历史样本
            previous_samples = nusc_reader.get_previous_samples(current_sample_token, self.nuscenes_history_samples_needed)
            
            if len(previous_samples) < 2:  # 至少需要2个历史点来计算轨迹
                logging.warning(f"Insufficient historical samples for token {current_sample_token}")
                return []
            
            # 获取当前样本的自车位姿作为参考
            current_sample = nusc_reader.get_sample_by_token(current_sample_token)
            current_sample_data = nusc_reader.get_sample_data(current_sample_token, 'CAM_FRONT')
            current_ego_pose = nusc_reader.get_ego_pose(current_sample_data['ego_pose_token'])
            
            # 计算历史轨迹点（相对于当前位置）
            historical_trajectory = []
            
            for i, prev_sample in enumerate(previous_samples):
                # 获取历史样本的自车位姿
                prev_sample_data = nusc_reader.get_sample_data(prev_sample['token'], 'CAM_FRONT')
                prev_ego_pose = nusc_reader.get_ego_pose(prev_sample_data['ego_pose_token'])
                
                # 计算相对位置和航向角
                x, y, heading = CoordinateTransformer.transform_global_to_ego(
                    prev_ego_pose, current_ego_pose
                )
                
                # 归一化航向角
                heading = CoordinateTransformer.normalize_angle(heading)
                
                # 计算时间差（负值表示过去）
                time_delta = (prev_sample['timestamp'] - current_sample['timestamp']) / 1e6
                
                historical_trajectory.append({
                    'x': float(x),
                    'y': float(y), 
                    'heading': float(heading),
                    'timestamp_delta': float(time_delta)
                })
            
            # 如果启用插值模式，插值到目标频率
            if self.interpolation_mode and self.sampling_rate > self.nuscenes_native_rate:
                interpolated_trajectory = self._interpolate_historical_trajectory(historical_trajectory)
                return interpolated_trajectory
            else:
                return historical_trajectory
                
        except Exception as e:
            logging.error(f"Error calculating historical trajectory for {current_sample_token}: {e}")
            return None
    
    def _interpolate_historical_trajectory(self, historical_trajectory: List[Dict]) -> List[Dict]:
        """
        插值历史轨迹到目标频率
        
        Args:
            historical_trajectory: 原始历史轨迹（2Hz）
            
        Returns:
            插值后的历史轨迹（12Hz）
        """
        if len(historical_trajectory) < 2:
            return historical_trajectory
        
        # 创建时间序列
        original_times = [wp['timestamp_delta'] for wp in historical_trajectory]
        
        # 创建12Hz插值时间点（从-1.0s到0s，间隔0.083s）
        target_times = [-self.history_horizon + i / self.sampling_rate 
                       for i in range(self.num_history_points)]
        
        # 插值位置坐标
        x_coords = [wp['x'] for wp in historical_trajectory]
        y_coords = [wp['y'] for wp in historical_trajectory]
        headings = [wp['heading'] for wp in historical_trajectory]
        
        x_interp = np.interp(target_times, original_times, x_coords)
        y_interp = np.interp(target_times, original_times, y_coords)
        
        # 航向角使用圆形插值
        heading_interp = self._interpolate_angles_circular(target_times, original_times, headings)
        
        # 构建插值后的轨迹
        interpolated_trajectory = []
        for i, t in enumerate(target_times):
            interpolated_trajectory.append({
                'x': float(x_interp[i]),
                'y': float(y_interp[i]),
                'heading': float(heading_interp[i]),
                'timestamp_delta': float(t)
            })
        
        return interpolated_trajectory
    
    def _interpolate_angles_circular(self, target_times: List[float], 
                                   original_times: List[float], 
                                   angles: List[float]) -> List[float]:
        """
        对角度进行圆形插值，处理-π/π边界
        
        Args:
            target_times: 目标时间点
            original_times: 原始时间点
            angles: 原始角度值
            
        Returns:
            插值后的角度值
        """
        # 展开角度以避免跳跃
        unwrapped_angles = np.unwrap(angles)
        
        # 线性插值
        interpolated_unwrapped = np.interp(target_times, original_times, unwrapped_angles)
        
        # 重新包装到[-π, π]范围
        return [CoordinateTransformer.normalize_angle(angle) for angle in interpolated_unwrapped]
    
    def calculate_historical_vehicle_states(self, current_sample_token: str, 
                                          nusc_reader: NuScenesReader) -> List[Dict[str, float]]:
        """
        计算历史车辆状态序列（用于prompt生成）
        
        Args:
            current_sample_token: 当前样本令牌
            nusc_reader: nuScenes读取器实例
            
        Returns:
            历史车辆状态列表，每个状态包含位置、速度、加速度、转向角等
        """
        historical_states = []
        
        try:
            # 获取历史轨迹
            historical_trajectory = self.calculate_historical_trajectory(current_sample_token, nusc_reader)
            
            if not historical_trajectory or len(historical_trajectory) < 2:
                return []
            
            # 获取当前样本信息
            current_sample = nusc_reader.get_sample_by_token(current_sample_token)
            scene_token = current_sample['scene_token']
            
            # 如果可用，获取CAN总线数据
            for i, hist_point in enumerate(historical_trajectory):
                # 计算历史时间戳
                hist_timestamp = current_sample['timestamp'] + int(hist_point['timestamp_delta'] * 1e6)
                
                # 初始化车辆状态
                vehicle_state = {
                    'x': hist_point['x'],
                    'y': hist_point['y'],
                    'heading': hist_point['heading'],
                    'timestamp_delta': hist_point['timestamp_delta'],
                    'velocity_x': 0.0,
                    'velocity_y': 0.0,
                    'acceleration_x': 0.0,
                    'acceleration_y': 0.0,
                    'steering_angle': 0.0,
                    'speed': 0.0
                }
                
                # 尝试从CAN总线数据获取详细车辆状态
                if nusc_reader.enable_can_bus:
                    can_data = nusc_reader.get_can_bus_data(scene_token, hist_timestamp)
                    can_vehicle_state = nusc_reader.extract_vehicle_state_from_can(can_data, hist_timestamp)
                    vehicle_state.update(can_vehicle_state)
                else:
                    # 从轨迹数据估算速度和加速度
                    if i > 0:
                        prev_point = historical_trajectory[i-1]
                        dt = hist_point['timestamp_delta'] - prev_point['timestamp_delta']
                        if dt > 0:
                            dx = hist_point['x'] - prev_point['x']
                            dy = hist_point['y'] - prev_point['y']
                            vehicle_state['velocity_x'] = dx / dt
                            vehicle_state['velocity_y'] = dy / dt
                            vehicle_state['speed'] = np.sqrt(dx**2 + dy**2) / dt
                            
                            # 计算加速度（如果有足够的点）
                            if i > 1:
                                prev_prev_point = historical_trajectory[i-2]
                                prev_dt = prev_point['timestamp_delta'] - prev_prev_point['timestamp_delta']
                                if prev_dt > 0:
                                    prev_dx = prev_point['x'] - prev_prev_point['x']
                                    prev_dy = prev_point['y'] - prev_prev_point['y']
                                    prev_vel_x = prev_dx / prev_dt
                                    prev_vel_y = prev_dy / prev_dt
                                    
                                    vehicle_state['acceleration_x'] = (vehicle_state['velocity_x'] - prev_vel_x) / dt
                                    vehicle_state['acceleration_y'] = (vehicle_state['velocity_y'] - prev_vel_y) / dt
                
                historical_states.append(vehicle_state)
            
            return historical_states
            
        except Exception as e:
            logging.error(f"Error calculating historical vehicle states: {e}")
            return []
    
    def _ensure_fixed_trajectory_length(self, trajectory: List[Dict]) -> List[Dict]:
        """
        确保轨迹具有固定长度（36个点）
        
        Args:
            trajectory: 输入轨迹
            
        Returns:
            调整后的固定长度轨迹
        """
        current_length = len(trajectory)
        target_length = self.num_waypoints
        
        if current_length == target_length:
            # 长度已正确
            return trajectory
        elif current_length > target_length:
            # 截断到目标长度
            logging.debug(f"Truncating trajectory from {current_length} to {target_length} points")
            return trajectory[:target_length]
        else:
            # 长度不足，需要扩展
            logging.debug(f"Extending trajectory from {current_length} to {target_length} points")
            
            if current_length == 0:
                # 特殊情况：空轨迹，生成静止轨迹
                return self._generate_stationary_trajectory()
            
            # 使用外推法扩展轨迹
            extended_trajectory = trajectory.copy()
            
            # 计算最后两个点的增量，用于外推
            if current_length >= 2:
                last_point = trajectory[-1]
                second_last_point = trajectory[-2]
                
                dx = last_point['x'] - second_last_point['x']
                dy = last_point['y'] - second_last_point['y']
                dh = CoordinateTransformer.normalize_angle(last_point['heading'] - second_last_point['heading'])
                
                # 基于采样率计算时间步长
                dt = 1.0 / self.sampling_rate
                
            else:
                # 只有一个点，假设静止
                dx, dy, dh = 0.0, 0.0, 0.0
                dt = 1.0 / self.sampling_rate
            
            # 扩展轨迹
            for i in range(current_length, target_length):
                last_point = extended_trajectory[-1]
                
                new_point = {
                    'x': last_point['x'] + dx,
                    'y': last_point['y'] + dy,
                    'heading': CoordinateTransformer.normalize_angle(last_point['heading'] + dh),
                    'timestamp_delta': i * dt
                }
                
                extended_trajectory.append(new_point)
            
            return extended_trajectory
    
    def _generate_stationary_trajectory(self) -> List[Dict]:
        """
        生成静止轨迹（用于特殊情况）
        
        Returns:
            36个静止点的轨迹
        """
        stationary_trajectory = []
        dt = 1.0 / self.sampling_rate
        
        for i in range(self.num_waypoints):
            point = {
                'x': 0.0,
                'y': 0.0,
                'heading': 0.0,
                'timestamp_delta': i * dt
            }
            stationary_trajectory.append(point)
        
        return stationary_trajectory