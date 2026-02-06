"""
对话式提示生成器

专门用于生成符合您理想格式的对话式提示：
- 支持多模态感知（RGB、深度、语义分割）
- 集成历史车辆状态数据
- 生成12Hz轨迹预测对话
- 符合自动驾驶场景的专业描述
"""

import os
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationTemplate:
    """对话模板"""
    name: str
    user_template: str
    assistant_template: str
    variables: List[str]
    description: str
    category: str


class ConversationPromptGenerator:
    """对话式提示生成器"""
    
    def __init__(self):
        self.templates = {}
        self.load_conversation_templates()
        
        # 摄像头描述映射
        self.camera_descriptions = {
            'CAM_FRONT': 'front view',
            'CAM_FRONT_LEFT': 'front-left view', 
            'CAM_FRONT_RIGHT': 'front-right view',
            'CAM_BACK': 'rear view',
            'CAM_BACK_LEFT': 'rear-left view',
            'CAM_BACK_RIGHT': 'rear-right view'
        }
        
        # 摄像头顺序（与理想格式一致）
        self.camera_order = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
    
    def load_conversation_templates(self):
        """加载对话模板"""
        
        # 基础多模态轨迹预测模板
        self.templates['multimodal_trajectory'] = ConversationTemplate(
            name="multimodal_trajectory",
            user_template=(
                "You are an autonomous driving agent. You have access to multi-modal sensory data "
                "from a vehicle's 6-camera system providing 360° coverage: {camera_views}. "
                "Additionally, you have corresponding depth maps and semantic segmentation maps for each camera view. "
                "Your task is to predict the vehicle's detailed future trajectory over the next 3 seconds "
                "at 12Hz sampling rate (36 waypoints total).\n\n"
                "Provided is the ego vehicle status recorded over the last 1.0 seconds "
                "(at 0.083-second intervals, 12Hz). This includes the x, y coordinates and heading angle "
                "of the ego vehicle in the ego-coordinate system. Positive x means forward direction, "
                "positive y means leftward direction, and heading angle is in radians. "
                "The data is presented in the format [x, y, heading]:\n\n"
                "{historical_states}\n\n"
                "Analyze the multi-modal sensor data (RGB images, depth maps, and semantic segmentation) "
                "from all 6 camera views to understand the surrounding environment, road structure, "
                "traffic conditions, and potential obstacles. Based on this comprehensive analysis and "
                "the vehicle's current motion pattern, predict the future trajectory."
            ),
            assistant_template=(
                "<PLANNING>Predicted future trajectory for the next 3 seconds "
                "(36 waypoints sampled at 12Hz, 0.083-second intervals), including position and "
                "orientation in ego-vehicle coordinate system. Positive x means forward direction, "
                "positive y means leftward direction, heading angle in radians. "
                "The output is formatted as [x, y, heading]: \n{trajectory_points}</PLANNING>"
            ),
            variables=['camera_views', 'historical_states', 'trajectory_points'],
            description="Multi-modal autonomous driving trajectory prediction with historical context",
            category="trajectory_conversation"
        )
        
        # 详细分析版本
        self.templates['detailed_analysis'] = ConversationTemplate(
            name="detailed_analysis",
            user_template=(
                "You are an expert autonomous vehicle path planner with access to comprehensive "
                "multi-modal sensory information. You can see the driving environment through a "
                "6-camera system: {camera_views}, providing complete 360° coverage around the vehicle. "
                "For each camera view, you also have depth perception data and semantic understanding "
                "of the scene elements.\n\n"
                "Historical vehicle dynamics over the past 1.0 seconds (12Hz sampling):\n"
                "{historical_states}\n\n"
                "Your mission: Analyze the complete sensory input to understand the current driving "
                "scenario, including road geometry, traffic participants, potential hazards, and "
                "environmental conditions. Then, predict a safe and efficient trajectory for the "
                "next 3 seconds, outputted as 36 precise waypoints at 12Hz frequency. Each waypoint "
                "should contain ego-centric coordinates [x, y, heading] where x is forward, y is leftward, "
                "and heading is in radians."
            ),
            assistant_template=(
                "<PLANNING>Based on comprehensive multi-modal analysis:\n\n"
                "Scene Assessment: {scene_analysis}\n\n"
                "Trajectory Plan: {trajectory_points}\n\n"
                "The trajectory represents 36 waypoints over 3 seconds (12Hz sampling) in ego-vehicle "
                "coordinates. Format: [x_forward, y_leftward, heading_radians]</PLANNING>"
            ),
            variables=['camera_views', 'historical_states', 'scene_analysis', 'trajectory_points'],
            description="Detailed multi-modal analysis with scene understanding",
            category="detailed_conversation"
        )
        
        # 安全导向版本
        self.templates['safety_focused'] = ConversationTemplate(
            name="safety_focused",
            user_template=(
                "You are a safety-critical autonomous driving system. Your primary responsibility "
                "is to ensure safe navigation while maintaining efficient progress. You have access "
                "to rich sensory data: {camera_views} providing 360° visual awareness, supplemented "
                "by depth estimation and semantic scene understanding.\n\n"
                "Current vehicle state history (1.0s, 12Hz sampling):\n"
                "{historical_states}\n\n"
                "Analyze all available multi-modal data to:\n"
                "1. Identify potential safety hazards and traffic participants\n"
                "2. Understand road structure and driving constraints\n"
                "3. Plan a collision-free trajectory respecting traffic rules\n"
                "4. Output 36 trajectory waypoints for the next 3 seconds (12Hz)\n\n"
                "Prioritize safety over speed. Each waypoint format: [x, y, heading] in ego coordinates."
            ),
            assistant_template=(
                "<PLANNING>Safety-prioritized trajectory planning:\n\n"
                "Risk Assessment: {risk_analysis}\n"
                "Planned Trajectory: {trajectory_points}\n\n"
                "36 waypoints over 3 seconds (0.083s intervals) in ego-vehicle coordinate system. "
                "Format: [x_forward, y_left, heading_radians]</PLANNING>"
            ),
            variables=['camera_views', 'historical_states', 'risk_analysis', 'trajectory_points'],
            description="Safety-focused trajectory planning with risk assessment",
            category="safety_conversation"
        )
    
    def generate_conversation_prompt(self, 
                                   template_name: str = "multimodal_trajectory",
                                   historical_states: List[Dict[str, float]] = None,
                                   future_trajectory: List[Dict[str, float]] = None,
                                   scene_context: Dict[str, Any] = None,
                                   **kwargs) -> Tuple[str, str]:
        """
        生成对话式提示（用户消息和助手响应）
        
        Args:
            template_name: 模板名称
            historical_states: 历史车辆状态列表
            future_trajectory: 未来轨迹（用于生成助手响应）
            scene_context: 场景上下文信息
            **kwargs: 其他模板变量
            
        Returns:
            (user_prompt, assistant_response) 元组
        """
        if template_name not in self.templates:
            template_name = "multimodal_trajectory"
        
        template = self.templates[template_name]
        
        try:
            # 准备模板变量
            template_vars = self._prepare_template_variables(
                historical_states, future_trajectory, scene_context, **kwargs
            )
            
            # 生成用户提示
            user_prompt = template.user_template.format(**template_vars)
            
            # 生成助手响应
            assistant_response = template.assistant_template.format(**template_vars)
            
            return user_prompt, assistant_response
            
        except KeyError as e:
            logger.warning(f"模板变量缺失 {e}，使用基础模板")
            return self.generate_conversation_prompt("multimodal_trajectory", 
                                                   historical_states, future_trajectory, 
                                                   scene_context, **kwargs)
        except Exception as e:
            logger.error(f"生成对话提示失败: {e}")
            raise
    
    def _prepare_template_variables(self, 
                                  historical_states: List[Dict[str, float]] = None,
                                  future_trajectory: List[Dict[str, float]] = None,
                                  scene_context: Dict[str, Any] = None,
                                  **kwargs) -> Dict[str, str]:
        """准备模板变量"""
        template_vars = {}
        
        # 生成摄像头视角描述
        template_vars['camera_views'] = self._format_camera_views()
        
        # 格式化历史状态
        if historical_states:
            template_vars['historical_states'] = self._format_historical_states(historical_states)
        else:
            template_vars['historical_states'] = "No historical data available"
        
        # 格式化轨迹点
        if future_trajectory:
            template_vars['trajectory_points'] = self._format_trajectory_points(future_trajectory)
        else:
            template_vars['trajectory_points'] = "No trajectory predicted"
        
        # 场景分析（如果需要）
        if scene_context:
            template_vars['scene_analysis'] = self._generate_scene_analysis(scene_context)
            template_vars['risk_analysis'] = self._generate_risk_analysis(scene_context)
        else:
            template_vars['scene_analysis'] = "Scene analysis pending"
            template_vars['risk_analysis'] = "Risk assessment pending"
        
        # 添加其他传入的变量
        template_vars.update(kwargs)
        
        return template_vars
    
    def _format_camera_views(self) -> str:
        """格式化摄像头视角描述"""
        view_descriptions = []
        for camera in self.camera_order:
            if camera in self.camera_descriptions:
                view_descriptions.append(f"{self.camera_descriptions[camera]} <image>")
        
        return ", ".join(view_descriptions)
    
    def _format_historical_states(self, historical_states: List[Dict[str, float]]) -> str:
        """格式化历史车辆状态"""
        if not historical_states:
            return "No historical data available"
        
        formatted_lines = []
        
        for state in historical_states:
            # 格式化时间戳
            time_delta = state.get('timestamp_delta', 0.0)
            if abs(time_delta) < 0.001:  # 当前时刻
                time_str = "(t-0.0s)"
            else:
                time_str = f"(t{time_delta:.3f}s)"
            
            # 格式化位置和角度 [x, y, heading]
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
    
    def _format_trajectory_points(self, trajectory: List[Dict[str, float]]) -> str:
        """格式化轨迹点"""
        if not trajectory:
            return "No trajectory available"
        
        trajectory_points = []
        for point in trajectory:
            x = point.get('x', 0.0)
            y = point.get('y', 0.0)
            heading = point.get('heading', 0.0)
            trajectory_points.append(f"[{x:.2f}, {y:.2f}, {heading:.3f}]")
        
        return ", ".join(trajectory_points)
    
    def _generate_scene_analysis(self, scene_context: Dict[str, Any]) -> str:
        """生成场景分析文本"""
        analysis_parts = []
        
        # 道路类型
        road_type = scene_context.get('road_type', 'urban road')
        analysis_parts.append(f"Road type: {road_type}")
        
        # 天气条件
        weather = scene_context.get('weather', 'clear')
        analysis_parts.append(f"Weather: {weather}")
        
        # 时间
        time_of_day = scene_context.get('time_of_day', 'daytime')
        analysis_parts.append(f"Time: {time_of_day}")
        
        # 交通状况
        traffic_density = scene_context.get('traffic_density', 'moderate')
        analysis_parts.append(f"Traffic: {traffic_density}")
        
        return ". ".join(analysis_parts)
    
    def _generate_risk_analysis(self, scene_context: Dict[str, Any]) -> str:
        """生成风险分析文本"""
        risks = []
        
        # 基于场景上下文生成风险评估
        traffic_density = scene_context.get('traffic_density', 'moderate')
        if traffic_density == 'heavy':
            risks.append("High traffic density requires careful following distance")
        
        weather = scene_context.get('weather', 'clear')
        if weather in ['rain', 'fog']:
            risks.append("Reduced visibility conditions detected")
        
        road_type = scene_context.get('road_type', 'urban')
        if road_type == 'highway':
            risks.append("High-speed environment requires extended planning horizon")
        elif road_type == 'intersection':
            risks.append("Intersection crossing requires multi-directional awareness")
        
        if not risks:
            risks.append("Normal driving conditions, standard precautions apply")
        
        return ". ".join(risks)
    
    def get_all_templates(self) -> Dict[str, ConversationTemplate]:
        """获取所有对话模板"""
        return self.templates.copy()
    
    def add_custom_template(self, template: ConversationTemplate):
        """添加自定义模板"""
        self.templates[template.name] = template
        logger.info(f"已添加自定义模板: {template.name}")
    
    def generate_varied_prompt(self, base_template: str = "multimodal_trajectory", 
                             add_variations: bool = True, **kwargs) -> Tuple[str, str]:
        """
        生成带变化的提示（保持多样性）
        
        Args:
            base_template: 基础模板名称
            add_variations: 是否添加变化
            **kwargs: 模板参数
            
        Returns:
            (user_prompt, assistant_response) 元组
        """
        # 如果启用变化，随机选择模板
        if add_variations:
            available_templates = list(self.templates.keys())
            if len(available_templates) > 1:
                # 有一定概率使用不同的模板
                if random.random() < 0.3:  # 30%概率使用其他模板
                    alternative_templates = [t for t in available_templates if t != base_template]
                    if alternative_templates:
                        base_template = random.choice(alternative_templates)
        
        return self.generate_conversation_prompt(base_template, **kwargs)


# 便利函数
def create_conversation_generator() -> ConversationPromptGenerator:
    """创建对话式提示生成器"""
    return ConversationPromptGenerator()


def generate_multimodal_conversation(historical_states: List[Dict[str, float]] = None,
                                   future_trajectory: List[Dict[str, float]] = None,
                                   template_name: str = "multimodal_trajectory") -> Tuple[str, str]:
    """
    生成多模态对话的便利函数
    
    Args:
        historical_states: 历史车辆状态
        future_trajectory: 未来轨迹
        template_name: 模板名称
        
    Returns:
        (user_prompt, assistant_response) 元组
    """
    generator = create_conversation_generator()
    return generator.generate_conversation_prompt(
        template_name, historical_states, future_trajectory
    )