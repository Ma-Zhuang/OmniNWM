#!/usr/bin/env python3
"""
ShareGPT数据处理管道

专门用于生成ShareGPT格式的多模态轨迹预测数据集的简化管道。
该脚本整合了所有新功能：
- CAN总线数据读取
- 历史轨迹计算
- 对话式prompt生成
- ShareGPT格式输出
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from src.core.nuscenes_reader import NuScenesReader
from src.core.trajectory_calculator import TrajectoryCalculator
from src.processors.multimodal_processor import MultiModalProcessor
from src.generators.conversation_prompt_generator import ConversationPromptGenerator
from src.generators.sharegpt_formatter import ShareGPTFormatter


class ShareGPTDataPipeline:
    """ShareGPT数据处理管道"""
    
    def __init__(self, config_path: str):
        """
        初始化ShareGPT数据管道
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._initialize_components()
        
        self.stats = {
            'total_samples': 0,
            'successful_samples': 0,
            'failed_samples': 0,
            'errors': []
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Failed to load config from {config_path}: {e}")
            raise
    
    def _setup_logging(self):
        """设置日志"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.get('logging', {}).get('conversation_log_file', 'sharegpt_pipeline.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("ShareGPT数据管道初始化")
    
    def _initialize_components(self):
        """初始化所有组件"""
        try:
            # 初始化nuScenes读取器（启用CAN总线）
            dataset_config = self.config['dataset']
            can_bus_enabled = dataset_config.get('enable_can_bus', True)
            
            self.nusc_reader = NuScenesReader(
                dataroot=dataset_config['nuscenes_dataroot'],
                version=dataset_config['nuscenes_version'],
                enable_can_bus=can_bus_enabled
            )
            
            # 初始化轨迹计算器（支持历史数据）
            trajectory_config = self.config['trajectory']
            self.trajectory_calculator = TrajectoryCalculator(
                prediction_horizon=trajectory_config['prediction_horizon'],
                sampling_rate=trajectory_config['sampling_rate'],
                interpolation_mode=trajectory_config['interpolation_mode'],
                history_horizon=trajectory_config.get('history_horizon', 1.0)
            )
            
            # 初始化多模态处理器
            # MultiModalProcessor期望接收一个字典配置
            self.multimodal_processor = MultiModalProcessor(self.config)
            
            # 初始化对话提示生成器
            self.prompt_generator = ConversationPromptGenerator()
            
            # 初始化ShareGPT格式化器
            conversation_config = self.config.get('conversation', {})
            include_metadata = conversation_config.get('include_debug_metadata', False)
            self.sharegpt_formatter = ShareGPTFormatter(include_metadata=include_metadata)
            
            self.logger.info("所有组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    def process_sample(self, sample_token: str, scene_token: str) -> Optional[Dict[str, Any]]:
        """
        处理单个样本，生成ShareGPT格式数据
        
        Args:
            sample_token: 样本令牌
            scene_token: 场景令牌
            
        Returns:
            ShareGPT格式数据或None（如果失败）
        """
        try:
            self.logger.debug(f"处理样本: {sample_token}")
            
            # 1. 计算未来轨迹
            future_trajectory = self.trajectory_calculator.calculate_future_trajectory(
                sample_token, self.nusc_reader
            )
            
            if not future_trajectory:
                self.logger.warning(f"无法计算未来轨迹: {sample_token}")
                return None
            
            # 2. 计算历史车辆状态
            historical_states = self.trajectory_calculator.calculate_historical_vehicle_states(
                sample_token, self.nusc_reader
            )
            
            # 3. 获取多模态数据路径
            camera_paths = self._get_camera_paths(sample_token)
            if not camera_paths:
                self.logger.warning(f"无法获取摄像头数据: {sample_token}")
                return None
            
            # 4. 生成对话式prompt
            conversation_config = self.config.get('conversation', {})
            template_type = conversation_config.get('template_type', 'multimodal_trajectory')
            
            user_prompt, assistant_response = self.prompt_generator.generate_conversation_prompt(
                template_name=template_type,
                historical_states=historical_states,
                future_trajectory=future_trajectory
            )
            
            # 5. 准备图像路径（按ShareGPT格式要求）
            images, depth_maps, semantic_maps = self._prepare_image_paths(camera_paths)
            
            # 检查是否所有模态都存在
            if images is None or depth_maps is None or semantic_maps is None:
                self.logger.error(f"缺少必需的深度图或语义图，跳过样本: {sample_token}")
                return None
            
            # 6. 生成ShareGPT格式数据
            sample_id = f"{scene_token[:8]}_{sample_token[:8]}"
            
            sharegpt_data = self.sharegpt_formatter.create_sharegpt_sample(
                sample_id=sample_id,
                scene_token=scene_token,
                sample_token=sample_token,
                images=images,
                depth_maps=depth_maps,
                semantic_maps=semantic_maps,
                user_prompt=user_prompt,
                assistant_response=assistant_response,
                metadata={
                    'trajectory_points': len(future_trajectory),
                    'historical_points': len(historical_states),
                    'can_bus_enabled': self.nusc_reader.enable_can_bus
                }
            )
            
            return sharegpt_data
            
        except Exception as e:
            self.logger.error(f"处理样本失败 {sample_token}: {e}")
            return None
    
    def _get_camera_paths(self, sample_token: str) -> Optional[Dict[str, str]]:
        """获取摄像头图像路径"""
        try:
            camera_paths = {}
            camera_channels = self.config['cameras']['channels']
            
            for camera in camera_channels:
                sample_data = self.nusc_reader.get_sample_data(sample_token, camera)
                image_path = os.path.join(self.nusc_reader.dataroot, sample_data['filename'])
                camera_paths[camera] = image_path
            
            return camera_paths
            
        except Exception as e:
            self.logger.error(f"获取摄像头路径失败: {e}")
            return None
    
    def _prepare_image_paths(self, camera_paths: Dict[str, str]) -> tuple:
        """准备按顺序排列的图像路径"""
        camera_order = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        images = []
        depth_maps = []
        semantic_maps = []
        
        output_dir = Path(self.config['dataset']['output_directory'])
        dataroot = Path(self.config['dataset']['nuscenes_dataroot'])
        
        for camera in camera_order:
            if camera in camera_paths:
                base_path = camera_paths[camera]
                
                # RGB图像路径
                images.append(base_path)
                
                # 构建深度和语义图路径（根据实际保存位置）
                rel_path = Path(base_path).relative_to(dataroot)
                
                # 深度图路径 - 在 output_directory/depth/ 下
                depth_path = output_dir / 'depth' / rel_path.parent / f"{rel_path.stem}_depth.png"
                if not depth_path.exists():
                    # 如果深度图不存在，必须失败，不能使用占位符
                    self.logger.error(f"深度图不存在: {depth_path}")
                    return None, None, None  # 返回None表示失败
                depth_maps.append(str(depth_path))
                
                # 语义图路径 - 在 output_directory/semantic/ 下
                semantic_path = output_dir / 'semantic' / rel_path.parent / f"{rel_path.stem}_semantic.png"
                if not semantic_path.exists():
                    # 如果语义图不存在，必须失败，不能使用占位符
                    self.logger.error(f"语义图不存在: {semantic_path}")
                    return None, None, None  # 返回None表示失败
                semantic_maps.append(str(semantic_path))
        
        return images, depth_maps, semantic_maps
    
    def run_pipeline(self, max_samples: Optional[int] = None, 
                    start_scene_idx: int = 0) -> Dict[str, Any]:
        """
        运行完整的数据处理管道
        
        Args:
            max_samples: 最大处理样本数（None表示处理所有）
            start_scene_idx: 开始场景索引
            
        Returns:
            处理统计信息
        """
        self.logger.info("开始ShareGPT数据管道处理")
        
        # 获取所有场景
        scenes = self.nusc_reader.get_scenes()[start_scene_idx:]
        
        output_dir = self.config['dataset']['output_directory']
        sharegpt_dir = os.path.join(output_dir, 'sharegpt_format')
        os.makedirs(sharegpt_dir, exist_ok=True)
        
        samples_processed = 0
        
        try:
            for scene_idx, scene in enumerate(scenes):
                self.logger.info(f"处理场景 {scene_idx + start_scene_idx + 1}/{len(scenes)}: {scene['name']}")
                
                # 获取场景中的所有样本
                samples = self.nusc_reader.get_samples_from_scene(scene['token'])
                
                for sample in samples:
                    if max_samples and samples_processed >= max_samples:
                        break
                    
                    # 处理样本
                    sharegpt_data = self.process_sample(sample['token'], scene['token'])
                    
                    if sharegpt_data:
                        # 保存ShareGPT格式文件
                        sample_id = sharegpt_data['id']
                        output_path = os.path.join(sharegpt_dir, f"conversation_{sample_id}.json")
                        
                        success = self.sharegpt_formatter.save_sharegpt_file(
                            sharegpt_data, output_path
                        )
                        
                        if success:
                            self.stats['successful_samples'] += 1
                        else:
                            self.stats['failed_samples'] += 1
                    else:
                        self.stats['failed_samples'] += 1
                    
                    samples_processed += 1
                    self.stats['total_samples'] += 1
                    
                    if samples_processed % 100 == 0:
                        self.logger.info(f"已处理 {samples_processed} 个样本")
                
                if max_samples and samples_processed >= max_samples:
                    break
        
        except KeyboardInterrupt:
            self.logger.info("用户中断处理")
        except Exception as e:
            self.logger.error(f"管道处理失败: {e}")
            self.stats['errors'].append(str(e))
        
        # 生成数据集清单
        try:
            self.sharegpt_formatter.create_dataset_manifest(sharegpt_dir)
        except Exception as e:
            self.logger.warning(f"生成数据集清单失败: {e}")
        
        # 输出统计信息
        self._print_statistics()
        
        return self.stats
    
    def _print_statistics(self):
        """打印处理统计信息"""
        self.logger.info("="*50)
        self.logger.info("ShareGPT数据管道处理完成")
        self.logger.info(f"总样本数: {self.stats['total_samples']}")
        self.logger.info(f"成功样本数: {self.stats['successful_samples']}")
        self.logger.info(f"失败样本数: {self.stats['failed_samples']}")
        if self.stats['total_samples'] > 0:
            success_rate = self.stats['successful_samples'] / self.stats['total_samples'] * 100
            self.logger.info(f"成功率: {success_rate:.2f}%")
        self.logger.info("="*50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ShareGPT格式数据集生成器")
    parser.add_argument('--config', '-c', required=True, help='配置文件路径')
    parser.add_argument('--max-samples', '-n', type=int, help='最大处理样本数')
    parser.add_argument('--start-scene', '-s', type=int, default=0, help='开始场景索引')
    
    args = parser.parse_args()
    
    try:
        # 创建并运行管道
        pipeline = ShareGPTDataPipeline(args.config)
        stats = pipeline.run_pipeline(
            max_samples=args.max_samples,
            start_scene_idx=args.start_scene
        )
        
        print("\\n处理完成！")
        print(f"成功生成 {stats['successful_samples']} 个ShareGPT格式样本")
        
    except Exception as e:
        print(f"管道运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()