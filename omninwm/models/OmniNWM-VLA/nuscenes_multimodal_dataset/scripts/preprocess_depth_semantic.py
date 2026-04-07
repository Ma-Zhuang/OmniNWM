#!/usr/bin/env python3
"""
预处理脚本：生成深度图和语义分割图
在生成ShareGPT数据集之前运行此脚本
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_gpu():
    """检查GPU状态"""
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return True
    else:
        print("⚠️  GPU不可用，将使用CPU（速度较慢）")
        return False

def initialize_processors(config_path):
    """初始化深度和语义处理器"""
    
    # 读取配置
    with open(config_path, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    
    # 初始化深度处理器
    from src.processors.depth_processor import DepthProcessor
    depth_config = config.get('depth', {})
    depth_processor = DepthProcessor(
        model_name=depth_config.get('model_name', 'ZoeDepth'),
        model_path=depth_config.get('model_path'),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_local_weights=True if depth_config.get('model_path') else False
    )
    print("✅ 深度处理器初始化成功")
    
    # 初始化语义分割处理器
    from src.processors.semantic_processor import SemanticProcessor
    semantic_config = config.get('semantic', {})
    semantic_processor = SemanticProcessor(
        model_name=semantic_config.get('model_name', 'SegFormer'),
        model_path=semantic_config.get('model_path'),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        use_local_weights=True if semantic_config.get('model_path') else False
    )
    print("✅ 语义分割处理器初始化成功")
    
    return depth_processor, semantic_processor, config

def process_single_image(image_path, depth_processor, semantic_processor, output_dir, dataroot):
    """处理单张图像"""
    
    # 创建输出目录
    depth_dir = output_dir / 'depth'
    semantic_dir = output_dir / 'semantic'
    depth_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建输出路径（使用配置中的dataroot而不是硬编码路径）
    rel_path = Path(image_path).relative_to(dataroot)
    depth_path = depth_dir / rel_path.parent / f"{rel_path.stem}_depth.png"
    semantic_path = semantic_dir / rel_path.parent / f"{rel_path.stem}_semantic.png"
    
    # 检查是否已处理
    if depth_path.exists() and semantic_path.exists():
        return True, "已存在"
    
    try:
        # 读取图像
        image = Image.open(image_path).convert('RGB')
        
        # 生成深度图
        if not depth_path.exists():
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            depth_map = depth_processor.process(image)  # 返回单位：米
            # 保存深度图（16位PNG，统一使用毫米单位）
            depth_mm = (depth_map * 1000).astype(np.uint16)
            Image.fromarray(depth_mm).save(depth_path)
        
        # 生成语义分割图
        if not semantic_path.exists():
            semantic_path.parent.mkdir(parents=True, exist_ok=True)
            semantic_map = semantic_processor.process(image)
            # 保存语义图（索引图）
            Image.fromarray(semantic_map.astype(np.uint8)).save(semantic_path)
        
        return True, "成功"
        
    except Exception as e:
        return False, str(e)

def process_nuscenes_dataset(config_path, max_samples=None, batch_size=1):
    """处理整个nuScenes数据集"""
    
    logger = setup_logging()
    
    print("="*70)
    print("🚀 nuScenes深度和语义图预处理")
    print("="*70)
    
    # 检查GPU
    check_gpu()
    
    # 初始化处理器
    print("\n初始化模型...")
    depth_processor, semantic_processor, config = initialize_processors(config_path)
    
    # 设置路径
    dataroot = Path(config['dataset']['nuscenes_dataroot'])
    output_dir = Path(config['dataset']['output_directory'])
    
    # 加载nuScenes
    print("\n加载nuScenes数据集...")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(
        version=config['dataset']['nuscenes_version'],
        dataroot=str(dataroot),
        verbose=False
    )
    print(f"✅ 加载 {len(nusc.scene)} 个场景")
    
    # 收集所有需要处理的图像
    print("\n收集图像路径...")
    camera_channels = config['cameras']['channels']
    image_paths = []
    
    for scene in tqdm(nusc.scene, desc="扫描场景"):
        # 获取场景中的所有样本
        sample_token = scene['first_sample_token']
        while sample_token:
            sample = nusc.get('sample', sample_token)
            
            # 获取所有摄像头的图像路径
            for camera in camera_channels:
                if camera in sample['data']:
                    sample_data = nusc.get('sample_data', sample['data'][camera])
                    image_path = dataroot / sample_data['filename']
                    if image_path.exists():
                        image_paths.append(image_path)
            
            sample_token = sample['next']
            
            # 限制样本数（用于测试）
            if max_samples and len(image_paths) >= max_samples * len(camera_channels):
                break
        
        if max_samples and len(image_paths) >= max_samples * len(camera_channels):
            break
    
    print(f"找到 {len(image_paths)} 张图像需要处理")
    
    # 处理图像
    print("\n开始处理图像...")
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 使用批处理提高效率
    with tqdm(total=len(image_paths), desc="处理进度") as pbar:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            for image_path in batch_paths:
                success, message = process_single_image(
                    image_path, 
                    depth_processor, 
                    semantic_processor,
                    output_dir,
                    dataroot  # 传递dataroot参数
                )
                
                if success:
                    if message == "已存在":
                        skip_count += 1
                    else:
                        success_count += 1
                else:
                    error_count += 1
                    logger.error(f"处理失败 {image_path}: {message}")
                
                pbar.update(1)
                
                # 定期清理GPU缓存
                if torch.cuda.is_available() and (i % 100 == 0):
                    torch.cuda.empty_cache()
    
    # 打印统计
    print("\n" + "="*70)
    print("📊 处理统计:")
    print("="*70)
    print(f"✅ 成功处理: {success_count} 张")
    print(f"⏭️  跳过（已存在）: {skip_count} 张")
    print(f"❌ 处理失败: {error_count} 张")
    print(f"📁 输出目录: {output_dir}")
    
    # 验证输出
    depth_files = list((output_dir / 'depth').rglob('*.png'))
    semantic_files = list((output_dir / 'semantic').rglob('*.png'))
    print(f"\n生成的文件:")
    print(f"  深度图: {len(depth_files)} 个")
    print(f"  语义图: {len(semantic_files)} 个")
    
    print("\n✅ 预处理完成！现在可以运行:")
    print("python scripts/build_sharegpt_dataset.py --config configs/sharegpt_dataset_config.yaml")

def main():
    parser = argparse.ArgumentParser(description='预处理nuScenes数据集的深度和语义图')
    parser.add_argument('--config', type=str, 
                       default='configs/sharegpt_dataset_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大处理样本数（用于测试）')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='批处理大小')
    
    args = parser.parse_args()
    
    process_nuscenes_dataset(
        config_path=args.config,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()