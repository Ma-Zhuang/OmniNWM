#!/usr/bin/env python3
"""
多GPU并行生成深度和语义图
完全独立的脚本，避免循环导入问题
"""

import os
import sys
import argparse
import logging
import pickle
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
import multiprocessing as mp
from tqdm import tqdm
import yaml

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

def setup_logging(gpu_id: int = None):
    """设置日志"""
    gpu_suffix = f"_gpu{gpu_id}" if gpu_id is not None else ""
    log_file = f"multi_gpu_generation{gpu_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger(f"GPU_{gpu_id}" if gpu_id is not None else "MAIN")
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器
    logger.handlers = []
    
    # 文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式化器
    formatter = logging.Formatter(
        f'%(asctime)s - GPU{gpu_id if gpu_id is not None else "MAIN"} - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def process_batch_on_gpu(args):
    """在指定GPU上处理一批图像"""
    gpu_id, image_batch, config_path, output_dir = args
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 设置日志
    logger = setup_logging(gpu_id)
    logger.info(f"GPU {gpu_id} 开始处理 {len(image_batch)} 张图像")
    
    # 导入必要的模块
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 读取配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化处理器
    from src.processors.depth_processor import DepthProcessor
    from src.processors.semantic_processor import SemanticProcessor
    
    depth_config = config.get('depth', {})
    depth_processor = DepthProcessor(
        model_name=depth_config.get('model_name', 'DPT'),
        model_path=depth_config.get('model_path'),
        device=device,
        use_local_weights=True if depth_config.get('model_path') else False
    )
    logger.info(f"GPU {gpu_id}: 深度处理器初始化成功")
    
    semantic_config = config.get('semantic', {})
    semantic_processor = SemanticProcessor(
        model_name=semantic_config.get('model_name', 'SegFormer'),
        model_path=semantic_config.get('model_path'),
        device=device,
        use_local_weights=True if semantic_config.get('model_path') else False
    )
    logger.info(f"GPU {gpu_id}: 语义处理器初始化成功")
    
    # 设置输出目录
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path(config['dataset']['output_directory'])
    
    depth_dir = output_path / 'depth'
    semantic_dir = output_path / 'semantic'
    depth_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    
    dataroot = Path(config['dataset']['nuscenes_dataroot'])
    
    success_count = 0
    error_count = 0
    
    # 处理每张图像
    for image_path, image_id in tqdm(image_batch, desc=f"GPU {gpu_id}", position=gpu_id):
        try:
            # 构建输出路径
            rel_path = Path(image_path).relative_to(dataroot)
            depth_path = depth_dir / rel_path.parent / f"{rel_path.stem}_depth.png"
            semantic_path = semantic_dir / rel_path.parent / f"{rel_path.stem}_semantic.png"
            
            # 创建输出目录
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            semantic_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 跳过已存在的文件
            if depth_path.exists() and semantic_path.exists():
                success_count += 1
                continue
            
            # 读取图像
            image = Image.open(image_path).convert('RGB')
            
            # 生成深度图
            if not depth_path.exists():
                depth_map = depth_processor.process(image)
                # 检查是否返回了有效的深度图
                if depth_map is None or np.all(depth_map == 0):
                    logger.error(f"GPU {gpu_id}: 深度图生成失败（返回空值）: {image_path}")
                    error_count += 1
                    continue
                # 保存为16位PNG（毫米单位）
                depth_mm = np.clip(depth_map * 1000, 0, 65535).astype(np.uint16)
                Image.fromarray(depth_mm).save(depth_path)
            
            # 生成语义图
            if not semantic_path.exists():
                semantic_map = semantic_processor.process(image)
                # 检查是否返回了有效的语义图
                if semantic_map is None or np.all(semantic_map == 0):
                    logger.error(f"GPU {gpu_id}: 语义图生成失败（返回空值）: {image_path}")
                    error_count += 1
                    continue
                # 保存为8位索引图
                Image.fromarray(semantic_map.astype(np.uint8)).save(semantic_path)
            
            success_count += 1
            
        except Exception as e:
            logger.error(f"GPU {gpu_id} 处理失败 {image_path}: {e}")
            error_count += 1
        
        # 定期清理GPU缓存
        if (success_count + error_count) % 100 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    logger.info(f"GPU {gpu_id} 完成: 成功 {success_count}, 失败 {error_count}")
    return success_count, error_count

def collect_all_images(config_path: str, max_samples: int = None) -> List[Tuple[str, str]]:
    """收集所有需要处理的图像"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    from nuscenes.nuscenes import NuScenes
    
    print("加载nuScenes数据集...")
    nusc = NuScenes(
        version=config['dataset']['nuscenes_version'],
        dataroot=config['dataset']['nuscenes_dataroot'],
        verbose=False
    )
    
    # 加载已处理的图像
    output_dir = Path(config['dataset']['output_directory'])
    progress_file = output_dir / 'generation_progress.pkl'
    processed_images = set()
    
    if progress_file.exists():
        with open(progress_file, 'rb') as f:
            processed_images = pickle.load(f)
        print(f"已加载进度: {len(processed_images)} 张图像已处理")
    
    image_info = []
    sample_count = 0
    camera_channels = config['cameras']['channels']
    dataroot = Path(config['dataset']['nuscenes_dataroot'])
    
    for scene in tqdm(nusc.scene, desc="收集图像路径"):
        sample_token = scene['first_sample_token']
        
        while sample_token:
            sample = nusc.get('sample', sample_token)
            
            for camera in camera_channels:
                if camera in sample['data']:
                    sample_data = nusc.get('sample_data', sample['data'][camera])
                    image_path = str(dataroot / sample_data['filename'])
                    
                    # 跳过已处理的图像
                    if image_path not in processed_images:
                        image_info.append((image_path, f"{sample_token}_{camera}"))
            
            sample_token = sample['next']
            sample_count += 1
            
            if max_samples and sample_count >= max_samples:
                break
        
        if max_samples and sample_count >= max_samples:
            break
    
    print(f"找到 {len(image_info)} 张待处理图像")
    return image_info

def main():
    parser = argparse.ArgumentParser(description='多GPU并行生成深度和语义图')
    parser.add_argument('--config', type=str, 
                       default='configs/production_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--num-gpus', type=int, default=4,
                       help='使用的GPU数量')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大处理样本数')
    
    args = parser.parse_args()
    
    # 设置multiprocessing使用spawn方式
    mp.set_start_method('spawn', force=True)
    
    # 收集所有图像
    all_images = collect_all_images(args.config, args.max_samples)
    
    if not all_images:
        print("✅ 所有图像已处理完成！")
        return
    
    # 检查可用GPU数量
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available_gpus)
    print(f"\n使用 {num_gpus} 个GPU (可用: {available_gpus})")
    
    # 将图像分配到各个GPU
    images_per_gpu = len(all_images) // num_gpus
    gpu_tasks = []
    
    for i in range(num_gpus):
        start_idx = i * images_per_gpu
        if i == num_gpus - 1:
            batch = all_images[start_idx:]
        else:
            batch = all_images[start_idx:start_idx + images_per_gpu]
        
        gpu_tasks.append((i, batch, args.config, args.output_dir))
    
    print(f"总共 {len(all_images)} 张图像，每个GPU处理约 {images_per_gpu} 张")
    print("="*70)
    print("🚀 开始多GPU并行处理")
    print("="*70)
    
    # 使用multiprocessing.Pool并行处理
    with mp.Pool(processes=num_gpus) as pool:
        results = pool.map(process_batch_on_gpu, gpu_tasks)
    
    # 统计结果
    total_success = sum(r[0] for r in results)
    total_error = sum(r[1] for r in results)
    
    print("\n" + "="*70)
    print("📊 多GPU处理完成统计:")
    print("="*70)
    
    for i, (success, error) in enumerate(results):
        print(f"GPU {i}: 成功 {success}, 失败 {error}")
    
    print(f"\n总计: 成功 {total_success}, 失败 {total_error}")
    
    if total_success > 0:
        print("\n✅ 深度和语义图生成完成！")
        print("\n下一步: 运行ShareGPT数据集生成")
        print("python scripts/build_sharegpt_dataset.py --config configs/sharegpt_dataset_config.yaml")

if __name__ == "__main__":
    main()