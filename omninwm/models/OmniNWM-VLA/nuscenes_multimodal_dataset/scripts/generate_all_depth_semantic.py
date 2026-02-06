#!/usr/bin/env python3
"""
批量生成所有深度和语义图
优化版本，支持断点续传和多GPU并行处理
"""

import os
import sys
import json
import argparse
import logging
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

class DepthSemanticGenerator:
    """深度和语义图批量生成器"""
    
    def __init__(self, config_path: str, output_dir: str = None, gpu_id: int = None):
        """初始化生成器
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
            gpu_id: 指定使用的GPU ID（None表示自动选择）
        """
        self.config_path = config_path
        self.gpu_id = gpu_id
        self.setup_logging()
        self.load_config()
        self.setup_output_dir(output_dir)
        self.setup_processors()
        self.setup_progress_tracking()
        
    def setup_logging(self):
        """设置日志"""
        gpu_suffix = f"_gpu{self.gpu_id}" if self.gpu_id is not None else ""
        log_file = f"depth_semantic_generation{gpu_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format=f'%(asctime)s - {f"GPU{self.gpu_id} - " if self.gpu_id is not None else ""}%(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self):
        """加载配置文件"""
        import yaml
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataroot = Path(self.config['dataset']['nuscenes_dataroot'])
        self.camera_channels = self.config['cameras']['channels']
        
    def setup_output_dir(self, output_dir: str = None):
        """设置输出目录"""
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(self.config['dataset']['output_directory'])
        
        self.depth_dir = self.output_dir / 'depth'
        self.semantic_dir = self.output_dir / 'semantic'
        
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"输出目录: {self.output_dir}")
        
    def setup_processors(self):
        """初始化处理器"""
        # 如果指定了GPU ID，设置CUDA设备
        if self.gpu_id is not None:
            torch.cuda.set_device(self.gpu_id)
            device = f'cuda:{self.gpu_id}'
            self.logger.info(f"使用GPU {self.gpu_id}")
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 初始化深度处理器
        from src.processors.depth_processor import DepthProcessor
        depth_config = self.config.get('depth', {})
        self.depth_processor = DepthProcessor(
            model_name=depth_config.get('model_name', 'ZoeDepth'),
            model_path=depth_config.get('model_path'),
            device=device,
            use_local_weights=True if depth_config.get('model_path') else False
        )
        self.logger.info("深度处理器初始化成功")
        
        # 初始化语义处理器
        from src.processors.semantic_processor import SemanticProcessor
        semantic_config = self.config.get('semantic', {})
        self.semantic_processor = SemanticProcessor(
            model_name=semantic_config.get('model_name', 'SegFormer'),
            model_path=semantic_config.get('model_path'),
            device=device,
            use_local_weights=True if semantic_config.get('model_path') else False
        )
        self.logger.info("语义处理器初始化成功")
        
    def setup_progress_tracking(self):
        """设置进度跟踪"""
        self.progress_file = self.output_dir / 'generation_progress.pkl'
        self.processed_images = set()
        
        # 加载已处理的图像列表
        if self.progress_file.exists():
            with open(self.progress_file, 'rb') as f:
                self.processed_images = pickle.load(f)
            self.logger.info(f"已加载进度: {len(self.processed_images)} 张图像已处理")
    
    def save_progress(self):
        """保存进度"""
        with open(self.progress_file, 'wb') as f:
            pickle.dump(self.processed_images, f)
    
    def collect_image_paths(self, max_samples: int = None) -> List[Tuple[str, str]]:
        """收集所有需要处理的图像路径"""
        from nuscenes.nuscenes import NuScenes
        
        # 加载nuScenes
        self.logger.info("加载nuScenes数据集...")
        nusc = NuScenes(
            version=self.config['dataset']['nuscenes_version'],
            dataroot=str(self.dataroot),
            verbose=False
        )
        
        image_info = []  # [(image_path, sample_token), ...]
        sample_count = 0
        
        for scene in tqdm(nusc.scene, desc="收集图像路径"):
            sample_token = scene['first_sample_token']
            
            while sample_token:
                sample = nusc.get('sample', sample_token)
                
                # 获取所有摄像头的图像
                for camera in self.camera_channels:
                    if camera in sample['data']:
                        sample_data = nusc.get('sample_data', sample['data'][camera])
                        image_path = str(self.dataroot / sample_data['filename'])
                        
                        # 检查是否已处理
                        if image_path not in self.processed_images:
                            image_info.append((image_path, f"{sample_token}_{camera}"))
                
                sample_token = sample['next']
                sample_count += 1
                
                if max_samples and sample_count >= max_samples:
                    break
            
            if max_samples and sample_count >= max_samples:
                break
        
        self.logger.info(f"找到 {len(image_info)} 张待处理图像")
        return image_info
    
    def process_image(self, image_path: str, image_id: str) -> Tuple[bool, str]:
        """处理单张图像"""
        try:
            # 构建输出路径
            rel_path = Path(image_path).relative_to(self.dataroot)
            depth_path = self.depth_dir / rel_path.parent / f"{rel_path.stem}_depth.png"
            semantic_path = self.semantic_dir / rel_path.parent / f"{rel_path.stem}_semantic.png"
            
            # 创建输出目录
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            semantic_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 读取图像
            image = Image.open(image_path).convert('RGB')
            
            # 生成深度图
            if not depth_path.exists():
                depth_map = self.depth_processor.process(image)  # 返回单位：米
                # 保存为16位PNG（统一使用毫米单位）
                depth_mm = np.clip(depth_map * 1000, 0, 65535).astype(np.uint16)
                Image.fromarray(depth_mm).save(depth_path)
            
            # 生成语义图
            if not semantic_path.exists():
                semantic_map = self.semantic_processor.process(image)
                # 保存为8位索引图
                Image.fromarray(semantic_map.astype(np.uint8)).save(semantic_path)
            
            return True, "成功"
            
        except Exception as e:
            return False, str(e)
    
    def run(self, max_samples: int = None, batch_size: int = 10):
        """运行批量生成"""
        print("\n" + "="*70)
        print("🚀 开始批量生成深度和语义图")
        print("="*70)
        
        # 收集图像路径
        image_info = self.collect_image_paths(max_samples)
        
        if not image_info:
            print("✅ 所有图像已处理完成！")
            return
        
        # 处理图像
        success_count = 0
        error_count = 0
        
        with tqdm(total=len(image_info), desc="处理进度") as pbar:
            for i, (image_path, image_id) in enumerate(image_info):
                # 处理图像
                success, message = self.process_image(image_path, image_id)
                
                if success:
                    success_count += 1
                    self.processed_images.add(image_path)
                else:
                    error_count += 1
                    self.logger.error(f"处理失败 {image_path}: {message}")
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': error_count
                })
                
                # 定期保存进度和清理缓存
                if (i + 1) % batch_size == 0:
                    self.save_progress()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # 最终保存进度
        self.save_progress()
        
        # 打印统计
        print("\n" + "="*70)
        print("📊 生成统计:")
        print("="*70)
        print(f"✅ 成功处理: {success_count} 张")
        print(f"❌ 处理失败: {error_count} 张")
        print(f"📁 深度图目录: {self.depth_dir}")
        print(f"📁 语义图目录: {self.semantic_dir}")
        
        # 验证输出
        depth_files = list(self.depth_dir.rglob('*.png'))
        semantic_files = list(self.semantic_dir.rglob('*.png'))
        print(f"\n生成的文件:")
        print(f"  深度图: {len(depth_files)} 个")
        print(f"  语义图: {len(semantic_files)} 个")
        
        if success_count > 0:
            print("\n✅ 深度和语义图生成完成！")
            print("\n下一步: 运行ShareGPT数据集生成")
            print("python scripts/build_sharegpt_dataset.py --config configs/sharegpt_dataset_config.yaml")

def process_images_on_gpu(args):
    """在指定GPU上处理一批图像
    使用单个参数以兼容multiprocessing.Pool
    """
    gpu_id, image_batch, config_path, output_dir = args
    
    # 设置当前进程使用的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 在子进程中导入必要的库
    import torch
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    # 创建生成器实例
    from scripts.generate_all_depth_semantic import DepthSemanticGenerator
    generator = DepthSemanticGenerator(
        config_path=config_path,
        output_dir=output_dir,
        gpu_id=None  # 使用环境变量设置的GPU
    )
    
    success_count = 0
    error_count = 0
    
    # 处理每张图像
    for image_path, image_id in tqdm(image_batch, desc=f"GPU {gpu_id}"):
        success, message = generator.process_image(image_path, image_id)
        
        if success:
            success_count += 1
            generator.processed_images.add(image_path)
        else:
            error_count += 1
            generator.logger.error(f"处理失败 {image_path}: {message}")
        
        # 定期保存进度和清理缓存
        if (success_count + error_count) % 10 == 0:
            generator.save_progress()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 最终保存进度
    generator.save_progress()
    generator.logger.info(f"GPU {gpu_id} 完成: 成功 {success_count}, 失败 {error_count}")
    
    return success_count, error_count

def run_multi_gpu(config_path: str, output_dir: str = None, 
                  max_samples: int = None, num_gpus: int = 4):
    """多GPU并行运行"""
    import yaml
    
    # 设置multiprocessing使用spawn方式以支持CUDA
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # 已经设置过了
    
    # 读取配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建一个临时生成器来收集图像路径
    temp_generator = DepthSemanticGenerator(config_path, output_dir)
    all_images = temp_generator.collect_image_paths(max_samples)
    
    if not all_images:
        print("✅ 所有图像已处理完成！")
        return
    
    # 检查可用GPU数量
    available_gpus = torch.cuda.device_count()
    num_gpus = min(num_gpus, available_gpus)
    print(f"\n使用 {num_gpus} 个GPU (可用: {available_gpus})")
    
    # 将图像分配到各个GPU
    images_per_gpu = len(all_images) // num_gpus
    gpu_tasks = []
    
    for i in range(num_gpus):
        start_idx = i * images_per_gpu
        if i == num_gpus - 1:
            # 最后一个GPU处理剩余的所有图像
            batch = all_images[start_idx:]
        else:
            batch = all_images[start_idx:start_idx + images_per_gpu]
        
        gpu_tasks.append((i, batch, config_path, output_dir))
    
    print(f"总共 {len(all_images)} 张图像，每个GPU处理约 {images_per_gpu} 张")
    print("="*70)
    print("🚀 开始多GPU并行处理")
    print("="*70)
    
    # 使用multiprocessing.Pool
    with mp.Pool(processes=num_gpus) as pool:
        results = pool.map(process_images_on_gpu, gpu_tasks)
    
    # 统计总结果
    total_success = sum(r[0] for r in results)
    total_error = sum(r[1] for r in results)
    
    for i, (success, error) in enumerate(results):
        print(f"GPU {i} 完成: 成功 {success}, 失败 {error}")
    
    # 打印最终统计
    print("\n" + "="*70)
    print("📊 多GPU处理完成统计:")
    print("="*70)
    print(f"✅ 成功处理: {total_success} 张")
    print(f"❌ 处理失败: {total_error} 张")
    
    if total_success > 0:
        print("\n✅ 深度和语义图生成完成！")
        print("\n下一步: 运行ShareGPT数据集生成")
        print("python scripts/build_sharegpt_dataset.py --config configs/sharegpt_dataset_config.yaml")

def main():
    parser = argparse.ArgumentParser(description='批量生成nuScenes深度和语义图')
    parser.add_argument('--config', type=str, 
                       default='configs/sharegpt_dataset_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录（默认使用配置文件中的路径）')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大处理样本数（用于测试）')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='批处理大小（多少张图后保存进度）')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='启用多GPU并行处理')
    parser.add_argument('--num-gpus', type=int, default=4,
                       help='使用的GPU数量（默认4个）')
    
    args = parser.parse_args()
    
    if args.multi_gpu:
        # 多GPU模式
        run_multi_gpu(
            config_path=args.config,
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            num_gpus=args.num_gpus
        )
    else:
        # 单GPU模式
        generator = DepthSemanticGenerator(
            config_path=args.config,
            output_dir=args.output_dir
        )
        
        generator.run(
            max_samples=args.max_samples,
            batch_size=args.batch_size
        )

if __name__ == "__main__":
    main()