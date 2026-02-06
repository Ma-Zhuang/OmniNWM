#!/usr/bin/env python3
"""
多GPU并行预计算视觉特征
使用accelerate实现分布式特征提取，8卡并行可以提速8倍
"""
import os
import json
import torch
import autoroot
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import argparse
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CLIPProcessor, CLIPVisionModel,
    SiglipProcessor, SiglipVisionModel,
    SegformerImageProcessor, SegformerModel
)
from ssr.utils.misc import colorize_depth, load_jsonl, str_datetime


class FeatureExtractionDataset(Dataset):
    """用于特征提取的数据集"""
    def __init__(self, data, data_dir):
        self.data = data
        self.data_dir = data_dir
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'index': idx,
            'item': item,
            'image_path': os.path.join(self.data_dir, item['image_path']),
            'depth_path': os.path.join(self.data_dir, item['depth_path']),
            'semantic_path': os.path.join(self.data_dir, item.get('semantic_path', ''))
        }


def parse_args():
    parser = argparse.ArgumentParser(description="多GPU并行预计算视觉特征")
    parser.add_argument("--data_dir", type=str, default="/code/VLA/datasets/SSR-CoT",
                       help="原始数据集路径")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="预计算特征保存路径")
    parser.add_argument("--clip_path", type=str, default="/code/VLA/models/clip-vit-large-patch14-336")
    parser.add_argument("--siglip_path", type=str, default="/code/VLA/models/siglip-so400m-patch14-384")
    parser.add_argument("--segformer_path", type=str, default="/code/VLA/models/segmentation_models/segformer-b5-finetuned-ade-640-640")
    parser.add_argument("--batch_size", type=int, default=16, help="每个GPU的批处理大小")
    parser.add_argument("--use_semantic", action="store_true", help="是否提取语义特征")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载进程数")
    return parser.parse_args()


def extract_features(batch, models, args, device):
    """提取一个批次的特征"""
    clip_processor, clip_model = models['clip']
    siglip_processor, siglip_model = models['siglip']
    segformer_processor, segformer_model = models['segformer']
    
    batch_results = []
    
    with torch.no_grad():
        for i in range(len(batch['index'])):
            try:
                # 加载图像
                image_path = batch['image_path'][i]
                depth_path = batch['depth_path'][i]
                
                if not os.path.exists(image_path) or not os.path.exists(depth_path):
                    continue
                
                raw_image = np.array(Image.open(image_path).convert("RGB"))
                raw_depth = colorize_depth(depth_path)
                
                # CLIP特征
                clip_inputs = clip_processor(images=raw_image, return_tensors="pt").to(device)
                image_embeds = clip_model(**clip_inputs).last_hidden_state.squeeze(0).cpu()
                
                # SigLIP特征
                siglip_inputs = siglip_processor(images=raw_depth, return_tensors="pt").to(device)
                depth_embeds = siglip_model(**siglip_inputs).last_hidden_state.squeeze(0).cpu()
                
                # SegFormer特征
                semantic_embeds = None
                if args.use_semantic and batch['semantic_path'][i]:
                    semantic_path = batch['semantic_path'][i]
                    if os.path.exists(semantic_path):
                        raw_semantic = np.array(Image.open(semantic_path).convert("RGB"))
                        segformer_inputs = segformer_processor(images=raw_semantic, return_tensors="pt").to(device)
                        semantic_outputs = segformer_model(**segformer_inputs)
                        semantic_features = semantic_outputs.last_hidden_state
                        B, C, H, W = semantic_features.shape
                        semantic_embeds = semantic_features.squeeze(0).permute(1, 2, 0).reshape(H * W, C).cpu()
                
                # 准备结果
                result = {
                    'index': batch['index'][i].item(),
                    'features': {
                        'image_embeds': image_embeds,
                        'depth_embeds': depth_embeds
                    },
                    'metadata': batch['item'][i]
                }
                
                if semantic_embeds is not None:
                    result['features']['semantic_embeds'] = semantic_embeds
                
                batch_results.append(result)
                
            except Exception as e:
                print(f"Error processing sample {batch['index'][i]}: {e}")
                continue
    
    return batch_results


def main():
    args = parse_args()
    
    # 初始化accelerator
    accelerator = Accelerator()
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"{str_datetime()} Initializing multi-GPU feature extraction")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Process index: {accelerator.process_index}")
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "precomputed_features")
    
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # 等待主进程创建目录
    accelerator.wait_for_everyone()
    
    # 加载模型
    if accelerator.is_main_process:
        print(f"{str_datetime()} Loading models on all GPUs...")
    
    # CLIP
    clip_processor = CLIPProcessor.from_pretrained(args.clip_path)
    clip_model = CLIPVisionModel.from_pretrained(args.clip_path).to(device).eval()
    
    # SigLIP
    siglip_processor = SiglipProcessor.from_pretrained(args.siglip_path)
    siglip_model = SiglipVisionModel.from_pretrained(args.siglip_path).to(device).eval()
    
    # SegFormer
    segformer_processor, segformer_model = None, None
    if args.use_semantic:
        segformer_processor = SegformerImageProcessor.from_pretrained(args.segformer_path)
        segformer_model = SegformerModel.from_pretrained(args.segformer_path).to(device).eval()
    
    models = {
        'clip': (clip_processor, clip_model),
        'siglip': (siglip_processor, siglip_model),
        'segformer': (segformer_processor, segformer_model)
    }
    
    # 读取数据集
    jsonl_path = os.path.join(args.data_dir, "ssr-cot.jsonl")
    if accelerator.is_main_process:
        print(f"{str_datetime()} Loading dataset from {jsonl_path}")
    
    data = load_jsonl(jsonl_path)
    
    # 创建数据集和数据加载器
    dataset = FeatureExtractionDataset(data, args.data_dir)
    
    # 创建DataLoader - 注意不要shuffle，保证顺序
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: {
            'index': torch.tensor([item['index'] for item in x]),
            'item': [item['item'] for item in x],
            'image_path': [item['image_path'] for item in x],
            'depth_path': [item['depth_path'] for item in x],
            'semantic_path': [item['semantic_path'] for item in x]
        }
    )
    
    # 使用accelerate准备dataloader
    dataloader = accelerator.prepare(dataloader)
    
    if accelerator.is_main_process:
        print(f"{str_datetime()} Starting feature extraction on {accelerator.num_processes} GPUs")
        print(f"Total samples: {len(dataset)}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batches: {len(dataloader)}")
    
    # 提取特征并实时保存
    saved_count = 0
    for batch in tqdm(dataloader, desc=f"GPU {accelerator.process_index}", disable=not accelerator.is_local_main_process):
        batch_results = extract_features(batch, models, args, device)
        
        # 实时保存每个批次的结果
        for result in batch_results:
            idx = result['index']
            
            # 保存tensor
            output_path = output_dir / f"sample_{idx:06d}.pt"
            torch.save(result['features'], output_path)
            
            # 保存元数据
            metadata_path = output_dir / f"sample_{idx:06d}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(result['metadata'], f, ensure_ascii=False, indent=2)
            
            saved_count += 1
        
        # 每100批打印一次进度（只在主进程）
        if accelerator.is_local_main_process and saved_count % 100 == 0:
            print(f"GPU {accelerator.process_index}: Saved {saved_count} features")
    
    # 等待所有进程完成
    accelerator.wait_for_everyone()
    
    # 主进程保存配置
    if accelerator.is_main_process:
        # 收集所有特征文件统计
        feature_files = list(output_dir.glob("sample_*.pt"))
        print(f"{str_datetime()} Total feature files created: {len(feature_files)}")
        
        # 获取特征维度
        if feature_files:
            sample_features = torch.load(feature_files[0])
            feature_dims = {
                'image_embeds': list(sample_features['image_embeds'].shape),
                'depth_embeds': list(sample_features['depth_embeds'].shape),
                'semantic_embeds': list(sample_features['semantic_embeds'].shape) if 'semantic_embeds' in sample_features else None
            }
        else:
            feature_dims = None
        
        # 保存配置
        config = {
            'total_samples': len(data),
            'use_semantic': args.use_semantic,
            'clip_path': args.clip_path,
            'siglip_path': args.siglip_path,
            'segformer_path': args.segformer_path if args.use_semantic else None,
            'data_dir': args.data_dir,
            'feature_dims': feature_dims,
            'num_gpus_used': accelerator.num_processes
        }
        
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"{str_datetime()} Config saved to {config_path}")
        print(f"{str_datetime()} Feature extraction completed!")
        print(f"Used {accelerator.num_processes} GPUs")


if __name__ == "__main__":
    main()