#!/usr/bin/env python3
"""
预计算所有视觉特征并保存到磁盘，避免训练时实时计算
这样可以将训练速度提升100倍以上
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
from transformers import (
    CLIPProcessor, CLIPVisionModel,
    SiglipProcessor, SiglipVisionModel,
    SegformerImageProcessor, SegformerModel
)
from ssr.utils.misc import colorize_depth, load_jsonl

def parse_args():
    parser = argparse.ArgumentParser(description="预计算视觉特征")
    parser.add_argument("--data_dir", type=str, default="/code/VLA/datasets/SSR-CoT",
                       help="原始数据集路径（包含图像文件和ssr-cot.jsonl）")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="预计算特征保存路径（默认为data_dir/precomputed_features）")
    parser.add_argument("--clip_path", type=str, default="/code/VLA/models/clip-vit-large-patch14-336")
    parser.add_argument("--siglip_path", type=str, default="/code/VLA/models/siglip-so400m-patch14-384")
    parser.add_argument("--segformer_path", type=str, default="/code/VLA/models/segmentation_models/segformer-b5-finetuned-ade-640-640")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    parser.add_argument("--use_semantic", action="store_true", help="是否提取语义特征")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start_idx", type=int, default=0, help="开始处理的样本索引（用于断点续传）")
    parser.add_argument("--end_idx", type=int, default=None, help="结束处理的样本索引")
    return parser.parse_args()

def load_models(args):
    """加载所有视觉编码器模型"""
    print("Loading CLIP model...")
    clip_processor = CLIPProcessor.from_pretrained(args.clip_path)
    clip_model = CLIPVisionModel.from_pretrained(args.clip_path).to(args.device).eval()
    
    print("Loading SigLIP model...")
    siglip_processor = SiglipProcessor.from_pretrained(args.siglip_path)
    siglip_model = SiglipVisionModel.from_pretrained(args.siglip_path).to(args.device).eval()
    
    segformer_processor = None
    segformer_model = None
    if args.use_semantic:
        print("Loading SegFormer model...")
        segformer_processor = SegformerImageProcessor.from_pretrained(args.segformer_path)
        segformer_model = SegformerModel.from_pretrained(args.segformer_path).to(args.device).eval()
    
    return {
        'clip': (clip_processor, clip_model),
        'siglip': (siglip_processor, siglip_model),
        'segformer': (segformer_processor, segformer_model) if args.use_semantic else (None, None)
    }

def extract_features_batch(batch_data, models, args):
    """批量提取视觉特征"""
    clip_processor, clip_model = models['clip']
    siglip_processor, siglip_model = models['siglip']
    segformer_processor, segformer_model = models['segformer']
    
    batch_features = []
    
    with torch.no_grad():
        for item in batch_data:
            features = {}
            
            try:
                # 加载图像
                image_path = os.path.join(args.data_dir, item['image_path'])
                depth_path = os.path.join(args.data_dir, item['depth_path'])
                
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue
                if not os.path.exists(depth_path):
                    print(f"Warning: Depth not found: {depth_path}")
                    continue
                
                raw_image = np.array(Image.open(image_path).convert("RGB"))
                raw_depth = colorize_depth(depth_path)
                
                # CLIP特征 (RGB)
                clip_inputs = clip_processor(images=raw_image, return_tensors="pt").to(args.device)
                image_embeds = clip_model(**clip_inputs).last_hidden_state.squeeze(0).cpu()
                features['image_embeds'] = image_embeds
                
                # SigLIP特征 (深度)
                siglip_inputs = siglip_processor(images=raw_depth, return_tensors="pt").to(args.device)
                depth_embeds = siglip_model(**siglip_inputs).last_hidden_state.squeeze(0).cpu()
                features['depth_embeds'] = depth_embeds
                
                # SegFormer特征 (语义)
                if args.use_semantic and 'semantic_path' in item:
                    semantic_path = os.path.join(args.data_dir, item['semantic_path'])
                    if os.path.exists(semantic_path):
                        raw_semantic = np.array(Image.open(semantic_path).convert("RGB"))
                        segformer_inputs = segformer_processor(images=raw_semantic, return_tensors="pt").to(args.device)
                        semantic_outputs = segformer_model(**segformer_inputs)
                        # [1, C, H, W] -> [H*W, C]
                        semantic_features = semantic_outputs.last_hidden_state
                        B, C, H, W = semantic_features.shape
                        semantic_embeds = semantic_features.squeeze(0).permute(1, 2, 0).reshape(H * W, C).cpu()
                        features['semantic_embeds'] = semantic_embeds
                    else:
                        print(f"Warning: Semantic not found: {semantic_path}")
                        features['semantic_embeds'] = None
                
                # 添加元数据
                features['index'] = item.get('index', -1)
                features['question'] = item['question']
                features['rationale'] = item['rationale']
                features['answer'] = item['answer']
                features['image_path'] = item['image_path']
                features['depth_path'] = item['depth_path']
                if 'semantic_path' in item:
                    features['semantic_path'] = item['semantic_path']
                
                batch_features.append(features)
                
            except Exception as e:
                print(f"Error processing sample {item.get('index', -1)}: {e}")
                continue
    
    return batch_features

def main():
    args = parse_args()
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, "precomputed_features")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # 加载模型
    models = load_models(args)
    
    # 读取数据集
    jsonl_path = os.path.join(args.data_dir, "ssr-cot.jsonl")
    print(f"Loading dataset from {jsonl_path}")
    data = load_jsonl(jsonl_path)
    
    # 添加索引
    for i, item in enumerate(data):
        item['index'] = i
    
    # 处理范围
    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx is not None else len(data)
    data = data[start_idx:end_idx]
    
    print(f"Processing samples {start_idx} to {end_idx} (total: {len(data)} samples)")
    
    # 检查已处理的样本
    processed_count = 0
    skipped_count = 0
    
    # 批量处理
    batch_size = args.batch_size
    num_batches = (len(data) + batch_size - 1) // batch_size
    
    print(f"Processing in {num_batches} batches...")
    
    for batch_idx in tqdm(range(num_batches), desc="Extracting features"):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(data))
        batch_data = data[batch_start:batch_end]
        
        # 提取特征
        batch_features = extract_features_batch(batch_data, models, args)
        
        # 实时保存特征（不要等到最后）
        for features in batch_features:
            idx = features['index']
            output_path = output_dir / f"sample_{idx:06d}.pt"
            
            # 检查是否已存在
            if output_path.exists():
                skipped_count += 1
                continue
            
            # 只保存tensor特征，元数据单独保存
            tensor_features = {
                'image_embeds': features['image_embeds'],
                'depth_embeds': features['depth_embeds']
            }
            if 'semantic_embeds' in features and features['semantic_embeds'] is not None:
                tensor_features['semantic_embeds'] = features['semantic_embeds']
            
            # 保存tensor
            torch.save(tensor_features, output_path)
            
            # 保存元数据
            metadata = {
                'index': idx,
                'question': features['question'],
                'rationale': features['rationale'],
                'answer': features['answer'],
                'image_path': features['image_path'],
                'depth_path': features['depth_path']
            }
            if 'semantic_path' in features:
                metadata['semantic_path'] = features['semantic_path']
            
            metadata_path = output_dir / f"sample_{idx:06d}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            processed_count += 1
    
    print(f"Features saved to {output_dir}")
    print(f"Processed: {processed_count}, Skipped: {skipped_count}")
    
    # 保存配置信息
    if processed_count > 0:
        # 读取一个样本获取维度信息
        sample_files = list(output_dir.glob("sample_*.pt"))
        if sample_files:
            sample_features = torch.load(sample_files[0])
            feature_dims = {
                'image_embeds': list(sample_features['image_embeds'].shape),
                'depth_embeds': list(sample_features['depth_embeds'].shape),
                'semantic_embeds': list(sample_features['semantic_embeds'].shape) if 'semantic_embeds' in sample_features else None
            }
        else:
            feature_dims = None
        
        config = {
            'total_samples': end_idx,
            'processed_range': [start_idx, end_idx],
            'use_semantic': args.use_semantic,
            'clip_path': args.clip_path,
            'siglip_path': args.siglip_path,
            'segformer_path': args.segformer_path if args.use_semantic else None,
            'data_dir': args.data_dir,
            'feature_dims': feature_dims
        }
        
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Config saved to {config_path}")
    
    print("Done!")

if __name__ == "__main__":
    main()