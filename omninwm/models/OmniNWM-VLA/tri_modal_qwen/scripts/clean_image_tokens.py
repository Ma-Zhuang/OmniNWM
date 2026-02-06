#!/usr/bin/env python3
"""
清理数据中的<image>标记
因为我们使用预融合的TMI特征，不需要单独的图像标记
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

def clean_image_tokens(data_path, output_path):
    """清理JSON数据中的<image>标记"""
    
    print(f"加载数据: {data_path}")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"处理 {len(data)} 条数据...")
    
    for item in tqdm(data):
        # 处理messages中的content
        if 'messages' in item:
            for message in item['messages']:
                if message.get('role') == 'user':
                    # 移除所有<image>标记
                    original_content = message['content']
                    
                    # 替换文本中的<image>描述 - 注意实际数据使用的是front-right而不是left
                    cleaned_content = original_content.replace(
                        "front view <image>, front-left view <image>, front-right view <image>, rear view <image>, rear-left view <image>, rear-right view <image>",
                        "front view, front-left view, front-right view, rear view, rear-left view, rear-right view"
                    )
                
                    # 再次确保没有遗留的<image>标记
                    cleaned_content = cleaned_content.replace("<image>", "")
                    
                    message['content'] = cleaned_content
                    
                    # 统计替换了多少个标记
                    num_replaced = original_content.count("<image>")
                    if num_replaced > 0:
                        # print(f"  移除了 {num_replaced} 个<image>标记")
                        pass
    
    print(f"保存清理后的数据到: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("✓ 数据清理完成！")

def main():
    parser = argparse.ArgumentParser(description="清理数据中的<image>标记")
    parser.add_argument("--train-data", type=str, 
                       default="/code/VLA/datasets/fused_features/train/train_with_tmi.json",
                       help="训练数据路径")
    parser.add_argument("--val-data", type=str,
                       default="/code/VLA/datasets/fused_features/val/val_with_tmi.json",
                       help="验证数据路径")
    parser.add_argument("--output-dir", type=str,
                       default="/code/VLA/datasets/fused_features",
                       help="输出目录")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # 清理训练数据
    if Path(args.train_data).exists():
        train_output = output_dir / "train" / "train_with_tmi_cleaned.json"
        clean_image_tokens(args.train_data, train_output)
    
    # 清理验证数据
    if Path(args.val_data).exists():
        val_output = output_dir / "val" / "val_with_tmi_cleaned.json"
        clean_image_tokens(args.val_data, val_output)

if __name__ == "__main__":
    main()