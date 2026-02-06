#!/bin/bash

# ShareGPT 数据集生成脚本 - 服务器版本
# 路径：/code/VLA/nuscenes_multimodal_dataset

echo "=========================================="
echo "开始生成 ShareGPT 数据集..."
echo "时间：$(date)"
echo "=========================================="

# 切换到项目目录
cd /code/VLA/nuscenes_multimodal_dataset

# 执行数据集生成
python scripts/build_sharegpt_dataset.py --config configs/sharegpt_dataset_config.yaml

echo "=========================================="
echo "数据集生成完成！"
echo "完成时间：$(date)"
echo "=========================================="