#!/bin/bash
# 使用SSR-MIDI-7B替代TMI进行特征提取
# 完全兼容原有的Stage 2训练流程

set -e

echo "=========================================="
echo "使用三模态MIDI进行特征提取"
echo "RGB + Depth + Semantic → 融合特征"
echo "=========================================="

# 基础路径
BASE_DIR="/code/VLA"
DATA_DIR="$BASE_DIR/datasets"
MODEL_DIR="$BASE_DIR/models"
OUTPUT_DIR="$DATA_DIR/fused_features"

# MIDI模型路径 (三模态训练后的模型)
MIDI_MODEL="/code/VLA/SSR/checkpoints/SSR-MIDI-trimodal/MIDI_tmi"

# 数据文件（使用您现有的数据）
TRAIN_DATA="/code/VLA/datasets/sharegpt_data/nuscenes_sharegpt_train.json"
VAL_DATA="/code/VLA/datasets/sharegpt_data/nuscenes_sharegpt_val.json"

# 输出路径（与原TMI相同）
TRAIN_OUTPUT="$OUTPUT_DIR/train"
VAL_OUTPUT="$OUTPUT_DIR/val"

# 创建输出目录
mkdir -p $TRAIN_OUTPUT/features
mkdir -p $VAL_OUTPUT/features

echo ""
echo "配置信息："
echo "  MIDI模型: $MIDI_MODEL"
echo "  训练数据: $TRAIN_DATA"
echo "  验证数据: $VAL_DATA"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 检查数据文件是否存在
if [ ! -f "$TRAIN_DATA" ]; then
    echo "错误：训练数据文件不存在: $TRAIN_DATA"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "错误：验证数据文件不存在: $VAL_DATA"
    exit 1
fi

# 检查MIDI模型是否存在
if [ ! -d "$MIDI_MODEL" ]; then
    echo "错误：MIDI模型目录不存在: $MIDI_MODEL"
    exit 1
fi

# 设置Python路径，确保能找到SSR模块
export PYTHONPATH=/code/VLA/SSR-main:$PYTHONPATH

# 解决protobuf版本冲突
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 步骤1：提取训练集特征（使用8卡并行，与TMI一致）
echo "==========================================="
echo "步骤1: 提取训练集特征"
echo "==========================================="

# 设置环境变量启用MIDI模式
export USE_MIDI_MODE=true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python /code/VLA/tri_modal_qwen/scripts/extract_tmi_features.py \
    --input_file $TRAIN_DATA \
    --output_file $TRAIN_OUTPUT/train_with_tmi.json \
    --tmi_checkpoint $MIDI_MODEL \
    --output_dir $TRAIN_OUTPUT/features \
    --num_gpus 8 \
    --device cuda

echo "训练集特征提取完成！"

# 步骤2：提取验证集特征
echo ""
echo "==========================================="
echo "步骤2: 提取验证集特征"
echo "==========================================="

python /code/VLA/tri_modal_qwen/scripts/extract_tmi_features.py \
    --input_file $VAL_DATA \
    --output_file $VAL_OUTPUT/val_with_tmi.json \
    --tmi_checkpoint $MIDI_MODEL \
    --output_dir $VAL_OUTPUT/features \
    --num_gpus 8 \
    --device cuda

echo "验证集特征提取完成！"

# 步骤3：清理数据（与TMI流程一致）
echo ""
echo "==========================================="
echo "步骤3: 清理数据以供LLaMA Factory使用"
echo "==========================================="

# 使用正确的参数格式调用清理脚本
python /code/VLA/tri_modal_qwen/scripts/clean_image_tokens.py \
    --train-data $TRAIN_OUTPUT/train_with_tmi.json \
    --val-data $VAL_OUTPUT/val_with_tmi.json \
    --output-dir $OUTPUT_DIR

# 步骤4：验证输出
echo ""
echo "==========================================="
echo "步骤4: 验证输出"
echo "==========================================="

# 检查文件
echo "检查生成的文件："
ls -lh $TRAIN_OUTPUT/train_with_tmi_cleaned.json
ls -lh $VAL_OUTPUT/val_with_tmi_cleaned.json

# 统计特征文件
echo ""
echo "特征文件统计："
echo "训练集特征数: $(ls $TRAIN_OUTPUT/features/*.npy 2>/dev/null | wc -l)"
echo "验证集特征数: $(ls $VAL_OUTPUT/features/*.npy 2>/dev/null | wc -l)"

# 验证特征维度
echo ""
echo "验证特征维度："
python -c "
import numpy as np
import os

# 检查第一个特征文件
train_features_dir = '$TRAIN_OUTPUT/features'
feature_files = [f for f in os.listdir(train_features_dir) if f.endswith('.npy')]
if feature_files:
    sample_feat = np.load(os.path.join(train_features_dir, feature_files[0]))
    print(f'特征形状: {sample_feat.shape}')
    print(f'特征dtype: {sample_feat.dtype}')
    
    # 检查是否需要调整维度
    if sample_feat.shape == (10, 3584):
        print('✓ 特征维度正确 (10个TOR, 每个3584维)')
    elif sample_feat.shape == (1, 3584):
        print('✓ 特征维度正确 (单个token, 3584维)')
    else:
        print(f'⚠️ 特征维度为 {sample_feat.shape}，期望 (10, 3584)')
else:
    print('❌ 未找到特征文件')
"

echo ""
echo "==========================================="
echo "✅ 三模态MIDI特征提取完成！"
echo "==========================================="
echo ""
echo "特征配置："
echo "- RGB特征: CLIP-ViT-L (~577 tokens, 1024维)"
echo "- Depth特征: SigLIP (~729 tokens, 1152维)"
echo "- Semantic特征: SegFormer-B5 (400 tokens, 512维)"
echo "- 融合后: 10个TOR tokens, 每个3584维 (Qwen2.5-7B hidden_size)"
echo ""
echo "下一步："
echo "1. 确认特征文件格式正确"
echo "2. 运行Stage 2训练："
echo "   bash /code/VLA/tri_modal_qwen/run_stage2_llama_factory.sh"
echo ""
echo "注意："
echo "- 生成的文件格式与TMI完全兼容"
echo "- 无需修改Stage 2的配置文件"
echo "- 特征字段名保持为 'tmi_features' 以兼容现有代码"
echo "==========================================="