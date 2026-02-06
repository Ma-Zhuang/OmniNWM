#!/bin/bash
# Stage 2 模型评估脚本
# 评估训练好的模型在验证集上的性能

set -e

echo "=========================================="
echo "Stage 2 三模态MIDI模型评估"
echo "模型: Qwen2.5-VL-7B + MIDI特征"
echo "=========================================="

# 配置路径
CHECKPOINT="/code/VLA/outputs/stage2_llama_factory/checkpoint-3500"  # 使用best model
VAL_DATA="/code/VLA/datasets/fused_features/val/val_with_tmi_cleaned.json"
FEATURE_DIR="/code/VLA/datasets/fused_features/val/features"
OUTPUT_DIR="/code/VLA/outputs/evaluation"

# 环境设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 使用8张GPU进行并行评估
export PYTHONPATH=/code/VLA/tri_modal_qwen:$PYTHONPATH

# 检查checkpoint是否存在
if [ ! -d "$CHECKPOINT" ]; then
    echo "❌ 错误: Checkpoint不存在: $CHECKPOINT"
    echo "可用的checkpoints:"
    ls -la /code/VLA/outputs/stage2_llama_factory/ | grep checkpoint
    exit 1
fi

# 检查数据文件
if [ ! -f "$VAL_DATA" ]; then
    echo "❌ 错误: 验证数据不存在: $VAL_DATA"
    exit 1
fi

# 检查特征目录
if [ ! -d "$FEATURE_DIR" ]; then
    echo "❌ 错误: 特征目录不存在: $FEATURE_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo ""
echo "评估配置:"
echo "  Checkpoint: $CHECKPOINT"
echo "  验证数据: $VAL_DATA"
echo "  特征目录: $FEATURE_DIR"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 统计数据
echo "数据统计:"
echo "  验证样本数: $(python -c "import json; print(len(json.load(open('$VAL_DATA'))))")"
echo "  特征文件数: $(ls $FEATURE_DIR/*.npy 2>/dev/null | wc -l)"
echo ""

# 运行评估
echo "开始评估..."

# 使用8卡并行评估完整验证集（每卡处理360个样本）
python /code/VLA/tri_modal_qwen/evaluate_stage2.py \
    --checkpoint $CHECKPOINT \
    --data_path $VAL_DATA \
    --feature_dir $FEATURE_DIR \
    --output_dir $OUTPUT_DIR \
    --use_parallel \
    --num_gpus 8 \
    --max_samples 2880  # 评估完整的验证集

# 备用：单GPU评估（调试用）
# python /code/VLA/tri_modal_qwen/evaluate_stage2.py \
#     --checkpoint $CHECKPOINT \
#     --data_path $VAL_DATA \
#     --feature_dir $FEATURE_DIR \
#     --output_dir $OUTPUT_DIR \
#     --device cuda \
#     --max_samples 100

echo ""
echo "=========================================="
echo "✅ 评估完成!"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  ls -la $OUTPUT_DIR/"
echo ""
echo "可视化结果:"
echo "  python /code/VLA/tri_modal_qwen/visualize_results.py --eval-dir $OUTPUT_DIR"
echo "=========================================="