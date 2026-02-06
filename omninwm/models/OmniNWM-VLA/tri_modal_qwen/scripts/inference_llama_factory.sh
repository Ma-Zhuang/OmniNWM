#!/bin/bash
# LLaMA Factory训练模型的推理脚本
# 支持TMI特征模式

set -e

# 配置路径
PROJECT_ROOT="/code/VLA/tri_modal_qwen"
MODEL_PATH="/code/VLA/outputs/stage2"  # LLaMA Factory训练输出
FEATURE_DIR="/code/VLA/outputs/fused_features/features"  # TMI特征目录
OUTPUT_DIR="/code/VLA/outputs/inference_results"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "================================================"
echo "LLaMA Factory模型推理（TMI特征模式）"
echo "================================================"
echo "模型路径: ${MODEL_PATH}"
echo "特征目录: ${FEATURE_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
echo "================================================"

# 单样本推理示例（使用TMI特征）
echo ""
echo "执行单样本推理..."
python ${PROJECT_ROOT}/scripts/inference.py \
    --model_path ${MODEL_PATH} \
    --use_tmi_features \
    --tmi_feature_path ${FEATURE_DIR}/sample_001_features.npy \
    --text_prompt "基于三模态感知信息，预测车辆的未来3秒轨迹。" \
    --output_dir ${OUTPUT_DIR} \
    --max_new_tokens 512 \
    --device cuda \
    --torch_dtype float16 \
    --save_visualization \
    --verbose

echo ""
echo "✓ 推理完成！"
echo "结果保存在: ${OUTPUT_DIR}"