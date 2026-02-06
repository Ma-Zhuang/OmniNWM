#!/bin/bash
# Stage 2: 使用LLaMA Factory进行SFT微调
# 输入：TMI融合特征
# 模型：标准Qwen2.5-VL-7B-Instruct
# 方法：LoRA微调

echo "=========================================="
echo "Stage 2: LLaMA Factory SFT with TMI Features"
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 使用8张GPU
export PYTHONPATH=/code/VLA/tri_modal_qwen:$PYTHONPATH

# 检查TMI特征文件是否存在（清理后的版本）
echo "检查TMI特征文件..."
if [ ! -f "/code/VLA/datasets/fused_features/train/train_with_tmi_cleaned.json" ]; then
    echo "⚠️ 清理后的训练集文件不存在，检查原始文件..."
    if [ -f "/code/VLA/datasets/fused_features/train/train_with_tmi.json" ]; then
        echo "原始文件存在，需要运行数据清理脚本:"
        echo "python /code/VLA/tri_modal_qwen/scripts/clean_image_tokens.py"
        exit 1
    else
        echo "❌ 错误: 训练集TMI特征文件不存在!"
        echo "请先运行Stage 1训练和特征提取"
        exit 1
    fi
fi

if [ ! -f "/code/VLA/datasets/fused_features/val/val_with_tmi_cleaned.json" ]; then
    echo "⚠️ 清理后的验证集文件不存在，检查原始文件..."
    if [ -f "/code/VLA/datasets/fused_features/val/val_with_tmi.json" ]; then
        echo "原始文件存在，需要运行数据清理脚本:"
        echo "python /code/VLA/tri_modal_qwen/scripts/clean_image_tokens.py"
        exit 1
    else
        echo "❌ 错误: 验证集TMI特征文件不存在!"
        echo "请先运行特征提取"
        exit 1
    fi
fi

# 检查特征文件目录
if [ ! -d "/code/VLA/datasets/fused_features/train/features" ]; then
    echo "❌ 错误: 训练集特征目录不存在!"
    exit 1
fi

if [ ! -d "/code/VLA/datasets/fused_features/val/features" ]; then
    echo "❌ 错误: 验证集特征目录不存在!"
    exit 1
fi

echo "✅ TMI特征文件和目录已就绪"

# 创建输出目录
OUTPUT_DIR="/code/VLA/outputs/stage2_llama_factory"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "配置信息："
echo "- 模型: Qwen2.5-VL-7B-Instruct (与MIDI架构一致)"
echo "- 训练方式: LoRA (rank=16)"
echo "- 训练数据: tri_modal_fused_train (23,038样本)"
echo "- 验证数据: tri_modal_fused_val (2,880样本)"
echo "- 特征维度: [10, 3584] (无需投影，完美对齐)"
echo "- 输出目录: $OUTPUT_DIR"
echo "=========================================="

# 不需要切换目录，llamafactory-cli可以在任何位置运行
echo "当前目录: $(pwd)"

# 使用YAML配置文件运行训练
echo "开始LLaMA Factory SFT训练..."

# 设置环境变量允许额外参数
export ALLOW_EXTRA_ARGS=1

# 多GPU训练 - 使用LLaMA Factory的标准方式
# 方式1: 使用llamafactory-cli + deepspeed（推荐，8卡训练）
FORCE_TORCHRUN=1 ALLOW_EXTRA_ARGS=1 llamafactory-cli train \
    /code/VLA/tri_modal_qwen/llama_factory_configs/stage2_end_to_end.yaml

# 方式2: 使用accelerate启动（备选方案）
# accelerate launch \
#     --config_file /code/VLA/tri_modal_qwen/accelerate_config.yaml \
#     --num_machines 1 \
#     --num_processes 8 \
#     --main_process_port 29500 \
#     --mixed_precision bf16 \
#     $(which llamafactory-cli) train \
#     --config_file /code/VLA/tri_modal_qwen/llama_factory_configs/stage2_end_to_end.yaml

# 方式3: 直接使用torchrun（如果llamafactory-cli有问题）
# torchrun --nproc_per_node 8 \
#     --master_port 29500 \
#     $(python -c "import llamafactory; import os; print(os.path.join(os.path.dirname(llamafactory.__file__), 'train.py'))") \
#     /code/VLA/tri_modal_qwen/llama_factory_configs/stage2_end_to_end.yaml

# 方式4: 单GPU测试（用于调试）
# CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
#     --config_file /code/VLA/tri_modal_qwen/llama_factory_configs/stage2_end_to_end.yaml

echo "✅ Stage 2 LLaMA Factory训练完成!"
echo "模型保存在: $OUTPUT_DIR"

# 显示最佳checkpoint
echo "=========================================="
echo "查找最佳模型..."
if [ -d "$OUTPUT_DIR/checkpoint-best" ]; then
    echo "最佳模型: $OUTPUT_DIR/checkpoint-best"
elif [ -d "$OUTPUT_DIR/best_model" ]; then
    echo "最佳模型: $OUTPUT_DIR/best_model"
else
    echo "最新checkpoint:"
    ls -lt $OUTPUT_DIR | grep checkpoint | head -5
fi

echo "=========================================="
echo "后续步骤："
echo "1. 查看TensorBoard: tensorboard --logdir=$OUTPUT_DIR --port=6006"
echo "2. 评估模型: bash run_stage2_evaluation.sh"
echo "3. 推理测试: bash run_inference.sh"
echo "=========================================="