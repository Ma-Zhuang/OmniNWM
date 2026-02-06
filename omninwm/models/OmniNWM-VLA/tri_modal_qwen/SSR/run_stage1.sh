#!/bin/bash
# 三模态MIDI快速训练脚本（使用预计算特征）

# 解决protobuf版本冲突
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 优化CUDA内存分配，避免碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "三模态MIDI快速训练（预计算特征模式）"
echo "LLM: Qwen2.5-7B → TOR输出: 3584维"
echo "=========================================="

# 步骤1：检查是否已有预计算特征
PRECOMPUTED_DIR="/code/VLA/datasets/SSR-CoT/precomputed_features"

if [ ! -d "$PRECOMPUTED_DIR" ]; then
    echo "[Step 1] 预计算特征不存在，开始提取..."
    echo "注意：这个过程只需要运行一次，之后可以重复使用"
    
    # 检测GPU数量
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "检测到 $NUM_GPUS 块GPU"
    
    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "使用 $NUM_GPUS 块GPU并行提取特征..."
        accelerate launch --num_processes=$NUM_GPUS \
            precompute_features_parallel.py \
            --data_dir /code/VLA/datasets/SSR-CoT \
            --clip_path /code/VLA/models/clip-vit-large-patch14-336 \
            --siglip_path /code/VLA/models/siglip-so400m-patch14-384 \
            --segformer_path /code/VLA/models/segmentation_models/segformer-b5-finetuned-ade-640-640 \
            --use_semantic \
            --batch_size 32
    else
        echo "使用单GPU提取特征..."
        python precompute_features.py \
            --data_dir /code/VLA/datasets/SSR-CoT \
            --clip_path /code/VLA/models/clip-vit-large-patch14-336 \
            --siglip_path /code/VLA/models/siglip-so400m-patch14-384 \
            --segformer_path /code/VLA/models/segmentation_models/segformer-b5-finetuned-ade-640-640 \
            --use_semantic \
            --batch_size 16
    fi
    
    echo "特征提取完成！"
else
    echo "[Step 1] 发现预计算特征目录，跳过特征提取"
fi

# 步骤2：使用预计算特征进行快速训练
echo ""
echo "[Step 2] 开始快速训练..."
echo "使用预计算特征，支持多进程数据加载"

accelerate launch --config_file scripts/fsdp.yaml \
    ssr/train/train_reasoning.py \
    --data_dir /code/VLA/datasets/SSR-CoT \
    --use_precomputed \
    --precomputed_dir $PRECOMPUTED_DIR \
    --mamba /code/VLA/models/state-spaces/mamba-130m-hf \
    --llm /code/VLA/models/Qwen2.5-7B \
    --use_semantic \
    --use_gradient_checkpointing \
    --epochs 2 \
    --batch_size_per_gpu 1 \
    --gradient_accumulation_steps 8 \
    --lr 2e-5 \
    --num_workers 4 \
    --output_dir checkpoints/SSR-MIDI-trimodal

echo ""
echo "=========================================="
echo "训练完成！"
echo "模型保存在: checkpoints/SSR-MIDI-trimodal"
echo "=========================================="