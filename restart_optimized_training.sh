#!/bin/bash

# 重启优化后的分离式训练
echo "🔄 重启优化后的分离式训练..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=2,3,4,5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

# 使用优化配置重启训练
python scripts/train_separated.py \
    --config configs/separated_training_optimized.yaml \
    --output-dir ./outputs/separated_optimized_v1 \
    --data-dir ./data/processed \
    --device auto \
    --use-cfg \
    --use-length-control \
    --stage both \
    --debug

echo "✅ 优化训练已启动"
