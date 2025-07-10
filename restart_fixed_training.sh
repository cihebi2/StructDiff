#!/bin/bash
# StructDiff修复后重启脚本

echo "🚀 启动修复后的StructDiff分离式训练..."

# 设置环境
export CUDA_VISIBLE_DEVICES=2,3,4,5
export PYTHONPATH="/home/qlyu/sequence/StructDiff-7.0.0:$PYTHONPATH"

# 检查数据
echo "📊 检查数据状态..."
ls -la ./data/processed/

# 启动训练
cd /home/qlyu/sequence/StructDiff-7.0.0

python scripts/train_separated.py \
    --config configs/separated_training_fixed_v2.yaml \
    --data-dir ./data/processed \
    --output-dir ./outputs/separated_fixed_v2 \
    --device auto \
    --stage both \
    --use-cfg \
    --use-length-control \
    --debug

echo "✅ 训练脚本已启动"
