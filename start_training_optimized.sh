#!/bin/bash
# 优化的训练启动脚本

echo "🚀 启动优化的ESMFold训练..."

# 设置内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0

# 运行内存监控（如果Python可用）
if command -v python3 &> /dev/null; then
    echo "📊 运行内存监控..."
    python3 memory_monitor.py
fi

echo "🧹 设置完成，启动训练..."

# 启动训练
python3 scripts/train_peptide_esmfold.py \
    --config configs/peptide_esmfold_config.yaml \
    --debug \
    "$@"

echo "✅ 训练完成"