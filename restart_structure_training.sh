#!/bin/bash
# StructDiff结构感知训练重启脚本

echo "🚀 启动结构感知StructDiff分离式训练..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=2,3,4,5
export PYTHONPATH="/home/qlyu/sequence/StructDiff-7.0.0:$PYTHONPATH"

# 内存优化设置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# 停止现有训练进程
echo "🛑 停止现有训练进程..."
pkill -f train_separated.py 2>/dev/null || true
sleep 2

# 清理GPU内存
echo "🧹 清理GPU内存..."
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU缓存已清理')
"

# 检查数据和缓存状态
echo "📊 检查数据和缓存状态..."
ls -la ./data/processed/ | head -5
echo "缓存目录状态:"
ls -la ./cache/ 2>/dev/null || echo "缓存目录不存在，将自动创建"

echo "🎯 开始结构感知训练..."
cd /home/qlyu/sequence/StructDiff-7.0.0

python scripts/train_separated.py \
    --config configs/structure_enabled_training.yaml \
    --data-dir ./data/processed \
    --output-dir ./outputs/structure_enabled_v1 \
    --device auto \
    --stage both \
    --use-cfg \
    --use-length-control \
    --debug

echo "✅ 训练已启动" 