#!/bin/bash

# GPU利用率优化 - 批次大小测试脚本
# 用于找到最优的批次大小以提升GPU利用率

echo "🚀 开始GPU利用率优化测试..."
echo "=========================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# 进入项目目录
cd /home/qlyu/sequence/StructDiff-7.0.0

# 激活conda环境
source /home/qlyu/miniconda3/etc/profile.d/conda.sh
conda activate cuda12.1

# 检查GPU状态
echo "📊 当前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits

echo ""
echo "🧪 启动批次大小优化测试..."
echo "=========================================="

# 运行测试
python test_optimal_batch_size.py

echo ""
echo "✅ 测试完成!"
echo "=========================================="

# 显示最终GPU状态
echo "📊 测试后GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits 