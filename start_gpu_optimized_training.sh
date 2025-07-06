#!/bin/bash

# GPU利用率优化训练启动脚本
# 目标：将GPU利用率从20%提升到70%以上，训练速度提升3-5倍

echo "🚀 启动GPU优化训练..."
echo "=========================================="

# 设置环境变量进行内存优化
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export OMP_NUM_THREADS=4
export CUDA_LAUNCH_BLOCKING=0

# 进入项目目录
cd /home/qlyu/sequence/StructDiff-7.0.0

# 激活conda环境
source /home/qlyu/miniconda3/etc/profile.d/conda.sh
conda activate cuda12.1

# 创建输出目录
mkdir -p outputs/gpu_optimized_training

# 清理GPU内存
echo "🧹 清理GPU内存..."
python -c "import torch; torch.cuda.empty_cache(); print('GPU内存清理完成')"

# 检查GPU状态
echo "📊 训练前GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits

echo ""
echo "🎯 优化配置:"
echo "  - 批次大小: 2 → 8 (4倍提升)"
echo "  - 梯度累积: 8 → 2 (4倍减少)"
echo "  - 数据加载工作进程: 0 → 4"
echo "  - 启用混合精度训练 (AMP)"
echo "  - 启用内存固定和预取"
echo "  - 启用模型编译优化"
echo ""

echo "📈 预期性能提升:"
echo "  - GPU利用率: 22% → 70%+"
echo "  - 训练速度: 3-5倍提升"
echo "  - 总训练时间: 缩短60-80%"
echo ""

echo "🚀 开始GPU优化训练..."
echo "=========================================="

# 启动训练 (使用nohup在后台运行)
nohup python full_train_gpu_optimized.py > outputs/gpu_optimized_training/console.log 2>&1 &

# 获取进程ID
TRAIN_PID=$!
echo "✅ 训练已启动，进程ID: $TRAIN_PID"

# 保存进程ID
echo $TRAIN_PID > outputs/gpu_optimized_training/train.pid

echo ""
echo "📊 监控命令:"
echo "  实时GPU状态: watch -n 1 nvidia-smi"
echo "  训练日志: tail -f outputs/gpu_optimized_training/training.log"
echo "  控制台输出: tail -f outputs/gpu_optimized_training/console.log"
echo ""

echo "🛑 停止训练命令:"
echo "  kill -TERM $TRAIN_PID"
echo "  或者: kill -TERM \$(cat outputs/gpu_optimized_training/train.pid)"
echo ""

# 等待几秒钟让训练启动
sleep 5

echo "📊 训练启动后GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits

echo ""
echo "✅ GPU优化训练已启动完成!"
echo "==========================================" 