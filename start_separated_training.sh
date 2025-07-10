#!/bin/bash

# StructDiff分离式训练启动脚本
# 基于CPL-Diff的两阶段训练策略

echo "🚀 开始StructDiff分离式训练"
echo "时间: $(date)"
echo "用户: $(whoami)"
echo "工作目录: $(pwd)"

# 环境检查
echo "📋 环境检查..."
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c "import torch; print(torch.__version__)")"
echo "CUDA版本: $(python -c "import torch; print(torch.version.cuda)")"
echo "GPU数量: $(nvidia-smi --list-gpus | wc -l)"

# 显示可用GPU
echo "🖥️  GPU信息:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits

# 检查可用GPU (2,3,4,5)
echo "🔍 检查目标GPU可用性..."
for gpu_id in 2 3 4 5; do
    gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id)
    gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
    echo "GPU $gpu_id: 使用率=${gpu_usage}%, 显存使用=${gpu_memory}MB"
done

# 设置CUDA可见设备（使用GPU 2,3,4,5）
export CUDA_VISIBLE_DEVICES=2,3,4,5
echo "✅ 设置CUDA可见设备: $CUDA_VISIBLE_DEVICES"

# 设置其他环境变量
export PYTHONPATH="/home/qlyu/sequence/StructDiff-7.0.0:$PYTHONPATH"
export OMP_NUM_THREADS=8

# 检查数据文件
echo "📁 检查数据文件..."
DATA_DIR="./data/processed"
if [ -f "$DATA_DIR/train.csv" ]; then
    echo "✅ 训练数据: $DATA_DIR/train.csv ($(wc -l < $DATA_DIR/train.csv) 行)"
else
    echo "❌ 训练数据文件不存在: $DATA_DIR/train.csv"
    exit 1
fi

if [ -f "$DATA_DIR/val.csv" ]; then
    echo "✅ 验证数据: $DATA_DIR/val.csv ($(wc -l < $DATA_DIR/val.csv) 行)"
else
    echo "⚠️  验证数据文件不存在: $DATA_DIR/val.csv"
fi

# 检查结构缓存
echo "🗂️  检查结构缓存..."
CACHE_DIR="./cache"
if [ -d "$CACHE_DIR/train" ]; then
    train_cache_count=$(find $CACHE_DIR/train -name "*.pkl" | wc -l)
    echo "✅ 训练结构缓存: $train_cache_count 个文件"
else
    echo "⚠️  训练结构缓存目录不存在: $CACHE_DIR/train"
fi

if [ -d "$CACHE_DIR/val" ]; then
    val_cache_count=$(find $CACHE_DIR/val -name "*.pkl" | wc -l)
    echo "✅ 验证结构缓存: $val_cache_count 个文件"
else
    echo "⚠️  验证结构缓存目录不存在: $CACHE_DIR/val"
fi

# 创建输出目录
OUTPUT_DIR="./outputs/separated_production_v1"
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/checkpoints
mkdir -p $OUTPUT_DIR/results
echo "📂 创建输出目录: $OUTPUT_DIR"

# 训练配置
CONFIG_FILE="configs/separated_training_production.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi
echo "⚙️  使用配置: $CONFIG_FILE"

# 构建训练命令
TRAIN_CMD="python scripts/train_separated.py \
    --config $CONFIG_FILE \
    --output-dir $OUTPUT_DIR \
    --data-dir $DATA_DIR \
    --device auto \
    --use-cfg \
    --use-length-control \
    --use-amp"

# 如果提供了阶段参数，添加到命令中
if [ ! -z "$1" ]; then
    if [ "$1" == "1" ] || [ "$1" == "2" ] || [ "$1" == "both" ]; then
        TRAIN_CMD="$TRAIN_CMD --stage $1"
        echo "🎯 训练阶段: $1"
    else
        echo "❌ 无效的阶段参数: $1 (支持: 1, 2, both)"
        exit 1
    fi
else
    echo "🎯 训练阶段: both (完整两阶段训练)"
fi

# 显示最终命令
echo "🔧 训练命令:"
echo "$TRAIN_CMD"

# 开始训练
echo ""
echo "🚀 开始分离式训练..."
echo "日志输出到: $OUTPUT_DIR/logs/training.log"
echo "==============================================="

# 执行训练（同时输出到终端和日志文件）
$TRAIN_CMD 2>&1 | tee $OUTPUT_DIR/logs/training.log

# 检查训练结果
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 训练成功完成！"
    echo "输出目录: $OUTPUT_DIR"
    
    # 显示生成的文件
    echo "📁 生成的文件:"
    find $OUTPUT_DIR -type f -name "*.pt" -o -name "*.json" -o -name "*.csv" | head -10
    
else
    echo ""
    echo "❌ 训练失败，退出码: $TRAIN_EXIT_CODE"
    echo "检查日志: $OUTPUT_DIR/logs/training.log"
    exit $TRAIN_EXIT_CODE
fi

echo "训练完成时间: $(date)" 