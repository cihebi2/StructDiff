#!/bin/bash

# StructDiff 多肽生成训练启动脚本

echo "=== StructDiff 多肽生成训练 ==="
echo "开始时间: $(date)"

# 进入项目目录
cd /home/qlyu/StructDiff

# 设置环境变量
export PYTHONPATH="/home/qlyu/StructDiff:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# 检查数据集
echo "检查数据集..."
if [ ! -f "data/processed/train.csv" ]; then
    echo "❌ 训练数据不存在: data/processed/train.csv"
    exit 1
fi

if [ ! -f "data/processed/val.csv" ]; then
    echo "❌ 验证数据不存在: data/processed/val.csv"
    exit 1
fi

echo "✓ 数据集检查通过"

# 检查配置文件
if [ ! -f "configs/peptide_esmfold_config.yaml" ]; then
    echo "❌ 配置文件不存在: configs/peptide_esmfold_config.yaml"
    exit 1
fi

echo "✓ 配置文件检查通过"

# 创建必要的目录
mkdir -p outputs
mkdir -p checkpoints
mkdir -p logs

echo "✓ 目录创建完成"

# 启动训练
echo "开始训练..."

# 选择训练模式
if [ "$1" = "debug" ]; then
    echo "🔧 Debug模式训练"
    python scripts/train_peptide_esmfold.py \
        --config configs/peptide_esmfold_config.yaml \
        --debug \
        --gpu 0
elif [ "$1" = "test" ]; then
    echo "🧪 测试模式训练 (3 epochs)"
    python scripts/train_peptide_esmfold.py \
        --config configs/peptide_esmfold_config.yaml \
        --test-run \
        --gpu 0
else
    echo "🚀 完整训练模式"
    python scripts/train_peptide_esmfold.py \
        --config configs/peptide_esmfold_config.yaml \
        --gpu 0
fi

echo "训练结束时间: $(date)"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "✅ 训练成功完成！"
    echo "检查输出目录: outputs/"
    echo "检查日志目录: logs/"
    echo "检查检查点: checkpoints/"
else
    echo "❌ 训练失败，请检查日志"
fi 