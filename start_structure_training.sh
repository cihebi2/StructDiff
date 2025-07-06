#!/bin/bash

# 启动结构特征训练脚本
echo "🚀 开始启动结构特征训练..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/home/qlyu/sequence/StructDiff-7.0.0:$PYTHONPATH"

# 切换到项目目录
cd /home/qlyu/sequence/StructDiff-7.0.0

# 检查预训练模型是否存在
if [ ! -f "outputs/full_training_200_esmfold_fixed/best_model.pt" ]; then
    echo "❌ 错误：未找到预训练模型文件"
    echo "请确保之前的训练已完成并生成了 best_model.pt 文件"
    exit 1
fi

echo "✅ 找到预训练模型文件"

# 检查GPU状态
echo "🔍 检查GPU状态..."
nvidia-smi

# 创建输出目录
mkdir -p outputs/structure_feature_training

# 启动训练
echo "🎯 启动结构特征训练..."
python full_train_with_structure_features_enabled.py

echo "✅ 结构特征训练启动完成！" 