#!/bin/bash

# 修复版本的结构特征训练启动脚本
echo "🚀 开始启动修复版本的结构特征训练..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/home/qlyu/sequence/StructDiff-7.0.0:$PYTHONPATH"

# 关键：设置PyTorch内存分配策略
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

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
mkdir -p outputs/structure_feature_training_fixed

# 强制清理GPU内存
echo "🧹 清理GPU内存..."
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f'GPU内存已清理，当前使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB')
"

# 启动修复版本的训练
echo "🎯 启动修复版本的结构特征训练..."
python full_train_with_structure_features_fixed_v2.py

echo "✅ 修复版本的结构特征训练启动完成！" 