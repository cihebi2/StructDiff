#!/bin/bash

echo "🚀 启动GPU优化训练 - 使用预计算结构特征"
echo "================================================"

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

# 检查GPU状态
echo "📊 当前GPU状态:"
nvidia-smi

# 进入工作目录
cd /home/qlyu/sequence/StructDiff-7.0.0

# 检查预计算缓存
echo ""
echo "🔍 检查结构特征缓存:"
if [ -d "./structure_cache" ]; then
    cache_size=$(du -sh ./structure_cache | cut -f1)
    cache_files=$(find ./structure_cache -name "*.pt" | wc -l)
    echo "✅ 缓存目录存在"
    echo "📁 缓存大小: $cache_size"
    echo "📄 缓存文件数: $cache_files"
else
    echo "❌ 缓存目录不存在，请先运行预计算脚本"
    exit 1
fi

echo ""
echo "🏃 开始优化训练..."
echo "配置说明:"
echo "- 批次大小: 16 (4倍提升)"
echo "- 数据加载工作进程: 4"
echo "- 混合精度训练: 启用"
echo "- 梯度累积: 2"
echo "- 使用GPU: cuda:1"
echo "- 预计算结构特征: 启用"

# 启动训练
python train_with_precomputed_features.py

echo ""
echo "训练完成!" 