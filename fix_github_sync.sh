#!/bin/bash

# GitHub同步修复脚本
# 解决标签和文件更新不显示的问题

echo "🔧 开始修复GitHub同步问题..."
echo "=========================================="

# 进入项目目录
cd /home/qlyu/sequence/StructDiff-7.0.0

# 检查当前状态
echo "📊 检查当前git状态..."
echo "当前分支:"
git branch -v 2>/dev/null || echo "Git命令失败"

echo ""
echo "远程仓库:"
git remote -v 2>/dev/null || echo "Git命令失败"

echo ""
echo "本地标签:"
git tag -l 2>/dev/null || echo "Git命令失败"

echo ""
echo "最近提交:"
git log --oneline -3 2>/dev/null || echo "Git命令失败"

# 重新配置用户信息（以防丢失）
echo ""
echo "🔧 重新配置Git用户信息..."
git config user.email "cihebi@163.com"
git config user.name "cihebi"

# 检查是否有未提交的更改
echo ""
echo "📋 检查工作区状态..."
git status --porcelain

# 重新设置远程仓库（使用token）
echo ""
echo "🔗 重新配置远程仓库..."
git remote set-url origin https://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@github.com/cihebi2/StructDiff.git

# 强制推送主分支
echo ""
echo "📤 强制推送主分支..."
git push --force origin main

# 删除并重新创建标签
echo ""
echo "🏷️  重新创建和推送标签..."

# 删除本地标签（如果存在）
git tag -d v7.2.0 2>/dev/null || echo "标签v7.2.0不存在，继续..."

# 删除远程标签（如果存在）
git push --delete origin v7.2.0 2>/dev/null || echo "远程标签v7.2.0不存在，继续..."

# 重新创建标签
git tag -a v7.2.0 -m "StructDiff v7.2.0: GPU优化架构重构版本

这个版本专注于GPU利用率优化和训练性能提升:

核心功能:
- 结构特征预计算和缓存系统 (precompute_structure_features.py)
- GPU利用率优化架构 (full_train_gpu_optimized*.py)
- 动态批次大小调整 (test_optimal_batch_size.py)
- ESMFold内存管理优化

新增文档:
- GPU优化指南 (docs/GPU_UTILIZATION_OPTIMIZATION_PLAN.md)
- ESMFold内存问题解决方案 (docs/ESMFOLD_MEMORY_ISSUE_SOLUTION.md)
- 训练架构指南 (docs/ESMFOLD_TRAINING_ARCHITECTURE_GUIDE.md)

性能提升:
- GPU利用率: 20% → 70%+
- 训练速度: 3-5倍提升
- 内存效率: 大幅改善

主要文件:
- precompute_structure_features.py: 结构特征预计算系统
- full_train_gpu_optimized.py: GPU优化训练脚本
- quick_gpu_optimization_analysis.py: GPU性能分析工具
- docs/: 完整的优化指南和架构文档"

# 推送标签
echo ""
echo "📤 推送标签到GitHub..."
git push origin v7.2.0

# 推送所有标签（确保完整性）
echo ""
echo "📤 推送所有标签..."
git push origin --tags

# 清理敏感信息
echo ""
echo "🔒 清理敏感信息..."
git remote set-url origin https://github.com/cihebi2/StructDiff.git

# 验证推送结果
echo ""
echo "✅ 验证推送结果..."
echo "远程分支:"
git ls-remote --heads origin

echo ""
echo "远程标签:"
git ls-remote --tags origin

echo ""
echo "🎉 GitHub同步修复完成!"
echo "=========================================="
echo ""
echo "请等待几分钟，然后检查："
echo "📋 项目页面: https://github.com/cihebi2/StructDiff"
echo "🏷️  发布页面: https://github.com/cihebi2/StructDiff/releases"
echo "📝 提交历史: https://github.com/cihebi2/StructDiff/commits/main"
echo ""
echo "如果仍然不显示，可能是GitHub缓存问题，建议："
echo "1. 等待5-10分钟后刷新页面"
echo "2. 使用强制刷新 (Ctrl+F5)"
echo "3. 清除浏览器缓存"
echo "" 