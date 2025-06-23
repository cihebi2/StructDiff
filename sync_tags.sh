#!/bin/bash

# Git 标签同步脚本
# 用于创建和推送缺失的版本标签到 GitHub

echo "🔍 检查当前标签状态..."

# 检查本地标签
echo "本地标签："
git tag

echo ""
echo "🏷️ 创建缺失的版本标签..."

# 根据提交信息创建对应的标签
echo "正在为 v5.2.0 创建标签..."
git tag v5.2.0 7a5f3bb -m "v5.2.0: 完整集成AlphaFold3自适应条件化机制"

echo "正在为 v5.3.0 创建标签..."
git tag v5.3.0 5d3b257 -m "v5.3.0: 完整的项目开发框架和ESMFold内存优化解决方案"

echo ""
echo "📤 推送所有标签到 GitHub..."

# 推送所有标签到远程仓库
git push origin --tags

echo ""
echo "✅ 标签同步完成！现在检查结果："
git tag

echo ""
echo "🌐 您现在可以在 GitHub 上看到以下标签："
echo "- v5.0.0"
echo "- v5.1.0" 
echo "- v5.2.0"
echo "- v5.3.0" 