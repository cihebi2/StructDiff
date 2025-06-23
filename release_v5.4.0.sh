#!/bin/bash

# StructDiff v5.4.0 版本发布脚本
# 完整的代码上传和版本管理流程

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

VERSION="v5.4.0"
COMMIT_MESSAGE="v5.4.0: 完善Git回滚工具和标签同步功能"

echo -e "${BLUE}🚀 开始 StructDiff $VERSION 版本发布流程${NC}"
echo -e "${BLUE}========================================${NC}"

# 1. 检查当前状态
echo -e "${YELLOW}📋 Step 1: 检查当前 Git 状态${NC}"
echo "当前分支信息："
git branch
echo ""
echo "工作区状态："
git status --porcelain

# 2. 更新版本文件
echo -e "${YELLOW}📝 Step 2: 更新版本信息${NC}"
echo "5.4.0" > VERSION
echo "✅ 版本文件已更新为 $VERSION"

# 3. 添加所有变更
echo -e "${YELLOW}📦 Step 3: 添加所有变更到暂存区${NC}"
git add .
echo "✅ 所有文件已添加到暂存区"

# 显示将要提交的文件
echo "将要提交的文件："
git diff --cached --name-only

# 4. 确认提交
echo -e "${YELLOW}⚠️  Step 4: 确认提交${NC}"
echo -e "${BLUE}提交信息: $COMMIT_MESSAGE${NC}"
echo -n "确认提交这些更改? (y/N): "
read -r confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo -e "${RED}❌ 操作已取消${NC}"
    exit 1
fi

# 5. 提交代码
echo -e "${YELLOW}💾 Step 5: 提交代码${NC}"
git commit -m "$COMMIT_MESSAGE"
echo "✅ 代码已提交"

# 6. 创建标签
echo -e "${YELLOW}🏷️  Step 6: 创建版本标签${NC}"
git tag $VERSION -m "$COMMIT_MESSAGE"
echo "✅ 标签 $VERSION 已创建"

# 7. 推送代码到远程仓库
echo -e "${YELLOW}📤 Step 7: 推送代码到 GitHub${NC}"
echo "推送主分支..."
git push origin main

echo "推送标签..."
git push origin --tags

echo -e "${GREEN}🎉 Step 8: 发布完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ StructDiff $VERSION 版本发布成功！${NC}"
echo ""
echo -e "${BLUE}📋 发布摘要:${NC}"
echo "• 版本: $VERSION"
echo "• 提交信息: $COMMIT_MESSAGE"
echo "• 推送状态: ✅ 完成"
echo "• GitHub 标签: ✅ 已同步"
echo ""
echo -e "${BLUE}🌐 您现在可以在 GitHub 上看到:${NC}"
echo "• 最新代码更新"
echo "• $VERSION 版本标签"
echo "• Release 页面中的新版本"

# 8. 显示最新的提交历史
echo ""
echo -e "${BLUE}📜 最新提交历史:${NC}"
GIT_PAGER=cat git log --oneline --graph --decorate -5

# 9. 显示所有标签
echo ""
echo -e "${BLUE}🏷️  所有版本标签:${NC}"
git tag --sort=-version:refname

echo ""
echo -e "${GREEN}🚀 发布流程全部完成！${NC}" 