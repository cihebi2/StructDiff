#!/bin/bash

# StructDiff v5.4.0 强制上传脚本
# 使用GitHub个人访问令牌进行身份验证

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

GITHUB_TOKEN="YOUR_GITHUB_TOKEN_HERE"
VERSION="v5.4.0"
COMMIT_MESSAGE="v5.4.0: 完善Git回滚工具和标签同步功能，强制同步所有更改"

echo -e "${BLUE}🚀 StructDiff $VERSION 强制上传流程${NC}"
echo -e "${BLUE}========================================${NC}"

# 1. 配置Git身份信息
echo -e "${YELLOW}👤 Step 1: 配置Git身份信息${NC}"
git config user.email "qlyu@example.com"
git config user.name "qlyu"
echo "✅ Git身份信息已配置"

# 2. 配置GitHub访问令牌
echo -e "${YELLOW}🔑 Step 2: 配置GitHub访问令牌${NC}"
git config credential.helper store
echo "https://qlyu:$GITHUB_TOKEN@github.com" > ~/.git-credentials
echo "✅ GitHub访问令牌已配置"

# 3. 检查当前状态
echo -e "${YELLOW}📋 Step 3: 检查当前状态${NC}"
git status

# 4. 添加所有变更（包括删除的文件）
echo -e "${YELLOW}📦 Step 4: 添加所有变更（包括删除）${NC}"
git add --all
echo "✅ 所有变更已添加到暂存区"

# 显示将要提交的更改
echo -e "${BLUE}将要提交的更改：${NC}"
git status --porcelain

# 5. 提交代码
echo -e "${YELLOW}💾 Step 5: 提交代码${NC}"
git commit -m "$COMMIT_MESSAGE"
echo "✅ 代码已提交"

# 6. 创建标签
echo -e "${YELLOW}🏷️  Step 6: 创建版本标签${NC}"
git tag $VERSION -m "$COMMIT_MESSAGE" || echo "标签可能已存在，继续执行..."
echo "✅ 标签 $VERSION 已创建"

# 7. 强制推送到远程仓库
echo -e "${YELLOW}📤 Step 7: 强制推送到GitHub${NC}"
echo "强制推送主分支..."
git push --force-with-lease origin main

echo "推送所有标签..."
git push origin --tags --force

# 8. 验证推送结果
echo -e "${YELLOW}🔍 Step 8: 验证推送结果${NC}"
git log --oneline -3

echo -e "${GREEN}🎉 Step 9: 强制上传完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ StructDiff $VERSION 强制上传成功！${NC}"
echo ""
echo -e "${BLUE}📋 上传摘要:${NC}"
echo "• 版本: $VERSION"
echo "• 提交信息: $COMMIT_MESSAGE"
echo "• 上传方式: 强制推送"
echo "• GitHub 认证: ✅ 令牌认证"
echo "• 删除文件处理: ✅ 已同步"

echo ""
echo -e "${GREEN}🚀 强制上传流程全部完成！${NC}" 