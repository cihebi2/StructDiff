#!/bin/bash

# StructDiff Git同步脚本
# 用于将GitHub上的最新代码同步到本地

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== StructDiff Git 代码同步 ===${NC}"

# 检查是否在git仓库中
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}错误: 当前目录不是Git仓库${NC}"
    exit 1
fi

# 显示当前状态
echo -e "${BLUE}检查当前状态...${NC}"
current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
echo -e "当前分支: ${GREEN}$current_branch${NC}"

# 检查工作区状态
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}警告: 工作区有未提交的修改${NC}"
    git status --short
    echo ""
    read -p "是否要暂存这些修改？ [y/N]: " stash_confirm
    if [[ "$stash_confirm" =~ ^[Yy]$ ]]; then
        git stash push -m "自动暂存 - $(date '+%Y-%m-%d %H:%M:%S')"
        echo -e "${GREEN}✓${NC} 修改已暂存"
        STASHED=true
    else
        echo -e "${YELLOW}注意: 未暂存的修改可能会在合并时产生冲突${NC}"
        STASHED=false
    fi
else
    echo -e "${GREEN}✓${NC} 工作区干净"
    STASHED=false
fi

echo ""

# 获取远程最新代码
echo -e "${BLUE}获取远程最新代码...${NC}"
git fetch origin
echo -e "${GREEN}✓${NC} 远程代码获取完成"

# 检查是否有更新
LOCAL_COMMIT=$(git rev-parse HEAD)
REMOTE_COMMIT=$(git rev-parse origin/$current_branch)

if [[ "$LOCAL_COMMIT" == "$REMOTE_COMMIT" ]]; then
    echo -e "${GREEN}✓${NC} 本地代码已是最新版本"
    if [[ "$STASHED" == true ]]; then
        echo -e "${BLUE}恢复暂存的修改...${NC}"
        git stash pop
        echo -e "${GREEN}✓${NC} 暂存的修改已恢复"
    fi
    exit 0
fi

# 显示将要合并的提交
echo -e "${BLUE}检查远程更新...${NC}"
COMMIT_COUNT=$(git rev-list --count HEAD..origin/$current_branch)
echo -e "发现 ${GREEN}$COMMIT_COUNT${NC} 个新提交"

echo ""
echo -e "${YELLOW}最新的提交:${NC}"
git log --oneline --decorate -5 origin/$current_branch

echo ""

# 确认合并
read -p "确认合并远程代码？ [y/N]: " merge_confirm
if [[ ! "$merge_confirm" =~ ^[Yy]$ ]]; then
    echo "取消同步"
    if [[ "$STASHED" == true ]]; then
        echo -e "${BLUE}恢复暂存的修改...${NC}"
        git stash pop
        echo -e "${GREEN}✓${NC} 暂存的修改已恢复"
    fi
    exit 0
fi

# 执行合并
echo -e "${BLUE}合并远程代码...${NC}"
if git merge origin/$current_branch --no-edit; then
    echo -e "${GREEN}✓${NC} 代码合并成功"
else
    echo -e "${RED}合并失败，存在冲突${NC}"
    echo -e "${YELLOW}请手动解决冲突后运行:${NC}"
    echo -e "  git add <冲突文件>"
    echo -e "  git commit"
    exit 1
fi

# 恢复暂存的修改
if [[ "$STASHED" == true ]]; then
    echo -e "${BLUE}恢复暂存的修改...${NC}"
    if git stash pop; then
        echo -e "${GREEN}✓${NC} 暂存的修改已恢复"
    else
        echo -e "${YELLOW}警告: 恢复暂存时有冲突，请手动处理${NC}"
    fi
fi

# 显示最终状态
echo ""
echo -e "${GREEN}=== 同步完成! ===${NC}"
NEW_COMMIT=$(git rev-parse HEAD)
echo -e "最新提交: ${GREEN}$(git log --oneline -1 $NEW_COMMIT)${NC}"

# 显示版本信息
if [[ -f "VERSION" ]]; then
    current_version=$(cat VERSION)
    echo -e "当前版本: ${GREEN}$current_version${NC}"
fi

echo ""
echo -e "${BLUE}有用的命令:${NC}"
echo -e "  查看更改历史: ${YELLOW}git log --oneline -10${NC}"
echo -e "  查看文件差异: ${YELLOW}git diff HEAD~$COMMIT_COUNT${NC}"
echo -e "  回滚到之前:   ${YELLOW}./git_rollback.sh -l${NC}" 