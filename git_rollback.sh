#!/bin/bash

# StructDiff Git回滚脚本
# 方便快速回滚到任何版本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo -e "${BLUE}StructDiff Git 回滚工具${NC}"
    echo ""
    echo "用法: $0 [选项] [版本号]"
    echo ""
    echo "选项:"
    echo "  -l, --list          列出所有版本"
    echo "  -s, --soft          软回滚（保留工作区修改）"
    echo "  -h, --hard          硬回滚（丢弃所有修改）"
    echo "  -p, --previous      回滚到上一版本"
    echo "  -c, --current       显示当前版本"
    echo "  --help              显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -l                    # 列出所有版本"
    echo "  $0 -c                    # 显示当前版本"
    echo "  $0 v1.2.3               # 回滚到指定版本"
    echo "  $0 -p                    # 回滚到上一版本"
    echo "  $0 -h v1.2.3            # 硬回滚到指定版本"
    echo ""
}

# 列出所有版本
list_versions() {
    echo -e "${BLUE}=== 所有版本 ===${NC}"
    echo ""
    
    # 获取当前版本
    current_commit=$(git rev-parse HEAD)
    
    # 列出所有标签，按版本排序
    git tag -l "v*" | sort -V | while read tag; do
        tag_commit=$(git rev-parse "$tag^{}")
        tag_date=$(git log -1 --format="%ci" "$tag")
        tag_message=$(git tag -l --format='%(contents:subject)' "$tag")
        
        if [[ "$tag_commit" == "$current_commit" ]]; then
            echo -e "${GREEN}→ $tag${NC} (当前版本)"
        else
            echo -e "  $tag"
        fi
        echo -e "    时间: ${YELLOW}$tag_date${NC}"
        echo -e "    说明: $tag_message"
        echo ""
    done
}

# 显示当前版本
show_current() {
    current_tag=$(git describe --tags --exact-match HEAD 2>/dev/null || echo "未标记")
    current_commit=$(git rev-parse --short HEAD)
    current_branch=$(git branch --show-current)
    
    echo -e "${BLUE}=== 当前状态 ===${NC}"
    echo -e "版本标签: ${GREEN}$current_tag${NC}"
    echo -e "提交哈希: ${YELLOW}$current_commit${NC}"
    echo -e "当前分支: ${BLUE}$current_branch${NC}"
    
    # 显示工作区状态
    if git diff-index --quiet HEAD --; then
        echo -e "工作区:   ${GREEN}干净${NC}"
    else
        echo -e "工作区:   ${YELLOW}有未提交的修改${NC}"
    fi
}

# 回滚到指定版本
rollback_to_version() {
    local version=$1
    local rollback_type=$2
    
    # 检查版本是否存在
    if ! git rev-parse "$version" >/dev/null 2>&1; then
        echo -e "${RED}错误: 版本 '$version' 不存在${NC}"
        echo "使用 -l 查看所有可用版本"
        exit 1
    fi
    
    # 显示版本信息
    echo -e "${BLUE}=== 回滚信息 ===${NC}"
    echo -e "目标版本: ${GREEN}$version${NC}"
    echo -e "回滚类型: ${YELLOW}$rollback_type${NC}"
    
    # 显示版本详情
    if git tag -l | grep -q "^$version$"; then
        tag_date=$(git log -1 --format="%ci" "$version")
        tag_message=$(git tag -l --format='%(contents:subject)' "$version")
        echo -e "版本时间: $tag_date"
        echo -e "版本说明: $tag_message"
    fi
    
    echo ""
    
    # 警告信息
    if [[ "$rollback_type" == "hard" ]]; then
        echo -e "${RED}警告: 硬回滚将丢失所有未提交的修改!${NC}"
    fi
    
    # 确认操作
    read -p "确认回滚？ [y/N]: " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "取消回滚"
        exit 0
    fi
    
    # 执行回滚
    case $rollback_type in
        "soft")
            git reset --soft "$version"
            echo -e "${GREEN}✓${NC} 软回滚完成（工作区修改已保留）"
            ;;
        "hard")
            git reset --hard "$version"
            echo -e "${GREEN}✓${NC} 硬回滚完成（所有修改已丢弃）"
            ;;
        *)
            # 默认检出到指定版本
            git checkout "$version"
            echo -e "${GREEN}✓${NC} 已切换到版本 $version"
            echo -e "${YELLOW}注意: 当前处于分离HEAD状态${NC}"
            echo -e "要回到主分支，请运行: ${BLUE}git checkout main${NC}"
            ;;
    esac
}

# 回滚到上一版本
rollback_to_previous() {
    local rollback_type=$1
    
    # 获取当前版本
    current_tag=$(git describe --tags --exact-match HEAD 2>/dev/null)
    
    if [[ -z "$current_tag" ]]; then
        echo -e "${RED}错误: 当前版本未标记，无法确定上一版本${NC}"
        exit 1
    fi
    
    # 获取上一个版本标签
    previous_tag=$(git tag -l "v*" | sort -V | grep -B1 "^$current_tag$" | head -1)
    
    if [[ -z "$previous_tag" || "$previous_tag" == "$current_tag" ]]; then
        echo -e "${RED}错误: 没有找到上一版本${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}当前版本: ${GREEN}$current_tag${NC}"
    echo -e "${BLUE}上一版本: ${GREEN}$previous_tag${NC}"
    
    rollback_to_version "$previous_tag" "$rollback_type"
}

# 参数解析
ACTION=""
VERSION=""
ROLLBACK_TYPE="checkout"

while [[ $# -gt 0 ]]; do
    case $1 in
        -l|--list)
            ACTION="list"
            shift
            ;;
        -c|--current)
            ACTION="current"
            shift
            ;;
        -s|--soft)
            ROLLBACK_TYPE="soft"
            shift
            ;;
        -h|--hard)
            ROLLBACK_TYPE="hard"
            shift
            ;;
        -p|--previous)
            ACTION="previous"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        v*)
            VERSION="$1"
            ACTION="rollback"
            shift
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 执行操作
case $ACTION in
    "list")
        list_versions
        ;;
    "current")
        show_current
        ;;
    "rollback")
        if [[ -z "$VERSION" ]]; then
            echo -e "${RED}错误: 请指定版本号${NC}"
            exit 1
        fi
        rollback_to_version "$VERSION" "$ROLLBACK_TYPE"
        ;;
    "previous")
        rollback_to_previous "$ROLLBACK_TYPE"
        ;;
    *)
        show_help
        ;;
esac 