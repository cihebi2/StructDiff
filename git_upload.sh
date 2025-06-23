#!/bin/bash

# StructDiff Git上传脚本
# 支持自动版本管理和完整代码上传

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 版本号文件
VERSION_FILE="VERSION"

# 获取当前版本号
get_current_version() {
    if [[ -f "$VERSION_FILE" ]]; then
        cat "$VERSION_FILE"
    else
        echo "1.0.0"
    fi
}

# 增加版本号
increment_version() {
    local version=$1
    local type=$2
    
    IFS='.' read -r major minor patch <<< "$version"
    
    case $type in
        "major")
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        "minor")
            minor=$((minor + 1))
            patch=0
            ;;
        "patch"|*)
            patch=$((patch + 1))
            ;;
    esac
    
    echo "$major.$minor.$patch"
}

# 显示帮助信息
show_help() {
    echo -e "${BLUE}StructDiff Git 上传工具${NC}"
    echo ""
    echo "用法: $0 [选项] <提交信息>"
    echo ""
    echo "选项:"
    echo "  -t, --type TYPE     版本类型 (major|minor|patch) [默认: patch]"
    echo "  -m, --message MSG   提交信息"
    echo "  -f, --force         强制推送"
    echo "  -b, --branch BRANCH 指定分支 [默认: main]"
    echo "  -h, --help          显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -t minor -m \"添加新功能\""
    echo "  $0 -t patch -m \"修复bug\""
    echo "  $0 -t major -m \"重大更新\""
    echo ""
}

# 参数解析
VERSION_TYPE="patch"
COMMIT_MESSAGE=""
FORCE_PUSH=false
BRANCH="main"

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            VERSION_TYPE="$2"
            shift 2
            ;;
        -m|--message)
            COMMIT_MESSAGE="$2"
            shift 2
            ;;
        -f|--force)
            FORCE_PUSH=true
            shift
            ;;
        -b|--branch)
            BRANCH="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            if [[ -z "$COMMIT_MESSAGE" ]]; then
                COMMIT_MESSAGE="$1"
            fi
            shift
            ;;
    esac
done

# 检查提交信息
if [[ -z "$COMMIT_MESSAGE" ]]; then
    echo -e "${RED}错误: 请提供提交信息${NC}"
    echo "使用 -h 查看帮助信息"
    exit 1
fi

# 获取当前版本并计算新版本
CURRENT_VERSION=$(get_current_version)
NEW_VERSION=$(increment_version "$CURRENT_VERSION" "$VERSION_TYPE")

echo -e "${BLUE}=== StructDiff Git 上传 ===${NC}"
echo -e "当前版本: ${YELLOW}$CURRENT_VERSION${NC}"
echo -e "新版本:   ${GREEN}$NEW_VERSION${NC}"
echo -e "分支:     ${BLUE}$BRANCH${NC}"
echo -e "提交信息: ${YELLOW}$COMMIT_MESSAGE${NC}"
echo ""

# 确认操作
read -p "确认上传？ [y/N]: " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "取消上传"
    exit 0
fi

echo -e "${BLUE}开始上传过程...${NC}"

# 1. 更新版本号文件
echo "$NEW_VERSION" > "$VERSION_FILE"
echo -e "${GREEN}✓${NC} 版本号已更新为 $NEW_VERSION"

# 2. 添加所有文件
echo -e "${BLUE}添加文件...${NC}"
git add .
echo -e "${GREEN}✓${NC} 所有文件已添加"

# 3. 检查状态
echo -e "${BLUE}检查Git状态...${NC}"
git status --short

# 4. 提交
FULL_COMMIT_MESSAGE="v$NEW_VERSION: $COMMIT_MESSAGE

详细信息:
- 版本: $NEW_VERSION ($VERSION_TYPE 版本更新)
- 时间: $(date '+%Y-%m-%d %H:%M:%S')
- 分支: $BRANCH

更新内容:
$COMMIT_MESSAGE"

git commit -m "$FULL_COMMIT_MESSAGE"
echo -e "${GREEN}✓${NC} 代码已提交"

# 5. 创建标签
TAG_NAME="v$NEW_VERSION"
git tag -a "$TAG_NAME" -m "Release $NEW_VERSION: $COMMIT_MESSAGE"
echo -e "${GREEN}✓${NC} 标签 $TAG_NAME 已创建"

# 6. 推送到远程
if [[ "$FORCE_PUSH" == true ]]; then
    echo -e "${YELLOW}强制推送代码和标签...${NC}"
    git push --force origin "$BRANCH"
    git push --force origin --tags
else
    echo -e "${BLUE}推送代码和标签...${NC}"
    git push origin "$BRANCH"
    git push origin --tags
fi

echo -e "${GREEN}✓${NC} 代码和标签已推送到远程仓库"

# 7. 显示完成信息
echo ""
echo -e "${GREEN}=== 上传完成! ===${NC}"
echo -e "版本:     ${GREEN}$NEW_VERSION${NC}"
echo -e "标签:     ${GREEN}$TAG_NAME${NC}"
echo -e "GitHub:   ${BLUE}https://github.com/cihebi2/StructDiff/releases/tag/$TAG_NAME${NC}"
echo ""
echo -e "${YELLOW}回滚命令:${NC}"
echo -e "  回滚到上一版本: ${BLUE}git reset --hard $TAG_NAME^${NC}"
echo -e "  查看所有版本:   ${BLUE}git tag -l${NC}"
echo -e "  切换到特定版本: ${BLUE}git checkout $TAG_NAME${NC}" 