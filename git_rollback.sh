#!/bin/bash

# Git 回滚操作脚本
# 提供多种 Git 回滚和撤销操作的便捷工具

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 显示帮助信息
show_help() {
    echo -e "${BLUE}Git 回滚操作脚本 - 使用说明${NC}"
    echo -e "${BLUE}===============================${NC}"
    echo ""
    echo -e "${GREEN}用法:${NC} $0 [选项] [参数]"
    echo ""
    echo -e "${GREEN}选项:${NC}"
    echo -e "  ${YELLOW}-h, --help${NC}              显示此帮助信息"
    echo -e "  ${YELLOW}-l, --list [数量]${NC}        列出最近的提交历史 (默认显示10条)"
    echo -e "  ${YELLOW}-c, --commit <hash>${NC}     回滚到指定的提交"
    echo -e "  ${YELLOW}-s, --soft <hash>${NC}       软回滚到指定提交 (保留更改在暂存区)"
    echo -e "  ${YELLOW}-m, --mixed <hash>${NC}      混合回滚到指定提交 (保留更改在工作区)"
    echo -e "  ${YELLOW}-H, --hard <hash>${NC}       硬回滚到指定提交 (丢弃所有更改)"
    echo -e "  ${YELLOW}-r, --revert <hash>${NC}     创建新提交来撤销指定提交"
    echo -e "  ${YELLOW}-u, --unstage [文件]${NC}    取消暂存文件 (默认取消所有)"
    echo -e "  ${YELLOW}-d, --discard [文件]${NC}    丢弃工作区更改 (默认丢弃所有)"
    echo -e "  ${YELLOW}-b, --branch <分支名>${NC}   回滚到指定分支的最新提交"
    echo -e "  ${YELLOW}-i, --interactive${NC}       交互式回滚模式"
    echo ""
    echo -e "${GREEN}示例:${NC}"
    echo -e "  $0 -l 20                     # 显示最近20条提交"
    echo -e "  $0 -c abc1234                # 回滚到提交 abc1234"
    echo -e "  $0 -s HEAD~1                 # 软回滚到上一个提交"
    echo -e "  $0 -H HEAD~2                 # 硬回滚到前两个提交"
    echo -e "  $0 -r abc1234                # 撤销提交 abc1234"
    echo -e "  $0 -u                        # 取消所有暂存的文件"
    echo -e "  $0 -d file.txt               # 丢弃 file.txt 的更改"
    echo -e "  $0 -b main                   # 回滚到 main 分支"
    echo -e "  $0 -i                        # 交互式模式"
    echo ""
}

# 检查是否在 Git 仓库中
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo -e "${RED}错误: 当前目录不是 Git 仓库${NC}"
        exit 1
    fi
}

# 显示提交历史
list_commits() {
    local count=${1:-10}
    echo -e "${BLUE}最近 $count 条提交历史:${NC}"
    echo -e "${BLUE}============================${NC}"
    GIT_PAGER=cat git log --oneline --graph --decorate -n "$count"
}

# 确认操作
confirm_action() {
    local message="$1"
    echo -e "${YELLOW}$message${NC}"
    echo -n "确认执行? (y/N): "
    read -r reply
    if [[ ! $reply =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}操作已取消${NC}"
        exit 0
    fi
}

# 回滚到指定提交
rollback_to_commit() {
    local commit_hash="$1"
    local reset_type="${2:-mixed}"
    
    if [ -z "$commit_hash" ]; then
        echo -e "${RED}错误: 请指定提交哈希${NC}"
        exit 1
    fi
    
    # 验证提交是否存在
    if ! git rev-parse --verify "$commit_hash" >/dev/null 2>&1; then
        echo -e "${RED}错误: 提交 '$commit_hash' 不存在${NC}"
        exit 1
    fi
    
    confirm_action "将要执行 $reset_type 回滚到提交: $commit_hash"
    
    case "$reset_type" in
        "soft")
            git reset --soft "$commit_hash"
            ;;
        "mixed")
            git reset --mixed "$commit_hash"
            ;;
        "hard")
            git reset --hard "$commit_hash"
            ;;
        *)
            git reset --mixed "$commit_hash"
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}成功回滚到提交: $commit_hash${NC}"
    else
        echo -e "${RED}回滚失败${NC}"
        exit 1
    fi
}

# 撤销指定提交
revert_commit() {
    local commit_hash="$1"
    
    if [ -z "$commit_hash" ]; then
        echo -e "${RED}错误: 请指定提交哈希${NC}"
        exit 1
    fi
    
    confirm_action "将要撤销提交: $commit_hash"
    
    git revert "$commit_hash" --no-edit
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}成功撤销提交: $commit_hash${NC}"
    else
        echo -e "${RED}撤销失败${NC}"
        exit 1
    fi
}

# 取消暂存文件
unstage_files() {
    local files="$1"
    
    if [ -z "$files" ]; then
        confirm_action "将要取消所有暂存的文件"
        git reset HEAD
    else
        confirm_action "将要取消暂存文件: $files"
        git reset HEAD "$files"
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}成功取消暂存${NC}"
    else
        echo -e "${RED}取消暂存失败${NC}"
        exit 1
    fi
}

# 丢弃工作区更改
discard_changes() {
    local files="$1"
    
    if [ -z "$files" ]; then
        confirm_action "将要丢弃所有工作区更改"
        git checkout -- .
    else
        confirm_action "将要丢弃文件更改: $files"
        git checkout -- "$files"
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}成功丢弃更改${NC}"
    else
        echo -e "${RED}丢弃更改失败${NC}"
        exit 1
    fi
}

# 回滚到分支
rollback_to_branch() {
    local branch="$1"
    
    if [ -z "$branch" ]; then
        echo -e "${RED}错误: 请指定分支名${NC}"
        exit 1
    fi
    
    confirm_action "将要切换到分支: $branch"
    
    git checkout "$branch"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}成功切换到分支: $branch${NC}"
    else
        echo -e "${RED}切换分支失败${NC}"
        exit 1
    fi
}

# 交互式模式
interactive_mode() {
    echo -e "${BLUE}Git 回滚交互式模式${NC}"
    echo -e "${BLUE}=====================${NC}"
    
    while true; do
        echo ""
        echo -e "${GREEN}请选择操作:${NC}"
        echo "1. 查看提交历史"
        echo "2. 软回滚到指定提交"
        echo "3. 混合回滚到指定提交"
        echo "4. 硬回滚到指定提交"
        echo "5. 撤销指定提交"
        echo "6. 取消暂存文件"
        echo "7. 丢弃工作区更改"
        echo "8. 切换分支"
        echo "0. 退出"
        
        echo -n "请选择 (0-8): "
        read -r choice
        
        case $choice in
            1)
                echo -n "显示多少条提交? (默认10): "
                read -r count
                list_commits "${count:-10}"
                ;;
            2)
                echo -n "请输入提交哈希: "
                read -r hash
                rollback_to_commit "$hash" "soft"
                ;;
            3)
                echo -n "请输入提交哈希: "
                read -r hash
                rollback_to_commit "$hash" "mixed"
                ;;
            4)
                echo -n "请输入提交哈希: "
                read -r hash
                rollback_to_commit "$hash" "hard"
                ;;
            5)
                echo -n "请输入提交哈希: "
                read -r hash
                revert_commit "$hash"
                ;;
            6)
                echo -n "请输入文件名 (留空表示所有): "
                read -r files
                unstage_files "$files"
                ;;
            7)
                echo -n "请输入文件名 (留空表示所有): "
                read -r files
                discard_changes "$files"
                ;;
            8)
                echo -n "请输入分支名: "
                read -r branch
                rollback_to_branch "$branch"
                ;;
            0)
                echo -e "${GREEN}退出交互式模式${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}无效选择${NC}"
                ;;
        esac
    done
}

# 主函数
main() {
    # 检查是否在 Git 仓库中
    check_git_repo
    
    # 如果没有参数，显示帮助
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -l|--list)
                count="$2"
                if [[ "$count" =~ ^[0-9]+$ ]]; then
                    shift
                else
                    count=10
                fi
                list_commits "$count"
                exit 0
                ;;
            -c|--commit)
                if [ -z "$2" ]; then
                    echo -e "${RED}错误: -c 选项需要提交哈希参数${NC}"
                    exit 1
                fi
                rollback_to_commit "$2" "mixed"
                shift
                exit 0
                ;;
            -s|--soft)
                if [ -z "$2" ]; then
                    echo -e "${RED}错误: -s 选项需要提交哈希参数${NC}"
                    exit 1
                fi
                rollback_to_commit "$2" "soft"
                shift
                exit 0
                ;;
            -m|--mixed)
                if [ -z "$2" ]; then
                    echo -e "${RED}错误: -m 选项需要提交哈希参数${NC}"
                    exit 1
                fi
                rollback_to_commit "$2" "mixed"
                shift
                exit 0
                ;;
            -H|--hard)
                if [ -z "$2" ]; then
                    echo -e "${RED}错误: -H 选项需要提交哈希参数${NC}"
                    exit 1
                fi
                rollback_to_commit "$2" "hard"
                shift
                exit 0
                ;;
            -r|--revert)
                if [ -z "$2" ]; then
                    echo -e "${RED}错误: -r 选项需要提交哈希参数${NC}"
                    exit 1
                fi
                revert_commit "$2"
                shift
                exit 0
                ;;
            -u|--unstage)
                unstage_files "$2"
                if [ -n "$2" ]; then
                    shift
                fi
                exit 0
                ;;
            -d|--discard)
                discard_changes "$2"
                if [ -n "$2" ]; then
                    shift
                fi
                exit 0
                ;;
            -b|--branch)
                if [ -z "$2" ]; then
                    echo -e "${RED}错误: -b 选项需要分支名参数${NC}"
                    exit 1
                fi
                rollback_to_branch "$2"
                shift
                exit 0
                ;;
            -i|--interactive)
                interactive_mode
                exit 0
                ;;
            *)
                echo -e "${RED}未知选项: $1${NC}"
                show_help
                exit 1
                ;;
        esac
        shift
    done
}

# 运行主函数
main "$@" 