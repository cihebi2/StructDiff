#!/bin/bash

# 脚本编码修复工具
# 用于修复Windows CRLF行尾字符问题

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== 脚本编码修复工具 ===${NC}"

if [[ $# -eq 0 ]]; then
    echo -e "${YELLOW}用法:${NC}"
    echo "  $0 <脚本文件>        # 修复单个文件"
    echo "  $0 --all            # 修复所有.sh文件"
    echo "  $0 --check <文件>    # 仅检查文件问题"
    exit 0
fi

# 检查文件是否有CRLF问题
check_file() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}错误: 文件 '$file' 不存在${NC}"
        return 1
    fi
    
    # 检查是否包含CRLF
    if grep -q $'\r' "$file"; then
        echo -e "${YELLOW}发现问题: '$file' 包含Windows CRLF行尾字符${NC}"
        return 0
    else
        echo -e "${GREEN}✓ '$file' 没有发现问题${NC}"
        return 1
    fi
}

# 修复文件
fix_file() {
    local file="$1"
    local backup="${file}.backup.$(date +%Y%m%d_%H%M%S)"
    
    echo -e "${BLUE}修复文件: $file${NC}"
    
    # 创建备份
    cp "$file" "$backup"
    echo -e "${GREEN}✓${NC} 备份已创建: $backup"
    
    # 使用tr命令删除回车符（更可靠）
    tr -d '\r' < "$backup" > "$file"
    
    # 确保脚本有执行权限
    chmod +x "$file"
    
    echo -e "${GREEN}✓${NC} 文件已修复并设置执行权限"
    
    # 验证修复结果
    if ! check_file "$file" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} 修复成功，文件现在应该可以正常运行"
    else
        echo -e "${RED}修复失败，请手动检查文件${NC}"
    fi
}

# 处理参数
case "$1" in
    --all)
        echo -e "${BLUE}检查和修复所有.sh文件...${NC}"
        found=false
        for file in *.sh; do
            if [[ -f "$file" ]]; then
                if check_file "$file"; then
                    fix_file "$file"
                    found=true
                fi
            fi
        done
        if [[ "$found" == false ]]; then
            echo -e "${GREEN}✓ 所有.sh文件都正常${NC}"
        fi
        ;;
    --check)
        if [[ -z "$2" ]]; then
            echo -e "${RED}错误: --check 需要指定文件名${NC}"
            exit 1
        fi
        check_file "$2"
        ;;
    *)
        file="$1"
        if check_file "$file"; then
            echo ""
            read -p "是否修复此文件？ [y/N]: " confirm
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                fix_file "$file"
            else
                echo "取消修复"
            fi
        fi
        ;;
esac 