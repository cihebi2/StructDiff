#!/bin/bash
# StructDiff项目备份脚本 (Linux/Mac版本)
# 创建日期：2025-08-04

echo "========================================"
echo "StructDiff项目备份工具"
echo "========================================"
echo

# 获取当前日期时间
backup_date=$(date +%Y%m%d_%H%M%S)

# 设置备份目录名
backup_name="StructDiff-8.0.0_backup_${backup_date}"
backup_path="../${backup_name}"

echo "备份目录: ${backup_path}"
echo

# 创建备份目录
if [ ! -d "${backup_path}" ]; then
    mkdir -p "${backup_path}"
    echo "创建备份目录..."
else
    echo "备份目录已存在，将覆盖..."
fi

echo
echo "开始备份文件..."
echo

# 使用rsync进行备份（排除大文件和临时文件）
rsync -av --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='outputs' \
    --exclude='logs' \
    --exclude='cache' \
    --exclude='data' \
    --exclude='.pytest_cache' \
    --exclude='*.pyc' \
    --exclude='*.pth' \
    --exclude='*.pkl' \
    --exclude='*.log' \
    . "${backup_path}/"

echo
echo "========================================"
echo "备份完成！"
echo "备份位置: ${backup_path}"
echo "========================================"
echo

# 创建备份信息文件
cat > "${backup_path}/BACKUP_INFO.txt" << EOF
备份信息
========
备份时间: $(date)
原始路径: $(pwd)
备份原因: 架构改进前的完整备份

排除的目录: .git, __pycache__, outputs, logs, cache, data, .pytest_cache
排除的文件: *.pyc, *.pth, *.pkl, *.log
EOF

echo "备份信息已保存到: ${backup_path}/BACKUP_INFO.txt"