@echo off
REM StructDiff项目备份脚本
REM 创建日期：2025-08-04

echo ========================================
echo StructDiff项目备份工具
echo ========================================
echo.

REM 获取当前日期时间
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set backup_date=%datetime:~0,8%_%datetime:~8,6%

REM 设置备份目录名
set backup_name=StructDiff-8.0.0_backup_%backup_date%
set backup_path=..\%backup_name%

echo 备份目录: %backup_path%
echo.

REM 创建备份目录
if not exist "%backup_path%" (
    mkdir "%backup_path%"
    echo 创建备份目录...
) else (
    echo 备份目录已存在，将覆盖...
)

echo.
echo 开始备份文件...
echo.

REM 使用robocopy进行备份（排除大文件和临时文件）
robocopy . "%backup_path%" /E /XD .git __pycache__ outputs logs cache data .pytest_cache /XF *.pyc *.pth *.pkl *.log /NFL /NDL

echo.
echo ========================================
echo 备份完成！
echo 备份位置: %backup_path%
echo ========================================
echo.

REM 创建备份信息文件
echo 备份信息 > "%backup_path%\BACKUP_INFO.txt"
echo ======== >> "%backup_path%\BACKUP_INFO.txt"
echo 备份时间: %date% %time% >> "%backup_path%\BACKUP_INFO.txt"
echo 原始路径: %cd% >> "%backup_path%\BACKUP_INFO.txt"
echo 备份原因: 架构改进前的完整备份 >> "%backup_path%\BACKUP_INFO.txt"
echo. >> "%backup_path%\BACKUP_INFO.txt"
echo 排除的目录: .git, __pycache__, outputs, logs, cache, data, .pytest_cache >> "%backup_path%\BACKUP_INFO.txt"
echo 排除的文件: *.pyc, *.pth, *.pkl, *.log >> "%backup_path%\BACKUP_INFO.txt"

pause