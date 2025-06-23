@echo off
REM StructDiff一键上传到GitHub脚本 (Windows)
echo.
echo ========================================
echo    StructDiff GitHub 一键上传工具
echo ========================================
echo.

REM 检查是否在正确目录
if not exist "README.md" (
    echo ❌ 错误: 请在StructDiff项目根目录运行此脚本
    pause
    exit /b 1
)

echo 🚀 开始上传StructDiff项目到GitHub...
echo.

REM 获取用户输入
set /p GITHUB_USERNAME="请输入您的GitHub用户名: "
echo.

echo 📦 初始化Git仓库...
git init
if errorlevel 1 (
    echo ❌ Git初始化失败，请确保已安装Git
    pause
    exit /b 1
)

git branch -M main

echo 🔗 配置远程仓库...
git remote remove origin 2>nul
git remote add origin https://github.com/%GITHUB_USERNAME%/StructDiff.git

echo 👤 配置Git用户信息...
set /p USER_NAME="请输入您的姓名: "
set /p USER_EMAIL="请输入您的邮箱: "
git config user.name "%USER_NAME%"
git config user.email "%USER_EMAIL%"

echo.
echo 📝 创建.gitignore文件...
(
echo # Python
echo __pycache__/
echo *.py[cod]
echo *$py.class
echo *.so
echo .Python
echo build/
echo develop-eggs/
echo dist/
echo downloads/
echo eggs/
echo .eggs/
echo lib/
echo lib64/
echo parts/
echo sdist/
echo var/
echo wheels/
echo *.egg-info/
echo .installed.cfg
echo *.egg
echo MANIFEST
echo.
echo # PyTorch
echo *.pth
echo *.pt
echo checkpoints/
echo outputs/
echo logs/
echo.
echo # Data
echo data/raw/
echo data/processed/
echo *.csv
echo *.json
echo *.pkl
echo *.h5
echo *.hdf5
echo.
echo # Jupyter Notebook
echo .ipynb_checkpoints
echo.
echo # Environment
echo .env
echo .venv
echo env/
echo venv/
echo ENV/
echo env.bak/
echo venv.bak/
echo.
echo # IDE
echo .vscode/
echo .idea/
echo *.swp
echo *.swo
echo *~
echo.
echo # OS
echo .DS_Store
echo Thumbs.db
echo.
echo # Evaluation results
echo evaluation_results/
echo generated_peptides.fasta
echo *_generated.fasta
echo *_demo.fasta
echo.
echo # Temporary files
echo temp/
echo tmp/
echo *.tmp
echo *.temp
echo.
echo # Model weights and large files
echo *.bin
echo *.safetensors
) > .gitignore

echo 📁 添加所有文件到Git...
git add .

echo 💾 创建提交...
git commit -m "集成CFG和长度采样器功能 - 完整版本

## 🎯 新增功能

### Classifier-Free Guidance (CFG)
- ✅ 实现与CPL-Diff论文一致的CFG机制
- ✅ 支持训练时条件丢弃和推理时引导采样
- ✅ 自适应引导强度和多级引导功能
- ✅ 性能优化的批量化CFG计算

### 长度分布采样器
- ✅ 支持5种分布类型（正态、均匀、Gamma、Beta、自定义）
- ✅ 条件相关的长度偏好设置
- ✅ 自适应长度控制和约束执行
- ✅ 温度控制的随机性调节

### CPL-Diff标准评估
- ✅ 完整的5个核心评估指标
- ✅ ESM-2伪困惑度、pLDDT、不稳定性指数、BLOSUM62相似性、活性预测
- ✅ 智能依赖检测和fallback机制
- ✅ 与原论文完全一致的评估标准

### AlphaFold3优化组件  
- ✅ AF3风格的时间步嵌入和自适应调节
- ✅ 增强的条件层归一化
- ✅ GLU激活和零初始化输出层
- ✅ 多方面自适应条件控制

## 📁 核心文件

- structdiff/models/classifier_free_guidance.py
- structdiff/sampling/length_sampler.py
- scripts/cpldiff_standard_evaluation.py
- scripts/cfg_length_integrated_sampling.py
- configs/cfg_length_config.yaml
- tests/test_cfg_length_integration.py

## 📖 文档

- CFG_LENGTH_INTEGRATION_GUIDE.md
- CPL_DIFF_EVALUATION_GUIDE.md
- EVALUATION_INTEGRATION_README.md

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

echo.
echo 🚀 推送到GitHub...
echo 注意: 接下来会要求输入GitHub认证信息
echo 用户名: %GITHUB_USERNAME%
echo 密码: 请使用您的GitHub Personal Access Token
echo.

git push -u origin main

if errorlevel 1 (
    echo.
    echo ❌ 推送失败，尝试解决方案...
    echo 🔄 可能的解决方案:
    echo 1. 确保GitHub上已创建StructDiff仓库
    echo 2. 检查网络连接
    echo 3. 确认访问令牌权限
    echo.
    echo 🔧 手动推送命令:
    echo git remote set-url origin https://YOUR_TOKEN@github.com/%GITHUB_USERNAME%/StructDiff.git
    echo git push -u origin main
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ 项目成功上传到GitHub!
echo 🔗 项目地址: https://github.com/%GITHUB_USERNAME%/StructDiff
echo.
echo 🎉 上传完成! 您现在可以在GitHub上查看您的项目了。
echo.
pause