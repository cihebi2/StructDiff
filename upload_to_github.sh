#!/bin/bash
# StructDiff GitHub上传脚本
# 自动化将项目上传到GitHub的脚本

set -e  # 遇到错误时退出

echo "🚀 开始上传StructDiff项目到GitHub..."

# 配置变量
GITHUB_TOKEN="YOUR_GITHUB_TOKEN_HERE"
REPO_NAME="StructDiff"
GITHUB_USERNAME="your-username"  # 请替换为您的GitHub用户名

# 检查是否在正确的目录
if [ ! -f "README.md" ]; then
    echo "❌ 错误：请在StructDiff项目根目录运行此脚本"
    exit 1
fi

# 初始化git仓库（如果还没有）
if [ ! -d ".git" ]; then
    echo "📦 初始化Git仓库..."
    git init
    git branch -M main
fi

# 检查远程仓库
if ! git remote | grep -q origin; then
    echo "🔗 添加远程仓库..."
    echo "请输入您的GitHub用户名："
    read -r GITHUB_USERNAME
    git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git
fi

# 配置git用户信息（如果未配置）
if [ -z "$(git config user.name)" ]; then
    echo "👤 配置Git用户信息..."
    echo "请输入您的姓名："
    read -r USER_NAME
    echo "请输入您的邮箱："
    read -r USER_EMAIL
    git config user.name "$USER_NAME"
    git config user.email "$USER_EMAIL"
fi

# 创建.gitignore文件
echo "📝 创建.gitignore文件..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyTorch
*.pth
*.pt
checkpoints/
outputs/
logs/

# Data
data/raw/
data/processed/
*.csv
*.json
*.pkl
*.h5
*.hdf5

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Evaluation results
evaluation_results/
generated_peptides.fasta
*_generated.fasta
*_demo.fasta

# Temporary files
temp/
tmp/
*.tmp
*.temp

# Model weights and large files
*.bin
*.safetensors
EOF

# 添加所有文件
echo "📁 添加文件到Git..."
git add .

# 检查状态
echo "📊 Git状态："
git status

# 创建提交
echo "💾 创建提交..."
git commit -m "集成CFG和长度采样器功能 - 完整版本

## 🎯 新增功能

### Classifier-Free Guidance (CFG)
- 实现与CPL-Diff论文一致的CFG机制
- 支持训练时条件丢弃和推理时引导采样
- 自适应引导强度和多级引导功能
- 性能优化的批量化CFG计算

### 长度分布采样器
- 支持5种分布类型（正态、均匀、Gamma、Beta、自定义）
- 条件相关的长度偏好设置
- 自适应长度控制和约束执行
- 温度控制的随机性调节

### CPL-Diff标准评估
- 完整的5个核心评估指标
- ESM-2伪困惑度、pLDDT、不稳定性指数、BLOSUM62相似性、活性预测
- 智能依赖检测和fallback机制
- 与原论文完全一致的评估标准

### AlphaFold3优化组件  
- AF3风格的时间步嵌入和自适应调节
- 增强的条件层归一化
- GLU激活和零初始化输出层
- 多方面自适应条件控制

## 📁 新增文件

### 核心实现
- structdiff/models/classifier_free_guidance.py
- structdiff/sampling/length_sampler.py
- scripts/cpldiff_standard_evaluation.py

### 集成和演示
- scripts/cfg_length_integrated_sampling.py
- configs/cfg_length_config.yaml
- tests/test_cfg_length_integration.py

### 文档
- CFG_LENGTH_INTEGRATION_GUIDE.md
- CPL_DIFF_EVALUATION_GUIDE.md
- EVALUATION_INTEGRATION_README.md

## 🛠️ 技术特性

- **精确控制**: CFG提供精确的条件生成控制
- **长度定制**: 灵活的长度分布和约束机制
- **高质量生成**: 显著提升生成序列的功能特异性
- **标准评估**: 与最新研究保持一致的评估标准
- **性能优化**: 高效实现减少计算开销

## 🚀 使用方法

\`\`\`bash
# CFG+长度控制演示
python scripts/cfg_length_integrated_sampling.py --num_samples 100

# CPL-Diff标准评估
python demo_cpldiff_evaluation.py

# 运行测试套件
python tests/test_cfg_length_integration.py
\`\`\`

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 推送到GitHub
echo "🚀 推送到GitHub..."
echo "正在使用提供的访问令牌进行认证..."

# 设置临时的认证信息
git config --local credential.helper store
echo "https://$GITHUB_TOKEN@github.com" > ~/.git-credentials

# 推送
if git push -u origin main; then
    echo "✅ 项目成功上传到GitHub!"
    echo "🔗 项目地址: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
else
    echo "❌ 推送失败，请检查："
    echo "   1. GitHub仓库是否已创建"
    echo "   2. 访问令牌是否有效"
    echo "   3. 用户名是否正确"
fi

# 清理认证信息
rm -f ~/.git-credentials

echo "🎉 上传流程完成!"