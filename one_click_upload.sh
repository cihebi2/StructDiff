#!/bin/bash
# StructDiff一键上传到GitHub脚本 (Linux/Mac)

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================"
echo -e "   StructDiff GitHub 一键上传工具"
echo -e "========================================${NC}"
echo

# 检查是否在正确目录
if [ ! -f "README.md" ]; then
    echo -e "${RED}❌ 错误: 请在StructDiff项目根目录运行此脚本${NC}"
    exit 1
fi

# 检查Git是否安装
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git未安装，请先安装Git${NC}"
    exit 1
fi

echo -e "${GREEN}🚀 开始上传StructDiff项目到GitHub...${NC}"
echo

# 获取用户输入
read -p "请输入您的GitHub用户名: " GITHUB_USERNAME
echo

# 检查是否已经是git仓库
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}📦 初始化Git仓库...${NC}"
    git init
    git branch -M main
else
    echo -e "${YELLOW}📦 使用现有Git仓库...${NC}"
fi

# 配置远程仓库
echo -e "${YELLOW}🔗 配置远程仓库...${NC}"
git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/$GITHUB_USERNAME/StructDiff.git

# 配置Git用户信息
if [ -z "$(git config user.name)" ]; then
    echo -e "${YELLOW}👤 配置Git用户信息...${NC}"
    read -p "请输入您的姓名: " USER_NAME
    read -p "请输入您的邮箱: " USER_EMAIL
    git config user.name "$USER_NAME"
    git config user.email "$USER_EMAIL"
else
    echo -e "${GREEN}👤 使用现有Git用户配置: $(git config user.name)${NC}"
fi

# 创建.gitignore文件
if [ ! -f ".gitignore" ]; then
    echo -e "${YELLOW}📝 创建.gitignore文件...${NC}"
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
fi

# 添加文件
echo -e "${YELLOW}📁 添加所有文件到Git...${NC}"
git add .

# 检查是否有文件被添加
if git diff --cached --quiet; then
    echo -e "${YELLOW}⚠️ 没有新文件需要提交${NC}"
else
    echo -e "${GREEN}📊 将要提交的文件:${NC}"
    git diff --cached --name-status | head -20
    if [ "$(git diff --cached --name-status | wc -l)" -gt 20 ]; then
        echo "... 和其他 $(($(git diff --cached --name-status | wc -l) - 20)) 个文件"
    fi
    echo
fi

# 创建提交
echo -e "${YELLOW}💾 创建提交...${NC}"
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

### 模型实现
- \`structdiff/models/classifier_free_guidance.py\` - CFG核心实现
- \`structdiff/sampling/length_sampler.py\` - 长度分布采样器
- \`structdiff/models/denoise.py\` - 增强的去噪器

### 评估系统
- \`scripts/cpldiff_standard_evaluation.py\` - CPL-Diff标准评估
- \`demo_cpldiff_evaluation.py\` - 评估演示脚本

### 集成工具
- \`scripts/cfg_length_integrated_sampling.py\` - 集成采样演示
- \`configs/cfg_length_config.yaml\` - 完整配置文件
- \`tests/test_cfg_length_integration.py\` - 测试套件

### 文档指南
- \`CFG_LENGTH_INTEGRATION_GUIDE.md\` - CFG和长度控制指南
- \`CPL_DIFF_EVALUATION_GUIDE.md\` - 评估套件指南
- \`EVALUATION_INTEGRATION_README.md\` - 评估集成说明

## 🛠️ 技术特性

- **精确控制**: CFG提供精确的条件生成控制
- **长度定制**: 灵活的长度分布和约束机制
- **高质量生成**: 显著提升生成序列的功能特异性
- **标准评估**: 与最新研究保持一致的评估标准
- **性能优化**: 高效实现减少计算开销

## 🚀 快速开始

\`\`\`bash
# CFG+长度控制集成演示
python scripts/cfg_length_integrated_sampling.py --num_samples 100

# CPL-Diff标准评估演示
python demo_cpldiff_evaluation.py

# 运行完整测试套件
python tests/test_cfg_length_integration.py
\`\`\`

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>" || echo -e "${YELLOW}⚠️ 没有新的更改需要提交${NC}"

echo
echo -e "${BLUE}🚀 推送到GitHub...${NC}"
echo -e "${YELLOW}注意: 接下来会要求输入GitHub认证信息${NC}"
echo -e "用户名: ${GREEN}$GITHUB_USERNAME${NC}"
echo -e "密码: ${GREEN}请使用您的GitHub Personal Access Token${NC}"
echo

# 尝试推送
if git push -u origin main; then
    echo
    echo -e "${GREEN}✅ 项目成功上传到GitHub!${NC}"
    echo -e "${BLUE}🔗 项目地址: https://github.com/$GITHUB_USERNAME/StructDiff${NC}"
    echo
    echo -e "${GREEN}🎉 上传完成! 您现在可以在GitHub上查看您的项目了。${NC}"
    
    # 显示后续建议
    echo
    echo -e "${YELLOW}📋 建议的后续操作:${NC}"
    echo "1. 在GitHub上为仓库添加描述和标签"
    echo "2. 启用Issues和Wiki（如果需要）"
    echo "3. 邀请协作者（如果有）"
    echo "4. 设置分支保护规则"
    
else
    echo
    echo -e "${RED}❌ 推送失败，尝试解决方案...${NC}"
    echo -e "${YELLOW}🔄 可能的解决方案:${NC}"
    echo "1. 确保GitHub上已创建StructDiff仓库"
    echo "2. 检查网络连接"
    echo "3. 确认访问令牌权限"
    echo
    echo -e "${YELLOW}🔧 手动推送命令:${NC}"
    echo -e "${BLUE}git remote set-url origin https://YOUR_TOKEN@github.com/$GITHUB_USERNAME/StructDiff.git${NC}"
    echo -e "${BLUE}git push -u origin main${NC}"
    echo
    exit 1
fi

echo