#!/bin/bash
# 使用GitHub CLI上传项目的脚本

echo "🚀 使用GitHub CLI上传StructDiff项目..."

# 检查GitHub CLI是否安装
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) 未安装"
    echo "请先安装GitHub CLI: https://cli.github.com/"
    exit 1
fi

# 使用提供的token进行认证
echo "🔐 配置GitHub CLI认证..."
echo "YOUR_GITHUB_TOKEN_HERE" | gh auth login --with-token

# 检查是否在git仓库中
if [ ! -d ".git" ]; then
    echo "📦 初始化Git仓库..."
    git init
    git branch -M main
fi

# 创建.gitignore
if [ ! -f ".gitignore" ]; then
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
fi

# 创建GitHub仓库
echo "🏗️ 创建GitHub仓库..."
gh repo create StructDiff --public --description "StructDiff: Structure-Aware Diffusion Model for Peptide Generation with CFG and Length Control" --clone=false

# 添加远程仓库
GITHUB_USERNAME=$(gh api user --jq .login)
git remote add origin https://github.com/$GITHUB_USERNAME/StructDiff.git

# 配置git用户信息
if [ -z "$(git config user.name)" ]; then
    USER_INFO=$(gh api user --jq '.name // .login')
    USER_EMAIL=$(gh api user --jq '.email // (.login + "@users.noreply.github.com")')
    git config user.name "$USER_INFO"
    git config user.email "$USER_EMAIL"
fi

# 添加文件并提交
echo "📁 添加文件到Git..."
git add .

echo "💾 创建提交..."
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

## 📁 新增文件

### 核心实现
- \`structdiff/models/classifier_free_guidance.py\`
- \`structdiff/sampling/length_sampler.py\`
- \`scripts/cpldiff_standard_evaluation.py\`

### 集成和演示
- \`scripts/cfg_length_integrated_sampling.py\`
- \`configs/cfg_length_config.yaml\`
- \`tests/test_cfg_length_integration.py\`

### 文档和指南
- \`CFG_LENGTH_INTEGRATION_GUIDE.md\`
- \`CPL_DIFF_EVALUATION_GUIDE.md\`
- \`EVALUATION_INTEGRATION_README.md\`

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

## 📖 文档

- [CFG和长度控制集成指南](CFG_LENGTH_INTEGRATION_GUIDE.md)
- [CPL-Diff评估套件指南](CPL_DIFF_EVALUATION_GUIDE.md)
- [评估指标集成说明](EVALUATION_INTEGRATION_README.md)

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 推送到GitHub
echo "🚀 推送到GitHub..."
if git push -u origin main; then
    echo "✅ 项目成功上传到GitHub!"
    echo "🔗 项目地址: https://github.com/$GITHUB_USERNAME/StructDiff"
    
    # 设置仓库描述和话题
    gh repo edit --description "StructDiff: Advanced Structure-Aware Diffusion Model for Peptide Generation with Classifier-Free Guidance and Adaptive Length Control"
    gh repo edit --add-topic "peptide-generation,diffusion-models,classifier-free-guidance,bioinformatics,machine-learning,pytorch,protein-design,antimicrobial-peptides"
    
    echo "🏷️ 仓库标签和描述已设置"
else
    echo "❌ 推送失败，请检查网络连接和认证信息"
fi

echo "🎉 GitHub CLI上传流程完成!"