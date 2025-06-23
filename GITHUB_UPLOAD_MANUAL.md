# StructDiff GitHub 手动上传指南

## 📋 前置准备

1. **GitHub账户**: 确保您有GitHub账户
2. **GitHub仓库**: 在GitHub上创建名为`StructDiff`的新仓库
3. **Git安装**: 确保本地安装了Git

## 🚀 上传步骤

### 步骤1: 准备项目目录

```bash
cd /mnt/c/Users/ciheb/Desktop/AA_work/代码融合/StructDiff
```

### 步骤2: 初始化Git仓库

```bash
# 初始化git仓库
git init

# 设置主分支名
git branch -M main
```

### 步骤3: 配置Git用户信息

```bash
# 替换为您的信息
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### 步骤4: 添加远程仓库

```bash
# 替换YOUR_USERNAME为您的GitHub用户名
git remote add origin https://github.com/YOUR_USERNAME/StructDiff.git
```

### 步骤5: 创建.gitignore文件

```bash
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
```

### 步骤6: 添加所有文件

```bash
# 添加所有文件到Git
git add .

# 检查状态
git status
```

### 步骤7: 创建提交

```bash
git commit -m "集成CFG和长度采样器功能 - 完整版本

## 🎯 新增功能

### Classifier-Free Guidance (CFG)
- ✅ 实现与CPL-Diff论文一致的CFG机制
- ✅ 支持训练时条件丢弃和推理时引导采样
- ✅ 自适应引导强度和多级引导功能

### 长度分布采样器
- ✅ 支持5种分布类型（正态、均匀、Gamma、Beta、自定义）
- ✅ 条件相关的长度偏好设置
- ✅ 自适应长度控制和约束执行

### CPL-Diff标准评估
- ✅ 完整的5个核心评估指标
- ✅ 与原论文完全一致的评估标准

### AlphaFold3优化组件  
- ✅ AF3风格的时间步嵌入和自适应调节
- ✅ 增强的条件层归一化

## 📁 文件结构

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

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 步骤8: 推送到GitHub

#### 方法A: 使用访问令牌 (推荐)

```bash
# 配置认证
git config credential.helper store

# 推送（会提示输入用户名和密码）
# 用户名：您的GitHub用户名
# 密码：使用您的GitHub Personal Access Token
git push -u origin main
```

#### 方法B: 直接在URL中包含令牌

```bash
# 先移除现有的远程仓库
git remote remove origin

# 添加包含令牌的远程仓库URL
git remote add origin https://YOUR_GITHUB_TOKEN@github.com/YOUR_USERNAME/StructDiff.git

# 推送
git push -u origin main
```

## 🔧 故障排除

### 问题1: 推送被拒绝
```bash
# 如果远程仓库有文件，先拉取
git pull origin main --allow-unrelated-histories

# 然后再推送
git push -u origin main
```

### 问题2: 认证失败
- 确保访问令牌有正确的权限
- 检查用户名是否正确
- 尝试重新生成访问令牌

### 问题3: 文件过大
```bash
# 如果有大文件，可以使用Git LFS
git lfs track "*.pth"
git lfs track "*.pt" 
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push -u origin main
```

## ✅ 验证上传成功

上传成功后，您应该能在GitHub上看到：

1. **所有项目文件**: 包括代码、配置、文档
2. **README.md**: 项目主页显示
3. **提交历史**: 显示提交信息
4. **分支**: main分支设为默认

## 🎉 完成后的工作

1. **设置仓库描述**: 在GitHub仓库页面添加项目描述
2. **添加标签**: 添加相关的topic标签
3. **启用Issues**: 如果需要问题跟踪
4. **设置Wiki**: 如果需要详细文档
5. **配置Actions**: 如果需要CI/CD

## 📞 获取帮助

如果遇到问题：
1. 检查网络连接
2. 确认GitHub访问令牌权限
3. 查看Git错误信息
4. 参考GitHub官方文档

---

**注意**: 请确保不要将访问令牌分享给他人，上传完成后建议定期更新令牌。