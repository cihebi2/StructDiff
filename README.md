# StructDiff 详细使用说明

## 项目简介

StructDiff 是一个基于扩散模型的肽段生成框架，通过整合序列和结构信息来生成高质量的功能性肽段。该模型结合了以下创新点：

* **结构感知生成**：利用 ESMFold 预测的 3D 结构信息指导生成
* **多尺度编码**：同时捕获局部和全局结构特征
* **条件生成**：支持生成特定类型的肽段（抗菌肽、抗真菌肽等）
* **灵活采样**：支持 DDPM、DDIM 和 PNDM 等多种采样方法
* **分类器自由引导 (CFG)**：实现精确的条件控制和生成质量提升
* **自适应长度控制**：支持多种长度分布和动态长度约束

## 目录

1. [安装指南](#安装指南)
2. [快速开始](#快速开始)
3. [数据准备](#数据准备)
4. [模型训练](#模型训练)
5. [肽段生成](#肽段生成)
6. [模型评估](#模型评估)
7. [CFG和长度控制](#CFG和长度控制)
8. [配置详解](#配置详解)
9. [API 文档](#api-文档)
10. [常见问题](#常见问题)
11. [示例脚本](#示例脚本)

## 安装指南

### 环境要求

* Python 3.8+
* PyTorch 2.0+
* CUDA 11.8+ (推荐使用 GPU)
* 至少 16GB RAM
* 至少 8GB GPU 内存

### 使用 Conda 安装（推荐）

```bash
# 克隆项目
git clone https://github.com/cihebi2/StructDiff.git
cd StructDiff

# 创建 Conda 环境
conda env create -f environment.yml

# 激活环境
conda activate structdiff

# 安装项目
pip install -e .
```

### 使用 pip 安装

```bash
# 克隆项目
git clone https://github.com/cihebi2/StructDiff.git
cd StructDiff

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

### 验证安装

```python
# 测试导入
python -c "import structdiff; print('StructDiff installed successfully')"

# 运行测试
pytest tests/
```

## 快速开始

### 1. 下载预训练模型（可选）

```bash
# 下载预训练的 StructDiff 模型
# wget https://example.com/structdiff_pretrained.pth -O checkpoints/structdiff_pretrained.pth
```

### 2. 生成肽段

```bash
# 使用预训练模型生成肽段
python scripts/generate.py \
    --checkpoint checkpoints/structdiff_pretrained.pth \
    --num_samples 100 \
    --output generated_peptides.fasta
```

### 3. Python API 使用

```python
from structdiff import StructDiff, GaussianDiffusion
from structdiff.utils import load_config

# 加载配置
config = load_config("configs/default.yaml")

# 创建模型
model = StructDiff(config)
diffusion = GaussianDiffusion(config.diffusion)

# 生成肽段
samples = model.sample(
    batch_size=10,
    seq_length=20,
    conditions={"peptide_type": "antimicrobial"}
)
```

## 数据准备

### 数据格式

StructDiff 支持以下输入格式：

1. **FASTA 格式**

```
>peptide_001|antimicrobial
ACDEFGHIKLMNPQRSTVWY
>peptide_002|antifungal
WYTVRSQPNMLKIHGFEDCA
```

2. **CSV 格式**

```csv
id,sequence,peptide_type
peptide_001,ACDEFGHIKLMNPQRSTVWY,antimicrobial
peptide_002,WYTVRSQPNMLKIHGFEDCA,antifungal
```

### 数据预处理

```bash
# 预处理肽段数据
python scripts/preprocess.py \
    --input data/raw/peptides.fasta \
    --output_dir data/processed \
    --min_length 5 \
    --max_length 50 \
    --split_ratios 0.8 0.1 0.1
```

## 模型训练

### 基础训练

```bash
# 使用默认配置训练
python scripts/train.py \
    --config configs/default.yaml \
    --data_dir data/processed
```

### 高级训练选项

```bash
# 自定义训练参数
python scripts/train.py \
    --config configs/default.yaml \
    --model.hidden_dim 1024 \
    --training.num_epochs 200 \
    --training.batch_size 64 \
    --training.lr 5e-5 \
    --diffusion.num_timesteps 1000
```

## 肽段生成

### 基础生成

```bash
# 生成肽段
python scripts/generate.py \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 1000 \
    --output results/generated_peptides.fasta
```

### 条件生成

```bash
# 生成特定类型的肽段
python scripts/generate.py \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 500 \
    --peptide_type antimicrobial \
    --min_length 10 \
    --max_length 30 \
    --output results/antimicrobial_peptides.fasta
```

### CFG 生成

```bash
# 使用分类器自由引导
python scripts/generate.py \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 200 \
    --guidance_scale 2.0 \
    --peptide_type antimicrobial \
    --output results/cfg_peptides.fasta
```

## 模型评估

### 基础评估

```bash
# 评估生成的肽段
python scripts/evaluate.py \
    --generated results/generated_peptides.fasta \
    --reference data/test.csv \
    --output results/evaluation_report.json
```

### 详细评估

```bash
# 包含结构和功能评估
python scripts/evaluate.py \
    --generated results/generated_peptides.fasta \
    --reference data/test.csv \
    --predict_structure \
    --compute_properties \
    --output results/detailed_evaluation.json
```

## CFG和长度控制

### CFG 配置

在配置文件中启用 CFG：

```yaml
diffusion:
  guidance_scale: 2.0
  use_cfg: true

model:
  use_classifier_free_guidance: true
  cfg_dropout_rate: 0.1
```

### 长度控制

```yaml
generation:
  length_control:
    enabled: true
    min_length: 8
    max_length: 40
    length_distribution: "uniform"  # uniform, normal, exponential
```

## 配置详解

### 模型配置

```yaml
model:
  hidden_dim: 768
  num_layers: 12
  num_heads: 12
  max_seq_length: 512
  use_structure_features: true
  structure_dim: 256
```

### 训练配置

```yaml
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 5e-5
  weight_decay: 0.01
  warmup_steps: 1000
  gradient_clipping: 1.0
```

### 扩散配置

```yaml
diffusion:
  num_timesteps: 1000
  beta_schedule: "cosine"
  loss_type: "mse"
  noise_schedule: "cosine"
```

## API 文档

### 核心类

#### StructDiff

主要的模型类，负责肽段生成。

```python
class StructDiff(nn.Module):
    def __init__(self, config):
        """初始化 StructDiff 模型"""
        
    def forward(self, x, t, conditions=None):
        """前向传播"""
        
    def sample(self, batch_size, seq_length, conditions=None):
        """生成肽段序列"""
```

#### GaussianDiffusion

扩散模型的实现。

```python
class GaussianDiffusion:
    def __init__(self, config):
        """初始化扩散过程"""
        
    def p_sample_loop(self, model, shape, conditions=None):
        """采样循环"""
```

### 实用函数

```python
from structdiff.utils import (
    load_config,
    save_sequences,
    compute_metrics,
    visualize_results
)
```

## 常见问题

### Q: 如何处理内存不足的问题？

A: 尝试以下方法：
- 减少批次大小：`--training.batch_size 16`
- 使用梯度累积：`--training.gradient_accumulation_steps 4`
- 启用混合精度：`--training.use_amp true`

### Q: 如何提高生成质量？

A: 可以尝试：
- 增加训练时间
- 使用更大的模型
- 调整 CFG 的引导强度
- 优化数据预处理

### Q: 支持哪些氨基酸？

A: 支持标准的 20 种氨基酸：
ACDEFGHIKLMNPQRSTVWY

## 示例脚本

### 完整训练流程

```bash
#!/bin/bash
# train_pipeline.sh

# 数据预处理
python scripts/preprocess.py \
    --input data/raw/peptides.fasta \
    --output_dir data/processed

# 训练模型
python scripts/train.py \
    --config configs/default.yaml \
    --data_dir data/processed \
    --experiment_name "my_experiment"

# 生成肽段
python scripts/generate.py \
    --checkpoint outputs/my_experiment/checkpoints/best.pth \
    --num_samples 1000 \
    --output results/generated.fasta

# 评估结果
python scripts/evaluate.py \
    --generated results/generated.fasta \
    --reference data/processed/test.csv \
    --output results/evaluation.json
```

### 批量实验

```python
# run_experiments.py
import itertools
import subprocess

# 定义超参数网格
param_grid = {
    'hidden_dim': [512, 768, 1024],
    'lr': [1e-4, 5e-5],
    'guidance_scale': [1.0, 2.0, 3.0]
}

# 运行实验
for params in itertools.product(*param_grid.values()):
    hidden_dim, lr, guidance_scale = params
    
    cmd = [
        "python", "scripts/train.py",
        "--config", "configs/default.yaml",
        "--model.hidden_dim", str(hidden_dim),
        "--training.lr", str(lr),
        "--diffusion.guidance_scale", str(guidance_scale)
    ]
    
    subprocess.run(cmd)
```

## 性能优化

### GPU 优化

```bash
# 启用 Flash Attention（如果可用）
export FLASH_ATTENTION=1

# 使用混合精度训练
python scripts/train.py \
    --config configs/default.yaml \
    --training.use_amp true
```

### 分布式训练

```bash
# 多 GPU 训练
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --config configs/default.yaml
```

## 贡献指南

欢迎对 StructDiff 做出贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 引用

如果您在研究中使用了 StructDiff，请引用：

```bibtex
@article{structdiff2024,
  title={StructDiff: Structure-Aware Diffusion Model for Peptide Generation},
  author={StructDiff Team},
  journal={bioRxiv},
  year={2024}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 联系方式

* 项目主页：https://github.com/cihebi2/StructDiff
* 问题反馈：https://github.com/cihebi2/StructDiff/issues
* 文档：详见 `docs/` 目录中的完整文档