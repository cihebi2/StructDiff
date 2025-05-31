# StructDiff 详细使用说明

## 项目简介

StructDiff 是一个基于扩散模型的肽段生成框架，通过整合序列和结构信息来生成高质量的功能性肽段。该模型结合了以下创新点：

* **结构感知生成** ：利用 ESMFold 预测的 3D 结构信息指导生成
* **多尺度编码** ：同时捕获局部和全局结构特征
* **条件生成** ：支持生成特定类型的肽段（抗菌肽、抗真菌肽等）
* **灵活采样** ：支持 DDPM、DDIM 和 PNDM 等多种采样方法

## 目录

1. [安装指南](https://claude.ai/chat/40816731-24f8-45d1-91e9-d99fc53aef62#%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97)
2. [快速开始](https://claude.ai/chat/40816731-24f8-45d1-91e9-d99fc53aef62#%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)
3. [数据准备](https://claude.ai/chat/40816731-24f8-45d1-91e9-d99fc53aef62#%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87)
4. [模型训练](https://claude.ai/chat/40816731-24f8-45d1-91e9-d99fc53aef62#%E6%A8%A1%E5%9E%8B%E8%AE%AD%E7%BB%83)
5. [肽段生成](https://claude.ai/chat/40816731-24f8-45d1-91e9-d99fc53aef62#%E8%82%BD%E6%AE%B5%E7%94%9F%E6%88%90)
6. [模型评估](https://claude.ai/chat/40816731-24f8-45d1-91e9-d99fc53aef62#%E6%A8%A1%E5%9E%8B%E8%AF%84%E4%BC%B0)
7. [配置详解](https://claude.ai/chat/40816731-24f8-45d1-91e9-d99fc53aef62#%E9%85%8D%E7%BD%AE%E8%AF%A6%E8%A7%A3)
8. [API 文档](https://claude.ai/chat/40816731-24f8-45d1-91e9-d99fc53aef62#api-%E6%96%87%E6%A1%A3)
9. [常见问题](https://claude.ai/chat/40816731-24f8-45d1-91e9-d99fc53aef62#%E5%B8%B8%E8%A7%81%E9%97%AE%E9%A2%98)
10. [示例脚本](https://claude.ai/chat/40816731-24f8-45d1-91e9-d99fc53aef62#%E7%A4%BA%E4%BE%8B%E8%84%9A%E6%9C%AC)

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
git clone https://github.com/yourusername/StructDiff.git
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
git clone https://github.com/yourusername/StructDiff.git
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
python -c "import structdiff; print(structdiff.__version__)"

# 运行测试
pytest tests/
```

## 快速开始

### 1. 下载预训练模型（可选）

```bash
# 下载预训练的 StructDiff 模型
wget https://example.com/structdiff_pretrained.pth -O checkpoints/structdiff_pretrained.pth
```

### 2. 生成肽段

```bash
# 使用预训练模型生成肽段
structdiff-generate \
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
structdiff-preprocess \
    --input data/raw/peptides.fasta \
    --output_dir data/processed \
    --min_length 5 \
    --max_length 50 \
    --split_ratios 0.8 0.1 0.1
```

### 结构预测（可选）

如果没有实验解析的结构，可以使用 ESMFold 预测：

```bash
# 添加 --predict_structures 标志
structdiff-preprocess \
    --input data/raw/peptides.fasta \
    --output_dir data/processed \
    --predict_structures
```

### 使用已有结构

如果有 PDB 文件：

```bash
structdiff-preprocess \
    --input data/raw/peptides.csv \
    --structure_dir data/structures \
    --output_dir data/processed
```

## 模型训练

### 基础训练

```bash
# 使用默认配置训练
structdiff-train \
    --config configs/default.yaml \
    --data.train_path data/processed/train.csv \
    --data.val_path data/processed/val.csv
```

### 高级训练选项

```bash
# 自定义训练参数
structdiff-train \
    --config configs/default.yaml \
    --model.hidden_dim 1024 \
    --training.num_epochs 200 \
    --training.batch_size 64 \
    --training.lr 5e-5 \
    --diffusion.num_timesteps 1000 \
    --experiment.name "large_model_experiment"
```

### 分布式训练

```bash
# 单节点多 GPU
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --config configs/default.yaml \
    --distributed
```

### 从检查点恢复训练

```bash
structdiff-train \
    --config configs/default.yaml \
    --resume checkpoints/experiment/checkpoint_epoch_50.pth
```

### 监控训练

使用 Weights & Biases：

```bash
# 配置 wandb
wandb login

# 训练时启用 wandb
structdiff-train \
    --config configs/default.yaml \
    --wandb.enabled true \
    --wandb.project "StructDiff" \
    --wandb.entity "your-entity"
```

使用 TensorBoard：

```bash
# 启动 TensorBoard
tensorboard --logdir logs/

# 在浏览器中访问 http://localhost:6006
```

## 肽段生成

### 基础生成

```bash
# 生成 100 个肽段
structdiff-generate \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 100 \
    --output generated.fasta
```

### 条件生成

```bash
# 生成特定类型的肽段
structdiff-generate \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 100 \
    --conditions.peptide_type antimicrobial \
    --output antimicrobial_peptides.fasta
```

### 结构引导生成

```bash
# 生成具有特定二级结构的肽段
structdiff-generate \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 100 \
    --structure_guidance.enabled true \
    --structure_guidance.target_helix_content 0.6 \
    --output helix_rich_peptides.fasta
```

### 控制生成长度

```bash
# 生成特定长度分布的肽段
structdiff-generate \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 100 \
    --length_distribution.type normal \
    --length_distribution.mean_length 25 \
    --length_distribution.std_length 5
```

### 采样方法选择

```bash
# 使用 DDIM 快速采样
structdiff-generate \
    --checkpoint checkpoints/best_model.pth \
    --sampling_method ddim \
    --num_inference_steps 50

# 使用 PNDM 更快速采样
structdiff-generate \
    --checkpoint checkpoints/best_model.pth \
    --sampling_method pndm \
    --num_inference_steps 25
```

### 批量生成与多样性控制

```python
from structdiff.generate import PeptideGenerator

# 创建生成器
generator = PeptideGenerator(checkpoint_path="checkpoints/best_model.pth")

# 生成多样化的肽段
peptides = generator.generate_diverse(
    num_samples=1000,
    temperature=1.2,  # 增加多样性
    top_k=50,         # Top-k 采样
    top_p=0.9,        # Nucleus 采样
    remove_duplicates=True
)

# 保存结果
generator.save_peptides(peptides, "diverse_peptides.fasta")
```

## 模型评估

### 基础评估

```bash
# 评估生成的肽段
structdiff-evaluate \
    --checkpoint checkpoints/best_model.pth \
    --test_data data/processed/test.csv \
    --metrics all
```

### 评估指标

1. **序列指标**
   * 困惑度 (Perplexity)
   * 准确率 (Accuracy)
   * 氨基酸分布 KL 散度
2. **结构指标**
   * 结构一致性分数
   * 二级结构准确率
   * pLDDT 分数（使用 ESMFold）
3. **功能指标**
   * 肽段类型分类准确率
   * 与已知功能肽段的相似度
4. **多样性指标**
   * 序列多样性
   * 结构多样性
   * 新颖性（与训练集的差异）

### 自定义评估

```python
from structdiff.evaluate import Evaluator

# 创建评估器
evaluator = Evaluator(
    model_checkpoint="checkpoints/best_model.pth",
    metrics=["perplexity", "diversity", "structure_consistency"]
)

# 评估生成的肽段
results = evaluator.evaluate_generated(
    generated_peptides="generated.fasta",
    reference_peptides="data/processed/test.csv"
)

# 生成评估报告
evaluator.generate_report(results, "evaluation_report.html")
```

## 配置详解

### 模型配置

```yaml
model:
  # 模型架构
  hidden_dim: 768          # 隐藏层维度
  num_layers: 12           # Transformer 层数
  num_attention_heads: 12  # 注意力头数
  
  # 序列编码器
  sequence_encoder:
    pretrained_model: "facebook/esm2_t33_650M_UR50D"
    freeze_encoder: false  # 是否冻结预训练参数
    use_lora: true        # 使用 LoRA 微调
  
  # 结构编码器
  structure_encoder:
    type: "multi_scale"
    local:
      kernel_sizes: [3, 5, 7]  # 局部特征卷积核
    global:
      num_layers: 4            # 全局注意力层数
```

### 训练配置

```yaml
training:
  # 基础设置
  num_epochs: 100
  batch_size: 32
  gradient_accumulation_steps: 1
  
  # 优化器
  optimizer:
    type: "AdamW"
    lr: 1e-4
    weight_decay: 0.01
  
  # 学习率调度
  scheduler:
    type: "cosine"
    num_warmup_steps: 1000
  
  # 混合精度训练
  use_amp: true
  amp_dtype: "float16"
```

### 扩散配置

```yaml
diffusion:
  # 扩散过程
  num_timesteps: 1000
  noise_schedule: "sqrt"  # linear, cosine, sqrt
  beta_start: 0.0001
  beta_end: 0.02
  
  # 采样设置
  sampling_method: "ddpm"
  guidance_scale: 1.0
```

## API 文档

### 核心类

#### StructDiff

```python
class StructDiff(nn.Module):
    """主模型类"""
  
    def __init__(self, config: DictConfig):
        """
        Args:
            config: 模型配置
        """
  
    def forward(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        timesteps: torch.Tensor,
        structures: Optional[Dict[str, torch.Tensor]] = None,
        conditions: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """前向传播"""
  
    def sample(
        self,
        batch_size: int,
        seq_length: int,
        conditions: Optional[Dict] = None,
        guidance_scale: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """生成样本"""
```

#### GaussianDiffusion

```python
class GaussianDiffusion:
    """高斯扩散过程"""
  
    def __init__(
        self,
        num_timesteps: int = 1000,
        noise_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        """初始化扩散过程"""
  
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向扩散：添加噪声"""
  
    def p_sample(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """反向扩散：去噪"""
```

### 工具函数

```python
# 配置管理
from structdiff.utils import load_config, save_config

config = load_config("configs/default.yaml")
save_config(config, "configs/experiment.yaml")

# 数据处理
from structdiff.data import PeptideDataset

dataset = PeptideDataset(
    data_path="data/processed/train.csv",
    structure_path="data/structures.pt",
    max_length=50
)

# 评估指标
from structdiff.metrics import compute_sequence_metrics

metrics = compute_sequence_metrics(
    generated_sequences,
    reference_sequences
)
```

## 常见问题

### 1. 内存不足错误

 **问题** ：训练时出现 CUDA out of memory 错误

 **解决方案** ：

```bash
# 减小批量大小
--training.batch_size 16

# 使用梯度累积
--training.gradient_accumulation_steps 4

# 启用梯度检查点
--model.gradient_checkpointing true
```

### 2. 生成质量差

 **问题** ：生成的肽段质量不高

 **解决方案** ：

* 增加训练轮数
* 使用更大的模型
* 调整噪声调度
* 使用结构信息引导

### 3. 训练不稳定

 **问题** ：训练损失震荡或发散

 **解决方案** ：

```bash
# 降低学习率
--training.optimizer.lr 5e-5

# 使用梯度裁剪
--training.max_grad_norm 1.0

# 使用 EMA
--training.use_ema true
```

### 4. 依赖安装问题

 **问题** ：ESMFold 安装失败

 **解决方案** ：

```bash
# 单独安装 fair-esm
pip install fair-esm[esmfold]

# 或从源码安装
git clone https://github.com/facebookresearch/esm
cd esm
pip install -e .
```

## 示例脚本

### 完整训练流程

```bash
#!/bin/bash
# train_pipeline.sh

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1

# 数据预处理
echo "Preprocessing data..."
structdiff-preprocess \
    --input data/raw/peptides.fasta \
    --output_dir data/processed \
    --predict_structures \
    --min_length 10 \
    --max_length 40

# 训练模型
echo "Training model..."
structdiff-train \
    --config configs/default.yaml \
    --experiment.name "peptide_diffusion_v1" \
    --data.train_path data/processed/train.csv \
    --data.val_path data/processed/val.csv \
    --training.num_epochs 150 \
    --wandb.enabled true

# 生成肽段
echo "Generating peptides..."
structdiff-generate \
    --checkpoint outputs/peptide_diffusion_v1/checkpoints/best.pth \
    --num_samples 1000 \
    --output results/generated_peptides.fasta

# 评估结果
echo "Evaluating results..."
structdiff-evaluate \
    --checkpoint outputs/peptide_diffusion_v1/checkpoints/best.pth \
    --test_data data/processed/test.csv \
    --generated results/generated_peptides.fasta \
    --output results/evaluation_report.json

echo "Pipeline completed!"
```

### 批量实验

```python
# run_experiments.py
import itertools
from pathlib import Path
import subprocess

# 定义超参数网格
param_grid = {
    'hidden_dim': [512, 768, 1024],
    'lr': [1e-4, 5e-5],
    'num_timesteps': [500, 1000],
    'noise_schedule': ['linear', 'cosine', 'sqrt']
}

# 生成所有组合
experiments = list(itertools.product(*param_grid.values()))

# 运行实验
for i, params in enumerate(experiments):
    hidden_dim, lr, timesteps, schedule = params
  
    exp_name = f"exp_{i}_h{hidden_dim}_lr{lr}_t{timesteps}_{schedule}"
  
    cmd = [
        "structdiff-train",
        "--config", "configs/default.yaml",
        "--experiment.name", exp_name,
        "--model.hidden_dim", str(hidden_dim),
        "--training.optimizer.lr", str(lr),
        "--diffusion.num_timesteps", str(timesteps),
        "--diffusion.noise_schedule", schedule,
        "--training.num_epochs", "50"  # 快速实验
    ]
  
    print(f"Running experiment: {exp_name}")
    subprocess.run(cmd)
```

### 生成特定功能肽段

```python
# generate_functional_peptides.py
from structdiff import StructDiff, GaussianDiffusion
from structdiff.utils import load_config, load_checkpoint
import torch

# 加载模型
config = load_config("configs/default.yaml")
model = StructDiff(config)
checkpoint = load_checkpoint("checkpoints/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 创建扩散模型
diffusion = GaussianDiffusion(config.diffusion)

# 生成不同类型的肽段
peptide_types = ['antimicrobial', 'antifungal', 'antiviral']
results = {}

for peptide_type in peptide_types:
    print(f"Generating {peptide_type} peptides...")
  
    # 设置条件
    conditions = {
        'peptide_type': peptide_type,
        'min_length': 15,
        'max_length': 30,
        'target_helix_content': 0.5 if peptide_type == 'antimicrobial' else None
    }
  
    # 生成样本
    samples = model.sample(
        batch_size=100,
        conditions=conditions,
        guidance_scale=2.0,
        temperature=0.9
    )
  
    results[peptide_type] = samples

# 保存结果
for peptide_type, samples in results.items():
    save_path = f"generated/{peptide_type}_peptides.fasta"
    save_peptides(samples, save_path)
```

### 结构分析

```python
# analyze_structures.py
from structdiff.analysis import StructureAnalyzer
import pandas as pd

# 创建分析器
analyzer = StructureAnalyzer()

# 加载生成的肽段
generated = pd.read_csv("results/generated_peptides.csv")

# 分析结构特征
structure_features = analyzer.analyze_batch(
    generated['sequence'].tolist(),
    predict_structure=True,
    compute_features=[
        'secondary_structure',
        'solvent_accessibility',
        'contact_map',
        'radius_of_gyration'
    ]
)

# 可视化结果
analyzer.plot_structure_distribution(
    structure_features,
    save_path="results/structure_analysis.png"
)

# 与天然肽段比较
natural = pd.read_csv("data/processed/test.csv")
comparison = analyzer.compare_distributions(
    generated_features=structure_features,
    natural_sequences=natural['sequence'].tolist()
)

print("Structure similarity score:", comparison['similarity_score'])
```

## 进阶使用

### 自定义模型架构

```python
# custom_model.py
from structdiff.models import StructDiff
from structdiff.models.layers import CustomAttention

class CustomStructDiff(StructDiff):
    """自定义 StructDiff 模型"""
  
    def __init__(self, config):
        super().__init__(config)
      
        # 替换注意力层
        self.cross_attention = CustomAttention(
            hidden_dim=config.model.hidden_dim,
            num_heads=16,  # 使用更多注意力头
            use_rotary_embeddings=True
        )
  
    def forward(self, *args, **kwargs):
        # 自定义前向传播逻辑
        pass
```

### 集成到现有流程

```python
# integration_example.py
from structdiff import StructDiff
from your_project import YourPipeline

class PeptideDesignPipeline(YourPipeline):
    def __init__(self, structdiff_checkpoint):
        super().__init__()
        self.generator = StructDiff.from_pretrained(structdiff_checkpoint)
  
    def design_peptides(self, target_properties):
        # 使用 StructDiff 生成候选肽段
        candidates = self.generator.sample(
            num_samples=1000,
            conditions=target_properties
        )
      
        # 应用你的筛选标准
        filtered = self.apply_filters(candidates)
      
        # 进行下游分析
        return self.analyze_candidates(filtered)
```

## 性能优化

### GPU 优化

```python
# 启用 Flash Attention
export FLASH_ATTENTION=1

# 使用混合精度训练
--training.use_amp true
--training.amp_dtype bfloat16  # 如果支持

# 使用编译优化（PyTorch 2.0+）
--model.compile true
```

### 数据加载优化

```python
# 增加数据加载器工作进程
--data.num_workers 8

# 启用预取
--data.prefetch_factor 4

# 使用持久化工作进程
--data.persistent_workers true
```

### 分布式训练优化

```bash
# 使用 DDP
torchrun --nproc_per_node=4 scripts/train.py --distributed

# 使用 DeepSpeed
deepspeed scripts/train.py --deepspeed_config configs/deepspeed.json
```

## 故障排除

### 调试模式

```bash
# 启用详细日志
--logging.level DEBUG

# 保存中间结果
--debug.save_intermediates true

# 使用小数据集测试
--debug.use_subset true
--debug.subset_size 100
```

### 性能分析

```python
# 使用 PyTorch Profiler
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True
) as prof:
    model.train_step(batch)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 贡献指南

欢迎对 StructDiff 做出贡献！请查看 [CONTRIBUTING.md](https://claude.ai/chat/CONTRIBUTING.md) 了解详情。

## 引用

如果您在研究中使用了 StructDiff，请引用：

```bibtex
@article{structdiff2024,
  title={StructDiff: Structure-Aware Diffusion Model for Peptide Generation},
  author={Your Name},
  journal={bioRxiv},
  year={2024}
}
```

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](https://claude.ai/chat/LICENSE) 文件。

## 联系方式

* 项目主页：https://github.com/yourusername/StructDiff
* 问题反馈：https://github.com/yourusername/StructDiff/issues
* 邮箱：your.email@example.com

# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04
#   F o r c e   u p d a t e :   0 5 / 3 1 / 2 0 2 5   1 5 : 1 4 : 2 0 
 
 
