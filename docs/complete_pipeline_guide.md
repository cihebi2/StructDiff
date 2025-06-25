# StructDiff 完整流水线系统详解

## 概述

`run_complete_pipeline.py` 是 StructDiff 项目的核心集成脚本，它将训练、生成、评估三个阶段无缝连接，形成一个端到端的肽段生成和评估流水线。该系统基于 CPL-Diff 启发的分离式训练策略，集成了最新的扩散模型技术和标准化评估体系。

---

## 🏗️ 系统架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                   完整流水线系统                              │
├─────────────────┬─────────────────┬─────────────────────────┤
│   训练阶段       │   生成阶段       │   评估阶段               │
│   Training      │   Generation    │   Evaluation            │
├─────────────────┼─────────────────┼─────────────────────────┤
│ • 分离式训练     │ • 序列生成       │ • CPL-Diff标准评估      │
│ • 模型检查点     │ • 多类型肽段     │ • 性能指标计算          │
│ • 训练监控       │ • 长度控制       │ • 报告生成              │
│ • 定期评估       │ • 引导采样       │ • 结果可视化            │
└─────────────────┴─────────────────┴─────────────────────────┘
```

---

## 📚 核心模块详解

### 🚀 1. 主控制模块

#### `main()` - 流水线主入口

**功能**：协调整个流水线的执行流程

**执行流程**：
```python
def main():
    args = parse_args()                    # 1. 解析命令行参数
    experiment_info = setup_experiment()   # 2. 设置实验环境
    config = load_config()                 # 3. 加载配置文件
    
    # 4. 执行三阶段流水线
    training_result = run_training_stage()
    generation_result = run_generation_stage()
    evaluation_result = run_evaluation_stage()
    
    # 5. 生成最终报告
    generate_final_report()
```

**输入**：
- 命令行参数
- 配置文件路径
- 数据目录路径

**输出**：
- 完整的实验报告
- 结构化的结果文件
- 日志和监控数据

---

### 🔧 2. 实验设置模块

#### `setup_experiment(args)` - 实验环境初始化

**功能**：为整个流水线创建统一的实验环境

```python
def setup_experiment(args):
    """
    设置实验环境，包括：
    - 创建实验目录结构
    - 配置日志系统
    - 设置随机种子
    - 检测和配置硬件设备
    """
```

**输入参数**：
```python
args = {
    'experiment_name': str,    # 实验名称
    'output_dir': str,         # 输出根目录
    'device': str,             # 设备类型 (cuda/cpu/auto)
    'seed': int                # 随机种子
}
```

**执行逻辑**：
1. **目录创建**：
   ```
   outputs/experiment_name/
   ├── training/          # 训练相关文件
   ├── generation/        # 生成的序列
   ├── evaluation/        # 评估结果
   ├── logs/             # 日志文件
   └── reports/          # 最终报告
   ```

2. **日志配置**：
   - 设置统一的日志格式
   - 配置文件和控制台输出
   - 记录实验元信息

3. **设备检测**：
   ```python
   if args.device == "auto":
       device = "cuda" if torch.cuda.is_available() else "cpu"
   ```

**输出**：
```python
{
    "experiment_dir": Path,    # 实验根目录
    "device": str,             # 最终使用的设备
    "experiment_name": str     # 实验名称
}
```

---

### 🎓 3. 训练阶段模块

#### `run_training_stage(args, config, experiment_info)` - 分离式训练执行

**功能**：执行基于 CPL-Diff 的两阶段分离式训练

```python
def run_training_stage(args, config, experiment_info):
    """
    执行完整的分离式训练流程：
    1. 模型和组件初始化
    2. 数据加载器创建
    3. 两阶段训练执行
    4. 检查点管理
    5. 训练监控和评估
    """
```

#### 3.1 模型初始化流程

```python
# 1. 分词器加载
tokenizer_name = config.model.sequence_encoder.pretrained_model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# 2. 模型创建
model = StructDiff(config.model).to(device)
diffusion = GaussianDiffusion(config.diffusion)

# 3. 训练管理器
trainer = SeparatedTrainingManager(
    config=training_config,
    model=model,
    diffusion=diffusion,
    device=device,
    tokenizer=tokenizer
)
```

#### 3.2 训练配置映射

**从YAML配置到训练配置**：
```python
training_config = SeparatedTrainingConfig(
    # 基础配置
    data_dir=args.data_dir,
    output_dir=experiment_dir / "training",
    checkpoint_dir=experiment_dir / "checkpoints",
    
    # 评估配置（新增）
    enable_evaluation=True,
    evaluate_every=5,
    auto_generate_after_training=True
)
```

#### 3.3 数据加载流程

```python
# 训练数据集
train_dataset = PeptideStructureDataset(
    data_path=data_dir / "train.csv",
    config=config,
    is_training=True
)

# 验证数据集（可选）
if val_path.exists():
    val_dataset = PeptideStructureDataset(
        data_path=data_dir / "val.csv",
        config=config,
        is_training=False
    )
```

#### 3.4 两阶段训练执行

**阶段1：去噪器训练**
```python
# 特点：
- 冻结序列编码器权重
- 专注训练去噪能力
- 使用较高学习率
- 定期进行CPL-Diff评估

# 训练目标：
学习从噪声中恢复有意义的特征表示
```

**阶段2：解码器训练**
```python
# 特点：
- 冻结去噪器权重
- 专注序列重建能力
- 使用较低学习率
- 更大的批次大小

# 训练目标：
学习将特征表示转换为氨基酸序列
```

**训练输入**：
- `train_loader`: 训练数据加载器
- `val_loader`: 验证数据加载器（可选）
- `config`: 模型和训练配置

**训练输出**：
```python
{
    "training_stats": {
        "stage1": {
            "losses": [float],           # 每个epoch的损失
            "val_losses": [float],       # 验证损失
            "evaluations": [dict],       # 定期评估结果
            "final_evaluation": dict     # 阶段结束评估
        },
        "stage2": {
            # 同stage1结构
        }
    },
    "checkpoint_path": str,              # 最佳模型检查点路径
    "training_config": SeparatedTrainingConfig
}
```

---

### 🎯 4. 生成阶段模块

#### `run_generation_stage(args, config, experiment_info, training_result)` - 序列生成执行

**功能**：基于训练好的模型生成多种类型的肽段序列

```python
def run_generation_stage(args, config, experiment_info, training_result):
    """
    执行序列生成流程：
    1. 加载训练好的模型
    2. 配置生成参数
    3. 批量生成不同类型肽段
    4. 序列后处理和保存
    """
```

#### 4.1 模型加载流程

```python
# 1. 检查点验证
checkpoint_path = training_result.get("checkpoint_path")
if not checkpoint_path or not Path(checkpoint_path).exists():
    logger.error("无法找到有效的检查点文件")
    return None

# 2. 模型重建和权重加载
checkpoint = torch.load(checkpoint_path, map_location=device)
model = StructDiff(config.model).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 3. 组件初始化
diffusion = GaussianDiffusion(config.diffusion)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
```

#### 4.2 生成流程详解

**多类型生成循环**：
```python
for peptide_type in args.peptide_types:  # ['antimicrobial', 'antifungal', 'antiviral']
    logger.info(f"生成 {peptide_type} 肽段...")
    
    sequences = []
    with torch.no_grad():
        for i in range(0, args.num_samples, 10):  # 批次生成
            # 1. 长度采样
            length = torch.randint(10, 30, (1,)).item()
            
            # 2. 噪声初始化
            seq_embeddings = torch.randn(1, length, hidden_size, device=device)
            attention_mask = torch.ones(1, length, device=device)
            
            # 3. 扩散去噪过程
            for t in reversed(range(0, 1000, 50)):  # DDIM采样
                timesteps = torch.tensor([t], device=device)
                noise_pred = model.denoiser(seq_embeddings, timesteps, attention_mask)
                seq_embeddings = seq_embeddings - 0.01 * noise_pred
            
            # 4. 序列解码
            sequence = decode_sequence(seq_embeddings, attention_mask, model, tokenizer)
            
            if sequence and len(sequence) >= 5:
                sequences.append(sequence)
```

**序列解码机制**：
```python
def decode_sequence(embeddings, attention_mask, model, tokenizer):
    """
    多层解码策略：
    1. 优先使用训练的序列解码器
    2. 回退到相似性匹配解码
    3. 最后使用随机回退方案
    """
    
    # 方法1：学习化解码
    if hasattr(model, 'sequence_decoder') and model.sequence_decoder:
        logits = model.sequence_decoder(embeddings, attention_mask)
        token_ids = torch.argmax(logits, dim=-1)
        sequence = tokenizer.decode(token_ids, skip_special_tokens=True)
        
    # 方法2：相似性解码（回退）
    else:
        # 基于嵌入相似性的解码逻辑
        
    # 方法3：随机回退
    # ...
    
    # 序列清理
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    clean_sequence = ''.join([c for c in sequence.upper() if c in amino_acids])
    return clean_sequence
```

#### 4.3 序列保存和组织

```python
# 为每种肽段类型保存序列
output_file = generation_dir / f"{peptide_type}_sequences.fasta"
with open(output_file, 'w') as f:
    for i, seq in enumerate(sequences):
        f.write(f">{peptide_type}_{i}\n{seq}\n")

all_generated[peptide_type] = {
    "sequences": sequences,      # 序列列表
    "count": len(sequences),     # 序列数量
    "file": str(output_file)     # 文件路径
}
```

**生成输入**：
- `checkpoint_path`: 训练好的模型检查点
- `args.peptide_types`: 要生成的肽段类型列表
- `args.num_samples`: 每种类型的生成数量
- `config`: 模型和生成配置

**生成输出**：
```python
{
    "antimicrobial": {
        "sequences": [str],      # 生成的序列列表
        "count": int,            # 序列数量
        "file": str              # FASTA文件路径
    },
    "antifungal": { ... },
    "antiviral": { ... }
}
```

---

### 🔬 5. 评估阶段模块

#### `run_evaluation_stage(args, experiment_info, generation_result)` - CPL-Diff标准评估

**功能**：对生成的序列进行全面的质量评估

```python
def run_evaluation_stage(args, experiment_info, generation_result):
    """
    执行CPL-Diff标准评估流程：
    1. 初始化评估器
    2. 加载生成的序列
    3. 运行五大核心指标评估
    4. 生成详细评估报告
    """
```

#### 5.1 评估器初始化

```python
evaluation_dir = experiment_info["experiment_dir"] / "evaluation"
evaluator = CPLDiffStandardEvaluator(output_dir=str(evaluation_dir))
```

#### 5.2 评估循环执行

```python
all_evaluations = {}

for peptide_type, gen_data in generation_result.items():
    sequences = gen_data["sequences"]
    
    # CPL-Diff标准评估
    eval_results = evaluator.comprehensive_cpldiff_evaluation(
        generated_sequences=sequences,
        reference_sequences=[],          # 使用内置参考序列
        peptide_type=peptide_type
    )
    
    # 生成详细报告
    report_name = f"{experiment_name}_{peptide_type}"
    evaluator.generate_cpldiff_report(eval_results, report_name)
    
    all_evaluations[peptide_type] = eval_results
```

#### 5.3 CPL-Diff 五大核心指标

**1. Pseudo Perplexity (伪困惑度)**
```python
指标含义: 使用ESM-2计算的序列自然度
期望值: 越低越好 (↓)
计算方法: 基于ESM-2的条件概率
质量指示: 序列的生物学合理性
```

**2. pLDDT Score (结构置信度)**
```python
指标含义: ESMFold预测的结构置信度
期望值: 越高越好 (↑)
计算方法: ESMFold预测置信度的平均值
质量指示: 序列可折叠性和结构稳定性
```

**3. Instability Index (不稳定性指数)**
```python
指标含义: modlAMP计算的蛋白质稳定性
期望值: 越低越好 (↓)
计算方法: 基于氨基酸组成的稳定性预测
质量指示: 蛋白质在体内的稳定性
```

**4. Similarity Score (相似性评分)**
```python
指标含义: 与已知序列的BLOSUM62相似性
期望值: 越低越好 (↓) - 表示更高的新颖性
计算方法: BLOSUM62矩阵计算序列相似性
质量指示: 生成序列的新颖性和多样性
```

**5. Activity Prediction (活性预测)**
```python
指标含义: 外部分类器预测的生物活性
期望值: 越高越好 (↑)
计算方法: 使用预训练的活性预测模型
质量指示: 序列的功能活性潜力
```

#### 5.4 评估报告生成

**HTML报告结构**：
```html
评估报告包含：
├── 执行摘要 (Executive Summary)
├── 核心指标总览 (Core Metrics Overview)
├── 详细指标分析 (Detailed Analysis)
│   ├── 伪困惑度分布图
│   ├── pLDDT分数分布
│   ├── 稳定性分析
│   ├── 相似性热图
│   └── 活性预测结果
├── 序列质量分析 (Sequence Quality Analysis)
├── 对比分析 (Comparative Analysis)
└── 结论和建议 (Conclusions & Recommendations)
```

**评估输入**：
- `generation_result`: 生成的序列数据
- `peptide_type`: 肽段类型
- `evaluation_dir`: 评估结果输出目录

**评估输出**：
```python
{
    "antimicrobial": {
        "cpldiff_core_metrics": {
            "pseudo_perplexity": {
                "mean_pseudo_perplexity": float,
                "std_pseudo_perplexity": float,
                "distribution": [float]
            },
            "plddt": {
                "mean_plddt": float,
                "std_plddt": float,
                "high_confidence_ratio": float  # pLDDT > 70的比例
            },
            "instability": {
                "mean_instability": float,
                "stable_ratio": float           # 稳定序列比例
            },
            "similarity": {
                "mean_similarity": float,
                "novelty_ratio": float          # 低相似性序列比例
            },
            "activity": {
                "mean_activity_score": float,
                "active_ratio": float           # 高活性序列比例
            }
        },
        "additional_metrics": {
            "information_entropy": float,
            "amino_acid_diversity": float,
            "length_distribution": dict
        },
        "quality_summary": {
            "overall_score": float,             # 综合质量评分
            "grade": str,                       # 质量等级 (A/B/C/D)
            "recommendations": [str]            # 改进建议
        }
    }
}
```

---

### 📊 6. 报告生成模块

#### `generate_final_report(args, experiment_info, results)` - 综合报告生成

**功能**：整合所有阶段的结果，生成全面的实验报告

```python
def generate_final_report(args, experiment_info, results):
    """
    生成最终实验报告：
    1. 整合三阶段结果数据
    2. 生成JSON详细报告
    3. 生成文本摘要报告
    4. 创建可视化图表
    """
```

#### 6.1 报告数据结构

```python
report_data = {
    "experiment_info": {
        "name": str,                    # 实验名称
        "timestamp": str,               # 执行时间
        "config": dict,                 # 完整配置
        "device": str,                  # 使用设备
        "duration": float               # 执行时长
    },
    "pipeline_results": {
        "training": {
            "stage1_stats": dict,       # 阶段1训练统计
            "stage2_stats": dict,       # 阶段2训练统计
            "checkpoint_path": str,     # 最佳检查点
            "training_duration": float  # 训练时长
        },
        "generation": {
            "antimicrobial": dict,      # 抗菌肽生成结果
            "antifungal": dict,         # 抗真菌肽生成结果
            "antiviral": dict,          # 抗病毒肽生成结果
            "generation_duration": float
        },
        "evaluation": {
            "antimicrobial": dict,      # 抗菌肽评估结果
            "antifungal": dict,         # 抗真菌肽评估结果
            "antiviral": dict,          # 抗病毒肽评估结果
            "evaluation_duration": float
        }
    }
}
```

#### 6.2 JSON详细报告

**文件**：`final_report.json`

**内容**：包含所有原始数据、统计信息、配置参数的完整记录

**用途**：程序化分析、数据挖掘、结果比较

#### 6.3 文本摘要报告

**文件**：`final_report.txt`

**内容示例**：
```text
StructDiff完整流水线实验报告
==================================================

实验名称: peptide_generation_20231201
执行时间: 2023-12-01 14:30:25
设备: NVIDIA RTX 4090
总耗时: 4.5小时

训练阶段结果:
--------------------
阶段1 - 去噪器训练:
  最终损失: 0.0245
  验证损失: 0.0312
  训练轮数: 200
  评估次数: 40

阶段2 - 解码器训练:
  最终损失: 0.0156
  验证损失: 0.0203
  训练轮数: 100
  评估次数: 20

生成阶段结果:
--------------------
抗菌肽: 1000 个序列
抗真菌肽: 1000 个序列
抗病毒肽: 1000 个序列
总生成时间: 15分钟

评估阶段结果:
--------------------
抗菌肽质量评估:
  伪困惑度: 8.45 (↓越低越好)
  pLDDT分数: 72.8 (↑越高越好)
  不稳定性: 32.1 (↓越低越好)
  相似性: 0.23 (↓越低越好，表示新颖性)
  活性预测: 0.78 (↑越高越好)
  
  综合评分: B+ (85/100)
  推荐度: 高质量，建议进一步实验验证

抗真菌肽质量评估:
  ...

抗病毒肽质量评估:
  ...

结论与建议:
--------------------
1. 模型在抗菌肽生成方面表现最佳
2. 建议调整抗真菌肽的训练参数
3. 序列多样性良好，新颖性较高
4. 推荐进行湿实验验证前10%的高质量序列

详细结果请查看相应目录中的具体文件。
```

---

## 🔄 模块间的数据流

### 数据流向图

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ 配置文件     │───→│ 实验环境设置  │───→│ 全局配置     │
│ YAML        │    │ setup_exp    │    │ experiment  │
└─────────────┘    └──────────────┘    └─────────────┘
                              │
                              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ 训练数据     │───→│ 训练阶段     │───→│ 模型检查点   │
│ CSV Files   │    │ training     │    │ checkpoint  │
└─────────────┘    └──────────────┘    └─────────────┘
                              │
                              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ 模型检查点   │───→│ 生成阶段     │───→│ 肽段序列     │
│ checkpoint  │    │ generation   │    │ sequences   │
└─────────────┘    └──────────────┘    └─────────────┘
                              │
                              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ 肽段序列     │───→│ 评估阶段     │───→│ 质量报告     │
│ sequences   │    │ evaluation   │    │ reports     │
└─────────────┘    └──────────────┘    └─────────────┘
```

### 关键数据传递

#### 1. 配置传递链

```python
YAML配置 → OmegaConf对象 → 各模块配置类 → 具体组件参数
```

#### 2. 模型传递链

```python
训练配置 → StructDiff模型 → 检查点文件 → 加载的模型 → 生成序列
```

#### 3. 评估传递链

```python
生成序列 → CPLDiff评估器 → 指标计算 → 评估结果 → 最终报告
```

---

## 🚀 使用指南

### 基础使用

#### 1. 完整流水线执行

```bash
# 标准执行
python scripts/run_complete_pipeline.py \
    --config configs/separated_training.yaml \
    --data-dir ./data/processed \
    --output-dir ./outputs/my_experiment

# 自定义实验名称
python scripts/run_complete_pipeline.py \
    --config configs/separated_training.yaml \
    --experiment-name antimicrobial_peptides_v2 \
    --num-samples 2000
```

#### 2. 部分流水线执行

```bash
# 仅生成和评估（跳过训练）
python scripts/run_complete_pipeline.py \
    --skip-training \
    --checkpoint-path ./checkpoints/best_model.pth \
    --num-samples 1000

# 仅训练（跳过生成和评估）
python scripts/run_complete_pipeline.py \
    --skip-generation \
    --skip-evaluation

# 自定义肽段类型
python scripts/run_complete_pipeline.py \
    --peptide-types antimicrobial antifungal \
    --num-samples 1500
```

### 高级使用

#### 1. 多实验批量执行

```bash
#!/bin/bash
# 批量实验脚本

experiments=(
    "antimicrobial_focused"
    "antifungal_focused" 
    "antiviral_focused"
)

for exp in "${experiments[@]}"; do
    python scripts/run_complete_pipeline.py \
        --experiment-name "$exp" \
        --peptide-types ${exp%_*} \
        --num-samples 2000
done
```

#### 2. 超参数扫描

```python
# 配置文件生成脚本
import yaml
from itertools import product

# 超参数网格
learning_rates = [1e-4, 5e-5, 1e-5]
guidance_scales = [1.5, 2.0, 2.5]

for lr, gs in product(learning_rates, guidance_scales):
    config = load_base_config()
    config['separated_training']['stage1']['learning_rate'] = lr
    config['classifier_free_guidance']['guidance_scale'] = gs
    
    exp_name = f"lr_{lr}_gs_{gs}"
    config_path = f"configs/sweep_{exp_name}.yaml"
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # 执行实验
    os.system(f"""
        python scripts/run_complete_pipeline.py \
            --config {config_path} \
            --experiment-name {exp_name}
    """)
```

#### 3. 分布式执行

```bash
# 使用多GPU训练
CUDA_VISIBLE_DEVICES=0,1 python scripts/run_complete_pipeline.py \
    --config configs/separated_training.yaml \
    --experiment-name multi_gpu_experiment

# 使用特定GPU
CUDA_VISIBLE_DEVICES=2 python scripts/run_complete_pipeline.py \
    --device cuda \
    --experiment-name gpu2_experiment
```

---

## 📁 输出文件结构

### 完整的输出目录结构

```
outputs/experiment_name/
├── training/                          # 训练阶段输出
│   ├── logs/                         # 训练日志
│   ├── evaluations/                  # 训练中评估
│   │   ├── stage1_epoch_5/          # 阶段1定期评估
│   │   ├── stage1_epoch_10/
│   │   ├── stage1_final/            # 阶段1最终评估
│   │   ├── stage2_epoch_5/          # 阶段2定期评估
│   │   └── stage2_final/            # 阶段2最终评估
│   ├── training_config.json         # 训练配置备份
│   ├── training_stats.json          # 训练统计数据
│   └── final_generated_sequences.fasta  # 训练后自动生成
│
├── checkpoints/                      # 模型检查点
│   ├── stage1_epoch_50.pth         # 阶段1检查点
│   ├── stage1_final.pth             # 阶段1最终模型
│   ├── stage2_epoch_20.pth         # 阶段2检查点
│   └── stage2_final.pth             # 阶段2最终模型
│
├── generation/                       # 生成阶段输出
│   ├── antimicrobial_sequences.fasta   # 抗菌肽序列
│   ├── antifungal_sequences.fasta      # 抗真菌肽序列
│   └── antiviral_sequences.fasta       # 抗病毒肽序列
│
├── evaluation/                       # 评估阶段输出
│   ├── antimicrobial/               # 抗菌肽评估结果
│   │   ├── cpldiff_evaluation.html     # 详细评估报告
│   │   ├── metrics_summary.json        # 指标汇总
│   │   ├── sequence_analysis.csv       # 序列分析
│   │   └── visualization_plots/        # 可视化图表
│   ├── antifungal/                  # 抗真菌肽评估结果
│   └── antiviral/                   # 抗病毒肽评估结果
│
├── logs/                            # 流水线日志
│   ├── pipeline.log                 # 主日志文件
│   ├── training.log                 # 训练详细日志
│   ├── generation.log               # 生成详细日志
│   └── evaluation.log               # 评估详细日志
│
├── reports/                         # 最终报告
│   ├── final_report.json            # 完整JSON报告
│   ├── final_report.txt             # 文本摘要报告
│   ├── executive_summary.html       # 执行摘要（HTML）
│   └── comparison_charts/           # 对比图表
│
└── metadata/                        # 元数据
    ├── experiment_config.yaml       # 实验配置备份
    ├── environment_info.json        # 环境信息
    ├── git_info.json               # 代码版本信息
    └── system_resources.json        # 硬件资源信息
```

### 关键文件说明

#### 1. 训练相关文件

- **`training_stats.json`**: 包含损失曲线、验证指标、评估历史
- **`stage1_final.pth`**: 阶段1最终模型，包含去噪器权重
- **`stage2_final.pth`**: 阶段2最终模型，包含完整模型权重

#### 2. 生成相关文件

- **`*_sequences.fasta`**: 标准FASTA格式的生成序列
- **序列命名规则**: `>{peptide_type}_{index}`

#### 3. 评估相关文件

- **`cpldiff_evaluation.html`**: 交互式HTML评估报告
- **`metrics_summary.json`**: 所有指标的数值总结
- **`sequence_analysis.csv`**: 每个序列的详细分析数据

#### 4. 报告相关文件

- **`final_report.json`**: 机器可读的完整报告
- **`final_report.txt`**: 人类可读的摘要报告

---

## ⚠️ 常见问题和故障排除

### 1. 内存相关问题

#### 问题：CUDA out of memory

**症状**：
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解决方案**：
```bash
# 方案1：减少批次大小
python scripts/run_complete_pipeline.py \
    --config configs/separated_training.yaml \
    --batch-size 16  # 从默认32减少

# 方案2：使用CPU模式
python scripts/run_complete_pipeline.py \
    --device cpu \
    --config configs/separated_training.yaml

# 方案3：修改配置文件
# 在separated_training.yaml中调整：
separated_training:
  stage1:
    batch_size: 8
  stage2:
    batch_size: 16
```

#### 问题：系统内存不足

**症状**：
```
OSError: [Errno 12] Cannot allocate memory
```

**解决方案**：
```yaml
# 减少数据加载进程
data:
  num_workers: 2        # 从4减少到2
  prefetch_factor: 1    # 从2减少到1

# 禁用内存锁定
resources:
  pin_memory: false
```

### 2. 模型加载问题

#### 问题：检查点文件损坏

**症状**：
```
RuntimeError: Error(s) in loading state_dict
```

**解决方案**：
```python
# 检查检查点文件
import torch

checkpoint_path = "path/to/checkpoint.pth"
try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print("检查点文件正常")
    print(f"包含的键: {checkpoint.keys()}")
except Exception as e:
    print(f"检查点文件损坏: {e}")
    # 使用备份检查点或重新训练
```

#### 问题：模型结构不匹配

**症状**：
```
RuntimeError: size mismatch for denoiser.layers.0.weight
```

**解决方案**：
```bash
# 确保配置文件与检查点匹配
# 或者删除检查点重新训练
rm -rf outputs/experiment_name/checkpoints/
python scripts/run_complete_pipeline.py --config configs/separated_training.yaml
```

### 3. 数据相关问题

#### 问题：数据文件缺失

**症状**：
```
FileNotFoundError: [Errno 2] No such file or directory: 'train.csv'
```

**解决方案**：
```bash
# 检查数据目录结构
ls -la data/processed/
# 应该包含：train.csv, val.csv (可选), test.csv (可选)

# 如果缺失，运行数据准备脚本
python scripts/prepare_data.py --data-dir data/raw --output-dir data/processed
```

#### 问题：数据格式错误

**症状**：
```
KeyError: 'sequence' or 'structure'
```

**解决方案**：
```python
# 检查CSV文件格式
import pandas as pd
df = pd.read_csv('data/processed/train.csv')
print(df.columns.tolist())
# 应该包含：['sequence', 'structure', 'peptide_type', 'activity']
```

### 4. 评估相关问题

#### 问题：评估器初始化失败

**症状**：
```
ImportError: No module named 'modlamp'
```

**解决方案**：
```bash
# 安装缺失的依赖
pip install modlamp
pip install biopython
pip install fair-esm

# 或使用conda
conda install -c bioconda modlamp
```

#### 问题：ESMFold内存不足

**症状**：
```
CUDA out of memory during ESMFold prediction
```

**解决方案**：
```yaml
# 在配置文件中禁用ESMFold
model:
  structure_encoder:
    use_esmfold: false

# 或者使用CPU模式的ESMFold
evaluation:
  use_cpu_esmfold: true
```

### 5. 性能优化建议

#### 1. 训练加速

```yaml
# 启用混合精度训练
training_enhancements:
  use_amp: true
  amp_dtype: "float16"

# 优化数据加载
data:
  num_workers: 8        # 根据CPU核心数调整
  pin_memory: true
  prefetch_factor: 2

# 使用LoRA减少参数量
model:
  sequence_encoder:
    use_lora: true
    lora_rank: 16       # 可以尝试8, 16, 32
```

#### 2. 生成加速

```bash
# 使用DDIM加速采样
# 在配置文件中：
diffusion:
  sampling_method: "ddim"
  ddim_steps: 20        # 从50减少到20

# 或者使用脚本参数
python scripts/run_complete_pipeline.py \
    --fast-sampling \
    --ddim-steps 20
```

#### 3. 评估加速

```python
# 减少评估样本数量（开发阶段）
evaluation:
  generation:
    num_samples: 100    # 从1000减少

# 并行评估
evaluation:
  parallel_workers: 4  # 启用多进程评估
```

---

## 📈 性能监控和调试

### 1. 实时监控

#### W&B监控

```python
# 在配置文件中启用
monitoring:
  wandb:
    enabled: true
    project: "StructDiff-Production"
    log_frequency: 50

# 查看实时训练状态
# 浏览器打开: https://wandb.ai/your-username/StructDiff-Production
```

#### TensorBoard监控

```bash
# 启动TensorBoard
tensorboard --logdir outputs/experiment_name/tensorboard

# 浏览器打开: http://localhost:6006
```

### 2. 调试模式

#### 开发调试

```yaml
# 在配置文件中启用调试
debug:
  enabled: true
  use_small_dataset: true
  small_dataset_size: 100
  detailed_logging: true
  save_intermediate_results: true
```

#### 详细日志

```bash
# 启用详细日志
python scripts/run_complete_pipeline.py \
    --config configs/separated_training.yaml \
    --log-level DEBUG \
    --save-intermediate
```

### 3. 性能分析

#### GPU利用率监控

```bash
# 在另一个终端运行
nvidia-smi -l 1

# 或使用专门的监控工具
pip install gpustat
gpustat -i 1
```

#### 内存使用分析

```python
import psutil
import GPUtil

def monitor_resources():
    """监控系统资源使用"""
    # CPU和内存
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    
    # GPU
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.memoryUtil*100:.1f}% memory, {gpu.load*100:.1f}% load")
```

---

## 🔮 扩展和定制

### 1. 添加新的肽段类型

```python
# 在generation阶段添加新类型
def run_generation_stage(...):
    # 添加新的肽段类型
    peptide_types = args.peptide_types + ['anticancer', 'antimalarial']
    
    for peptide_type in peptide_types:
        # 根据类型调整生成参数
        if peptide_type == 'anticancer':
            length_range = (15, 35)  # 抗癌肽通常较长
        elif peptide_type == 'antimalarial':
            length_range = (10, 25)  # 抗疟疾肽中等长度
```

### 2. 集成新的评估指标

```python
# 扩展评估器
class ExtendedEvaluator(CPLDiffStandardEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def evaluate_additional_metrics(self, sequences):
        """添加新的评估指标"""
        results = {}
        
        # 示例：添加疏水性评估
        hydrophobicity_scores = []
        for seq in sequences:
            score = self.calculate_hydrophobicity(seq)
            hydrophobicity_scores.append(score)
        
        results['hydrophobicity'] = {
            'mean': np.mean(hydrophobicity_scores),
            'std': np.std(hydrophobicity_scores),
            'distribution': hydrophobicity_scores
        }
        
        return results
```

### 3. 自定义报告格式

```python
def generate_custom_report(results, output_path):
    """生成自定义格式的报告"""
    
    # LaTeX报告
    latex_template = r"""
    \documentclass{article}
    \begin{document}
    \title{StructDiff Experiment Report}
    \section{Results}
    Training Loss: {{ training_loss }}
    Generation Count: {{ generation_count }}
    \end{document}
    """
    
    # 使用Jinja2模板引擎
    from jinja2 import Template
    template = Template(latex_template)
    
    report_content = template.render(
        training_loss=results['training']['final_loss'],
        generation_count=results['generation']['total_count']
    )
    
    with open(output_path / "report.tex", 'w') as f:
        f.write(report_content)
```

---

## 📚 相关文档链接

- [配置文件详解](./separated_training_config_guide.md) - 详细的配置参数说明
- [CPL-Diff评估体系](./cpldiff_evaluation_guide.md) - 评估指标和标准
- [模型架构说明](./model_architecture_guide.md) - StructDiff模型设计
- [数据准备指南](./data_preparation_guide.md) - 数据格式和预处理
- [部署和生产指南](./deployment_guide.md) - 生产环境部署建议

---

*本文档将持续更新，如有疑问或建议，请提交Issue或联系项目维护团队。*