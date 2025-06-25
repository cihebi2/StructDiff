# StructDiff 分离式训练配置文件详解

## 概述

`separated_training.yaml` 是 StructDiff 分离式训练系统的核心配置文件。该配置系统基于 CPL-Diff 的两阶段训练策略设计，将去噪器训练和序列解码器训练分离，以提高训练效率和模型性能。

---

## 📋 配置结构总览

```yaml
experiment:           # 实验基础信息
model:               # 模型架构配置
diffusion:           # 扩散过程配置
separated_training:  # 分离式训练核心配置
data:                # 数据处理配置
length_control:      # 长度控制机制
classifier_free_guidance:  # CFG引导配置
training_enhancements:     # 训练增强技术
evaluation:          # 评估系统配置
output:              # 输出管理配置
monitoring:          # 实验监控配置
debug:               # 调试和开发配置
resources:           # 硬件资源配置
```

---

## 📝 详细配置参数说明

### 🔬 1. 实验配置 (experiment)

```yaml
experiment:
  name: "structdiff_separated_training"    # 实验名称，用于文件命名和日志标识
  description: "两阶段分离式训练..."        # 实验描述，用于文档和报告
  project: "StructDiff-Separated"         # 项目名称，用于组织相关实验
  seed: 42                                 # 随机种子，确保结果可复现
```

**参数说明：**
- `name`: 实验的唯一标识符，会出现在日志、检查点文件名中
- `description`: 详细描述实验目的和方法，用于生成报告
- `project`: 项目级别的分组，便于管理多个相关实验
- `seed`: 控制所有随机过程的种子，确保实验可重复

### 🧠 2. 模型配置 (model)

#### 2.1 序列编码器 (sequence_encoder)

```yaml
sequence_encoder:
  pretrained_model: "facebook/esm2_t6_8M_UR50D"  # 预训练ESM-2模型
  freeze_encoder: false                          # 是否冻结编码器权重
  use_lora: true                                 # 启用LoRA微调
  lora_rank: 16                                  # LoRA低秩矩阵维度
  lora_alpha: 32                                 # LoRA缩放因子
  lora_dropout: 0.1                              # LoRA dropout率
```

**参数详解：**
- `pretrained_model`: ESM-2预训练模型路径，提供蛋白质序列的预训练表示
- `freeze_encoder`: 在阶段1训练时动态设置为`true`，避免破坏预训练知识
- `use_lora`: 使用LoRA (Low-Rank Adaptation) 进行参数高效微调
- `lora_rank`: 控制LoRA矩阵的秩，越小参数越少但表达能力越弱
- `lora_alpha`: LoRA的学习率缩放，通常设为rank的2倍
- `lora_dropout`: 防止LoRA层过拟合的dropout率

#### 2.2 结构编码器 (structure_encoder)

```yaml
structure_encoder:
  type: "multi_scale"                    # 多尺度结构编码器
  hidden_dim: 256                        # 隐藏层维度
  use_esmfold: false                     # 是否使用ESMFold预测结构
  
  local:                                 # 局部结构特征
    hidden_dim: 256
    num_layers: 3
    kernel_sizes: [3, 5, 7]              # 多尺度卷积核
    dropout: 0.1
    
  global:                                # 全局结构特征
    hidden_dim: 512
    num_attention_heads: 8
    num_layers: 4
    dropout: 0.1
    
  fusion:                                # 特征融合
    method: "attention"                  # 融合方法
    hidden_dim: 256
```

**设计思路：**
- `multi_scale`: 同时捕捉局部和全局结构信息
- `local`: 使用多尺度卷积捕捉短程结构模式
- `global`: 使用Transformer捕捉长程依赖关系
- `fusion`: 通过注意力机制融合多尺度特征

#### 2.3 去噪器 (denoiser)

```yaml
denoiser:
  hidden_dim: 768                        # 隐藏层维度，与ESM-2对齐
  num_layers: 12                         # Transformer层数
  num_heads: 12                          # 注意力头数
  dropout: 0.1                           # Dropout率
  use_cross_attention: true              # 启用跨注意力机制
```

**架构要点：**
- `hidden_dim: 768`: 与ESM-2的768维特征对齐，便于特征融合
- `num_layers: 12`: 足够的层数来学习复杂的去噪模式
- `use_cross_attention`: 允许序列特征与结构特征交互

#### 2.4 序列解码器 (sequence_decoder)

```yaml
sequence_decoder:
  hidden_dim: 768                        # 与去噪器维度保持一致
  num_layers: 6                          # 解码器层数（比去噪器少）
  vocab_size: 33                         # ESM-2词汇表大小
  dropout: 0.1                           # Dropout率
```

**设计考虑：**
- `num_layers: 6`: 比去噪器层数少，专注于序列重建任务
- `vocab_size: 33`: 对应ESM-2的氨基酸词汇表

### 🌊 3. 扩散过程配置 (diffusion)

```yaml
diffusion:
  num_timesteps: 1000                    # 扩散步数
  noise_schedule: "sqrt"                 # 噪声调度类型
  beta_start: 0.0001                     # 初始噪声水平
  beta_end: 0.02                         # 最终噪声水平
  
  sampling_method: "ddpm"                # 采样方法
  ddim_steps: 50                         # DDIM加速采样步数
```

**参数含义：**
- `noise_schedule: "sqrt"`: CPL-Diff推荐的平方根调度，提供更好的训练稳定性
- `beta_start/end`: 控制噪声添加的速度和强度
- `ddim_steps: 50`: 推理时使用DDIM加速，从1000步减少到50步

### 🎯 4. 分离式训练配置 (separated_training)

#### 4.1 阶段1：去噪器训练

```yaml
stage1:
  epochs: 200                            # 训练轮数
  batch_size: 32                         # 批次大小
  learning_rate: 1e-4                    # 学习率
  warmup_steps: 1000                     # 预热步数
  gradient_clip: 1.0                     # 梯度裁剪
  
  optimizer:
    type: "AdamW"                        # 优化器类型
    weight_decay: 0.01                   # 权重衰减
    betas: [0.9, 0.999]                  # Adam动量参数
    eps: 1e-8                            # 数值稳定性参数
    
  scheduler:
    type: "cosine"                       # 余弦学习率调度
    eta_min: 1e-6                        # 最小学习率
```

**训练策略：**
- **目标**: 训练去噪器学习从噪声中恢复特征
- **特点**: 冻结序列编码器，专注于去噪能力
- **学习率**: 较高的初始学习率，快速学习去噪模式

#### 4.2 阶段2：解码器训练

```yaml
stage2:
  epochs: 100                            # 较少的训练轮数
  batch_size: 64                         # 更大的批次大小
  learning_rate: 5e-5                    # 更低的学习率
  warmup_steps: 500                      # 较少的预热步数
  gradient_clip: 0.5                     # 更严格的梯度裁剪
```

**训练策略：**
- **目标**: 训练序列解码器将特征转换为氨基酸序列
- **特点**: 冻结去噪器，专注于序列重建
- **学习率**: 更低的学习率，精细调整解码能力

### 📊 5. 数据配置 (data)

```yaml
data:
  data_dir: "./data/processed"           # 数据目录
  train_file: "train.csv"                # 训练集文件
  val_file: "val.csv"                    # 验证集文件
  test_file: "test.csv"                  # 测试集文件
  
  max_length: 50                         # 最大序列长度
  min_length: 5                          # 最小序列长度
  
  num_workers: 4                         # 数据加载进程数
  pin_memory: true                       # 启用内存锁定
  prefetch_factor: 2                     # 预取因子
```

**数据处理要点：**
- `max/min_length`: 控制序列长度范围，过滤异常数据
- `num_workers`: 多进程数据加载，提高I/O效率
- `pin_memory`: 将数据锁定在内存中，加速GPU传输

### 📏 6. 长度控制配置 (length_control)

```yaml
length_control:
  enabled: true                          # 启用长度控制
  min_length: 5                          # 最小生成长度
  max_length: 50                         # 最大生成长度
  
  analyze_training_data: true            # 分析训练数据长度分布
  save_distributions: true               # 保存分布统计
  length_penalty_weight: 0.1             # 长度惩罚权重
  
  type_specific_lengths:                 # 肽段类型特定长度
    antimicrobial: [20, 8]               # [均值, 标准差]
    antifungal: [25, 10]
    antiviral: [30, 12]
    general: [25, 5]
```

**长度控制机制：**
- **统计驱动**: 基于训练数据的真实长度分布
- **类型特定**: 不同肽段类型有不同的长度偏好
- **生成约束**: 在生成过程中施加长度约束

### 🎨 7. 分类器自由引导配置 (classifier_free_guidance)

```yaml
classifier_free_guidance:
  enabled: true                          # 启用CFG
  dropout_prob: 0.1                      # 条件丢弃概率
  guidance_scale: 2.0                    # 引导强度
  
  adaptive_guidance: true                # 自适应引导
  guidance_schedule: "cosine"            # 引导调度策略
```

**CFG工作原理：**
- **训练时**: 随机丢弃条件信息，学习无条件和有条件生成
- **推理时**: 使用引导强度增强条件控制能力
- **自适应**: 根据生成质量动态调整引导强度

### 🚀 8. 训练增强配置 (training_enhancements)

```yaml
training_enhancements:
  use_amp: true                          # 混合精度训练
  amp_dtype: "float16"                   # AMP数据类型
  
  use_ema: true                          # 指数移动平均
  ema_decay: 0.9999                      # EMA衰减率
  ema_update_every: 10                   # EMA更新频率
  
  gradient_accumulation_steps: 1         # 梯度累积步数
  
  save_every: 1000                       # 检查点保存频率
  validate_every: 500                    # 验证频率
  log_every: 100                         # 日志记录频率
  max_checkpoints: 5                     # 最大检查点数量
```

**优化技术：**
- **AMP**: 使用float16减少内存占用，提高训练速度
- **EMA**: 保持参数的移动平均，提高模型稳定性
- **梯度累积**: 模拟更大的批次大小

### 📈 9. 评估配置 (evaluation)

```yaml
evaluation:
  metrics:                               # CPL-Diff标准评估指标
    - pseudo_perplexity                  # ESM-2伪困惑度 ↓
    - plddt_score                        # ESMFold结构置信度 ↑
    - instability_index                  # modlAMP不稳定性指数 ↓
    - similarity_score                   # BLOSUM62相似性 ↓
    - activity_prediction                # 外部分类器活性预测 ↑
    - information_entropy                # 信息熵
    - novelty_ratio                      # 新颖性比例
  
  generation:
    num_samples: 1000                    # 评估样本数量
    guidance_scale: 2.0                  # 生成时引导强度
    temperature: 1.0                     # 采样温度
    use_length_control: true             # 使用长度控制
    
  evaluate_every: 5                      # 评估频率（每5个epoch）
```

**评估指标解读：**
- **↓**: 越低越好的指标
- **↑**: 越高越好的指标
- **核心指标**: 前5个是CPL-Diff论文的标准评估指标
- **辅助指标**: 后2个提供额外的质量评估

### 📁 10. 输出配置 (output)

```yaml
output:
  base_dir: "./outputs/separated_training"      # 基础输出目录
  checkpoint_dir: "./outputs/.../checkpoints"   # 检查点目录
  log_dir: "./outputs/.../logs"                 # 日志目录
  results_dir: "./outputs/.../results"          # 结果目录
  
  save_model_config: true               # 保存模型配置
  save_training_stats: true            # 保存训练统计
  save_generated_samples: true         # 保存生成样本
```

**文件组织结构：**
```
outputs/separated_training/
├── checkpoints/          # 模型检查点
├── logs/                 # 训练日志
├── results/              # 评估结果
├── generated_samples/    # 生成的序列
└── config_backup.yaml   # 配置备份
```

### 📊 11. 监控配置 (monitoring)

#### 11.1 Weights & Biases

```yaml
wandb:
  enabled: true                          # 启用W&B
  project: "StructDiff-Separated"        # W&B项目名
  entity: null                           # W&B实体（团队/用户）
  tags: ["separated-training", ...]      # 实验标签
  
  log_gradients: false                   # 记录梯度（消耗存储）
  log_parameters: false                  # 记录参数（消耗存储）
  log_frequency: 100                     # 记录频率
```

#### 11.2 TensorBoard

```yaml
tensorboard:
  enabled: true                          # 启用TensorBoard
  log_dir: "./outputs/.../tensorboard"   # TensorBoard日志目录
```

### 🐛 12. 调试配置 (debug)

```yaml
debug:
  enabled: false                         # 启用调试模式
  use_small_dataset: false               # 使用小数据集
  small_dataset_size: 1000               # 小数据集大小
  save_intermediate_results: false       # 保存中间结果
  detailed_logging: false                # 详细日志记录
```

**调试模式特点：**
- **快速验证**: 使用小数据集快速验证代码
- **详细输出**: 记录更多中间状态信息
- **开发友好**: 便于调试和开发新功能

### ⚙️ 13. 资源配置 (resources)

```yaml
resources:
  gpu_memory_fraction: 0.9               # GPU内存使用比例
  allow_growth: true                     # 允许GPU内存动态增长
  
  num_threads: 8                         # CPU线程数
  
  pin_memory: true                       # 内存锁定
  non_blocking: true                     # 非阻塞数据传输
```

**资源优化：**
- **GPU管理**: 控制GPU内存使用，避免OOM
- **CPU优化**: 合理设置线程数，提高并行效率
- **内存优化**: 使用内存锁定和非阻塞传输

---

## 🎯 配置使用指南

### 快速开始

```bash
# 使用默认配置
python scripts/train_separated.py --config configs/separated_training.yaml

# 自定义实验名称
python scripts/train_separated.py \
    --config configs/separated_training.yaml \
    --experiment-name my_experiment
```

### 配置定制

#### 1. 调整训练强度

```yaml
# 快速验证配置
separated_training:
  stage1:
    epochs: 20          # 减少训练轮数
    batch_size: 16      # 减少批次大小
  stage2:
    epochs: 10

# 生产环境配置
separated_training:
  stage1:
    epochs: 500         # 增加训练轮数
    batch_size: 64      # 增加批次大小
  stage2:
    epochs: 200
```

#### 2. 启用/禁用功能

```yaml
# 最小配置（快速测试）
length_control:
  enabled: false        # 禁用长度控制
classifier_free_guidance:
  enabled: false        # 禁用CFG
training_enhancements:
  use_amp: false        # 禁用混合精度
  use_ema: false        # 禁用EMA

# 完整功能配置
evaluation:
  evaluate_every: 1     # 每个epoch都评估
monitoring:
  wandb:
    enabled: true       # 启用完整监控
    log_gradients: true
```

### 性能调优建议

#### 内存优化

```yaml
# 小内存环境
separated_training:
  stage1:
    batch_size: 8       # 减少批次大小
    gradient_accumulation_steps: 4  # 增加梯度累积

training_enhancements:
  use_amp: true         # 启用混合精度
  amp_dtype: "float16"

model:
  sequence_encoder:
    use_lora: true      # 使用LoRA减少参数
    lora_rank: 8        # 降低LoRA秩
```

#### 速度优化

```yaml
# 高速训练配置
data:
  num_workers: 16       # 增加数据加载进程
  prefetch_factor: 4    # 增加预取

training_enhancements:
  use_amp: true         # 启用混合精度
  
resources:
  pin_memory: true      # 启用内存锁定
  non_blocking: true    # 非阻塞传输
```

---

## ⚠️ 常见问题和注意事项

### 1. 内存不足

**问题**: `CUDA out of memory`

**解决方案**:
```yaml
# 减少批次大小
separated_training:
  stage1:
    batch_size: 16    # 从32减少到16
  stage2:
    batch_size: 32    # 从64减少到32

# 启用梯度累积模拟大批次
training_enhancements:
  gradient_accumulation_steps: 2
```

### 2. 训练不稳定

**问题**: 损失震荡或NaN

**解决方案**:
```yaml
# 减小学习率
separated_training:
  stage1:
    learning_rate: 5e-5  # 从1e-4减少

# 增强梯度裁剪
separated_training:
  stage1:
    gradient_clip: 0.5   # 从1.0减少

# 启用EMA稳定训练
training_enhancements:
  use_ema: true
  ema_decay: 0.999
```

### 3. 评估失败

**问题**: 评估过程出错

**解决方案**:
```yaml
# 减少评估样本数量
evaluation:
  generation:
    num_samples: 100   # 从1000减少

# 调整评估频率
evaluation:
  evaluate_every: 10   # 从5增加到10
```

### 4. 配置兼容性

**重要提示**:
- ESM-2模型名称必须正确，确保能从HuggingFace下载
- 数据文件路径必须存在且格式正确
- GPU内存设置要根据实际硬件调整
- 不同阶段的批次大小可以不同，stage2通常可以更大

---

## 📚 扩展阅读

- [CPL-Diff原论文](https://arxiv.org/abs/xxxx) - 了解分离式训练的理论基础
- [ESM-2文档](https://github.com/facebookresearch/esm) - 了解序列编码器
- [扩散模型教程](https://arxiv.org/abs/2006.11239) - 了解扩散过程原理
- [LoRA论文](https://arxiv.org/abs/2106.09685) - 了解参数高效微调

---

*本文档持续更新，如有疑问请查看项目README或提交Issue。*