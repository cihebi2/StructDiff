# Classifier-Free Guidance + 长度分布采样器集成指南

## 📋 概述

本文档介绍了为StructDiff项目新集成的两个核心功能：

1. **Classifier-Free Guidance (CFG)** - 分类器自由引导机制
2. **Length Distribution Sampler** - 长度分布采样器

这两个功能的集成显著提升了StructDiff在多肽生成方面的控制能力和生成质量。

## 🎯 核心功能特性

### Classifier-Free Guidance (CFG)

#### 1. 基础原理
- **无分类器引导**: 通过训练时的条件丢弃和推理时的双路预测实现引导
- **CPL-Diff兼容**: 实现与CPL-Diff论文完全一致的CFG机制
- **自适应引导**: 支持时间步相关的动态引导强度调整

#### 2. 核心特性
```python
# CFG配置示例
cfg_config = CFGConfig(
    dropout_prob=0.15,          # 训练时条件丢弃概率
    guidance_scale=2.5,         # 默认引导强度
    adaptive_guidance=True,     # 自适应引导
    multi_level_guidance=False, # 多级引导（实验性）
    guidance_schedule="cosine"  # 引导强度调度
)
```

#### 3. 技术实现
- **训练时**: 随机丢弃条件信息，模型学习有条件和无条件生成
- **推理时**: 结合有条件和无条件预测，引导生成方向
- **公式**: `ε_guided = ε_uncond + w × (ε_cond - ε_uncond)`

### Length Distribution Sampler (长度分布采样器)

#### 1. 支持的分布类型
- **正态分布**: 适合自然长度分布
- **均匀分布**: 等概率长度采样
- **Gamma分布**: 适合右偏分布
- **Beta分布**: 适合有界分布
- **自定义分布**: 用户定义的离散分布

#### 2. 核心特性
```python
# 长度采样器配置示例
length_config = LengthSamplerConfig(
    min_length=5,
    max_length=50,
    distribution_type="normal",
    normal_mean=25.0,
    normal_std=8.0,
    use_adaptive_sampling=True,
    condition_dependent=True
)
```

#### 3. 自适应长度控制
- **条件相关**: 不同肽类型使用不同长度偏好
- **温度调节**: 控制长度采样的随机性
- **长度约束**: 生成过程中强制执行长度限制

## 🛠️ 使用方法

### 1. 基础CFG使用

```python
from structdiff.models.classifier_free_guidance import CFGConfig, ClassifierFreeGuidance

# 创建CFG配置
cfg_config = CFGConfig(
    dropout_prob=0.1,
    guidance_scale=2.5,
    adaptive_guidance=True
)

# 集成到现有模型
cfg = ClassifierFreeGuidance(cfg_config)

# 训练时条件处理
processed_conditions = cfg.prepare_conditions(
    conditions, batch_size, training=True
)

# 推理时引导采样
guided_output = cfg.guided_denoising(
    model, x_t, t, conditions, guidance_scale=2.5
)
```

### 2. 基础长度采样

```python
from structdiff.sampling.length_sampler import AdaptiveLengthSampler, LengthSamplerConfig

# 创建长度采样器
length_config = LengthSamplerConfig(
    distribution_type="normal",
    normal_mean=25.0,
    normal_std=8.0
)
length_sampler = AdaptiveLengthSampler(length_config)

# 采样长度
lengths = length_sampler.sample_lengths(
    batch_size=16,
    conditions=conditions,
    temperature=1.0
)

# 创建长度掩码
mask = constrainer.create_length_mask(lengths, max_length)
```

### 3. 集成使用示例

```python
from scripts.cfg_length_integrated_sampling import CFGLengthIntegratedSampler

# 创建集成采样器
sampler = CFGLengthIntegratedSampler(
    denoiser=denoiser,
    diffusion=diffusion,
    cfg_config=cfg_config,
    length_config=length_config
)

# 集成采样配置
sampling_config = IntegratedSamplingConfig(
    cfg_guidance_scale=2.5,
    length_distribution="normal",
    length_mean=25.0,
    peptide_types=['antimicrobial', 'antifungal', 'antiviral']
)

# 执行采样
results = sampler.sample_with_cfg_and_length(sampling_config)
```

## 📊 配置文件使用

### 完整配置示例 (`configs/cfg_length_config.yaml`)

```yaml
# Classifier-Free Guidance配置
classifier_free_guidance:
  enabled: true
  dropout_prob: 0.15
  guidance_scale: 2.5
  adaptive_guidance: true
  multi_level_guidance: false

# 长度分布采样器配置
length_sampler:
  enabled: true
  distribution_type: "normal"
  normal_mean: 25.0
  normal_std: 8.0
  condition_dependent: true
  peptide_type_length_prefs:
    antimicrobial: [20.0, 8.0]
    antifungal: [25.0, 10.0]
    antiviral: [30.0, 12.0]

# 采样配置
sampling:
  use_cfg: true
  cfg_guidance_scale: 2.5
  use_length_sampler: true
  length_distribution: "normal"
```

## 🔧 模型架构集成

### 去噪器增强

现有的`StructureAwareDenoiser`已经增强以支持CFG：

```python
# 创建支持CFG的去噪器
denoiser = StructureAwareDenoiser(
    seq_hidden_dim=768,
    struct_hidden_dim=768,
    denoiser_config=denoiser_config,
    cfg_config=cfg_config  # 添加CFG配置
)

# 前向传播支持CFG参数
output = denoiser(
    noisy_embeddings, timesteps, attention_mask,
    conditions=conditions,
    use_cfg=True,              # 启用CFG
    guidance_scale=2.5,        # 引导强度
    timestep_idx=step_idx,     # 用于自适应引导
    total_timesteps=total_steps
)
```

### 训练流程修改

```python
# CFG训练模式
if cfg_enabled and model.training:
    # 自动应用条件丢弃
    loss = model(x_t, t, attention_mask, conditions=conditions)
    
# CFG推理模式
if cfg_enabled and not model.training:
    # 自动应用引导采样
    output = model(
        x_t, t, attention_mask, 
        conditions=conditions,
        use_cfg=True,
        guidance_scale=guidance_scale
    )
```

## 📈 性能优化建议

### 1. CFG优化

```python
# 批量化CFG计算
batch_size = x_t.shape[0]
x_doubled = torch.cat([x_t, x_t], dim=0)
t_doubled = torch.cat([t, t], dim=0)
cond_doubled = torch.cat([cond, uncond], dim=0)

# 单次前向传播
output_doubled = model(x_doubled, t_doubled, cond_doubled)
cond_out, uncond_out = output_doubled.chunk(2, dim=0)

# CFG组合
guided_out = uncond_out + guidance_scale * (cond_out - uncond_out)
```

### 2. 长度采样优化

```python
# 预计算长度分布
length_probs = sampler.get_length_probabilities(conditions)

# 批量长度采样
lengths = torch.multinomial(length_probs, 1).squeeze(-1)

# 高效掩码创建
positions = torch.arange(max_length).unsqueeze(0)
mask = positions < lengths.unsqueeze(1)
```

### 3. 内存优化

```python
# 梯度检查点
torch.utils.checkpoint.checkpoint(
    cfg_forward_func, x_t, t, conditions
)

# 混合精度
with torch.cuda.amp.autocast():
    output = cfg.guided_denoising(model, x_t, t, conditions)
```

## 🧪 测试和验证

### 运行测试套件

```bash
# 运行完整测试
python tests/test_cfg_length_integration.py

# 运行性能测试
python -m pytest tests/test_cfg_length_integration.py::TestPerformance -v

# 运行集成演示
python scripts/cfg_length_integrated_sampling.py --num_samples 100
```

### 验证指标

#### CFG验证
- **引导效果**: 不同引导强度下的生成差异
- **条件控制**: 条件准确性和多样性平衡
- **计算效率**: CFG相对于标准采样的开销

#### 长度采样验证
- **分布拟合**: 采样长度分布与目标分布的拟合度
- **条件响应**: 不同条件下长度分布的差异
- **约束满足**: 长度约束的执行效果

### 质量评估

```python
# 使用CPL-Diff标准评估
from scripts.cpldiff_standard_evaluation import CPLDiffStandardEvaluator

evaluator = CPLDiffStandardEvaluator()
results = evaluator.comprehensive_cpldiff_evaluation(
    generated_sequences=cfg_length_generated,
    reference_sequences=reference_data,
    peptide_type='antimicrobial'
)

# 对比CFG效果
cfg_results = evaluate_with_cfg(guidance_scale=2.5)
no_cfg_results = evaluate_with_cfg(guidance_scale=1.0)
```

## 🎯 应用场景

### 1. 条件生成增强
- **精确控制**: 通过CFG精确控制生成的肽类型
- **质量提升**: 提高生成序列的功能特异性
- **多样性平衡**: 在控制性和多样性之间找到平衡

### 2. 长度定制生成
- **目标长度**: 生成特定长度范围的肽段
- **分布匹配**: 匹配自然肽段的长度分布
- **约束满足**: 满足下游应用的长度要求

### 3. 多目标优化
- **联合控制**: 同时控制类型、长度、性质
- **梯度引导**: 使用多个目标的联合引导
- **自适应调整**: 根据生成进度动态调整引导强度

## 🔮 未来扩展

### 1. 高级CFG功能
- **层级引导**: 不同层级的条件引导
- **动态引导**: 基于生成质量的自适应引导
- **多模态引导**: 结合序列、结构、功能的联合引导

### 2. 长度采样增强
- **序列相关长度**: 基于已生成部分预测剩余长度
- **结构约束长度**: 基于结构预测调整长度分布
- **功能优化长度**: 基于功能预测优化长度选择

### 3. 集成优化
- **端到端训练**: CFG和长度采样的联合训练
- **强化学习**: 使用RL进一步优化生成策略
- **对抗训练**: 引入判别器提升生成质量

---

## 💡 最佳实践

1. **CFG引导强度**: 建议从2.0-3.0开始，根据任务调整
2. **条件丢弃率**: 训练时使用10-15%的丢弃率
3. **长度分布选择**: 根据数据特性选择合适的分布类型
4. **自适应引导**: 在长序列生成时启用自适应引导
5. **性能监控**: 定期评估CFG对计算效率的影响

这套集成的CFG和长度采样器系统为StructDiff提供了强大的条件生成能力，使其能够生成更高质量、更可控的功能性肽段。