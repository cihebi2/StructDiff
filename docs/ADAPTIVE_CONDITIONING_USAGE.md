# AlphaFold3自适应条件化使用示例

## 1. 基本使用

```python
# 在训练脚本中添加条件强度控制
conditions = {
    'peptide_type': torch.tensor([0, 1, 2]),  # antimicrobial, antifungal, antiviral
    'condition_strength': torch.tensor([[1.0], [0.8], [1.2]])  # 可调强度
}

# 去噪器自动使用自适应条件化
denoised, cross_attn = model.denoiser(
    noisy_embeddings=noisy_emb,
    timesteps=timesteps,
    attention_mask=mask,
    structure_features=struct_features,
    conditions=conditions
)
```

## 2. 高级条件控制

```python
# 条件强度调度 (在训练循环中)
def get_adaptive_strength(epoch, total_epochs):
    # 余弦调度：从0.8到1.2
    progress = epoch / total_epochs
    strength = 0.8 + 0.4 * (1 + math.cos(math.pi * progress)) / 2
    return strength

# 条件混合策略
def mix_conditions(conditions, mixing_prob=0.1):
    if random.random() < mixing_prob:
        # 随机混合不同条件
        mixed_conditions = conditions.clone()
        mixed_conditions[torch.randperm(len(conditions))] = conditions
        return mixed_conditions
    return conditions
```

## 3. 条件特异性评估

```python
# 评估不同条件的生成效果
for condition_type in ['antimicrobial', 'antifungal', 'antiviral']:
    # 生成特定条件的序列
    generated_sequences = generate_with_condition(
        model=model,
        condition=condition_type,
        num_samples=100,
        strength=1.0
    )
    
    # 评估条件特异性
    specificity_score = evaluate_condition_specificity(
        sequences=generated_sequences,
        target_condition=condition_type
    )
    print(f"{condition_type} specificity: {specificity_score:.3f}")
```

## 4. 条件插值实验

```python
# 测试条件间的平滑过渡
def interpolate_conditions(model, condition_a, condition_b, steps=10):
    results = []
    for i in range(steps):
        alpha = i / (steps - 1)
        # 在embedding空间插值
        interpolated_condition = (1 - alpha) * condition_a + alpha * condition_b
        
        sequences = generate_with_interpolated_condition(
            model, interpolated_condition
        )
        results.append(sequences)
    return results

# 抗菌 -> 抗病毒 的渐变生成
antimicrobial_to_antiviral = interpolate_conditions(
    model, 
    condition_a=torch.tensor([0]),  # antimicrobial
    condition_b=torch.tensor([2])   # antiviral
)
```

## 5. 配置参数说明

```yaml
adaptive_conditioning:
  enabled: true
  condition_dim_ratio: 0.5          # 条件维度 = 隐藏维度 * 0.5
  
  multi_aspect_control:
    charge_control: true             # 电荷特征控制
    hydrophobic_control: true        # 疏水性控制
    structure_control: true          # 结构特征控制
    functional_control: true         # 功能特征控制
  
  biological_initialization:
    antimicrobial_bias: 0.5         # 正电荷偏向初始化
    antifungal_bias: 0.0            # 中性初始化
    antiviral_bias: -0.2            # 疏水偏向初始化
  
  strength_control:
    enabled: true
    default_strength: 1.0           # 默认强度
    adaptive_strength_learning: true # 学习自适应强度
```

## 6. 预期改进效果

- 🎯 **更精确的条件控制**: 细粒度调节不同功能特性
- 🧬 **生物学合理性**: 基于已知AMP特征的初始化
- ⚡ **训练稳定性**: 零初始化策略确保收敛
- 📊 **可解释性**: 分离的条件信号便于分析
- 🔧 **灵活性**: 支持条件强度调节和混合策略
