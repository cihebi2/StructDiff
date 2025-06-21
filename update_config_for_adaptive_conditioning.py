#!/usr/bin/env python3
"""
为自适应条件化更新配置文件
"""

import yaml
from pathlib import Path

def update_config_file():
    """更新配置文件以支持自适应条件化"""
    
    config_file = "configs/peptide_esmfold_config.yaml"
    
    # 读取现有配置
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 添加自适应条件化配置
    adaptive_conditioning_config = {
        'adaptive_conditioning': {
            'enabled': True,
            'condition_dim_ratio': 0.5,  # condition_dim = hidden_dim * ratio
            'num_condition_types': 4,   # antimicrobial, antifungal, antiviral, unconditioned
            'dropout': 0.1,
            
            # 多层次条件控制
            'multi_aspect_control': {
                'charge_control': True,
                'hydrophobic_control': True, 
                'structure_control': True,
                'functional_control': True
            },
            
            # 生物学启发的初始化
            'biological_initialization': {
                'antimicrobial_bias': 0.5,    # 正电荷偏向
                'antifungal_bias': 0.0,       # 中性平衡
                'antiviral_bias': -0.2,       # 轻微疏水偏向
                'unconditioned_bias': 0.0     # 完全中性
            },
            
            # 自适应强度控制
            'strength_control': {
                'enabled': True,
                'default_strength': 1.0,
                'strength_range': [0.1, 2.0],
                'adaptive_strength_learning': True
            },
            
            # 零初始化策略
            'zero_initialization': {
                'primary_gate_bias': -2.0,   # sigmoid(-2) ≈ 0.12
                'fine_gate_bias': -3.0,      # sigmoid(-3) ≈ 0.05
                'condition_networks_zero_init': True
            }
        }
    }
    
    # 更新模型配置
    if 'model' not in config:
        config['model'] = {}
    
    config['model'].update(adaptive_conditioning_config)
    
    # 更新去噪器配置以启用自适应条件化
    if 'denoiser' in config['model']:
        config['model']['denoiser']['use_adaptive_conditioning'] = True
        config['model']['denoiser']['enhanced_layer_norm'] = True
        config['model']['denoiser']['conditional_zero_init'] = True
    
    # 添加训练时的条件化策略
    conditioning_training = {
        'conditioning_training': {
            # 条件强度调度
            'strength_scheduling': {
                'enabled': True,
                'initial_strength': 0.8,
                'final_strength': 1.2,
                'warmup_steps': 1000,
                'schedule_type': 'cosine'  # linear, cosine, exponential
            },
            
            # 条件混合策略
            'condition_mixing': {
                'enabled': True,
                'mixing_probability': 0.1,  # 10%概率混合不同条件
                'unconditioned_probability': 0.1  # 10%概率无条件训练
            },
            
            # 自适应损失权重
            'adaptive_loss_weights': {
                'condition_consistency_weight': 0.1,
                'condition_specificity_weight': 0.05,
                'condition_smoothness_weight': 0.02
            }
        }
    }
    
    if 'training' not in config:
        config['training'] = {}
    
    config['training'].update(conditioning_training)
    
    # 更新评估配置
    evaluation_updates = {
        'condition_specific_evaluation': {
            'enabled': True,
            'evaluate_per_condition': True,
            'condition_transfer_evaluation': True,  # 评估条件间迁移
            'condition_interpolation_test': True    # 测试条件插值
        }
    }
    
    if 'evaluation' not in config:
        config['evaluation'] = {}
    
    config['evaluation'].update(evaluation_updates)
    
    # 保存更新的配置
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"✅ 配置文件已更新: {config_file}")
    
    # 创建自适应条件化的专用配置文件
    adaptive_config_file = "configs/peptide_adaptive_conditioning.yaml"
    
    # 基于原配置创建专门的自适应条件化配置
    adaptive_config = config.copy()
    adaptive_config['experiment']['name'] = "peptide_adaptive_conditioning"
    adaptive_config['experiment']['project'] = "StructDiff-AdaptiveConditioning"
    
    # 增强自适应条件化的参数
    adaptive_config['model']['adaptive_conditioning']['strength_control']['adaptive_strength_learning'] = True
    adaptive_config['model']['adaptive_conditioning']['multi_aspect_control'] = {
        'charge_control': True,
        'hydrophobic_control': True,
        'structure_control': True, 
        'functional_control': True,
        'fine_grained_modulation': True
    }
    
    with open(adaptive_config_file, 'w', encoding='utf-8') as f:
        yaml.dump(adaptive_config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"✅ 专用配置文件已创建: {adaptive_config_file}")
    
    return True

def create_usage_examples():
    """创建使用示例"""
    
    examples_content = '''# AlphaFold3自适应条件化使用示例

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
'''

    with open("ADAPTIVE_CONDITIONING_USAGE.md", 'w', encoding='utf-8') as f:
        f.write(examples_content)
    
    print("✅ 使用示例文档已创建: ADAPTIVE_CONDITIONING_USAGE.md")

def main():
    """主函数"""
    print("🔧 为自适应条件化更新配置...")
    
    try:
        # 更新配置文件
        update_config_file()
        
        # 创建使用示例
        create_usage_examples()
        
        print("\n🎉 自适应条件化配置更新完成!")
        print("\n📋 更新内容:")
        print("  ✅ 主配置文件已更新 (peptide_esmfold_config.yaml)")
        print("  ✅ 专用配置文件已创建 (peptide_adaptive_conditioning.yaml)")
        print("  ✅ 使用示例文档已创建 (ADAPTIVE_CONDITIONING_USAGE.md)")
        
        print("\n🚀 下一步:")
        print("  1. 运行测试: python3 test_adaptive_conditioning.py")
        print("  2. 使用新配置训练: --config configs/peptide_adaptive_conditioning.yaml")
        print("  3. 观察条件特异性改进效果")
        
    except Exception as e:
        print(f"❌ 配置更新失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()