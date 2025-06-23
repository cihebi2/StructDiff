#!/usr/bin/env python3
"""
测试AlphaFold3自适应条件化集成
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_af3_adaptive_conditioning():
    """测试AF3自适应条件化组件"""
    print("🧪 测试AF3自适应条件化组件...")
    
    try:
        from structdiff.models.layers.alphafold3_embeddings import (
            AF3AdaptiveConditioning,
            AF3EnhancedConditionalLayerNorm,
            AF3ConditionalZeroInit
        )
        
        batch_size = 4
        seq_len = 20
        hidden_dim = 256
        condition_dim = hidden_dim // 2
        
        # Test adaptive conditioning
        adaptive_cond = AF3AdaptiveConditioning(
            hidden_dim=hidden_dim,
            condition_dim=condition_dim,
            num_condition_types=4
        )
        
        # Test condition indices: 0=antimicrobial, 1=antifungal, 2=antiviral, 3=unconditioned
        condition_indices = torch.tensor([0, 1, 2, 3])
        conditioning_signals = adaptive_cond(condition_indices)
        
        print(f"✅ AF3AdaptiveConditioning: 输入{condition_indices.shape} -> 输出{len(conditioning_signals)}个信号")
        print(f"   Enhanced condition: {conditioning_signals['enhanced_condition'].shape}")
        print(f"   Charge signal: {conditioning_signals['charge_signal'].shape}")
        print(f"   Hydrophobic signal: {conditioning_signals['hydrophobic_signal'].shape}")
        
        # Test enhanced conditional layer norm
        enhanced_norm = AF3EnhancedConditionalLayerNorm(hidden_dim, condition_dim)
        x = torch.randn(batch_size, seq_len, hidden_dim)
        norm_out = enhanced_norm(x, conditioning_signals)
        print(f"✅ AF3EnhancedConditionalLayerNorm: {x.shape} -> {norm_out.shape}")
        
        # Test conditional zero init
        zero_init = AF3ConditionalZeroInit(hidden_dim, hidden_dim, condition_dim)
        zero_out = zero_init(x, conditioning_signals)
        print(f"✅ AF3ConditionalZeroInit: {x.shape} -> {zero_out.shape}")
        
        return True
    except Exception as e:
        print(f"❌ AF3自适应条件化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_denoiser_integration():
    """测试去噪器集成"""
    print("\n🔧 测试去噪器集成...")
    
    try:
        from structdiff.models.denoise import StructureAwareDenoiser
        
        # 配置
        denoiser_config = type('obj', (object,), {
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1,
            'use_cross_attention': True
        })()
        
        # 创建去噪器
        denoiser = StructureAwareDenoiser(
            seq_hidden_dim=768,  # ESM-2 hidden dim
            struct_hidden_dim=320,  # 结构特征维度
            denoiser_config=denoiser_config
        )
        
        # 测试输入
        batch_size = 2
        seq_len = 20
        noisy_embeddings = torch.randn(batch_size, seq_len, 768)
        timesteps = torch.randint(0, 1000, (batch_size,))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        structure_features = torch.randn(batch_size, seq_len, 320)
        
        # 测试条件
        conditions = {
            'peptide_type': torch.tensor([0, 1]),  # antimicrobial, antifungal
            'condition_strength': torch.tensor([[1.0], [0.8]])  # 可选强度控制
        }
        
        # 前向传播
        denoised, cross_attn = denoiser(
            noisy_embeddings=noisy_embeddings,
            timesteps=timesteps,
            attention_mask=attention_mask,
            structure_features=structure_features,
            conditions=conditions
        )
        
        print(f"✅ Enhanced StructureAwareDenoiser: 输入{noisy_embeddings.shape} -> 输出{denoised.shape}")
        print(f"   参数数量: {sum(p.numel() for p in denoiser.parameters()):,}")
        print(f"   自适应条件化已集成")
        
        return True
    except Exception as e:
        print(f"❌ 去噪器集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_condition_specificity():
    """测试条件特异性"""
    print("\n🎯 测试条件特异性...")
    
    try:
        from structdiff.models.layers.alphafold3_embeddings import AF3AdaptiveConditioning
        
        adaptive_cond = AF3AdaptiveConditioning(
            hidden_dim=256,
            condition_dim=128,
            num_condition_types=4
        )
        
        # 测试不同条件类型的输出差异
        condition_types = ["antimicrobial", "antifungal", "antiviral", "unconditioned"]
        
        for i, cond_type in enumerate(condition_types):
            condition_indices = torch.tensor([i])
            signals = adaptive_cond(condition_indices)
            
            enhanced_cond = signals['enhanced_condition']
            charge_signal = signals['charge_signal']
            hydrophobic_signal = signals['hydrophobic_signal']
            
            print(f"📊 {cond_type}:")
            print(f"   Enhanced condition mean: {enhanced_cond.mean().item():.4f}")
            print(f"   Charge signal mean: {charge_signal.mean().item():.4f}")
            print(f"   Hydrophobic signal mean: {hydrophobic_signal.mean().item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ 条件特异性测试失败: {e}")
        return False

def test_performance_impact():
    """测试性能影响"""
    print("\n⚡ 测试性能影响...")
    
    try:
        import time
        from structdiff.models.denoise import StructureAwareDenoiser
        
        # 配置
        config = type('obj', (object,), {
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1,
            'use_cross_attention': True
        })()
        
        # 创建去噪器
        denoiser = StructureAwareDenoiser(768, 320, config)
        
        # 测试数据
        batch_size = 8
        seq_len = 32
        noisy_embeddings = torch.randn(batch_size, seq_len, 768)
        timesteps = torch.randint(0, 1000, (batch_size,))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        structure_features = torch.randn(batch_size, seq_len, 320)
        conditions = {'peptide_type': torch.randint(0, 4, (batch_size,))}
        
        # 预热
        for _ in range(5):
            with torch.no_grad():
                _ = denoiser(noisy_embeddings, timesteps, attention_mask, structure_features, conditions)
        
        # 性能测试
        num_runs = 20
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = denoiser(noisy_embeddings, timesteps, attention_mask, structure_features, conditions)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        print(f"✅ 性能测试完成:")
        print(f"   平均推理时间: {avg_time*1000:.2f}ms")
        print(f"   吞吐量: {batch_size/avg_time:.1f} samples/s")
        
        # 参数分析
        total_params = sum(p.numel() for p in denoiser.parameters())
        adaptive_params = sum(p.numel() for n, p in denoiser.named_parameters() if 'adaptive' in n.lower())
        
        print(f"   总参数: {total_params:,}")
        print(f"   自适应条件化参数: {adaptive_params:,} ({adaptive_params/total_params*100:.1f}%)")
        
        return True
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 测试AlphaFold3自适应条件化集成")
    print("=" * 60)
    
    tests = [
        ("AF3自适应条件化组件", test_af3_adaptive_conditioning),
        ("去噪器集成", test_denoiser_integration),
        ("条件特异性", test_condition_specificity),
        ("性能影响", test_performance_impact),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"❌ {name}: 测试过程出错 - {e}")
            results[name] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 测试总结:")
    
    success_count = 0
    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
        if result:
            success_count += 1
    
    print(f"\n总体结果: {'✅ 所有测试通过' if success_count == len(results) else '❌ 存在问题'}")
    print(f"成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if success_count == len(results):
        print("\n🎉 AlphaFold3自适应条件化已成功集成!")
        print("\n💡 新功能亮点:")
        print("  🎯 多层次条件控制 (电荷、疏水性、结构、功能)")
        print("  🧬 生物学启发的初始化模式")
        print("  ⚡ 零初始化保证训练稳定性")
        print("  🔧 细粒度功能性条件调制")
        print("  📊 可解释的条件信号分离")
    else:
        print("\n⚠️ 请解决测试失败的问题")

if __name__ == "__main__":
    main()