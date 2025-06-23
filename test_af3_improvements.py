#!/usr/bin/env python3
"""
测试AlphaFold3改进组件
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_af3_noise_schedule():
    """测试AlphaFold3噪声调度"""
    print("测试AF3噪声调度...")
    
    from structdiff.diffusion.noise_schedule import get_noise_schedule
    
    # 测试所有噪声调度类型
    schedules = ["linear", "cosine", "sqrt", "alphafold3"]
    num_timesteps = 100
    
    for schedule in schedules:
        try:
            betas = get_noise_schedule(schedule, num_timesteps)
            print(f"✅ {schedule}: shape={betas.shape}, range=[{betas.min():.6f}, {betas.max():.6f}]")
            
            # 检查是否单调递增
            if schedule != "alphafold3":  # AF3调度可能不完全单调
                is_monotonic = np.all(np.diff(betas) >= 0)
                print(f"   单调递增: {is_monotonic}")
                
        except Exception as e:
            print(f"❌ {schedule}: 错误 - {e}")
    
    print()

def test_af3_embeddings():
    """测试AlphaFold3嵌入层"""
    print("测试AF3嵌入层...")
    
    from structdiff.models.layers.alphafold3_embeddings import (
        AF3FourierEmbedding, 
        AF3TimestepEmbedding,
        AF3AdaptiveLayerNorm
    )
    
    batch_size = 4
    seq_len = 20
    hidden_dim = 256
    
    # 测试Fourier嵌入
    try:
        fourier_emb = AF3FourierEmbedding(embedding_dim=hidden_dim)
        timesteps = torch.randint(0, 1000, (batch_size,))
        fourier_out = fourier_emb(timesteps)
        print(f"✅ AF3FourierEmbedding: {fourier_out.shape}")
    except Exception as e:
        print(f"❌ AF3FourierEmbedding: {e}")
    
    # 测试时间嵌入
    try:
        time_emb = AF3TimestepEmbedding(hidden_dim)
        timesteps = torch.randint(0, 1000, (batch_size,))
        time_out = time_emb(timesteps)
        print(f"✅ AF3TimestepEmbedding: {time_out.shape}")
    except Exception as e:
        print(f"❌ AF3TimestepEmbedding: {e}")
    
    # 测试自适应LayerNorm
    try:
        adaptive_norm = AF3AdaptiveLayerNorm(hidden_dim, condition_dim=64)
        x = torch.randn(batch_size, seq_len, hidden_dim)
        condition = torch.randn(batch_size, 64)
        norm_out = adaptive_norm(x, condition)
        print(f"✅ AF3AdaptiveLayerNorm: {norm_out.shape}")
    except Exception as e:
        print(f"❌ AF3AdaptiveLayerNorm: {e}")
    
    print()

def test_glu_feedforward():
    """测试GLU前馈网络"""
    print("测试GLU前馈网络...")
    
    from structdiff.models.layers.mlp import FeedForward
    
    batch_size = 4
    seq_len = 20
    hidden_dim = 256
    
    # 测试标准FFN
    try:
        standard_ffn = FeedForward(
            hidden_dim=hidden_dim,
            use_gate=False
        )
        x = torch.randn(batch_size, seq_len, hidden_dim)
        standard_out = standard_ffn(x)
        print(f"✅ 标准FFN: {standard_out.shape}")
    except Exception as e:
        print(f"❌ 标准FFN: {e}")
    
    # 测试GLU FFN
    try:
        glu_ffn = FeedForward(
            hidden_dim=hidden_dim,
            use_gate=True,
            activation="silu"
        )
        x = torch.randn(batch_size, seq_len, hidden_dim)
        glu_out = glu_ffn(x)
        print(f"✅ GLU FFN: {glu_out.shape}")
    except Exception as e:
        print(f"❌ GLU FFN: {e}")
    
    print()

def test_denoiser_integration():
    """测试去噪器集成"""
    print("测试去噪器集成...")
    
    try:
        from structdiff.models.denoise import StructureAwareDenoiser
        
        # 配置
        denoiser_config = type('obj', (object,), {
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 6,
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
        
        # 前向传播
        denoised, cross_attn = denoiser(
            noisy_embeddings=noisy_embeddings,
            timesteps=timesteps,
            attention_mask=attention_mask,
            structure_features=structure_features
        )
        
        print(f"✅ StructureAwareDenoiser: 输入{noisy_embeddings.shape} -> 输出{denoised.shape}")
        print(f"   参数数量: {sum(p.numel() for p in denoiser.parameters()):,}")
        
    except Exception as e:
        print(f"❌ StructureAwareDenoiser: {e}")
        import traceback
        traceback.print_exc()
    
    print()

def test_performance_comparison():
    """性能对比测试"""
    print("性能对比测试...")
    
    from structdiff.models.layers.mlp import FeedForward
    import time
    
    batch_size = 16
    seq_len = 50
    hidden_dim = 512
    num_runs = 100
    
    x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 标准FFN
    standard_ffn = FeedForward(hidden_dim=hidden_dim, use_gate=False)
    
    # GLU FFN
    glu_ffn = FeedForward(hidden_dim=hidden_dim, use_gate=True, activation="silu")
    
    # 预热
    for _ in range(10):
        _ = standard_ffn(x)
        _ = glu_ffn(x)
    
    # 测试标准FFN
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(num_runs):
        _ = standard_ffn(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    standard_time = time.time() - start_time
    
    # 测试GLU FFN
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(num_runs):
        _ = glu_ffn(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    glu_time = time.time() - start_time
    
    print(f"标准FFN时间: {standard_time:.4f}s ({standard_time/num_runs*1000:.2f}ms/run)")
    print(f"GLU FFN时间: {glu_time:.4f}s ({glu_time/num_runs*1000:.2f}ms/run)")
    print(f"速度比: {standard_time/glu_time:.2f}x")
    
    # 参数数量对比
    standard_params = sum(p.numel() for p in standard_ffn.parameters())
    glu_params = sum(p.numel() for p in glu_ffn.parameters())
    
    print(f"标准FFN参数: {standard_params:,}")
    print(f"GLU FFN参数: {glu_params:,}")
    print(f"参数比: {glu_params/standard_params:.2f}x")
    
    print()

def main():
    """主测试函数"""
    print("🧪 测试AlphaFold3改进组件\n")
    print("=" * 50)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")
    
    # 运行测试
    test_af3_noise_schedule()
    test_af3_embeddings()
    test_glu_feedforward()
    test_denoiser_integration()
    test_performance_comparison()
    
    print("✅ 所有测试完成!")

if __name__ == "__main__":
    main()