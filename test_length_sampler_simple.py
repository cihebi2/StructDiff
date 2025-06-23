#!/usr/bin/env python3
"""
简化的长度采样器测试脚本
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from structdiff.sampling.length_sampler import (
    LengthSamplerConfig, AdaptiveLengthSampler
)

def test_length_sampler_device():
    """测试长度采样器的设备一致性"""
    print("🧪 测试长度采样器设备一致性...")
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建配置
    config = LengthSamplerConfig(
        min_length=5,
        max_length=50,
        distribution_type="normal",
        normal_mean=25.0,
        normal_std=8.0
    )
    
    # 创建采样器
    sampler = AdaptiveLengthSampler(config)
    
    print("✓ 测试概率计算...")
    # 测试概率计算
    probs = sampler.get_length_probabilities(device=device)
    print(f"  概率张量形状: {probs.shape}")
    print(f"  概率张量设备: {probs.device}")
    print(f"  概率总和: {torch.sum(probs, dim=-1)}")
    
    # 验证概率和
    prob_sums = torch.sum(probs, dim=-1)
    expected_ones = torch.ones(probs.shape[0], device=device)
    
    if torch.allclose(prob_sums, expected_ones):
        print("✅ 概率计算设备一致性测试通过！")
    else:
        print("❌ 概率计算设备一致性测试失败！")
        return False
    
    print("✓ 测试长度采样...")
    # 测试长度采样
    lengths = sampler.sample_lengths(
        batch_size=10,
        temperature=1.0,
        device=device
    )
    print(f"  采样长度形状: {lengths.shape}")
    print(f"  采样长度设备: {lengths.device}")
    print(f"  采样长度范围: {lengths.min()} - {lengths.max()}")
    
    if lengths.device.type == device.split(':')[0]:
        print("✅ 长度采样设备一致性测试通过！")
        return True
    else:
        print("❌ 长度采样设备一致性测试失败！")
        return False

def test_length_sampler_functionality():
    """测试长度采样器功能"""
    print("\n🧪 测试长度采样器功能...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = LengthSamplerConfig(
        min_length=10,
        max_length=30,
        distribution_type="normal",
        normal_mean=20.0,
        normal_std=5.0
    )
    
    sampler = AdaptiveLengthSampler(config)
    
    # 多次采样测试
    batch_size = 100
    lengths = sampler.sample_lengths(
        batch_size=batch_size,
        temperature=1.0,
        device=device
    )
    
    # 统计
    mean_length = lengths.float().mean()
    std_length = lengths.float().std()
    
    print(f"✓ 采样统计:")
    print(f"  平均长度: {mean_length:.2f} (期望: {config.normal_mean})")
    print(f"  标准差: {std_length:.2f} (期望: {config.normal_std})")
    print(f"  最小长度: {lengths.min()} (限制: {config.min_length})")
    print(f"  最大长度: {lengths.max()} (限制: {config.max_length})")
    
    # 验证范围
    if lengths.min() >= config.min_length and lengths.max() <= config.max_length:
        print("✅ 长度范围限制正确！")
    else:
        print("❌ 长度范围限制失败！")
        return False
    
    # 验证均值大致正确
    if abs(mean_length - config.normal_mean) < 3.0:
        print("✅ 长度分布统计合理！")
        return True
    else:
        print("❌ 长度分布统计异常！")
        return False

if __name__ == "__main__":
    print("=== 长度采样器简化测试 ===")
    
    try:
        # 设备一致性测试
        success1 = test_length_sampler_device()
        
        # 功能测试
        success2 = test_length_sampler_functionality()
        
        if success1 and success2:
            print("\n🎉 所有长度采样器测试通过！")
        else:
            print("\n❌ 部分测试失败")
            
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()