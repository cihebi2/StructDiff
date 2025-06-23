#!/usr/bin/env python3
"""
简化的CFG测试脚本
"""

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from structdiff.models.classifier_free_guidance import CFGConfig, ClassifierFreeGuidance

def test_cfg_basic():
    """基础CFG测试"""
    print("🧪 开始CFG基础测试...")
    
    # 创建CFG配置
    cfg_config = CFGConfig(
        dropout_prob=0.1,
        guidance_scale=2.0
    )
    
    # 创建CFG实例
    cfg = ClassifierFreeGuidance(cfg_config)
    
    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 测试数据
    batch_size = 4
    seq_length = 20
    hidden_dim = 256
    
    x_t = torch.randn(batch_size, seq_length, hidden_dim, device=device)
    t = torch.randint(0, 100, (batch_size,), device=device)
    conditions = {
        'peptide_type': torch.tensor([0, 1, 2, 0], device=device)
    }
    
    # 创建能够区分条件的模型函数
    def test_model(x_input, time_step, cond_input):
        """测试模型：能够区分有条件和无条件输入"""
        # 基础输出
        base_output = x_input * 0.9
        
        if cond_input is None:
            # 无条件
            return base_output * 1.2
        elif 'peptide_type' in cond_input:
            peptide_type = cond_input['peptide_type']
            if isinstance(peptide_type, torch.Tensor):
                # 检查是否无条件
                if torch.any(peptide_type == -1):
                    return base_output * 1.2
                else:
                    # 有条件：根据类型调整
                    type_factor = 1.0 + 0.1 * peptide_type.float().mean()
                    return base_output * type_factor
        
        return base_output
    
    print("✓ 测试引导去噪...")
    
    # 测试不同引导强度
    guidance_scales = [1.0, 1.5, 2.0, 3.0]
    outputs = []
    
    for scale in guidance_scales:
        output = cfg.guided_denoising(test_model, x_t, t, conditions, scale)
        outputs.append(output)
        print(f"  引导强度 {scale}: 输出均值 = {output.mean().item():.4f}")
    
    # 验证引导效果
    base_output = outputs[0]  # guidance_scale = 1.0
    guided_output = outputs[-1]  # guidance_scale = 3.0
    
    print(f"✓ 基础输出（scale=1.0）均值: {base_output.mean().item():.4f}")
    print(f"✓ 引导输出（scale=3.0）均值: {guided_output.mean().item():.4f}")
    print(f"✓ 差异: {torch.abs(guided_output - base_output).mean().item():.4f}")
    
    # 验证输出确实不同
    if torch.allclose(guided_output, base_output, rtol=1e-3):
        print("❌ 警告：引导输出与基础输出几乎相同！CFG可能未正常工作")
        return False
    else:
        print("✅ CFG引导效果验证成功！")
        return True

def test_cfg_unconditional_creation():
    """测试无条件批次创建"""
    print("\n🧪 测试无条件批次创建...")
    
    cfg_config = CFGConfig()
    cfg = ClassifierFreeGuidance(cfg_config)
    
    batch_size = 4
    uncond_batch = cfg._create_unconditional_batch(batch_size)
    
    print(f"✓ 无条件批次: {uncond_batch}")
    
    # 验证
    assert 'peptide_type' in uncond_batch
    assert 'is_unconditional' in uncond_batch
    assert torch.all(uncond_batch['peptide_type'] == -1)
    assert torch.all(uncond_batch['is_unconditional'] == True)
    
    print("✅ 无条件批次创建测试通过！")
    return True

if __name__ == "__main__":
    print("=== CFG 简化测试 ===")
    
    try:
        # 基础CFG测试
        success1 = test_cfg_basic()
        
        # 无条件批次测试
        success2 = test_cfg_unconditional_creation()
        
        if success1 and success2:
            print("\n🎉 所有CFG测试通过！")
        else:
            print("\n❌ 部分测试失败")
            
    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()