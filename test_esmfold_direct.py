#!/usr/bin/env python3
"""
测试修复后的 ESMFoldWrapper
"""

import torch
from structdiff.models.esmfold_wrapper import ESMFoldWrapper

def test_esmfold_wrapper():
    """测试修复后的 ESMFoldWrapper"""
    print("=== 测试修复后的 ESMFoldWrapper ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试序列
    test_sequence = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETARDLLISEQNVVNGITKGEMLPVSDTTGFPYT"
    
    print("\n1. 初始化 ESMFoldWrapper...")
    try:
        wrapper = ESMFoldWrapper(device=device)
        print(f"   ESMFold 可用: {wrapper.available}")
        
        if not wrapper.available:
            print("   ❌ ESMFold 不可用，检查初始化过程")
            return False
        
    except Exception as e:
        print(f"   ❌ 初始化失败: {e}")
        return False
    
    print("\n2. 测试结构预测...")
    try:
        features = wrapper.predict_structure(test_sequence)
        
        print("   ✓ 预测成功！")
        print(f"   特征键: {list(features.keys())}")
        
        for key, value in features.items():
            if torch.is_tensor(value):
                print(f"     {key}: {value.shape}")
            else:
                print(f"     {key}: {type(value)}")
        
        # 检查关键特征
        if 'positions' in features:
            print(f"   原子坐标形状: {features['positions'].shape}")
        
        if 'plddt' in features:
            plddt = features['plddt']
            print(f"   pLDDT: 平均={plddt.mean():.2f}, 最小={plddt.min():.2f}, 最大={plddt.max():.2f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 预测失败: {e}")
        return False

if __name__ == "__main__":
    success = test_esmfold_wrapper()
    
    if success:
        print("\n🎉 ESMFoldWrapper 测试成功！")
    else:
        print("\n❌ ESMFoldWrapper 测试失败") 