#!/usr/bin/env python3
"""
分离式训练快速验证脚本
验证核心组件是否正常工作
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试导入"""
    print("🧪 测试导入...")
    
    try:
        from structdiff.training.separated_training import SeparatedTrainingManager, SeparatedTrainingConfig
        print("✅ 分离式训练模块导入成功")
        
        from structdiff.training.length_controller import (
            LengthDistributionAnalyzer, AdaptiveLengthController, LengthAwareDataCollator
        )
        print("✅ 长度控制器模块导入成功")
        
        return True
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_config_creation():
    """测试配置创建"""
    print("\n🧪 测试配置创建...")
    
    try:
        from structdiff.training.separated_training import SeparatedTrainingConfig
        
        config = SeparatedTrainingConfig(
            stage1_epochs=10,
            stage2_epochs=5,
            use_cfg=True,
            use_length_control=True
        )
        
        print(f"✅ 配置创建成功: stage1_epochs={config.stage1_epochs}")
        return True
    except Exception as e:
        print(f"❌ 配置创建失败: {e}")
        return False

def test_length_controller():
    """测试长度控制器基础功能"""
    print("\n🧪 测试长度控制器...")
    
    try:
        from structdiff.training.length_controller import AdaptiveLengthController
        import torch
        
        controller = AdaptiveLengthController(min_length=5, max_length=50)
        
        # 测试长度采样
        lengths = controller.sample_target_lengths(
            batch_size=10,
            peptide_types=['antimicrobial', 'antifungal']
        )
        
        print(f"✅ 长度采样成功: shape={lengths.shape}, range=[{lengths.min()}, {lengths.max()}]")
        
        # 测试长度掩码
        mask = controller.create_length_mask(lengths, 60)
        print(f"✅ 长度掩码创建成功: shape={mask.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 长度控制器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始分离式训练快速验证")
    
    tests = [
        test_imports,
        test_config_creation,
        test_length_controller,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有基础测试通过！分离式训练组件工作正常")
        print("\n下一步:")
        print("1. 运行完整测试: python test_separated_training.py")
        print("2. 准备数据并开始训练: python scripts/train_separated.py")
    else:
        print("❌ 部分测试失败，请检查实现")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)