#!/usr/bin/env python3
"""
ESMFold生产环境测试脚本
验证结构预测功能在当前硬件环境下的可用性
"""

import os
import sys
import torch
import numpy as np
import time
from pathlib import Path

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 使用GPU 2进行测试

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from structdiff.models.esmfold_wrapper import ESMFoldWrapper
from structdiff.utils.logger import setup_logger, get_logger

setup_logger(level="INFO")
logger = get_logger(__name__)

def test_esmfold_basic():
    """基础ESMFold功能测试"""
    logger.info("🧪 开始ESMFold基础功能测试")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 测试序列
    test_sequences = [
        "MKTFFGREDLG",  # 短序列 (11 aa)
        "MKTFFGREDLGKYKLLACYRGFQDLFETKGFFEDEPKLLNQGYRKQVKMLPGFDPFLFRRWCNMSCF",  # 中等长度 (65 aa)
    ]
    
    try:
        # 初始化ESMFold
        logger.info("初始化ESMFold包装器...")
        start_time = time.time()
        esmfold = ESMFoldWrapper(device=device)
        init_time = time.time() - start_time
        
        logger.info(f"ESMFold初始化耗时: {init_time:.2f}s")
        logger.info(f"ESMFold可用性: {esmfold.available}")
        
        if not esmfold.available:
            logger.error("❌ ESMFold不可用，测试失败")
            return False
        
        # 测试不同长度的序列
        for i, sequence in enumerate(test_sequences):
            logger.info(f"\n🧪 测试序列 {i+1}: 长度 {len(sequence)}")
            
            # 预测结构
            start_time = time.time()
            features = esmfold.predict_structure(sequence)
            pred_time = time.time() - start_time
            
            logger.info(f"预测耗时: {pred_time:.2f}s")
            
            # 验证输出
            assert 'positions' in features, "缺少positions特征"
            assert 'plddt' in features, "缺少plddt特征"
            assert 'distance_matrix' in features, "缺少distance_matrix特征"
            assert 'angles' in features, "缺少angles特征"
            
            # 检查形状
            seq_len = len(sequence)
            assert features['positions'].shape[0] == seq_len, f"positions形状错误: {features['positions'].shape}"
            assert features['plddt'].shape[0] == seq_len, f"plddt形状错误: {features['plddt'].shape}"
            assert features['distance_matrix'].shape == (seq_len, seq_len), f"distance_matrix形状错误: {features['distance_matrix'].shape}"
            assert features['angles'].shape == (seq_len, 10), f"angles形状错误: {features['angles'].shape}"
            
            # 检查数值范围
            assert torch.all(features['plddt'] >= 0) and torch.all(features['plddt'] <= 100), "plddt值超出范围"
            assert torch.all(features['distance_matrix'] >= 0), "distance_matrix包含负值"
            
            logger.info(f"✓ 序列 {i+1} 测试通过")
            logger.info(f"  - 平均pLDDT: {features['plddt'].mean().item():.2f}")
            logger.info(f"  - 位置范围: {features['positions'].min().item():.2f} - {features['positions'].max().item():.2f}")
        
        logger.info("✅ ESMFold基础功能测试全部通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ ESMFold测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_memory_usage():
    """测试内存使用情况"""
    logger.info("\n🧪 开始内存使用测试")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        logger.info(f"初始GPU内存: {initial_memory / 1024**3:.2f} GB")
    
    try:
        # 初始化ESMFold
        esmfold = ESMFoldWrapper(device=device)
        
        if device.type == 'cuda':
            after_init_memory = torch.cuda.memory_allocated(device)
            init_memory_usage = (after_init_memory - initial_memory) / 1024**3
            logger.info(f"ESMFold加载后GPU内存: {after_init_memory / 1024**3:.2f} GB")
            logger.info(f"ESMFold内存占用: {init_memory_usage:.2f} GB")
        
        if not esmfold.available:
            logger.warning("ESMFold不可用，跳过内存测试")
            return True
        
        # 测试不同批次大小
        test_sequence = "MKTFFGREDLGKYKLLACYRGFQDLFETKGFFEDEPKLLNQGYRKQV"
        
        for batch_size in [1, 2, 4]:
            logger.info(f"\n测试批次大小: {batch_size}")
            
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                before_memory = torch.cuda.memory_allocated(device)
            
            # 模拟批次处理
            for i in range(batch_size):
                features = esmfold.predict_structure(test_sequence)
            
            if device.type == 'cuda':
                after_memory = torch.cuda.memory_allocated(device)
                batch_memory = (after_memory - before_memory) / 1024**3
                logger.info(f"批次处理后GPU内存: {after_memory / 1024**3:.2f} GB")
                logger.info(f"批次内存增量: {batch_memory:.2f} GB")
        
        logger.info("✅ 内存使用测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 内存测试失败: {e}")
        return False

def test_cache_strategy():
    """测试缓存策略"""
    logger.info("\n🧪 开始缓存策略测试")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    try:
        esmfold = ESMFoldWrapper(device=device)
        
        if not esmfold.available:
            logger.warning("ESMFold不可用，跳过缓存测试")
            return True
        
        test_sequence = "MKTFFGREDLGKYKLL"
        
        # 第一次预测（无缓存）
        start_time = time.time()
        features1 = esmfold.predict_structure(test_sequence)
        first_time = time.time() - start_time
        
        # 第二次预测（可能有缓存）
        start_time = time.time()
        features2 = esmfold.predict_structure(test_sequence)
        second_time = time.time() - start_time
        
        logger.info(f"第一次预测耗时: {first_time:.2f}s")
        logger.info(f"第二次预测耗时: {second_time:.2f}s")
        
        # 验证结果一致性
        assert torch.allclose(features1['plddt'], features2['plddt'], atol=1e-5), "缓存结果不一致"
        
        logger.info("✅ 缓存策略测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 缓存测试失败: {e}")
        return False

def test_error_handling():
    """测试错误处理"""
    logger.info("\n🧪 开始错误处理测试")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    try:
        esmfold = ESMFoldWrapper(device=device)
        
        # 测试空序列
        empty_features = esmfold.predict_structure("")
        assert empty_features is not None, "空序列处理失败"
        
        # 测试无效字符
        invalid_sequence = "MKTFFGREDLGXYZ"  # 包含无效氨基酸
        invalid_features = esmfold.predict_structure(invalid_sequence)
        assert invalid_features is not None, "无效序列处理失败"
        
        # 测试超长序列
        long_sequence = "M" * 1000  # 1000个氨基酸
        long_features = esmfold.predict_structure(long_sequence)
        assert long_features is not None, "超长序列处理失败"
        
        logger.info("✅ 错误处理测试通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ 错误处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始ESMFold生产环境测试")
    
    # GPU信息
    if torch.cuda.is_available():
        logger.info(f"GPU设备: {torch.cuda.get_device_name()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("使用CPU模式")
    
    tests = [
        ("基础功能测试", test_esmfold_basic),
        ("内存使用测试", test_memory_usage),
        ("缓存策略测试", test_cache_strategy),
        ("错误处理测试", test_error_handling),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"开始: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name} 通过")
            else:
                logger.error(f"❌ {test_name} 失败")
                
        except Exception as e:
            logger.error(f"❌ {test_name} 异常: {e}")
            results[test_name] = False
    
    # 总结
    logger.info(f"\n{'='*50}")
    logger.info("测试总结")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！ESMFold生产环境就绪")
        return True
    else:
        logger.error("⚠️  部分测试失败，请检查配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 