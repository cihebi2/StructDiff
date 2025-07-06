#!/usr/bin/env python3
"""
批次大小优化测试脚本
用于找到最优的批次大小以提升GPU利用率
"""

import os
import sys
import time
import torch
import torch.nn as nn
import gc
import json
from datetime import datetime
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# 添加项目路径
sys.path.append('/home/qlyu/sequence/StructDiff-7.0.0')

from src.models.struct_diff import StructDiff
from src.data.peptide_dataset import PeptideDataset
from src.data.collator import PeptideCollator
from src.utils.esmfold_wrapper import ESMFoldWrapper
from src.utils.config import Config

def create_test_batch(batch_size, dataset, collator):
    """创建测试批次"""
    # 从数据集中随机选择样本
    indices = torch.randperm(len(dataset))[:batch_size]
    samples = [dataset[i] for i in indices]
    
    # 使用collator处理批次
    batch = collator(samples)
    
    # 移动到GPU
    for key in batch:
        if torch.is_tensor(batch[key]):
            batch[key] = batch[key].cuda()
    
    return batch

def test_batch_sizes():
    """测试不同批次大小的性能"""
    
    print("🚀 开始批次大小优化测试...")
    print("=" * 60)
    
    # 初始化配置
    config = Config()
    config.model.structure_prediction_enabled = True
    config.training.batch_size = 2  # 基础批次大小
    
    # 初始化ESMFold
    print("📥 初始化ESMFold...")
    esmfold_wrapper = ESMFoldWrapper(device="cuda:1")
    
    # 初始化模型
    print("🔧 初始化模型...")
    model = StructDiff(config.model)
    model.cuda()
    model.train()
    
    # 强制设置共享ESMFold实例
    model.esmfold_wrapper = esmfold_wrapper
    
    # 初始化数据集
    print("📊 初始化数据集...")
    train_dataset = PeptideDataset(
        data_path=config.data.train_data_path,
        max_length=config.data.max_length,
        structure_prediction_enabled=config.model.structure_prediction_enabled
    )
    
    collator = PeptideCollator(
        max_length=config.data.max_length,
        structure_prediction_enabled=config.model.structure_prediction_enabled
    )
    
    # 初始化优化器和损失缩放器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    # 测试不同批次大小
    batch_sizes = [2, 4, 6, 8, 10, 12, 16]
    results = {}
    
    print("\n🧪 开始测试不同批次大小...")
    print("-" * 60)
    
    for bs in batch_sizes:
        print(f"\n📏 测试批次大小: {bs}")
        
        try:
            # 清理GPU内存
            torch.cuda.empty_cache()
            gc.collect()
            
            # 记录初始内存
            initial_memory = torch.cuda.memory_allocated() / 1e9
            
            # 创建测试批次
            test_batch = create_test_batch(bs, train_dataset, collator)
            
            # 创建随机时间步
            timesteps = torch.randint(0, 1000, (bs,), device='cuda')
            
            # 测试前向传播
            start_time = time.time()
            
            with autocast():
                outputs = model(
                    sequences=test_batch['sequences'],
                    attention_mask=test_batch['attention_mask'],
                    timesteps=timesteps,
                    structures=test_batch.get('structures'),
                    conditions=test_batch.get('conditions'),
                    return_loss=True
                )
                loss = outputs['total_loss']
            
            forward_time = time.time() - start_time
            
            # 测试反向传播
            backward_start = time.time()
            scaler.scale(loss).backward()
            backward_time = time.time() - backward_start
            
            total_time = forward_time + backward_time
            peak_memory = torch.cuda.memory_allocated() / 1e9
            memory_used = peak_memory - initial_memory
            
            # 计算性能指标
            throughput = bs / total_time
            samples_per_second = bs / total_time
            
            results[bs] = {
                'forward_time': forward_time,
                'backward_time': backward_time,
                'total_time': total_time,
                'memory_used': memory_used,
                'peak_memory': peak_memory,
                'throughput': throughput,
                'samples_per_second': samples_per_second,
                'loss': loss.item()
            }
            
            print(f"  ✅ 成功")
            print(f"  ⏱️  总时间: {total_time:.2f}s (前向: {forward_time:.2f}s, 反向: {backward_time:.2f}s)")
            print(f"  💾 内存使用: {memory_used:.1f}GB (峰值: {peak_memory:.1f}GB)")
            print(f"  🚀 吞吐量: {throughput:.2f} samples/s")
            print(f"  📉 损失: {loss.item():.6f}")
            
            # 清理梯度
            optimizer.zero_grad()
            
            # 检查内存是否超出限制
            if peak_memory > 22:  # 保持2GB安全边界
                print(f"  ⚠️  内存使用过高 ({peak_memory:.1f}GB > 22GB)")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  ❌ 内存不足: {str(e)}")
                results[bs] = {'error': 'OOM', 'message': str(e)}
                break
            else:
                print(f"  ❌ 其他错误: {str(e)}")
                results[bs] = {'error': 'Other', 'message': str(e)}
                break
    
    # 分析结果
    print("\n" + "=" * 60)
    print("📊 测试结果分析")
    print("=" * 60)
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        # 找到最优批次大小
        best_throughput = max(valid_results.values(), key=lambda x: x['throughput'])
        best_bs_throughput = [k for k, v in valid_results.items() if v['throughput'] == best_throughput['throughput']][0]
        
        # 找到内存效率最高的批次大小
        memory_efficient = min(valid_results.values(), key=lambda x: x['memory_used'] / x['throughput'])
        best_bs_memory = [k for k, v in valid_results.items() if v == memory_efficient][0]
        
        print(f"\n🏆 最优批次大小分析:")
        print(f"  🚀 最高吞吐量: 批次大小 {best_bs_throughput} ({best_throughput['throughput']:.2f} samples/s)")
        print(f"  💾 内存效率最高: 批次大小 {best_bs_memory} ({memory_efficient['throughput']:.2f} samples/s, {memory_efficient['memory_used']:.1f}GB)")
        
        # 推荐批次大小
        recommended_bs = best_bs_throughput
        if valid_results[best_bs_throughput]['peak_memory'] > 20:
            # 如果最高吞吐量的批次大小内存使用过高，选择内存效率高的
            recommended_bs = best_bs_memory
        
        print(f"\n💡 推荐批次大小: {recommended_bs}")
        print(f"  - 吞吐量: {valid_results[recommended_bs]['throughput']:.2f} samples/s")
        print(f"  - 内存使用: {valid_results[recommended_bs]['memory_used']:.1f}GB")
        print(f"  - 总时间: {valid_results[recommended_bs]['total_time']:.2f}s")
        
        # 预期性能提升
        current_bs = 2
        if current_bs in valid_results and recommended_bs in valid_results:
            current_perf = valid_results[current_bs]['throughput']
            new_perf = valid_results[recommended_bs]['throughput']
            improvement = (new_perf / current_perf - 1) * 100
            
            print(f"\n📈 预期性能提升:")
            print(f"  - 当前批次大小 {current_bs}: {current_perf:.2f} samples/s")
            print(f"  - 推荐批次大小 {recommended_bs}: {new_perf:.2f} samples/s")
            print(f"  - 性能提升: {improvement:.1f}%")
    
    # 详细结果表格
    print(f"\n📋 详细结果:")
    print("-" * 100)
    print(f"{'批次大小':<8} {'总时间(s)':<10} {'内存(GB)':<10} {'吞吐量(s/s)':<12} {'状态':<10}")
    print("-" * 100)
    
    for bs, result in results.items():
        if 'error' in result:
            print(f"{bs:<8} {'N/A':<10} {'N/A':<10} {'N/A':<12} {result['error']:<10}")
        else:
            print(f"{bs:<8} {result['total_time']:<10.2f} {result['memory_used']:<10.1f} {result['throughput']:<12.2f} {'成功':<10}")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"batch_size_optimization_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 结果已保存到: {results_file}")
    
    return results

if __name__ == "__main__":
    try:
        results = test_batch_sizes()
        print("\n✅ 批次大小优化测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc() 