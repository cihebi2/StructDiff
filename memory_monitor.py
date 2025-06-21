#!/usr/bin/env python3
"""
GPU内存监控和清理脚本
"""

import os
import gc

def monitor_memory():
    """监控内存使用"""
    print("🔍 内存监控报告")
    print("=" * 40)
    
    # 检查torch是否可用
    try:
        import torch
        
        # GPU内存
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                print(f"GPU {i}:")
                print(f"  已分配: {allocated:.2f}GB")
                print(f"  已预留: {reserved:.2f}GB") 
                print(f"  总容量: {total:.2f}GB")
                print(f"  使用率: {allocated/total*100:.1f}%")
        else:
            print("CUDA 不可用")
    except ImportError:
        print("PyTorch 未安装")
    
    # 系统内存
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"\n系统内存:")
        print(f"  已使用: {memory.used/1024**3:.2f}GB")
        print(f"  总容量: {memory.total/1024**3:.2f}GB")
        print(f"  使用率: {memory.percent:.1f}%")
    except ImportError:
        print("\npsutil 未安装，无法检查系统内存")

def cleanup_memory():
    """清理内存"""
    print("\n🧹 开始内存清理...")
    
    # Python垃圾回收
    collected = gc.collect()
    print(f"  垃圾回收: {collected} 个对象")
    
    # GPU内存清理
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("  GPU缓存已清理")
    except ImportError:
        print("  PyTorch 未安装，跳过GPU清理")
    
    print("✅ 内存清理完成")

def optimize_pytorch_memory():
    """优化PyTorch内存设置"""
    print("\n⚙️ 优化PyTorch内存设置...")
    
    # 设置环境变量
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # 设置内存分数限制
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)  # 使用90%的GPU内存
            print("  设置GPU内存使用限制为90%")
    except ImportError:
        print("  PyTorch 未安装，跳过GPU内存优化")
    
    print("✅ PyTorch内存优化完成")

def main():
    """主函数"""
    monitor_memory()
    cleanup_memory()
    optimize_pytorch_memory()
    
    print("\n📋 建议:")
    print("  1. 在训练前运行此脚本")
    print("  2. 设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print("  3. 减小批次大小或序列长度")
    print("  4. 考虑使用CPU进行ESMFold评估")

if __name__ == "__main__":
    main()