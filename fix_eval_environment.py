#!/usr/bin/env python3
"""
快速修复评估问题的脚本
"""

import torch
import gc
import os

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print(f"🧹 GPU内存已清理，当前使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

def set_environment_for_eval():
    """设置环境变量优化评估"""
    # PyTorch内存管理
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 禁用一些调试功能
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    print("✅ 环境变量已优化")

if __name__ == "__main__":
    print("🔧 运行评估修复...")
    set_environment_for_eval()
    clear_gpu_memory()
    print("✅ 评估环境已优化，可以重新运行训练脚本")
