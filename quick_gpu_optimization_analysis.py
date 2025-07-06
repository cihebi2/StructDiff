#!/usr/bin/env python3
"""
快速GPU优化分析脚本
分析当前训练状态并提供具体的优化建议
"""

import subprocess
import json
import time
import psutil
import re
from datetime import datetime

def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.split(', ')
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_used': int(parts[2]),
                    'memory_total': int(parts[3]),
                    'utilization': int(parts[4]),
                    'power_draw': float(parts[5]),
                    'power_limit': float(parts[6]),
                    'temperature': int(parts[7])
                })
        return gpus
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return []

def get_process_info():
    """获取当前训练进程信息"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line.strip() and 'python' in line:
                parts = line.split(', ')
                pid = int(parts[0])
                
                # 获取进程详细信息
                try:
                    proc = psutil.Process(pid)
                    cmd = ' '.join(proc.cmdline())
                    
                    processes.append({
                        'pid': pid,
                        'name': parts[1],
                        'gpu_memory': int(parts[2]),
                        'command': cmd,
                        'cpu_percent': proc.cpu_percent(),
                        'memory_percent': proc.memory_percent()
                    })
                except:
                    pass
        
        return processes
    except Exception as e:
        print(f"获取进程信息失败: {e}")
        return []

def analyze_training_bottleneck():
    """分析训练瓶颈"""
    print("🔍 GPU利用率分析报告")
    print("=" * 60)
    
    # 获取GPU信息
    gpus = get_gpu_info()
    processes = get_process_info()
    
    if not gpus:
        print("❌ 无法获取GPU信息")
        return
    
    print(f"📊 GPU状态 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("-" * 60)
    
    training_gpu = None
    for gpu in gpus:
        print(f"GPU {gpu['index']}: {gpu['name']}")
        print(f"  💾 内存: {gpu['memory_used']}/{gpu['memory_total']} MB ({gpu['memory_used']/gpu['memory_total']*100:.1f}%)")
        print(f"  🔥 利用率: {gpu['utilization']}%")
        print(f"  ⚡ 功率: {gpu['power_draw']:.0f}/{gpu['power_limit']:.0f} W ({gpu['power_draw']/gpu['power_limit']*100:.1f}%)")
        print(f"  🌡️  温度: {gpu['temperature']}°C")
        
        # 找到正在训练的GPU
        if gpu['memory_used'] > 10000:  # 大于10GB内存使用
            training_gpu = gpu
        print()
    
    if not training_gpu:
        print("❌ 未找到正在训练的GPU")
        return
    
    print(f"🎯 检测到训练GPU: GPU {training_gpu['index']}")
    print("-" * 60)
    
    # 分析瓶颈
    bottlenecks = []
    optimizations = []
    
    # 1. GPU利用率分析
    if training_gpu['utilization'] < 30:
        bottlenecks.append("GPU利用率过低")
        optimizations.extend([
            "增加批次大小 (batch_size)",
            "启用混合精度训练 (AMP)",
            "增加数据加载器工作进程数 (num_workers)",
            "启用内存固定 (pin_memory=True)"
        ])
    
    # 2. 功率使用分析
    power_usage = training_gpu['power_draw'] / training_gpu['power_limit']
    if power_usage < 0.5:
        bottlenecks.append("GPU功率使用不足")
        optimizations.extend([
            "增加计算密度",
            "优化数据加载流水线",
            "减少CPU-GPU同步等待"
        ])
    
    # 3. 内存使用分析
    memory_usage = training_gpu['memory_used'] / training_gpu['memory_total']
    if memory_usage > 0.9:
        bottlenecks.append("GPU内存使用过高")
        optimizations.extend([
            "启用梯度检查点",
            "减少批次大小",
            "优化内存管理"
        ])
    elif memory_usage < 0.6:
        bottlenecks.append("GPU内存使用不足")
        optimizations.extend([
            "增加批次大小",
            "增加模型复杂度",
            "启用数据预取"
        ])
    
    # 4. 进程分析
    print("🔍 训练进程分析:")
    print("-" * 60)
    
    training_processes = [p for p in processes if p['gpu_memory'] > 1000]
    for proc in training_processes:
        print(f"PID {proc['pid']}: {proc['name']}")
        print(f"  📄 命令: {proc['command'][:80]}...")
        print(f"  💾 GPU内存: {proc['gpu_memory']} MB")
        print(f"  🖥️  CPU: {proc['cpu_percent']:.1f}%")
        print(f"  💿 RAM: {proc['memory_percent']:.1f}%")
        print()
    
    # 输出分析结果
    print("🚨 检测到的性能瓶颈:")
    print("-" * 60)
    for i, bottleneck in enumerate(bottlenecks, 1):
        print(f"{i}. {bottleneck}")
    
    print(f"\n💡 推荐的优化措施:")
    print("-" * 60)
    for i, opt in enumerate(set(optimizations), 1):
        print(f"{i}. {opt}")
    
    # 具体优化建议
    print(f"\n🎯 具体优化建议:")
    print("-" * 60)
    
    current_memory_gb = training_gpu['memory_used'] / 1024
    total_memory_gb = training_gpu['memory_total'] / 1024
    
    print(f"📊 当前状态:")
    print(f"  - GPU利用率: {training_gpu['utilization']}%")
    print(f"  - 功率使用: {training_gpu['power_draw']:.0f}W / {training_gpu['power_limit']:.0f}W")
    print(f"  - 内存使用: {current_memory_gb:.1f}GB / {total_memory_gb:.1f}GB")
    
    print(f"\n🚀 立即可执行的优化:")
    
    # 批次大小建议
    if training_gpu['utilization'] < 50:
        if current_memory_gb < 20:
            suggested_batch_size = 8
            print(f"1. 增加批次大小到 {suggested_batch_size} (当前可能是2)")
        else:
            suggested_batch_size = 6
            print(f"1. 适度增加批次大小到 {suggested_batch_size}")
    
    # 数据加载优化
    if training_gpu['utilization'] < 30:
        print("2. 启用多进程数据加载:")
        print("   - num_workers=4")
        print("   - pin_memory=True")
        print("   - prefetch_factor=2")
    
    # 混合精度训练
    if power_usage < 0.6:
        print("3. 启用混合精度训练 (AMP):")
        print("   - 可提升20-30%性能")
        print("   - 减少内存使用")
    
    # 梯度累积优化
    print("4. 优化梯度累积策略:")
    print("   - 减少累积步数，增加批次大小")
    print("   - 当前: batch_size=2, accumulation=8")
    print("   - 建议: batch_size=8, accumulation=2")
    
    print(f"\n📈 预期性能提升:")
    print("-" * 60)
    
    if training_gpu['utilization'] < 30:
        print("🎯 目标GPU利用率: 70-85%")
        print("🎯 目标功率使用: 250-350W")
        print("🎯 预期训练速度提升: 3-5倍")
        print("🎯 预期总训练时间缩短: 60-80%")
    
    return {
        'gpu_info': training_gpu,
        'bottlenecks': bottlenecks,
        'optimizations': optimizations,
        'processes': training_processes
    }

def create_optimization_config():
    """创建优化配置建议"""
    print(f"\n⚙️  优化配置建议:")
    print("-" * 60)
    
    config_suggestions = {
        "training": {
            "batch_size": 8,  # 从2增加到8
            "gradient_accumulation_steps": 2,  # 从8减少到2
            "num_workers": 4,  # 从0增加到4
            "pin_memory": True,  # 启用
            "prefetch_factor": 2,  # 启用预取
            "use_amp": True,  # 启用混合精度
            "persistent_workers": True  # 持久化工作进程
        },
        "model": {
            "gradient_checkpointing": True,  # 启用梯度检查点
            "compile_model": True  # 启用模型编译
        },
        "optimization": {
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "grad_clip_norm": 1.0
        }
    }
    
    print("建议的配置参数:")
    for section, params in config_suggestions.items():
        print(f"\n[{section}]")
        for key, value in params.items():
            print(f"  {key} = {value}")
    
    return config_suggestions

if __name__ == "__main__":
    try:
        print("🚀 开始GPU优化分析...")
        analysis = analyze_training_bottleneck()
        
        if analysis:
            config = create_optimization_config()
            
            # 保存分析结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"gpu_optimization_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'analysis': analysis,
                    'config_suggestions': config
                }, f, indent=2)
            
            print(f"\n💾 详细报告已保存到: {report_file}")
            print("\n✅ GPU优化分析完成!")
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc() 