#!/usr/bin/env python3
"""
GPU资源监控脚本
监控GPU可用内存，当满足训练要求时发出通知
"""

import time
import subprocess
import re
from datetime import datetime

def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv'], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')[1:]  # 跳过表头
        
        gpu_info = []
        for line in lines:
            parts = line.split(', ')
            gpu_id = int(parts[0])
            memory_used = int(parts[1].replace(' MiB', ''))  # MB
            memory_total = int(parts[2].replace(' MiB', ''))  # MB
            utilization = int(parts[3].replace(' %', ''))  # %
            
            memory_free = memory_total - memory_used
            memory_usage_percent = (memory_used / memory_total) * 100
            
            gpu_info.append({
                'id': gpu_id,
                'memory_used': memory_used,
                'memory_total': memory_total,
                'memory_free': memory_free,
                'memory_usage_percent': memory_usage_percent,
                'utilization': utilization
            })
        
        return gpu_info
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return []

def check_training_feasibility(gpu_info, min_memory_gb=8):
    """检查是否有GPU满足训练要求"""
    min_memory_mb = min_memory_gb * 1024
    
    suitable_gpus = []
    for gpu in gpu_info:
        if gpu['memory_free'] >= min_memory_mb:
            suitable_gpus.append(gpu)
    
    return suitable_gpus

def main():
    print("🔍 GPU资源监控器")
    print("=" * 50)
    print("监控GPU可用内存，寻找适合训练的时机")
    print("训练需要约8GB GPU内存用于ESM2模型")
    print("按 Ctrl+C 停止监控")
    print()
    
    check_interval = 30  # 30秒检查一次
    
    while True:
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{current_time}] 检查GPU状态...")
            
            gpu_info = get_gpu_info()
            if not gpu_info:
                print("❌ 无法获取GPU信息")
                time.sleep(check_interval)
                continue
            
            # 显示当前状态
            print("\n📊 当前GPU状态:")
            for gpu in gpu_info:
                status = "🟢 可用" if gpu['memory_free'] >= 8192 else "🔴 忙碌"
                print(f"  GPU {gpu['id']}: {status}")
                print(f"    内存: {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB "
                      f"({gpu['memory_usage_percent']:.1f}% 已用)")
                print(f"    可用: {gpu['memory_free']:.0f} MB ({gpu['memory_free']/1024:.1f} GB)")
                print(f"    利用率: {gpu['utilization']}%")
                print()
            
            # 检查是否有可用GPU
            suitable_gpus = check_training_feasibility(gpu_info, min_memory_gb=8)
            
            if suitable_gpus:
                print("🎉 发现可用GPU!")
                print("=" * 50)
                for gpu in suitable_gpus:
                    print(f"✅ GPU {gpu['id']} 满足训练要求:")
                    print(f"   可用内存: {gpu['memory_free']/1024:.1f} GB")
                    print(f"   利用率: {gpu['utilization']}%")
                
                print("\n🚀 可以启动训练了!")
                print("建议使用以下命令启动训练:")
                best_gpu = max(suitable_gpus, key=lambda x: x['memory_free'])
                print(f"cd /home/qlyu/sequence/StructDiff-7.0.0")
                
                # 根据可用内存推荐配置
                available_gb = best_gpu['memory_free'] / 1024
                if available_gb >= 15:
                    batch_size = 16
                    accumulation = 2
                elif available_gb >= 10:
                    batch_size = 8
                    accumulation = 4
                else:
                    batch_size = 4
                    accumulation = 8
                
                print(f"# 推荐配置: batch_size={batch_size}, accumulation_steps={accumulation}")
                print(f"# 使用GPU {best_gpu['id']} (可用内存: {available_gb:.1f} GB)")
                
                # 创建临时配置文件
                config_script = f"""
# 修改train_with_precomputed_features.py中的配置:
config = {{
    'batch_size': {batch_size},
    'num_workers': 2 if {batch_size} >= 8 else 0,
    'accumulation_steps': {accumulation},
    'device': 'cuda:{best_gpu["id"]}',
    # ... 其他配置保持不变
}}
"""
                print("\n推荐配置:")
                print(config_script)
                
                break
            else:
                print("⏳ 暂无可用GPU，继续监控...")
            
            print(f"下次检查时间: {check_interval}秒后")
            print("-" * 50)
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n\n👋 监控已停止")
            break
        except Exception as e:
            print(f"❌ 监控过程中发生错误: {e}")
            time.sleep(check_interval)

if __name__ == "__main__":
    main() 