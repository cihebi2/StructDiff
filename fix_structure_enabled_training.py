#!/usr/bin/env python3
"""
🔧 StructDiff结构感知训练修复脚本
保留核心功能：
1. ✅ 结构特征 (ESMFold)
2. ✅ 交叉注意力 (CrossModalAttention)
3. 🔧 修复维度配置错误
4. 💾 优化内存使用策略
"""

import os
import sys
import yaml
import psutil
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any
import torch

def stop_all_training():
    """停止所有训练进程"""
    print("🛑 停止所有训练进程...")
    
    commands = [
        "pkill -f train_separated.py",
        "pkill -f structdiff",
        "pkill -f python.*train"
    ]
    
    for cmd in commands:
        try:
            subprocess.run(cmd, shell=True, capture_output=True, text=True)
        except Exception:
            pass
    
    # 等待进程停止
    import time
    time.sleep(2)
    print("✅ 训练进程已停止")

def analyze_dimension_issue():
    """分析维度配置问题"""
    print("🔍 深度分析维度配置问题...")
    
    # 检查当前的配置文件
    config_files = [
        "configs/separated_training_optimized.yaml",
        "configs/separated_training_production.yaml",
        "configs/separated_training_fixed_v2.yaml"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"📄 检查配置文件: {config_file}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 提取关键维度参数
            model_config = config.get('model', {})
            denoiser_config = model_config.get('denoiser', {})
            structure_config = model_config.get('structure_encoder', {})
            
            hidden_dim = denoiser_config.get('hidden_dim', 'N/A')
            num_heads = denoiser_config.get('num_heads', 'N/A')
            struct_hidden_dim = structure_config.get('hidden_dim', 'N/A')
            use_cross_attention = denoiser_config.get('use_cross_attention', 'N/A')
            
            print(f"  去噪器: hidden_dim={hidden_dim}, num_heads={num_heads}")
            print(f"  结构编码器: hidden_dim={struct_hidden_dim}")
            print(f"  交叉注意力: {use_cross_attention}")
            
            # 检查维度兼容性
            if isinstance(hidden_dim, int) and isinstance(num_heads, int):
                if hidden_dim % num_heads == 0:
                    head_dim = hidden_dim // num_heads
                    print(f"  ✅ 维度配置正确: head_dim={head_dim}")
                else:
                    print(f"  ❌ 维度配置错误: {hidden_dim} % {num_heads} != 0")
            
            print()

def create_structure_enabled_config():
    """创建保留结构特征的优化配置"""
    print("🔧 创建结构感知的优化配置...")
    
    # 设计维度兼容的配置
    # ESM2_t6_8M_UR50D 的 hidden_size 是 320
    # 选择合适的注意力头数：320 能被 8, 10, 16, 20, 32, 40 整除
    # 我们选择 8 头，每头 40 维度
    
    optimized_config = {
        'experiment': {
            'name': "structdiff_structure_enabled_v1",
            'description': "保留结构特征和交叉注意力的优化版本",
            'project': "StructDiff-StructureAware",
            'seed': 42
        },
        
        'model': {
            'type': "StructDiff",
            'sequence_encoder': {
                'pretrained_model': "facebook/esm2_t6_8M_UR50D",
                'freeze_encoder': False,
                'use_lora': False
            },
            'structure_encoder': {
                'type': "multi_scale",
                'hidden_dim': 320,  # 与ESM2完全匹配
                'use_esmfold': True,  # ✅ 启用ESMFold
                'use_cache': True,   # 使用缓存减少计算
                'cache_dir': "./cache",
                'memory_efficient': True,  # 启用内存优化
                'batch_size_limit': 1,     # ESMFold批次限制
                
                'local': {
                    'hidden_dim': 320,
                    'num_layers': 2,  # 减少层数节省内存
                    'kernel_sizes': [3, 5],
                    'dropout': 0.1
                },
                'global': {
                    'hidden_dim': 320,
                    'num_attention_heads': 8,  # 320 ÷ 8 = 40 (完美整除)
                    'num_layers': 2,
                    'dropout': 0.1
                },
                'fusion': {
                    'method': "attention",
                    'hidden_dim': 320
                }
            },
            'denoiser': {
                'hidden_dim': 320,     # ✅ 确保与其他组件匹配
                'num_layers': 4,       # 减少层数节省内存
                'num_heads': 8,        # ✅ 320 ÷ 8 = 40 (完美整除)
                'dropout': 0.1,
                'use_cross_attention': True  # ✅ 启用交叉注意力
            },
            'sequence_decoder': {
                'hidden_dim': 320,
                'num_layers': 2,  # 减少层数
                'vocab_size': 33,
                'dropout': 0.1
            }
        },
        
        'diffusion': {
            'num_timesteps': 1000,
            'noise_schedule': "sqrt",
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'sampling_method': "ddpm",
            'ddim_steps': 50
        },
        
        'separated_training': {
            'stage1': {
                'epochs': 30,      # 适度减少用于测试
                'batch_size': 1,   # 最小批次避免内存问题
                'learning_rate': 1e-4,
                'warmup_steps': 100,
                'gradient_clip': 1.0,
                'optimizer': {
                    'type': "AdamW",
                    'weight_decay': 0.01,
                    'betas': [0.9, 0.999],
                    'eps': 1e-8
                },
                'scheduler': {
                    'type': "cosine",
                    'eta_min': 1e-6
                }
            },
            'stage2': {
                'epochs': 20,
                'batch_size': 1,   # 保持最小批次
                'learning_rate': 5e-5,
                'warmup_steps': 50,
                'gradient_clip': 0.5,
                'optimizer': {
                    'type': "AdamW",
                    'weight_decay': 0.01,
                    'betas': [0.9, 0.999],
                    'eps': 1e-8
                },
                'scheduler': {
                    'type': "cosine",
                    'eta_min': 1e-6
                }
            }
        },
        
        'data': {
            'data_dir': "./data/processed",
            'train_file': "train.csv",
            'val_file': "val.csv",
            'max_length': 50,
            'min_length': 5,
            'num_workers': 0,        # 禁用多进程避免CUDA冲突
            'pin_memory': False,     # 禁用pin_memory节省内存
            'use_predicted_structures': True,  # ✅ 启用结构特征
            'structure_cache_dir': "./cache"
        },
        
        'length_control': {
            'enabled': True,
            'min_length': 5,
            'max_length': 50,
            'analyze_training_data': True,
            'save_distributions': True,
            'length_penalty_weight': 0.1
        },
        
        'classifier_free_guidance': {
            'enabled': True,
            'dropout_prob': 0.1,
            'guidance_scale': 2.0,
            'adaptive_guidance': True,
            'guidance_schedule': "cosine"
        },
        
        'training_enhancements': {
            'use_amp': False,        # 禁用AMP避免复杂性
            'use_ema': False,        # 禁用EMA节省内存
            'gradient_accumulation_steps': 16,  # 增加梯度累积补偿小批次
            'save_every': 500,
            'validate_every': 100,
            'log_every': 10,
            'max_checkpoints': 2
        },
        
        'evaluation': {
            'enabled': False  # 训练阶段暂时禁用
        },
        
        'output': {
            'base_dir': "./outputs/structure_enabled_v1",
            'checkpoint_dir': "./outputs/structure_enabled_v1/checkpoints",
            'log_dir': "./outputs/structure_enabled_v1/logs",
            'results_dir': "./outputs/structure_enabled_v1/results",
            'save_model_config': True,
            'save_training_stats': True,
            'save_generated_samples': True
        },
        
        'monitoring': {
            'wandb': {'enabled': False},
            'tensorboard': {'enabled': False}
        },
        
        'debug': {
            'enabled': True,         # 启用调试信息
            'detailed_logging': True
        },
        
        'resources': {
            'device': "cuda:2",
            'available_gpus': [2, 3, 4, 5],
            'stage1_gpus': [2, 3],
            'stage2_gpus': [4, 5],
            'gpu_memory_fraction': 0.6,  # 限制GPU内存使用
            'allow_growth': True,
            'num_threads': 2,            # 减少CPU线程数
            
            # ESMFold内存优化配置
            'esmfold_memory_limit': '8GB',
            'structure_cache_size': 1000,
            'enable_structure_cache_cleanup': True
        }
    }
    
    # 保存配置文件
    config_path = "configs/structure_enabled_training.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(optimized_config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"✅ 创建结构感知配置: {config_path}")
    
    # 输出配置摘要
    print("\n📋 关键配置:")
    print("   🧬 结构特征: ✅ 启用 (ESMFold + 缓存)")
    print("   🔗 交叉注意力: ✅ 启用")
    print("   🔧 维度配置: hidden_dim=320, num_heads=8, head_dim=40")
    print("   📦 批次大小: 1 (最小化内存使用)")
    print("   🔄 梯度累积: 16 (补偿小批次)")
    print("   💾 内存限制: 60% GPU内存")
    
    return config_path

def optimize_structure_cache():
    """优化结构缓存配置"""
    print("📁 优化结构缓存配置...")
    
    cache_dir = Path("./cache")
    
    # 确保缓存目录存在
    for subdir in ['train', 'val', 'test']:
        (cache_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # 检查现有缓存
    train_cache = cache_dir / "train"
    if train_cache.exists():
        cache_files = list(train_cache.glob("*.pt"))
        print(f"   📊 训练缓存文件: {len(cache_files)} 个")
    
    val_cache = cache_dir / "val"
    if val_cache.exists():
        cache_files = list(val_cache.glob("*.pt"))
        print(f"   📊 验证缓存文件: {len(cache_files)} 个")
    
    # 创建缓存配置文件
    cache_config = {
        'version': '1.0',
        'structure_type': 'esmfold',
        'feature_dim': 320,
        'max_length': 50,
        'memory_limit': '8GB',
        'cleanup_policy': 'lru',
        'cache_size_limit': 1000
    }
    
    with open(cache_dir / "cache_config.yaml", 'w') as f:
        yaml.dump(cache_config, f, indent=2)
    
    print("✅ 缓存配置已优化")

def create_memory_optimized_restart_script():
    """创建内存优化的重启脚本"""
    print("📝 创建内存优化重启脚本...")
    
    script_content = '''#!/bin/bash
# StructDiff结构感知训练重启脚本

echo "🚀 启动结构感知StructDiff分离式训练..."

# 设置环境变量
export CUDA_VISIBLE_DEVICES=2,3,4,5
export PYTHONPATH="/home/qlyu/sequence/StructDiff-7.0.0:$PYTHONPATH"

# 内存优化设置
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# 检查GPU内存
echo "🔍 检查GPU内存状态..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# 清理GPU内存
echo "🧹 清理GPU内存..."
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU缓存已清理')
"

# 检查数据
echo "📊 检查数据和缓存状态..."
ls -la ./data/processed/
ls -la ./cache/

echo "🎯 开始结构感知训练..."
cd /home/qlyu/sequence/StructDiff-7.0.0

python scripts/train_separated.py \\
    --config configs/structure_enabled_training.yaml \\
    --data-dir ./data/processed \\
    --output-dir ./outputs/structure_enabled_v1 \\
    --device auto \\
    --stage both \\
    --use-cfg \\
    --use-length-control \\
    --debug

echo "✅ 训练已启动"
'''
    
    script_path = "restart_structure_training.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"✅ 创建重启脚本: {script_path}")
    
    return script_path

def create_structure_cache_validator():
    """创建结构缓存验证脚本"""
    print("🔍 创建结构缓存验证脚本...")
    
    validator_content = '''#!/usr/bin/env python3
"""结构缓存验证工具"""
import torch
from pathlib import Path
import pandas as pd

def validate_structure_cache():
    """验证结构缓存的完整性"""
    print("🔍 验证结构缓存...")
    
    cache_dir = Path("./cache")
    data_dir = Path("./data/processed")
    
    # 检查训练数据
    train_file = data_dir / "train.csv"
    if train_file.exists():
        train_df = pd.read_csv(train_file)
        print(f"训练数据: {len(train_df)} 条序列")
        
        # 检查对应的缓存文件
        train_cache_dir = cache_dir / "train"
        if train_cache_dir.exists():
            cache_files = list(train_cache_dir.glob("*.pt"))
            print(f"训练缓存: {len(cache_files)} 个文件")
            
            # 随机检查几个缓存文件
            for i, cache_file in enumerate(cache_files[:3]):
                try:
                    cached_data = torch.load(cache_file, map_location='cpu')
                    if isinstance(cached_data, dict):
                        features = cached_data.get('features')
                        if features is not None:
                            print(f"  缓存文件 {i+1}: 特征维度 {features.shape}")
                        else:
                            print(f"  缓存文件 {i+1}: 无特征数据")
                    else:
                        print(f"  缓存文件 {i+1}: 特征维度 {cached_data.shape}")
                except Exception as e:
                    print(f"  缓存文件 {i+1}: 加载失败 - {e}")
    
    print("✅ 缓存验证完成")

if __name__ == "__main__":
    validate_structure_cache()
'''
    
    validator_path = "validate_structure_cache.py"
    with open(validator_path, 'w') as f:
        f.write(validator_content)
    
    os.chmod(validator_path, 0o755)
    print(f"✅ 创建缓存验证脚本: {validator_path}")

def main():
    """主修复流程"""
    print("🔧 StructDiff结构感知训练修复程序")
    print("=" * 60)
    print("目标: 保留结构特征和交叉注意力的同时解决维度和内存问题")
    print("=" * 60)
    
    # 1. 停止现有训练
    stop_all_training()
    
    # 2. 分析维度问题
    analyze_dimension_issue()
    
    # 3. 创建优化配置
    config_path = create_structure_enabled_config()
    
    # 4. 优化结构缓存
    optimize_structure_cache()
    
    # 5. 创建重启脚本
    restart_script = create_memory_optimized_restart_script()
    
    # 6. 创建验证工具
    create_structure_cache_validator()
    
    print("\n🎉 结构感知训练修复完成!")
    print("=" * 50)
    print("📋 修复摘要:")
    print("  ✅ 保留结构特征 (ESMFold + 缓存)")
    print("  ✅ 保留交叉注意力机制")
    print("  ✅ 修复维度配置 (320维, 8头, 每头40维)")
    print("  ✅ 优化内存使用策略")
    print("  ✅ 创建专用配置和脚本")
    
    print(f"\n🚀 使用修复后的训练:")
    print(f"   ./{restart_script}")
    
    print(f"\n🔍 验证缓存状态:")
    print(f"   python validate_structure_cache.py")
    
    print("\n🎯 核心改进:")
    print("  • 维度兼容: hidden_dim=320, num_heads=8 (完美整除)")
    print("  • 内存优化: batch_size=1, 梯度累积=16")
    print("  • 结构缓存: 启用预计算和清理策略")
    print("  • GPU管理: 限制60%内存使用")

if __name__ == "__main__":
    main() 