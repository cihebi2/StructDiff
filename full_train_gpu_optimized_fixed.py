#!/usr/bin/env python3
"""
GPU利用率优化版本训练脚本 - 修复版本
目标：将GPU利用率从20%提升到70%以上，训练速度提升3-5倍
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import logging
import time
import gc
from datetime import datetime
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

# 设置环境变量进行内存优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'

# 添加项目路径
sys.path.append('/home/qlyu/sequence/StructDiff-7.0.0')

from structdiff.data.dataset import PeptideStructureDataset
from structdiff.data.collator import PeptideStructureCollator
from structdiff.utils.config import load_config
from structdiff.utils.logger import setup_logger, get_logger
from structdiff.models.structdiff import StructDiff
from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
from structdiff.models.esmfold_wrapper import ESMFoldWrapper

def clear_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def setup_optimized_environment():
    """设置优化环境"""
    # 启用PyTorch优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 设置多线程
    torch.set_num_threads(4)
    
    print("✅ 优化环境设置完成")

def check_gpu_availability():
    """检查GPU可用性和内存"""
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA不可用")
    
    device_count = torch.cuda.device_count()
    print(f"📊 检测到 {device_count} 个GPU")
    
    # 选择最适合的GPU
    best_gpu = 1  # 默认使用GPU 1
    max_free_memory = 0
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1e9
        
        # 检查当前内存使用
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated() / 1e9
        free_memory = total_memory - allocated
        
        print(f"GPU {i}: {props.name}, 总内存: {total_memory:.1f}GB, 可用: {free_memory:.1f}GB")
        
        if free_memory > max_free_memory and free_memory > 15:  # 至少需要15GB
            best_gpu = i
            max_free_memory = free_memory
    
    if max_free_memory < 15:
        raise RuntimeError(f"❌ 没有足够的GPU内存 (需要至少15GB)")
    
    print(f"🎯 选择GPU {best_gpu}进行训练")
    return best_gpu

def setup_shared_esmfold(device):
    """创建共享的ESMFold实例"""
    print("🔄 创建共享ESMFold实例...")
    
    # 清理内存
    clear_memory()
    
    try:
        shared_esmfold = ESMFoldWrapper(device=device)
        
        if shared_esmfold.available:
            print("✅ 共享ESMFold实例创建成功")
            print(f"ESMFold内存使用: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        else:
            print("❌ ESMFold实例创建失败")
            shared_esmfold = None
            
    except Exception as e:
        print(f"❌ ESMFold初始化失败: {e}")
        shared_esmfold = None
    
    return shared_esmfold

def create_optimized_data_loaders(config, shared_esmfold):
    """创建优化的数据加载器"""
    print("🔄 创建优化数据加载器...")
    
    # 创建数据集
    train_dataset = PeptideStructureDataset(
        data_path="/home/qlyu/sequence/StructDiff-7.0.0/data/processed/train.csv",
        config=config,
        is_training=True,
        cache_dir="./cache/train",
        shared_esmfold=shared_esmfold
    )
    
    val_dataset = PeptideStructureDataset(
        data_path="/home/qlyu/sequence/StructDiff-7.0.0/data/processed/val.csv",
        config=config,
        is_training=False,
        cache_dir="./cache/val",
        shared_esmfold=shared_esmfold
    )
    
    # 创建数据整理器
    collator = PeptideStructureCollator(config)
    
    # 优化的数据加载器配置
    optimized_batch_size = 8  # 从2增加到8
    num_workers = 4          # 从0增加到4
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=optimized_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,           # 启用内存固定
        prefetch_factor=2,         # 预取因子
        persistent_workers=True,   # 持久化工作进程
        collate_fn=collator,
        drop_last=True            # 丢弃最后不完整的批次
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=optimized_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        collate_fn=collator,
        drop_last=False
    )
    
    print(f"✅ 优化数据加载器创建完成")
    print(f"📊 批次大小: {optimized_batch_size} (4倍提升)")
    print(f"📊 工作进程: {num_workers}")
    print(f"📊 训练数据集: {len(train_dataset)} 样本")
    print(f"📊 验证数据集: {len(val_dataset)} 样本")
    
    return train_loader, val_loader

def create_optimized_model(config, device, shared_esmfold):
    """创建优化的模型"""
    print("🔄 创建优化模型...")
    
    # 临时禁用模型内部ESMFold加载
    original_enabled = config.model.structure_encoder.get('use_esmfold', False)
    config.model.structure_encoder.use_esmfold = False
    
    # 创建模型
    model = StructDiff(config.model).to(device)
    
    # 恢复设置并强制设置ESMFold实例
    config.model.structure_encoder.use_esmfold = original_enabled
    
    # 如果有共享的ESMFold实例，手动设置到模型中
    if shared_esmfold and shared_esmfold.available:
        print("🔗 设置共享ESMFold实例到模型...")
        # 设置ESMFold实例
        if hasattr(model.structure_encoder, 'esmfold'):
            model.structure_encoder.esmfold = shared_esmfold
        if hasattr(model.structure_encoder, '_esmfold'):
            model.structure_encoder._esmfold = shared_esmfold
        
        # 确保ESMFold被标记为可用
        model.structure_encoder.use_esmfold = True
        print("✅ 共享ESMFold实例设置完成")
    
    # 启用训练模式
    model.train()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    return model

def move_to_device(obj, device):
    """递归地将对象移动到指定设备"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    else:
        return obj

def optimized_training_step(model, diffusion, batch, device, scaler, gradient_accumulation_steps=2):
    """优化的训练步骤"""
    try:
        # 移动数据到设备
        batch = move_to_device(batch, device)
        
        # 检查基本字段
        if 'sequences' not in batch or 'attention_mask' not in batch:
            return None, float('inf')
        
        # 采样时间步
        batch_size = batch['sequences'].shape[0]
        timesteps = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
        
        # 前向传播 (使用混合精度)
        with autocast():
            outputs = model(
                sequences=batch['sequences'],
                attention_mask=batch['attention_mask'],
                timesteps=timesteps,
                structures=batch.get('structures'),
                conditions=batch.get('conditions'),
                return_loss=True
            )
            
            # 获取损失
            if 'total_loss' in outputs:
                loss = outputs['total_loss']
            elif 'diffusion_loss' in outputs:
                loss = outputs['diffusion_loss']
            elif 'loss' in outputs:
                loss = outputs['loss']
            else:
                return None, float('inf')
            
            loss = loss / gradient_accumulation_steps
        
        # 检查损失是否为NaN
        if torch.isnan(loss) or torch.isinf(loss):
            return None, float('inf')
        
        # 反向传播
        scaler.scale(loss).backward()
        
        return outputs, loss.item() * gradient_accumulation_steps
        
    except Exception as e:
        print(f"❌ 训练步骤失败: {e}")
        return None, float('inf')

def optimized_training_loop(model, diffusion, train_loader, val_loader, optimizer, scheduler, scaler, device, num_epochs=100):
    """优化的训练循环"""
    print("🚀 开始GPU优化训练...")
    
    # 训练配置
    gradient_accumulation_steps = 2  # 从8减少到2
    log_interval = 10  # 更频繁的日志
    
    print(f"📊 训练配置:")
    print(f"  总epoch: {num_epochs}")
    print(f"  批次大小: {train_loader.batch_size}")
    print(f"  梯度累积步数: {gradient_accumulation_steps}")
    print(f"  有效批次大小: {train_loader.batch_size * gradient_accumulation_steps}")
    
    # 创建输出目录
    output_dir = "outputs/gpu_optimized_training"
    os.makedirs(output_dir, exist_ok=True)
    
    # 性能监控
    training_stats = {
        'epoch_times': [],
        'batch_times': [],
        'memory_usage': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch_start_time = time.time()
            
            # 执行优化的训练步骤
            outputs, loss = optimized_training_step(
                model, diffusion, batch, device, scaler, gradient_accumulation_steps
            )
            
            if outputs is not None:
                total_loss += loss
                num_batches += 1
            
            # 梯度更新
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 优化器步骤
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # 学习率调度
                if scheduler:
                    scheduler.step()
            
            # 记录性能指标
            batch_time = time.time() - batch_start_time
            training_stats['batch_times'].append(batch_time)
            
            # 记录GPU内存使用
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(device) / 1e9
                training_stats['memory_usage'].append(memory_used)
            
            # 更新进度条
            if outputs is not None:
                progress_bar.set_postfix({
                    'Loss': f'{loss:.4f}',
                    'GPU_Mem': f'{memory_used:.1f}GB',
                    'Time': f'{batch_time:.2f}s'
                })
            
            # 定期清理内存
            if batch_idx % 50 == 0:
                clear_memory()
        
        # Epoch结束统计
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        avg_batch_time = sum(training_stats['batch_times'][-num_batches:]) / num_batches if num_batches > 0 else 0
        
        training_stats['epoch_times'].append(epoch_time)
        
        print(f"📊 Epoch {epoch+1} 完成:")
        print(f"  平均损失: {avg_loss:.6f}")
        print(f"  Epoch时间: {epoch_time:.2f}s")
        print(f"  平均批次时间: {avg_batch_time:.2f}s")
        print(f"  预计剩余时间: {epoch_time * (num_epochs - epoch - 1) / 3600:.1f}小时")
        
        # 验证
        if epoch % 5 == 0:  # 每5个epoch验证一次
            val_loss = validate_model(model, diffusion, val_loader, device, scaler)
            print(f"📊 验证损失: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'training_stats': training_stats
                }, f'{output_dir}/best_model.pth')
                
                print(f"💾 最佳模型已保存 (验证损失: {best_val_loss:.6f})")
        
        # 保存检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict(),
                'training_stats': training_stats
            }, f'{output_dir}/checkpoint_epoch_{epoch}.pth')
    
    # 保存最终统计
    with open(f'{output_dir}/training_stats.json', 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    print("🎉 GPU优化训练完成!")
    
    # 性能分析
    analyze_training_performance(training_stats)

def validate_model(model, diffusion, val_loader, device, scaler):
    """验证模型"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            outputs, loss = optimized_training_step(model, diffusion, batch, device, scaler, 1)
            if outputs is not None:
                total_loss += loss
                num_batches += 1
            
            # 清理内存
            clear_memory()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    model.train()
    return avg_loss

def analyze_training_performance(training_stats):
    """分析训练性能"""
    print("📊 训练性能分析:")
    
    if training_stats['batch_times']:
        avg_batch_time = sum(training_stats['batch_times']) / len(training_stats['batch_times'])
        print(f"  平均批次时间: {avg_batch_time:.2f}s")
        
        # 与原始配置比较 (原始: batch_size=2, 约2s/batch)
        original_batch_time = 2.0
        speedup = original_batch_time / avg_batch_time
        print(f"  训练速度提升: {speedup:.2f}x")
    
    if training_stats['memory_usage']:
        avg_memory = sum(training_stats['memory_usage']) / len(training_stats['memory_usage'])
        max_memory = max(training_stats['memory_usage'])
        print(f"  平均GPU内存使用: {avg_memory:.1f}GB")
        print(f"  峰值GPU内存使用: {max_memory:.1f}GB")

def main():
    """主函数"""
    try:
        print("🚀 启动GPU优化训练...")
        
        # 设置优化环境
        setup_optimized_environment()
        
        # 检查GPU
        device_id = check_gpu_availability()
        device = torch.device(f'cuda:{device_id}')
        
        # 加载配置
        config_path = "/home/qlyu/sequence/StructDiff-7.0.0/configs/separated_training.yaml"
        config = load_config(config_path)
        
        # 启用结构特征和GPU优化
        config.data.use_predicted_structures = True
        config.model.structure_encoder.use_esmfold = True
        
        print("✅ 配置加载成功，已启用结构特征和GPU优化")
        
        # 创建共享ESMFold实例
        shared_esmfold = setup_shared_esmfold(device)
        
        if not shared_esmfold or not shared_esmfold.available:
            print("❌ 共享ESMFold实例创建失败")
            return
        
        # 创建优化的数据加载器
        train_loader, val_loader = create_optimized_data_loaders(config, shared_esmfold)
        
        # 创建优化的模型
        model = create_optimized_model(config, device, shared_esmfold)
        
        # 创建扩散过程
        diffusion = GaussianDiffusion(
            num_timesteps=config.diffusion.num_timesteps,
            noise_schedule=config.diffusion.noise_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end
        )
        
        # 加载预训练模型
        print("🔄 加载预训练模型...")
        checkpoint_path = "/home/qlyu/sequence/StructDiff-7.0.0/outputs/full_training_200_esmfold_fixed/best_model.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print("✅ 预训练模型加载成功")
        else:
            print("⚠️ 预训练模型未找到，从头开始训练")
        
        # 创建优化器和调度器
        # 只优化非ESMFold参数
        trainable_params = []
        for name, param in model.named_parameters():
            if not ('esmfold' in name.lower() and 'structure_encoder' in name):
                trainable_params.append(param)
            else:
                param.requires_grad = False
        
        optimizer = optim.AdamW(
            trainable_params,
            lr=5e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        
        # 创建梯度缩放器
        scaler = GradScaler()
        
        # 开始优化训练
        optimized_training_loop(
            model, diffusion, train_loader, val_loader, 
            optimizer, scheduler, scaler, device, num_epochs=100
        )
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 