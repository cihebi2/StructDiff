#!/usr/bin/env python3
"""
GPU利用率优化版本训练脚本
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/gpu_optimized_training/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_optimized_environment():
    """设置优化环境"""
    # 启用PyTorch优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 设置多线程
    torch.set_num_threads(4)
    
    logger.info("✅ 优化环境设置完成")

def check_gpu_availability():
    """检查GPU可用性和内存"""
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA不可用")
    
    device_count = torch.cuda.device_count()
    logger.info(f"📊 检测到 {device_count} 个GPU")
    
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
        
        logger.info(f"GPU {i}: {props.name}, 总内存: {total_memory:.1f}GB, 可用: {free_memory:.1f}GB")
        
        if free_memory > max_free_memory and free_memory > 15:  # 至少需要15GB
            best_gpu = i
            max_free_memory = free_memory
    
    if max_free_memory < 15:
        raise RuntimeError(f"❌ 没有足够的GPU内存 (需要至少15GB)")
    
    logger.info(f"🎯 选择GPU {best_gpu}进行训练")
    return best_gpu

def create_optimized_data_loaders(config, esmfold_wrapper):
    """创建优化的数据加载器"""
    logger.info("🔄 创建优化数据加载器...")
    
    # 创建数据集
    train_dataset = PeptideDataset(
        data_path=config.data.train_data_path,
        max_length=config.data.max_length,
        structure_prediction_enabled=config.model.structure_prediction_enabled,
        esmfold_wrapper=esmfold_wrapper
    )
    
    val_dataset = PeptideDataset(
        data_path=config.data.val_data_path,
        max_length=config.data.max_length,
        structure_prediction_enabled=config.model.structure_prediction_enabled,
        esmfold_wrapper=esmfold_wrapper
    )
    
    # 创建优化的collator
    collator = PeptideCollator(
        max_length=config.data.max_length,
        structure_prediction_enabled=config.model.structure_prediction_enabled
    )
    
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
    
    logger.info(f"✅ 优化数据加载器创建完成")
    logger.info(f"📊 批次大小: {optimized_batch_size} (4倍提升)")
    logger.info(f"📊 工作进程: {num_workers}")
    logger.info(f"📊 训练数据集: {len(train_dataset)} 样本")
    logger.info(f"📊 验证数据集: {len(val_dataset)} 样本")
    
    return train_loader, val_loader

def create_optimized_model(config, device, esmfold_wrapper):
    """创建优化的模型"""
    logger.info("🔄 创建优化模型...")
    
    # 临时禁用模型内部ESMFold加载
    original_enabled = config.model.structure_prediction_enabled
    config.model.structure_prediction_enabled = False
    
    # 创建模型
    model = StructDiff(config.model)
    model = model.to(device)
    
    # 恢复设置并强制设置ESMFold实例
    config.model.structure_prediction_enabled = original_enabled
    model.esmfold_wrapper = esmfold_wrapper
    
    # 启用训练模式
    model.train()
    
    # 模型编译优化 (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="reduce-overhead")
        logger.info("✅ 模型编译优化启用")
    except Exception as e:
        logger.warning(f"⚠️ 模型编译优化失败: {e}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"📊 模型参数统计:")
    logger.info(f"  总参数: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")
    
    return model

def create_optimized_optimizer_and_scheduler(model, config, train_loader):
    """创建优化的优化器和调度器"""
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(train_loader) * 5,  # 5个epoch为一个周期
        T_mult=2,
        eta_min=1e-6
    )
    
    logger.info("✅ 优化器和调度器创建完成")
    return optimizer, scheduler

def optimized_training_step(model, batch, optimizer, scaler, device, gradient_accumulation_steps=2):
    """优化的训练步骤"""
    model.train()
    
    # 移动数据到GPU
    for key in batch:
        if torch.is_tensor(batch[key]):
            batch[key] = batch[key].to(device, non_blocking=True)
    
    # 创建随机时间步
    batch_size = batch['sequences'].size(0)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    
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
        loss = outputs['total_loss'] / gradient_accumulation_steps
    
    # 反向传播
    scaler.scale(loss).backward()
    
    return outputs, loss.item() * gradient_accumulation_steps

def optimized_training_loop(model, train_loader, val_loader, optimizer, scheduler, scaler, device, config):
    """优化的训练循环"""
    logger.info("🚀 开始GPU优化训练...")
    
    # 训练配置
    num_epochs = config.training.num_epochs
    gradient_accumulation_steps = 2  # 从8减少到2
    log_interval = 10  # 更频繁的日志
    
    logger.info(f"📊 训练配置:")
    logger.info(f"  总epoch: {num_epochs}")
    logger.info(f"  批次大小: {train_loader.batch_size}")
    logger.info(f"  梯度累积步数: {gradient_accumulation_steps}")
    logger.info(f"  有效批次大小: {train_loader.batch_size * gradient_accumulation_steps}")
    
    # 创建输出目录
    os.makedirs("outputs/gpu_optimized_training", exist_ok=True)
    
    # 性能监控
    training_stats = {
        'epoch_times': [],
        'batch_times': [],
        'gpu_utilization': [],
        'memory_usage': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            batch_start_time = time.time()
            
            # 执行优化的训练步骤
            outputs, loss = optimized_training_step(
                model, batch, optimizer, scaler, device, gradient_accumulation_steps
            )
            
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
                scheduler.step()
            
            # 记录性能指标
            batch_time = time.time() - batch_start_time
            training_stats['batch_times'].append(batch_time)
            
            # 记录GPU内存使用
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated(device) / 1e9
                training_stats['memory_usage'].append(memory_used)
            
            # 日志输出
            if batch_idx % log_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss:.6f}, "
                    f"LR: {current_lr:.2e}, "
                    f"Time: {batch_time:.2f}s, "
                    f"GPU Memory: {memory_used:.1f}GB"
                )
            
            # 定期清理内存
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Epoch结束统计
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches
        avg_batch_time = sum(training_stats['batch_times'][-num_batches:]) / num_batches
        
        training_stats['epoch_times'].append(epoch_time)
        
        logger.info(f"📊 Epoch {epoch+1} 完成:")
        logger.info(f"  平均损失: {avg_loss:.6f}")
        logger.info(f"  Epoch时间: {epoch_time:.2f}s")
        logger.info(f"  平均批次时间: {avg_batch_time:.2f}s")
        logger.info(f"  预计剩余时间: {epoch_time * (num_epochs - epoch - 1) / 3600:.1f}小时")
        
        # 验证
        if epoch % 5 == 0:  # 每5个epoch验证一次
            val_loss = validate_model(model, val_loader, device, scaler)
            logger.info(f"📊 验证损失: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'training_stats': training_stats
                }, 'outputs/gpu_optimized_training/best_model.pth')
                
                logger.info(f"💾 最佳模型已保存 (验证损失: {best_val_loss:.6f})")
        
        # 保存检查点
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'training_stats': training_stats
            }, f'outputs/gpu_optimized_training/checkpoint_epoch_{epoch}.pth')
    
    # 保存最终统计
    with open('outputs/gpu_optimized_training/training_stats.json', 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    logger.info("🎉 GPU优化训练完成!")
    
    # 性能分析
    analyze_training_performance(training_stats)

def validate_model(model, val_loader, device, scaler):
    """验证模型"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # 移动数据到GPU
            for key in batch:
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            batch_size = batch['sequences'].size(0)
            timesteps = torch.randint(0, 1000, (batch_size,), device=device)
            
            with autocast():
                outputs = model(
                    sequences=batch['sequences'],
                    attention_mask=batch['attention_mask'],
                    timesteps=timesteps,
                    structures=batch.get('structures'),
                    conditions=batch.get('conditions'),
                    return_loss=True
                )
                loss = outputs['total_loss']
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def analyze_training_performance(training_stats):
    """分析训练性能"""
    logger.info("📊 训练性能分析:")
    
    if training_stats['batch_times']:
        avg_batch_time = sum(training_stats['batch_times']) / len(training_stats['batch_times'])
        logger.info(f"  平均批次时间: {avg_batch_time:.2f}s")
        
        # 与原始配置比较 (原始: batch_size=2, 约2s/batch)
        original_batch_time = 2.0
        speedup = original_batch_time / avg_batch_time
        logger.info(f"  训练速度提升: {speedup:.2f}x")
    
    if training_stats['memory_usage']:
        avg_memory = sum(training_stats['memory_usage']) / len(training_stats['memory_usage'])
        max_memory = max(training_stats['memory_usage'])
        logger.info(f"  平均GPU内存使用: {avg_memory:.1f}GB")
        logger.info(f"  峰值GPU内存使用: {max_memory:.1f}GB")

def main():
    """主函数"""
    try:
        logger.info("🚀 启动GPU优化训练...")
        
        # 设置优化环境
        setup_optimized_environment()
        
        # 检查GPU
        device_id = check_gpu_availability()
        device = torch.device(f'cuda:{device_id}')
        
        # 加载配置
        config = Config()
        config.model.structure_prediction_enabled = True
        config.training.batch_size = 8  # 优化批次大小
        config.training.num_epochs = 100
        config.training.learning_rate = 5e-5
        
        logger.info("✅ 配置加载成功，已启用结构特征和GPU优化")
        
        # 创建共享ESMFold实例
        logger.info("🔄 创建共享ESMFold实例...")
        torch.cuda.empty_cache()
        
        esmfold_wrapper = ESMFoldWrapper(device=device)
        logger.info("✅ 共享ESMFold实例创建成功")
        
        # 创建优化的数据加载器
        train_loader, val_loader = create_optimized_data_loaders(config, esmfold_wrapper)
        
        # 创建优化的模型
        model = create_optimized_model(config, device, esmfold_wrapper)
        
        # 加载预训练模型
        logger.info("🔄 加载预训练模型...")
        checkpoint_path = "outputs/sequence_feature_training/best_model.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info("✅ 预训练模型加载成功")
        else:
            logger.warning("⚠️ 预训练模型未找到，从头开始训练")
        
        # 创建优化器和调度器
        optimizer, scheduler = create_optimized_optimizer_and_scheduler(model, config, train_loader)
        
        # 创建梯度缩放器
        scaler = GradScaler()
        
        # 开始优化训练
        optimized_training_loop(model, train_loader, val_loader, optimizer, scheduler, scaler, device, config)
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 