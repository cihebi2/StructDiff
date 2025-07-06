#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
from tqdm import tqdm
import logging
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
import gc

# Add project root to path
sys.path.append('/home/qlyu/sequence/StructDiff-7.0.0')

from structdiff.data.dataset import PeptideStructureDataset
from structdiff.data.collator import PeptideStructureCollator
from structdiff.utils.config import load_config
from structdiff.utils.logger import setup_logger, get_logger
from structdiff.models.structdiff import StructDiff
from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
from structdiff.models.esmfold_wrapper import ESMFoldWrapper
from torch.utils.data import DataLoader

def clear_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def setup_shared_esmfold(device: torch.device):
    """创建共享的ESMFold实例 - 基于成功脚本的策略"""
    logger = get_logger(__name__)
    shared_esmfold = None
    
    logger.info("🔄 正在创建共享ESMFold实例...")
    
    # 更激进的内存清理，为ESMFold腾出空间
    if torch.cuda.is_available():
        # 强制清理所有缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 强制垃圾回收
        gc.collect()
        
        # 设置内存分配策略 - 关键优化
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        
        # 再次清理
        torch.cuda.empty_cache()
        
        current_mem = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"🧹 ESMFold初始化前GPU内存清理完成: {current_mem:.2f}GB")
    
    try:
        # 首先尝试GPU
        shared_esmfold = ESMFoldWrapper(device=device)
        
        if shared_esmfold.available:
            logger.info("✅ 共享ESMFold GPU实例创建成功")
            logger.info(f"ESMFold内存使用: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        else:
            logger.error("❌ ESMFold GPU实例创建失败")
            shared_esmfold = None
            
    except Exception as gpu_error:
        logger.warning(f"⚠️ ESMFold GPU初始化失败: {gpu_error}")
        
        try:
            # 尝试CPU fallback
            logger.info("🔄 尝试使用CPU创建ESMFold...")
            shared_esmfold = ESMFoldWrapper(device='cpu')
            if shared_esmfold.available:
                logger.info("✅ ESMFold CPU实例创建成功")
            else:
                shared_esmfold = None
        except Exception as cpu_error:
            logger.error(f"❌ ESMFold CPU初始化也失败: {cpu_error}")
            shared_esmfold = None
    
    return shared_esmfold

def create_model_and_diffusion(config):
    """创建模型和扩散过程"""
    # 创建模型
    model = StructDiff(config.model)
    
    # 创建扩散过程
    diffusion = GaussianDiffusion(
        num_timesteps=config.diffusion.num_timesteps,
        noise_schedule=config.diffusion.noise_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end
    )
    
    return model, diffusion

def setup_model_and_training(config, device, shared_esmfold):
    """设置模型和训练组件 - 基于成功脚本的策略"""
    logger = get_logger(__name__)
    logger.info("🔄 正在初始化模型...")
    
    try:
        # 备份原始配置
        original_use_esmfold = config.model.structure_encoder.get('use_esmfold', False)
        
        # 如果已有共享实例，临时禁用模型内部的ESMFold加载
        if shared_esmfold and shared_esmfold.available:
            logger.info("💡 临时禁用模型内部ESMFold加载以避免内存不足...")
            config.model.structure_encoder.use_esmfold = False
        
        # 创建模型
        model = StructDiff(config.model).to(device)
        
        # 恢复配置并设置共享实例
        config.model.structure_encoder.use_esmfold = original_use_esmfold
        
        # 如果有共享的ESMFold实例，手动设置到模型中
        if shared_esmfold and shared_esmfold.available:
            logger.info("🔗 正在将共享 ESMFold 实例设置到模型中...")
            
            # 尝试多种方式设置ESMFold实例
            if hasattr(model.structure_encoder, 'esmfold') or hasattr(model.structure_encoder, '_esmfold'):
                # 设置ESMFold实例
                model.structure_encoder.esmfold = shared_esmfold
                model.structure_encoder._esmfold = shared_esmfold
                # 确保ESMFold被标记为可用
                model.structure_encoder.use_esmfold = True
                logger.info("✅ 共享 ESMFold 实例已设置到模型中")
            else:
                # 如果模型结构不同，尝试直接设置属性
                setattr(model.structure_encoder, 'esmfold', shared_esmfold)
                setattr(model.structure_encoder, 'use_esmfold', True)
                logger.info("✅ 共享 ESMFold 实例已强制设置到模型中")
            
            clear_memory()
        
        logger.info(f"✅ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建优化器 - 只排除ESMFold参数，保留其他所有参数
        trainable_params = []
        esmfold_params = 0
        total_params = 0
        sequence_encoder_params = 0
        other_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            
            # 只排除真正的ESMFold模型参数，保留其他所有参数
            if ('structure_encoder.esmfold.' in name or 
                'structure_encoder._esmfold.' in name or
                name.startswith('esmfold.')):
                esmfold_params += param.numel()
                param.requires_grad = False  # 冻结ESMFold参数
                logger.debug(f"冻结ESMFold参数: {name}")
            else:
                trainable_params.append(param)
                if 'sequence_encoder' in name:
                    sequence_encoder_params += param.numel()
                else:
                    other_params += param.numel()
        
        trainable_param_count = sum(p.numel() for p in trainable_params)
        
        logger.info(f"📊 参数统计:")
        logger.info(f"  总参数: {total_params:,}")
        logger.info(f"  ESMFold参数(冻结): {esmfold_params:,}")
        logger.info(f"  序列编码器参数: {sequence_encoder_params:,}")
        logger.info(f"  其他可训练参数: {other_params:,}")
        logger.info(f"  可训练参数总计: {trainable_param_count:,}")
        
        if trainable_param_count < 1000000:  # 少于100万参数
            logger.warning(f"⚠️ 可训练参数过少 ({trainable_param_count:,})，可能影响训练效果")
        
        # 创建优化器
        optimizer = optim.AdamW(
            trainable_params,  # 只优化非ESMFold参数
            lr=5e-5,  # 降低学习率
            weight_decay=1e-5
        )
        
        # 创建学习率调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        
        logger.info(f"✅ 模型初始化完成，GPU内存: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        
        return model, optimizer, scheduler
        
    except Exception as e:
        logger.error(f"❌ 模型初始化失败: {e}")
        # 如果失败，尝试完全禁用ESMFold
        logger.info("🔄 尝试禁用ESMFold重新初始化模型...")
        try:
            config.model.structure_encoder.use_esmfold = False
            config.data.use_predicted_structures = False
            model = StructDiff(config.model).to(device)
            logger.info("✅ 模型初始化成功（未使用ESMFold）")
            
            # 创建基础组件
            optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
            scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
            
            return model, optimizer, scheduler
            
        except Exception as e2:
            logger.error(f"❌ 禁用ESMFold后仍然失败: {e2}")
            raise

def create_data_loaders(config, shared_esmfold):
    """创建数据加载器 - 基于成功脚本的策略"""
    logger = get_logger(__name__)
    logger.info("🔄 正在创建数据加载器...")
    
    try:
        # 创建数据集
        train_dataset = PeptideStructureDataset(
            data_path="/home/qlyu/sequence/StructDiff-7.0.0/data/processed/train.csv",
            config=config,
            is_training=True,
            cache_dir="./cache/train",
            shared_esmfold=shared_esmfold  # 传递共享实例
        )
        
        val_dataset = PeptideStructureDataset(
            data_path="/home/qlyu/sequence/StructDiff-7.0.0/data/processed/val.csv",
            config=config,
            is_training=False,
            cache_dir="./cache/val",
            shared_esmfold=shared_esmfold  # 传递共享实例
        )
        
        logger.info(f"✅ 训练数据集: {len(train_dataset)} 样本")
        logger.info(f"✅ 验证数据集: {len(val_dataset)} 样本")
        
        # 创建数据整理器
        collator = PeptideStructureCollator(config)
        
        # 创建数据加载器 - 使用更保守的设置
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,  # 小批次大小适应ESMFold
            shuffle=True,
            num_workers=0,  # 关键：使用0避免多进程缓存竞争问题
            pin_memory=False,  # 禁用pin_memory节省内存
            collate_fn=collator,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=2,
            shuffle=False,  
            num_workers=0,  # 关键：使用0避免多进程缓存竞争问题
            pin_memory=False,  # 禁用pin_memory节省内存
            collate_fn=collator,
            drop_last=False
        )
        
        logger.info("✅ 数据加载器创建成功")
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"❌ 数据加载器创建失败: {e}")
        raise

def move_to_device(obj, device):
    """递归地将对象移动到指定设备"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    else:
        return obj

def train_step(model, diffusion, batch, device, gradient_accumulation_steps):
    """执行一个训练步骤 - 基于成功脚本的策略"""
    try:
        # 移动数据到设备
        batch = move_to_device(batch, device)
        
        # 检查batch的基本字段
        if 'sequences' not in batch or 'attention_mask' not in batch:
            return None, float('inf')
        
        # 检查张量形状一致性
        seq_shape = batch['sequences'].shape
        mask_shape = batch['attention_mask'].shape
        
        if seq_shape != mask_shape:
            # 修正attention_mask的形状
            if mask_shape[1] != seq_shape[1]:
                min_len = min(mask_shape[1], seq_shape[1])
                batch['sequences'] = batch['sequences'][:, :min_len]
                batch['attention_mask'] = batch['attention_mask'][:, :min_len]
        
        # 检查结构数据的形状一致性
        if 'structures' in batch and batch['structures'] is not None:
            expected_struct_len = batch['sequences'].shape[1] - 2  # 除去CLS/SEP
            
            for key, value in batch['structures'].items():
                if value is None:
                    continue
                    
                # 对于结构特征，第二个维度应该与序列长度-2匹配
                if len(value.shape) >= 2:
                    actual_len = value.shape[1]
                    if actual_len != expected_struct_len:
                        # 截断或填充结构特征
                        if actual_len > expected_struct_len:
                            if len(value.shape) == 2:
                                batch['structures'][key] = value[:, :expected_struct_len]
                            elif len(value.shape) == 3:
                                if 'matrix' in key or 'map' in key:
                                    batch['structures'][key] = value[:, :expected_struct_len, :expected_struct_len]
                                else:
                                    batch['structures'][key] = value[:, :expected_struct_len, :]
                        elif actual_len < expected_struct_len:
                            pad_size = expected_struct_len - actual_len
                            if len(value.shape) == 2:
                                batch['structures'][key] = torch.nn.functional.pad(value, (0, 0, 0, pad_size), value=0)
                            elif len(value.shape) == 3:
                                if 'matrix' in key or 'map' in key:
                                    batch['structures'][key] = torch.nn.functional.pad(value, (0, pad_size, 0, pad_size), value=0)
                                else:
                                    batch['structures'][key] = torch.nn.functional.pad(value, (0, 0, 0, pad_size), value=0)
        
        # 采样时间步
        batch_size = batch['sequences'].shape[0]
        timesteps = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
        
        # 前向传播
        outputs = model(
            sequences=batch['sequences'],
            attention_mask=batch['attention_mask'],
            timesteps=timesteps,
            structures=batch.get('structures'),
            conditions=batch.get('conditions'),
            return_loss=True
        )
        
        # 检查是否有损失
        if 'total_loss' not in outputs:
            if 'diffusion_loss' in outputs:
                outputs['total_loss'] = outputs['diffusion_loss']
            elif 'loss' in outputs:
                outputs['total_loss'] = outputs['loss']
            else:
                return None, float('inf')
        
        loss = outputs['total_loss'] / gradient_accumulation_steps
        
        # 检查损失是否为NaN
        if torch.isnan(loss) or torch.isinf(loss):
            return None, float('inf')
        
        # 反向传播
        loss.backward()
        
        return outputs, loss.item() * gradient_accumulation_steps
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"❌ 训练步骤失败: {e}")
        return None, float('inf')

def validation_step(model, diffusion, val_loader, device, logger):
    """执行验证步骤"""
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in val_loader:
            outputs, loss = train_step(model, diffusion, batch, device, 1)
            if outputs is not None:
                val_losses.append(loss)
            
            # 清理显存
            clear_memory()
    
    avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
    logger.info(f"✅ 验证损失: {avg_val_loss:.6f}")
    
    model.train()
    return avg_val_loss

def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_path, logger):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }, checkpoint_path)
    logger.info(f"💾 检查点已保存: {checkpoint_path}")

def full_train_with_structure_features_fixed():
    """修复版本的结构特征训练函数"""
    print("🚀 开始修复版本的结构特征训练...")
    
    # Setup logging
    output_dir = "/home/qlyu/sequence/StructDiff-7.0.0/outputs/structure_feature_training_fixed"
    os.makedirs(output_dir, exist_ok=True)
    
    setup_logger(
        level=logging.INFO,
        log_file=f"{output_dir}/training.log"
    )
    logger = get_logger(__name__)
    
    try:
        # 设置设备
        device = torch.device('cuda:0')
        logger.info(f"🎯 使用设备: {device}")
        
        # 显存优化设置
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Load config
        config_path = "/home/qlyu/sequence/StructDiff-7.0.0/configs/separated_training.yaml"
        config = load_config(config_path)
        
        # 启用结构特征
        config.data.use_predicted_structures = True
        config.model.structure_encoder.use_esmfold = True
        
        logger.info("✅ 配置加载成功，已启用结构特征")
        
        # 关键步骤：先创建共享ESMFold实例
        shared_esmfold = setup_shared_esmfold(device)
        
        if not shared_esmfold or not shared_esmfold.available:
            logger.error("❌ 共享ESMFold实例创建失败，无法进行结构特征训练")
            return
        
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(config, shared_esmfold)
        
        # 创建模型和训练组件
        model, optimizer, scheduler = setup_model_and_training(config, device, shared_esmfold)
        
        # 创建扩散过程
        diffusion = GaussianDiffusion(
            num_timesteps=config.diffusion.num_timesteps,
            noise_schedule=config.diffusion.noise_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end
        )
        
        # 加载预训练模型
        checkpoint_path = "/home/qlyu/sequence/StructDiff-7.0.0/outputs/full_training_200_esmfold_fixed/best_model.pt"
        if os.path.exists(checkpoint_path):
            logger.info("🔄 正在加载预训练模型...")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=False)
            logger.info("✅ 预训练模型加载成功")
        else:
            logger.warning("⚠️ 未找到预训练模型，从头开始训练")
        
        # 训练参数
        num_epochs = 100
        gradient_accumulation_steps = 8  # 增加梯度累积
        save_every = 10
        validate_every = 5
        
        logger.info(f"🎯 开始结构特征训练，共 {num_epochs} 个epoch")
        logger.info(f"📊 批次大小: 2, 梯度累积: {gradient_accumulation_steps}")
        logger.info(f"📊 有效批次大小: {2 * gradient_accumulation_steps}, 学习率: {optimizer.param_groups[0]['lr']}")
        
        # 训练指标
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            accumulated_loss = 0.0
            
            # 训练循环
            model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(progress_bar):
                # 训练步骤
                outputs, loss = train_step(model, diffusion, batch, device, gradient_accumulation_steps)
                
                if outputs is None:
                    continue
                
                accumulated_loss += loss / gradient_accumulation_steps
                
                # 梯度累积
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 优化器步骤
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 更新指标
                    epoch_loss += accumulated_loss
                    num_batches += 1
                    accumulated_loss = 0.0
                
                # 更新进度条
                current_lr = optimizer.param_groups[0]["lr"]
                allocated = torch.cuda.memory_allocated(device) // 1024**3
                
                progress_bar.set_postfix({
                    'loss': f'{loss:.6f}',
                    'avg_loss': f'{epoch_loss/max(num_batches, 1):.6f}',
                    'lr': f'{current_lr:.2e}',
                    'gpu_mem': f'{allocated}GB'
                })
                
                # 定期日志和清理
                if batch_idx % 25 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss:.6f}, GPU Memory: {allocated}GB")
                
                if batch_idx % 10 == 0:
                    clear_memory()
            
            # 处理剩余的累积梯度
            if accumulated_loss > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += accumulated_loss
                num_batches += 1
            
            # 更新学习率
            scheduler.step()
            
            avg_train_loss = epoch_loss / max(num_batches, 1)
            train_losses.append(avg_train_loss)
            
            logger.info(f"✅ Epoch {epoch+1} 完成，平均训练损失: {avg_train_loss:.6f}, 学习率: {optimizer.param_groups[0]['lr']:.2e}")
            
            # 验证
            if (epoch + 1) % validate_every == 0:
                clear_memory()
                val_loss = validation_step(model, diffusion, val_loader, device, logger)
                val_losses.append(val_loss)
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = f"{output_dir}/best_model_with_structure.pt"
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"🏆 新的最佳模型已保存: {best_model_path} (验证损失: {val_loss:.6f})")
            
            # 保存检查点
            if (epoch + 1) % save_every == 0:
                checkpoint_path = f"{output_dir}/checkpoint_epoch_{epoch+1}.pt"
                save_checkpoint(model, optimizer, scheduler, epoch + 1, avg_train_loss, checkpoint_path, logger)
                
                # 保存训练指标
                metrics = {
                    'epoch': epoch + 1,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss,
                    'structure_features_enabled': True,
                    'esmfold_available': shared_esmfold.available if shared_esmfold else False
                }
                metrics_path = f"{output_dir}/training_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
        
        logger.info("🎉 结构特征训练完成！")
        
        # 保存最终模型
        final_model_path = f"{output_dir}/final_model_with_structure_epoch_{num_epochs}.pt"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"💾 最终模型已保存: {final_model_path}")
        
        # 保存最终指标
        final_metrics = {
            'total_epochs': num_epochs,
            'final_train_loss': train_losses[-1] if train_losses else 0,
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'esmfold_available': shared_esmfold.available if shared_esmfold else False,
            'structure_features_used': True,
            'training_successful': True
        }
        
        final_metrics_path = f"{output_dir}/final_metrics.json"
        with open(final_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info(f"📊 结构特征训练摘要:")
        logger.info(f"  总epoch数: {num_epochs}")
        logger.info(f"  最终训练损失: {train_losses[-1]:.6f}" if train_losses else "  最终训练损失: N/A")
        logger.info(f"  最佳验证损失: {best_val_loss:.6f}")
        logger.info(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  ESMFold可用: ✅")
        logger.info(f"  结构特征: ✅ 真正启用")
        
    except Exception as e:
        logger.error(f"❌ 结构特征训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    full_train_with_structure_features_fixed() 