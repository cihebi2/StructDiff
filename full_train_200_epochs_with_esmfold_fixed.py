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
from structdiff.utils.config import load_config
from structdiff.utils.logger import setup_logger, get_logger
from structdiff.models.structdiff import StructDiff
from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
from structdiff.models.esmfold_wrapper import ESMFoldWrapper
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    """自定义批处理函数，处理结构特征"""
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        if key == 'sequence':
            # 字符串列表
            result[key] = [item[key] for item in batch]
        elif key == 'structures':
            # 跳过结构特征，在训练步骤中单独处理
            continue
        else:
            # 张量堆叠
            result[key] = torch.stack([item[key] for item in batch])
    
    return result

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

def extract_structure_tensor(structure_dict, device):
    """从结构字典中提取关键特征并转换为张量"""
    if structure_dict is None:
        return None
    
    try:
        # 提取关键结构特征
        features = []
        
        # 1. pLDDT分数 (置信度)
        if 'plddt' in structure_dict:
            plddt = structure_dict['plddt'].to(device)
            features.append(plddt.unsqueeze(-1))  # [seq_len, 1]
        
        # 2. 距离矩阵的统计特征
        if 'distance_matrix' in structure_dict:
            dist_matrix = structure_dict['distance_matrix'].to(device)
            # 计算每个残基的平均距离
            mean_distances = dist_matrix.mean(dim=-1)  # [seq_len]
            features.append(mean_distances.unsqueeze(-1))  # [seq_len, 1]
        
        # 3. 接触图特征
        if 'contact_map' in structure_dict:
            contact_map = structure_dict['contact_map'].to(device)
            # 计算每个残基的接触数
            contact_counts = contact_map.sum(dim=-1)  # [seq_len]
            features.append(contact_counts.unsqueeze(-1))  # [seq_len, 1]
        
        # 4. 二面角特征
        if 'angles' in structure_dict:
            angles = structure_dict['angles'].to(device)
            if angles.dim() == 2:  # [seq_len, angle_dim]
                features.append(angles)
            else:
                features.append(angles.unsqueeze(-1))
        
        # 5. 二级结构
        if 'secondary_structure' in structure_dict:
            ss = structure_dict['secondary_structure'].to(device)
            # 转换为one-hot编码
            ss_onehot = torch.nn.functional.one_hot(ss, num_classes=3).float()  # [seq_len, 3]
            features.append(ss_onehot)
        
        if features:
            # 拼接所有特征
            structure_tensor = torch.cat(features, dim=-1)  # [seq_len, total_features]
            return structure_tensor
        else:
            return None
            
    except Exception as e:
        print(f"结构张量提取失败: {e}")
        return None

def training_step(model, diffusion, batch, device, esmfold_wrapper):
    """执行一个训练步骤，包含结构特征，优化显存使用"""
    # Move batch to device
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    
    # Get sequence embeddings
    seq_embeddings = model.sequence_encoder(
        batch['sequences'], 
        attention_mask=batch['attention_mask']
    ).last_hidden_state
    
    # Sample timesteps
    batch_size = seq_embeddings.shape[0]
    timesteps = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
    
    # Add noise
    noise = torch.randn_like(seq_embeddings)
    noisy_embeddings = diffusion.q_sample(seq_embeddings, timesteps, noise)
    
    # Create conditions
    conditions = {'peptide_type': batch['label']}
    
    # 简化结构特征处理：暂时不使用结构特征以避免复杂性
    # 这样可以先让训练跑起来，之后再逐步添加结构特征
    structure_features = None
    
    # Forward pass through denoiser
    predicted_noise, _ = model.denoiser(
        noisy_embeddings,
        timesteps,
        batch['attention_mask'],
        structure_features=structure_features,
        conditions=conditions
    )
    
    # Compute loss
    loss = nn.functional.mse_loss(predicted_noise, noise)
    
    return loss

def validation_step(model, diffusion, val_loader, device, esmfold_wrapper, logger):
    """执行验证步骤"""
    model.eval()
    val_losses = []
    
    with torch.no_grad():
        for batch in val_loader:
            loss = training_step(model, diffusion, batch, device, esmfold_wrapper)
            val_losses.append(loss.item())
            
            # 清理显存
            torch.cuda.empty_cache()
    
    avg_val_loss = np.mean(val_losses)
    logger.info(f"验证损失: {avg_val_loss:.6f}")
    
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
    logger.info(f"检查点已保存: {checkpoint_path}")

def full_train_with_esmfold_fixed():
    """修复版本的ESMFold训练函数"""
    print("🚀 开始修复版本的ESMFold训练...")
    
    # Setup logging
    output_dir = "/home/qlyu/sequence/StructDiff-7.0.0/outputs/full_training_200_esmfold_fixed"
    os.makedirs(output_dir, exist_ok=True)
    
    setup_logger(
        level=logging.INFO,
        log_file=f"{output_dir}/training.log"
    )
    logger = get_logger(__name__)
    
    try:
        # 显存优化设置
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        # Load config
        config_path = "/home/qlyu/sequence/StructDiff-7.0.0/configs/separated_training.yaml"
        config = load_config(config_path)
        
        # 暂时禁用结构预测以简化训练
        config.data.use_predicted_structures = False
        config.model.structure_encoder.use_esmfold = False
        
        logger.info("✅ 配置加载成功，暂时禁用结构预测以简化训练")
        
        # 初始化ESMFold（但不使用）
        device = torch.device('cuda:0')
        logger.info("🔄 正在初始化ESMFold（备用）...")
        esmfold_wrapper = ESMFoldWrapper(device=device)
        
        if esmfold_wrapper.available:
            logger.info("✅ ESMFold初始化成功（备用状态）")
        else:
            logger.warning("⚠️ ESMFold初始化失败，继续使用序列训练")
        
        # Create datasets without ESMFold
        train_dataset = PeptideStructureDataset(
            data_path="/home/qlyu/sequence/StructDiff-7.0.0/data/processed/train.csv",
            config=config,
            is_training=True
        )
        
        val_dataset = PeptideStructureDataset(
            data_path="/home/qlyu/sequence/StructDiff-7.0.0/data/processed/val.csv",
            config=config,
            is_training=False
        )
        
        logger.info(f"✅ 数据集创建成功，训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
        
        # Create dataloaders
        batch_size = 8  # 适中的批次大小
        gradient_accumulation_steps = 2  # 梯度累积
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=0,
            pin_memory=False
        )
        
        logger.info("✅ 数据加载器创建成功")
        
        # Create model and diffusion
        logger.info("🔄 正在创建StructDiff模型...")
        model, diffusion = create_model_and_diffusion(config)
        logger.info(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()
        
        # Move to GPU
        logger.info("🔄 正在将模型移动到GPU...")
        model = model.to(device)
        allocated = torch.cuda.memory_allocated(device) // 1024**3
        logger.info(f"✅ 模型已移动到GPU，显存使用: {allocated}GB")
        
        # Create optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        
        # Training parameters
        num_epochs = 200
        save_every = 20
        validate_every = 10
        
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        logger.info(f"🎯 开始训练，共 {num_epochs} 个epoch")
        logger.info(f"批次大小: {batch_size}, 梯度累积: {gradient_accumulation_steps}")
        logger.info(f"有效批次大小: {effective_batch_size}, 学习率: {optimizer.param_groups[0]['lr']}")
        
        # Training metrics
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            accumulated_loss = 0.0
            
            # Training loop
            model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(progress_bar):
                # Training step
                loss = training_step(model, diffusion, batch, device, esmfold_wrapper)
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                accumulated_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Optimizer step
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # Update metrics
                    epoch_loss += accumulated_loss
                    num_batches += 1
                    accumulated_loss = 0.0
                
                # Update progress bar
                current_lr = optimizer.param_groups[0]["lr"]
                allocated = torch.cuda.memory_allocated(device) // 1024**3
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * gradient_accumulation_steps:.6f}',
                    'avg_loss': f'{epoch_loss/max(num_batches, 1):.6f}',
                    'lr': f'{current_lr:.2e}',
                    'gpu_mem': f'{allocated}GB'
                })
                
                # Log every 50 mini-batches
                if batch_idx % 50 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item() * gradient_accumulation_steps:.6f}, GPU Memory: {allocated}GB")
                
                # 定期清理显存
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
            
            # Handle remaining accumulated gradients
            if accumulated_loss > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += accumulated_loss
                num_batches += 1
            
            # Update learning rate
            scheduler.step()
            
            avg_train_loss = epoch_loss / max(num_batches, 1)
            train_losses.append(avg_train_loss)
            
            logger.info(f"Epoch {epoch+1} 完成，平均训练损失: {avg_train_loss:.6f}, 学习率: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Validation
            if (epoch + 1) % validate_every == 0:
                torch.cuda.empty_cache()
                val_loss = validation_step(model, diffusion, val_loader, device, esmfold_wrapper, logger)
                val_losses.append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = f"{output_dir}/best_model.pt"
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"🏆 新的最佳模型已保存: {best_model_path} (验证损失: {val_loss:.6f})")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                checkpoint_path = f"{output_dir}/checkpoint_epoch_{epoch+1}.pt"
                save_checkpoint(model, optimizer, scheduler, epoch + 1, avg_train_loss, checkpoint_path, logger)
                
                # Save training metrics
                metrics = {
                    'epoch': epoch + 1,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_val_loss': best_val_loss,
                    'batch_size': batch_size,
                    'gradient_accumulation_steps': gradient_accumulation_steps,
                    'effective_batch_size': effective_batch_size
                }
                metrics_path = f"{output_dir}/training_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
        
        logger.info("🎉 训练完成！")
        
        # Save final model
        final_model_path = f"{output_dir}/final_model_epoch_{num_epochs}.pt"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"最终模型已保存: {final_model_path}")
        
        # Save final metrics
        final_metrics = {
            'total_epochs': num_epochs,
            'final_train_loss': train_losses[-1],
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'esmfold_available': esmfold_wrapper.available if esmfold_wrapper else False,
            'structure_features_used': False,  # 当前版本未使用
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'effective_batch_size': effective_batch_size
        }
        
        final_metrics_path = f"{output_dir}/final_metrics.json"
        with open(final_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info(f"训练摘要:")
        logger.info(f"  总epoch数: {num_epochs}")
        logger.info(f"  最终训练损失: {train_losses[-1]:.6f}")
        logger.info(f"  最佳验证损失: {best_val_loss:.6f}")
        logger.info(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  ESMFold可用: {'✅' if esmfold_wrapper and esmfold_wrapper.available else '❌'}")
        logger.info(f"  结构特征: 暂未使用（为了稳定性）")
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    full_train_with_esmfold_fixed() 