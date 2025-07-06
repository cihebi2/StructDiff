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
            # 处理结构特征 - 如果存在的话
            structures = []
            for item in batch:
                if key in item and item[key] is not None:
                    structures.append(item[key])
                else:
                    structures.append(None)
            result[key] = structures
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

def training_step(model, diffusion, batch, device, esmfold_wrapper):
    """执行一个训练步骤，包含结构特征"""
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
    
    # Prepare structure features
    structure_features = None
    if 'structures' in batch and batch['structures'] is not None:
        # 处理结构特征 - 如果批次中有结构数据
        try:
            # 简单处理：为每个序列预测结构
            structure_features = []
            for i, seq in enumerate(batch['sequence']):
                if batch['structures'][i] is not None:
                    structure_features.append(batch['structures'][i])
                else:
                    # 实时预测结构
                    try:
                        struct_feat = esmfold_wrapper.predict_structure(seq)
                        structure_features.append(struct_feat)
                    except:
                        structure_features.append(None)
        except Exception as e:
            print(f"结构特征处理错误: {e}")
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

def full_train_with_esmfold():
    """完整的200 epoch训练函数，启用ESMFold"""
    print("🚀 开始完整的200 epoch训练（启用ESMFold）...")
    
    # Setup logging
    output_dir = "/home/qlyu/sequence/StructDiff-7.0.0/outputs/full_training_200_esmfold"
    os.makedirs(output_dir, exist_ok=True)
    
    setup_logger(
        level=logging.INFO,
        log_file=f"{output_dir}/training.log"
    )
    logger = get_logger(__name__)
    
    try:
        # Load config
        config_path = "/home/qlyu/sequence/StructDiff-7.0.0/configs/separated_training.yaml"
        config = load_config(config_path)
        
        # 启用结构预测
        config.data.use_predicted_structures = True
        config.model.structure_encoder.use_esmfold = True
        
        logger.info("✅ 配置加载成功，已启用ESMFold结构预测")
        
        # 初始化ESMFold
        device = torch.device('cuda:0')
        logger.info("🔄 正在初始化ESMFold...")
        esmfold_wrapper = ESMFoldWrapper(device=device)
        
        if esmfold_wrapper.available:
            logger.info("✅ ESMFold初始化成功")
        else:
            logger.error("❌ ESMFold初始化失败")
            raise RuntimeError("ESMFold不可用")
        
        # Create datasets with ESMFold
        train_dataset = PeptideStructureDataset(
            data_path="/home/qlyu/sequence/StructDiff-7.0.0/data/processed/train.csv",
            config=config,
            is_training=True,
            shared_esmfold=esmfold_wrapper
        )
        
        val_dataset = PeptideStructureDataset(
            data_path="/home/qlyu/sequence/StructDiff-7.0.0/data/processed/val.csv",
            config=config,
            is_training=False,
            shared_esmfold=esmfold_wrapper
        )
        
        logger.info(f"✅ 数据集创建成功，训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
        
        # Create dataloaders with custom collate function
        batch_size = 8  # 减小批次大小以适应ESMFold的显存需求
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=0  # 避免多进程问题
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=0
        )
        
        logger.info("✅ 数据加载器创建成功")
        
        # Create model and diffusion
        model, diffusion = create_model_and_diffusion(config)
        logger.info(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # Move to GPU
        model = model.to(device)
        logger.info("✅ 模型已移动到GPU")
        
        # Create optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)  # 降低学习率
        scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        
        # Training parameters
        num_epochs = 200
        save_every = 10  # 每10个epoch保存一次
        validate_every = 5  # 每5个epoch验证一次
        
        logger.info(f"🎯 开始训练，共 {num_epochs} 个epoch")
        logger.info(f"批次大小: {batch_size}, 学习率: {optimizer.param_groups[0]['lr']}")
        logger.info(f"预计显存使用: 15-20GB (ESMFold + 模型)")
        
        # Training metrics
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Training loop
            model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()
                
                # Training step
                loss = training_step(model, diffusion, batch, device, esmfold_wrapper)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'avg_loss': f'{epoch_loss/num_batches:.6f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'gpu_mem': f'{torch.cuda.memory_allocated(device)//1024**3}GB'
                })
                
                # Log every 20 batches (更频繁的日志)
                if batch_idx % 20 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}, GPU Memory: {torch.cuda.memory_allocated(device)//1024**3}GB")
                
                # 清理显存
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Update learning rate
            scheduler.step()
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            logger.info(f"Epoch {epoch+1} 完成，平均训练损失: {avg_train_loss:.6f}, 学习率: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Validation
            if (epoch + 1) % validate_every == 0:
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
                    'best_val_loss': best_val_loss
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
            'esmfold_enabled': True
        }
        
        final_metrics_path = f"{output_dir}/final_metrics.json"
        with open(final_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info(f"训练摘要:")
        logger.info(f"  总epoch数: {num_epochs}")
        logger.info(f"  最终训练损失: {train_losses[-1]:.6f}")
        logger.info(f"  最佳验证损失: {best_val_loss:.6f}")
        logger.info(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  ESMFold已启用: ✅")
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    full_train_with_esmfold() 