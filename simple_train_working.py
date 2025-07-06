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

# Add project root to path
sys.path.append('/home/qlyu/sequence/StructDiff-7.0.0')

from structdiff.data.dataset import PeptideStructureDataset
from structdiff.utils.config import load_config
from structdiff.utils.logger import setup_logger, get_logger
from structdiff.models.structdiff import StructDiff
from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
from torch.utils.data import DataLoader

def custom_collate_fn(batch):
    """自定义批处理函数，处理可能的结构特征不匹配"""
    # 收集所有键
    keys = batch[0].keys()
    result = {}
    
    for key in keys:
        if key == 'structures':
            # 跳过结构特征，避免形状不匹配问题
            continue
        elif key == 'sequence':
            # 字符串列表
            result[key] = [item[key] for item in batch]
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

def training_step(model, diffusion, batch, device):
    """执行一个训练步骤"""
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
    
    # Forward pass through denoiser
    predicted_noise, _ = model.denoiser(
        noisy_embeddings,
        timesteps,
        batch['attention_mask'],
        structure_features=None,  # 明确设置为None
        conditions=conditions
    )
    
    # Compute loss
    loss = nn.functional.mse_loss(predicted_noise, noise)
    
    return loss

def simple_train():
    """简化的训练函数"""
    print("🚀 开始简化训练...")
    
    # Setup logging
    setup_logger(
        level=logging.INFO,
        log_file="/home/qlyu/sequence/StructDiff-7.0.0/outputs/separated_training/simple_training.log"
    )
    logger = get_logger(__name__)
    
    try:
        # Load config
        config_path = "/home/qlyu/sequence/StructDiff-7.0.0/configs/separated_training.yaml"
        config = load_config(config_path)
        
        # 强制禁用结构预测
        config.data.use_predicted_structures = False
        
        logger.info("✅ 配置加载成功，已禁用结构预测")
        
        # Create dataset
        dataset = PeptideStructureDataset(
            data_path="/home/qlyu/sequence/StructDiff-7.0.0/data/processed/train.csv",
            config=config,
            is_training=True
        )
        
        logger.info(f"✅ 数据集创建成功，大小: {len(dataset)}")
        
        # Create dataloader with custom collate function
        dataloader = DataLoader(
            dataset, 
            batch_size=8,  # 较小的批次大小
            shuffle=True,
            collate_fn=custom_collate_fn,
            num_workers=0  # 避免多进程问题
        )
        
        logger.info("✅ 数据加载器创建成功")
        
        # Create model and diffusion
        model, diffusion = create_model_and_diffusion(config)
        logger.info(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # Move to GPU (when CUDA_VISIBLE_DEVICES=1, the visible device index is 0)
        device = torch.device('cuda:0')
        model = model.to(device)
        logger.info("✅ 模型已移动到GPU")
        
        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        # Training loop
        model.train()
        num_epochs = 5  # 少量epoch用于测试
        
        logger.info(f"🎯 开始训练，共 {num_epochs} 个epoch")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()
                
                # Training step
                loss = training_step(model, diffusion, batch, device)
                
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
                    'avg_loss': f'{epoch_loss/num_batches:.6f}'
                })
                
                # Log every 10 batches
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.6f}")
                
                # Early break for testing (remove this for full training)
                if batch_idx >= 50:  # 只训练50个批次用于测试
                    break
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1} 完成，平均损失: {avg_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % 2 == 0:
                checkpoint_path = f"/home/qlyu/sequence/StructDiff-7.0.0/outputs/separated_training/checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                logger.info(f"检查点已保存: {checkpoint_path}")
        
        logger.info("🎉 训练完成！")
        
        # Save final model
        final_model_path = "/home/qlyu/sequence/StructDiff-7.0.0/outputs/separated_training/final_model.pt"
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"最终模型已保存: {final_model_path}")
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # Create output directory
    os.makedirs("/home/qlyu/sequence/StructDiff-7.0.0/outputs/separated_training", exist_ok=True)
    
    # Run training
    simple_train() 