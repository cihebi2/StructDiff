#!/usr/bin/env python3
"""
简化的StructDiff训练脚本
基于测试验证的配置，进行真正的StructDiff模型训练
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from structdiff.models.structdiff import StructDiff
from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
from structdiff.data.dataset import PeptideStructureDataset
from structdiff.utils.logger import get_logger

logger = get_logger(__name__)

def custom_collate_fn(batch):
    """自定义collate函数，处理不同长度的序列"""
    # 获取最大长度
    max_len = max(item['sequences'].shape[0] for item in batch)
    
    # 准备批次数据
    sequences = []
    attention_masks = []
    labels = []
    
    for item in batch:
        seq = item['sequences']
        # 截断或填充到固定长度
        if seq.shape[0] > max_len:
            seq = seq[:max_len]
        elif seq.shape[0] < max_len:
            # 使用pad_token_id填充（通常是0）
            pad_length = max_len - seq.shape[0]
            seq = torch.cat([seq, torch.zeros(pad_length, dtype=seq.dtype)], dim=0)
        
        sequences.append(seq)
        
        # 创建attention mask
        attention_mask = torch.ones(max_len, dtype=torch.bool)
        if item['sequences'].shape[0] < max_len:
            attention_mask[item['sequences'].shape[0]:] = False
        attention_masks.append(attention_mask)
        
        labels.append(item['label'])
    
    return {
        'sequences': torch.stack(sequences),
        'attention_mask': torch.stack(attention_masks),
        'peptide_type': torch.stack(labels)
    }

def create_model_config():
    """创建经过验证的模型配置"""
    return OmegaConf.create({
        "model": {
            "type": "StructDiff",
            "sequence_encoder": {
                "pretrained_model": "facebook/esm2_t6_8M_UR50D",
                "freeze_encoder": False,  # 允许微调
                "use_lora": True,  # 使用LoRA
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1
            },
            "structure_encoder": {
                "type": "multi_scale",
                "hidden_dim": 256,
                "use_esmfold": False,  # 完全禁用结构特征
                "local": {
                    "hidden_dim": 256,
                    "num_layers": 3,
                    "kernel_sizes": [3, 5, 7],
                    "dropout": 0.1
                },
                "global": {
                    "hidden_dim": 256,
                    "num_attention_heads": 8,
                    "num_layers": 3,
                    "dropout": 0.1
                },
                "fusion": {
                    "method": "attention",
                    "hidden_dim": 256
                }
            },
            "denoiser": {
                "hidden_dim": 320,  # 匹配ESM2_t6_8M
                "num_layers": 6,    # 增加层数以提升能力
                "num_heads": 8,     # 增加注意力头
                "dropout": 0.1,
                "use_cross_attention": False  # 禁用结构交叉注意力
            }
        },
        "diffusion": {
            "num_timesteps": 1000,  # 标准扩散步数
            "noise_schedule": "cosine",  # 更好的噪声调度
            "beta_start": 0.0001,
            "beta_end": 0.02
        },
        "data": {
            "max_length": 50,
            "use_predicted_structures": False
        }
    })

def create_data_config():
    """创建数据配置"""
    return OmegaConf.create({
        "data": {
            "max_length": 50,
            "use_predicted_structures": False
        },
        "model": {
            "sequence_encoder": {
                "pretrained_model": "facebook/esm2_t6_8M_UR50D"
            },
            "structure_encoder": {
                "use_esmfold": False
            }
        }
    })

def train_epoch(model, dataloader, diffusion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # 移动数据到设备
        sequences = batch['sequences'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        conditions = {'peptide_type': batch['peptide_type'].to(device)}
        
        # 获取序列嵌入
        with torch.no_grad():
            embeddings = model.sequence_encoder(sequences, attention_mask)
            # 取最后一层的隐藏状态，移除CLS和SEP token
            embeddings = embeddings.last_hidden_state[:, 1:-1, :]  # [batch, seq_len-2, hidden]
        
        # 前向扩散过程：添加噪声
        batch_size, seq_len, hidden_dim = embeddings.shape
        timesteps = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(embeddings)
        noisy_embeddings = diffusion.q_sample(embeddings, timesteps, noise)
        
        # 预测噪声
        optimizer.zero_grad()
        predicted_noise = model.denoiser(
            noisy_embeddings=noisy_embeddings,
            timesteps=timesteps,
            attention_mask=attention_mask[:, 1:-1],  # 移除CLS和SEP的mask
            structure_features=None,
            conditions=conditions
        )[0]  # 只取预测结果
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # 日志
        if batch_idx % 50 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss

def validate(model, dataloader, diffusion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['sequences'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            conditions = {'peptide_type': batch['peptide_type'].to(device)}
            
            # 获取序列嵌入
            embeddings = model.sequence_encoder(sequences, attention_mask)
            embeddings = embeddings.last_hidden_state[:, 1:-1, :]
            
            # 前向扩散
            batch_size, seq_len, hidden_dim = embeddings.shape
            timesteps = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
            noise = torch.randn_like(embeddings)
            noisy_embeddings = diffusion.q_sample(embeddings, timesteps, noise)
            
            # 预测噪声
            predicted_noise = model.denoiser(
                noisy_embeddings=noisy_embeddings,
                timesteps=timesteps,
                attention_mask=attention_mask[:, 1:-1],
                structure_features=None,
                conditions=conditions
            )[0]
            
            # 计算损失
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='简化StructDiff训练')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('--device', type=str, default='cuda:4', help='训练设备')
    parser.add_argument('--output-dir', type=str, default='./outputs/structdiff_simplified', help='输出目录')
    parser.add_argument('--use-amp', action='store_true', help='使用混合精度训练')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建配置
    model_config = create_model_config()
    data_config = create_data_config()
    
    # 创建数据集
    logger.info("创建数据集...")
    train_dataset = PeptideStructureDataset(
        data_path="data/processed/train.csv",
        config=data_config,
        is_training=True
    )
    
    val_dataset = PeptideStructureDataset(
        data_path="data/processed/val.csv", 
        config=data_config,
        is_training=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 禁用多进程避免CUDA冲突
        pin_memory=False,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=custom_collate_fn
    )
    
    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    logger.info("创建模型...")
    model = StructDiff(model_config.model).to(device)
    logger.info(f"模型参数: {model.count_parameters():,}")
    
    # 创建扩散过程
    diffusion = GaussianDiffusion(
        num_timesteps=model_config.diffusion.num_timesteps,
        noise_schedule=model_config.diffusion.noise_schedule,
        beta_start=model_config.diffusion.beta_start,
        beta_end=model_config.diffusion.beta_end
    )
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    # 训练循环
    logger.info("开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_loader, diffusion, optimizer, device, epoch)
        
        # 验证
        val_loss = validate(model, val_loader, diffusion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, time={epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': model_config
            }, os.path.join(args.output_dir, 'best_model.pt'))
            logger.info(f"保存最佳模型，验证损失: {val_loss:.6f}")
        
        # 定期保存检查点
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': model_config
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))
    
    logger.info("训练完成！")
    logger.info(f"最佳验证损失: {best_val_loss:.6f}")

if __name__ == "__main__":
    main() 