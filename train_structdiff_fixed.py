#!/usr/bin/env python3
"""
修复版的StructDiff训练脚本
基于成功的调试结果，添加详细的进度输出和错误处理
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import argparse

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from structdiff.models.structdiff import StructDiff
from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
from structdiff.data.dataset import PeptideStructureDataset

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
        "sequence_encoder": {
            "pretrained_model": "facebook/esm2_t6_8M_UR50D",
            "freeze_encoder": False,  # 允许微调
            "use_lora": True,  # 使用LoRA
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1
        },
        "structure_encoder": {
            "use_esmfold": False,  # 完全禁用结构特征
            "hidden_dim": 256
        },
        "denoiser": {
            "hidden_dim": 320,  # 匹配ESM2_t6_8M
            "num_layers": 6,    # 增加层数
            "num_heads": 8,     # 增加注意力头
            "dropout": 0.1,
            "use_cross_attention": False  # 禁用结构交叉注意力
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

def train_epoch(model, dataloader, diffusion, optimizer, device, epoch, total_epochs):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        batch_start_time = time.time()
        
        # 移动数据到设备
        sequences = batch['sequences'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        conditions = {'peptide_type': batch['peptide_type'].to(device)}
        
        # 获取序列嵌入
        with torch.no_grad():
            embeddings = model.sequence_encoder(sequences, attention_mask)
            embeddings = embeddings.last_hidden_state[:, 1:-1, :]
        
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
            attention_mask=attention_mask[:, 1:-1],
            structure_features=None,
            conditions=conditions
        )[0]
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        batch_time = time.time() - batch_start_time
        
        # 实时日志输出
        if batch_idx % 10 == 0:
            elapsed_time = time.time() - start_time
            progress = (batch_idx + 1) / num_batches
            eta = elapsed_time / progress * (1 - progress) if progress > 0 else 0
            
            print(f"Epoch {epoch+1}/{total_epochs} | "
                  f"Batch {batch_idx+1}/{num_batches} ({progress*100:.1f}%) | "
                  f"Loss: {loss.item():.6f} | "
                  f"Time: {batch_time:.2f}s | "
                  f"ETA: {eta/60:.1f}min")
            sys.stdout.flush()
    
    avg_loss = total_loss / num_batches
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1} 完成 | 平均损失: {avg_loss:.6f} | 时间: {epoch_time/60:.2f}分钟")
    return avg_loss

def validate(model, dataloader, diffusion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    print("开始验证...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
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
            
            if batch_idx % 50 == 0:
                print(f"验证进度: {batch_idx+1}/{num_batches}")
                sys.stdout.flush()
    
    avg_loss = total_loss / num_batches
    print(f"验证完成 | 平均损失: {avg_loss:.6f}")
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='修复版StructDiff训练')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='训练设备')
    parser.add_argument('--output-dir', type=str, default='./outputs/structdiff_fixed', help='输出目录')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 开始StructDiff训练 (修复版)")
    print("=" * 60)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"✓ 使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"✓ 输出目录: {args.output_dir}")
    
    # 创建配置
    print("📋 创建配置...")
    model_config = create_model_config()
    data_config = create_data_config()
    print("✓ 配置创建完成")
    
    # 创建数据集
    print("📊 创建数据集...")
    sys.stdout.flush()
    
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
    
    print(f"✓ 训练集: {len(train_dataset)} 样本")
    print(f"✓ 验证集: {len(val_dataset)} 样本")
    
    # 创建数据加载器
    print("📥 创建数据加载器...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
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
    print(f"✓ 数据加载器创建完成，每个epoch {len(train_loader)} 个批次")
    
    # 创建模型
    print("🧠 创建模型...")
    sys.stdout.flush()
    model = StructDiff(model_config).to(device)
    print(f"✓ 模型创建完成，参数数量: {model.count_parameters():,}")
    
    # 创建扩散过程
    print("🌀 创建扩散过程...")
    diffusion = GaussianDiffusion(
        num_timesteps=1000,
        noise_schedule="cosine",
        beta_start=0.0001,
        beta_end=0.02
    )
    print("✓ 扩散过程创建完成")
    
    # 创建优化器
    print("⚙️ 创建优化器...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    print(f"✓ 优化器创建完成，学习率: {args.lr}")
    
    # 训练循环
    print("\n" + "=" * 60)
    print("🏋️ 开始训练循环")
    print("=" * 60)
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n📈 Epoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        # 训练
        train_loss = train_epoch(model, train_loader, diffusion, optimizer, device, epoch, args.epochs)
        
        # 验证
        val_loss = validate(model, val_loader, diffusion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 记录
        epoch_time = time.time() - epoch_start_time
        print(f"\n📊 Epoch {epoch+1} 总结:")
        print(f"  训练损失: {train_loss:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"  学习率: {current_lr:.2e}")
        print(f"  用时: {epoch_time/60:.2f} 分钟")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': model_config
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"💾 保存最佳模型，验证损失: {val_loss:.6f}")
        
        # 定期保存检查点
        if epoch % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': model_config
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'))
        
        sys.stdout.flush()
    
    print("\n" + "=" * 60)
    print("🎉 训练完成！")
    print(f"💯 最佳验证损失: {best_val_loss:.6f}")
    print(f"📁 模型保存在: {args.output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main() 