#!/usr/bin/env python3
"""
调试版本的StructDiff训练脚本
添加详细日志输出，帮助定位训练卡住的问题
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

def custom_collate_fn(batch):
    """自定义collate函数，处理不同长度的序列"""
    print(f"DEBUG: Processing batch of size {len(batch)}")
    
    # 获取最大长度
    max_len = max(item['sequences'].shape[0] for item in batch)
    print(f"DEBUG: Max sequence length in batch: {max_len}")
    
    # 准备批次数据
    sequences = []
    attention_masks = []
    labels = []
    
    for i, item in enumerate(batch):
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
    
    print(f"DEBUG: Collated batch successfully")
    return {
        'sequences': torch.stack(sequences),
        'attention_mask': torch.stack(attention_masks),
        'peptide_type': torch.stack(labels)
    }

def main():
    parser = argparse.ArgumentParser(description='调试StructDiff训练')
    parser.add_argument('--device', type=str, default='cuda:0', help='训练设备')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🐛 开始调试StructDiff训练")
    print("=" * 50)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"✓ 使用设备: {device}")
    
    # 数据配置
    data_config = OmegaConf.create({
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
    
    print("✓ 配置创建完成")
    
    # 创建数据集
    print("📊 创建数据集...")
    sys.stdout.flush()
    
    try:
        from structdiff.data.dataset import PeptideStructureDataset
        
        train_dataset = PeptideStructureDataset(
            data_path="data/processed/train.csv",
            config=data_config,
            is_training=True
        )
        print(f"✓ 训练数据集创建完成: {len(train_dataset)} 样本")
        
        # 创建数据加载器
        print("📥 创建数据加载器...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,  # 很小的批次大小
            shuffle=False,  # 禁用随机打乱便于调试
            num_workers=0,
            pin_memory=False,
            collate_fn=custom_collate_fn
        )
        print("✓ 数据加载器创建完成")
        
        # 测试数据加载
        print("🔄 测试数据加载...")
        sys.stdout.flush()
        
        for i, batch in enumerate(train_loader):
            print(f"✓ 成功加载第 {i+1} 个批次")
            print(f"  序列形状: {batch['sequences'].shape}")
            print(f"  标签: {batch['peptide_type']}")
            
            if i >= 2:  # 只测试前几个批次
                break
        
        print("✓ 数据加载测试完成")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建简化的模型配置
    print("🧠 创建模型...")
    sys.stdout.flush()
    
    model_config = OmegaConf.create({
        "sequence_encoder": {
            "pretrained_model": "facebook/esm2_t6_8M_UR50D",
            "freeze_encoder": True,  # 固定编码器减少训练复杂度
            "use_lora": False
        },
        "structure_encoder": {
            "use_esmfold": False,
            "hidden_dim": 256
        },
        "denoiser": {
            "hidden_dim": 320,
            "num_layers": 2,  # 只用2层便于调试
            "num_heads": 4,
            "dropout": 0.1,
            "use_cross_attention": False
        }
    })
    
    try:
        from structdiff.models.structdiff import StructDiff
        
        print("  初始化StructDiff模型...")
        sys.stdout.flush()
        model = StructDiff(model_config)
        print(f"✓ 模型创建成功，参数数量: {model.count_parameters():,}")
        
        print("  移动模型到设备...")
        sys.stdout.flush()
        model = model.to(device)
        print("✓ 模型移动到设备完成")
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建扩散过程
    print("🌀 创建扩散过程...")
    sys.stdout.flush()
    
    try:
        from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
        
        diffusion = GaussianDiffusion(
            num_timesteps=100,  # 减少时间步便于调试
            noise_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02
        )
        print("✓ 扩散过程创建完成")
        
    except Exception as e:
        print(f"❌ 扩散过程创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试单个训练步骤
    print("🏋️ 测试单个训练步骤...")
    sys.stdout.flush()
    
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # 获取一个批次
        batch = next(iter(train_loader))
        print("✓ 获取训练批次成功")
        
        # 移动数据到设备
        print("  移动数据到设备...")
        sequences = batch['sequences'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        conditions = {'peptide_type': batch['peptide_type'].to(device)}
        print("✓ 数据移动完成")
        
        # 获取序列嵌入
        print("  获取序列嵌入...")
        sys.stdout.flush()
        with torch.no_grad():
            embeddings = model.sequence_encoder(sequences, attention_mask)
            embeddings = embeddings.last_hidden_state[:, 1:-1, :]
        print(f"✓ 序列嵌入获取完成，形状: {embeddings.shape}")
        
        # 前向扩散
        print("  执行前向扩散...")
        batch_size, seq_len, hidden_dim = embeddings.shape
        timesteps = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(embeddings)
        noisy_embeddings = diffusion.q_sample(embeddings, timesteps, noise)
        print("✓ 前向扩散完成")
        
        # 预测噪声
        print("  预测噪声...")
        sys.stdout.flush()
        model.train()
        optimizer.zero_grad()
        
        predicted_noise = model.denoiser(
            noisy_embeddings=noisy_embeddings,
            timesteps=timesteps,
            attention_mask=attention_mask[:, 1:-1],
            structure_features=None,
            conditions=conditions
        )[0]
        print("✓ 噪声预测完成")
        
        # 计算损失和反向传播
        print("  计算损失...")
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        print(f"✓ 损失计算完成: {loss.item():.6f}")
        
        print("  反向传播...")
        loss.backward()
        optimizer.step()
        print("✓ 反向传播完成")
        
        print("=" * 50)
        print("🎉 所有测试步骤完成！训练流程正常。")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ 训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 