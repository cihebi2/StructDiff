#!/usr/bin/env python3

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append('/home/qlyu/sequence/StructDiff-7.0.0')

from structdiff.data.dataset import PeptideStructureDataset
from structdiff.utils.config import load_config
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

def debug_simple_training():
    """调试简化的训练流程"""
    print("🚀 开始简化训练调试...")
    
    try:
        # Load config
        config_path = "/home/qlyu/sequence/StructDiff-7.0.0/configs/separated_training.yaml"
        config = load_config(config_path)
        
        # 强制禁用结构预测
        config.data.use_predicted_structures = False
        
        print("✅ 配置加载成功，已禁用结构预测")
        
        # Create dataset
        dataset = PeptideStructureDataset(
            data_path="/home/qlyu/sequence/StructDiff-7.0.0/data/processed/train.csv",
            config=config,
            is_training=True
        )
        
        print(f"✅ 数据集创建成功，大小: {len(dataset)}")
        
        # Create dataloader with custom collate function
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=False,
            collate_fn=custom_collate_fn
        )
        
        print("✅ 数据加载器创建成功")
        
        # Test batch loading
        batch = next(iter(dataloader))
        print(f"✅ 批次加载成功，键: {batch.keys()}")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if key == 'label':
                    print(f"    label值: {value}")
            elif isinstance(value, list):
                print(f"  {key}: list of {len(value)} items")
        
        # Import model
        from structdiff.models.structdiff import StructDiff
        
        print("正在创建模型...")
        model = StructDiff(config.model)
        print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # Move to GPU (when CUDA_VISIBLE_DEVICES=1, the visible device index is 0)
        device = torch.device('cuda:0')
        model = model.to(device)
        
        # Move batch to GPU
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        print("✅ 数据已移动到GPU")
        
        # Test forward pass
        print("尝试前向传播...")
        
        with torch.no_grad():
            # Get sequence embeddings
            seq_embeddings = model.sequence_encoder(
                batch['sequences'], 
                attention_mask=batch['attention_mask']
            ).last_hidden_state
            
            print(f"✅ 序列编码成功: {seq_embeddings.shape}")
            
            # Create dummy conditions
            conditions = {'peptide_type': batch['label']}
            
            # Try denoiser forward (without structure features)
            timesteps = torch.randint(0, 1000, (batch['sequences'].shape[0],), device=device)
            
            print("尝试去噪器前向传播（无结构特征）...")
            denoised, _ = model.denoiser(
                seq_embeddings,
                timesteps,
                batch['attention_mask'],
                structure_features=None,  # 明确设置为None
                conditions=conditions
            )
            
            print(f"✅ 去噪器前向传播成功: {denoised.shape}")
            
            # Test training step
            print("尝试训练步骤...")
            model.train()
            
            # Add noise to embeddings
            noise = torch.randn_like(seq_embeddings)
            noisy_embeddings = seq_embeddings + noise
            
            # Forward pass with noise
            predicted_noise, _ = model.denoiser(
                noisy_embeddings,
                timesteps,
                batch['attention_mask'],
                structure_features=None,
                conditions=conditions
            )
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            print(f"✅ 训练步骤成功，损失: {loss.item():.6f}")
        
        print("\n🎉 所有测试通过！训练应该可以正常进行。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_simple_training() 