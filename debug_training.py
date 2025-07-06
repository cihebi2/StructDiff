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

def debug_data_loading():
    """Debug data loading and check label values"""
    print("🔍 调试数据加载...")
    
    # Load data directly
    train_path = "/home/qlyu/sequence/StructDiff-7.0.0/data/processed/train.csv"
    df = pd.read_csv(train_path)
    
    print(f"数据集形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    if 'label' in df.columns:
        print(f"Label值范围: {df['label'].min()} - {df['label'].max()}")
        print(f"Label值分布:\n{df['label'].value_counts().sort_index()}")
        print(f"Label中的NaN: {df['label'].isna().sum()}")
        print(f"Label数据类型: {df['label'].dtype}")
        
        # Check for any unusual values
        unique_labels = df['label'].unique()
        print(f"唯一的label值: {unique_labels}")
        
        # Check if any labels are outside expected range [0, 1, 2]
        invalid_labels = df[(df['label'] < 0) | (df['label'] > 2)]
        if len(invalid_labels) > 0:
            print(f"⚠️ 发现无效的label值:")
            print(invalid_labels[['sequence', 'label']].head())
    else:
        print("❌ 没有找到label列")
        print("前5行数据:")
        print(df.head())

def debug_model_creation():
    """Debug model creation"""
    print("\n🔍 调试模型创建...")
    
    try:
        # Load config
        config_path = "/home/qlyu/sequence/StructDiff-7.0.0/configs/separated_training.yaml"
        config = load_config(config_path)
        
        # Import model classes
        from structdiff.models.structdiff import StructDiff
        
        print("正在创建模型...")
        model = StructDiff(config.model)
        print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
        
        return model, config
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def debug_forward_pass(model, config):
    """Debug forward pass with sample data"""
    print("\n🔍 调试前向传播...")
    
    try:
        # Create dataset
        dataset = PeptideStructureDataset(
            data_path="/home/qlyu/sequence/StructDiff-7.0.0/data/processed/train.csv",
            config=config,
            is_training=True
        )
        
        # Create dataloader with small batch
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Get one batch
        batch = next(iter(dataloader))
        print(f"批次数据键: {batch.keys()}")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
                if key == 'label':
                    print(f"  label值: {value}")
                    print(f"  label范围: {value.min()} - {value.max()}")
            else:
                print(f"{key}: {type(value)}")
        
        # Move to GPU
        device = torch.device('cuda:1')
        model = model.to(device)
        
        # Move batch to GPU
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        print("数据已移动到GPU")
        
        # Try forward pass
        print("尝试前向传播...")
        
        # Create dummy conditions
        conditions = {'peptide_type': batch['label']}
        
        with torch.no_grad():
            # Try model forward (without denoiser for now)
            seq_embeddings = model.sequence_encoder(
                batch['sequences'], 
                attention_mask=batch['attention_mask']
            ).last_hidden_state
            
            print(f"✅ 序列编码成功: {seq_embeddings.shape}")
            
            # Try denoiser forward
            timesteps = torch.randint(0, 1000, (batch['sequences'].shape[0],), device=device)
            
            print("尝试去噪器前向传播...")
            denoised, _ = model.denoiser(
                seq_embeddings,
                timesteps,
                batch['attention_mask'],
                conditions=conditions
            )
            
            print(f"✅ 去噪器前向传播成功: {denoised.shape}")
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🚀 开始调试训练问题...")
    
    # Debug data loading
    debug_data_loading()
    
    # Debug model creation
    model, config = debug_model_creation()
    
    if model is not None and config is not None:
        # Debug forward pass
        debug_forward_pass(model, config)
    
    print("\n✅ 调试完成")

if __name__ == "__main__":
    main() 