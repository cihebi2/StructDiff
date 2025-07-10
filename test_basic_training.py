#!/usr/bin/env python3
"""
基础StructDiff训练测试脚本
测试最简化的序列扩散模型训练，完全禁用结构特征
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from structdiff.models.structdiff import StructDiff
from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
from structdiff.data.dataset import PeptideStructureDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

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
        'peptide_type': torch.stack(labels)  # 为了与模型接口一致，命名为peptide_type
    }

def test_basic_model():
    """测试基础模型初始化和前向传播"""
    print("🧪 测试基础模型初始化...")
    
    # 设备 - 先用CPU测试
    device = torch.device('cpu')  # 暂时使用CPU
    print(f"使用设备: {device}")
    
    # 最简化的配置
    config = OmegaConf.create({
        "model": {
            "type": "StructDiff",
            "sequence_encoder": {
                "pretrained_model": "facebook/esm2_t6_8M_UR50D",
                "freeze_encoder": True,  # 固定编码器
                "use_lora": False
            },
            "structure_encoder": {
                "type": "multi_scale",
                "hidden_dim": 256,
                "use_esmfold": False,  # 完全禁用
                "local": {
                    "hidden_dim": 256,
                    "num_layers": 2,
                    "kernel_sizes": [3, 5],
                    "dropout": 0.1
                },
                "global": {
                    "hidden_dim": 256,
                    "num_attention_heads": 4,
                    "num_layers": 2,
                    "dropout": 0.1
                },
                "fusion": {
                    "method": "attention",
                    "hidden_dim": 256
                }
            },
            "denoiser": {
                "hidden_dim": 320,  # 匹配ESM2_t6_8M
                "num_layers": 4,    # 减少层数
                "num_heads": 4,     # 减少注意力头
                "dropout": 0.1,
                "use_cross_attention": False  # 禁用结构交叉注意力
            }
        },
        "diffusion": {
            "num_timesteps": 100,  # 减少时间步
            "noise_schedule": "linear",
            "beta_start": 0.0001,
            "beta_end": 0.02
        },
        "data": {
            "max_length": 50,
            "use_predicted_structures": False  # 完全禁用结构特征
        }
    })
    
    try:
        # 创建模型
        model = StructDiff(config.model).to(device)
        print(f"✅ 模型初始化成功，参数数量: {model.count_parameters():,}")
        
        # 创建扩散过程
        diffusion = GaussianDiffusion(
            num_timesteps=config.diffusion.num_timesteps,
            noise_schedule=config.diffusion.noise_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end
        )
        print("✅ 扩散过程初始化成功")
        
        # 测试前向传播
        batch_size = 2
        seq_length = 20
        
        # 创建测试数据
        test_sequences = torch.randint(0, 32, (batch_size, seq_length + 2), device=device)  # +2 for CLS/SEP
        attention_mask = torch.ones(batch_size, seq_length + 2, device=device)
        timesteps = torch.randint(0, config.diffusion.num_timesteps, (batch_size,), device=device)
        conditions = {'peptide_type': torch.randint(0, 3, (batch_size,), device=device)}
        
        print("🔄 测试前向传播...")
        with torch.no_grad():
            outputs = model(
                sequences=test_sequences,
                attention_mask=attention_mask,
                timesteps=timesteps,
                structures=None,  # 无结构特征
                conditions=conditions,
                return_loss=False
            )
        
        print(f"✅ 前向传播成功")
        print(f"   输出嵌入形状: {outputs['denoised_embeddings'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n📊 测试数据加载...")
    
    try:
        # 完整配置结构 - 修复配置路径问题
        config = OmegaConf.create({
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
        
        # 创建数据集（禁用结构预测）
        dataset = PeptideStructureDataset(
            data_path="data/processed/train.csv",
            config=config,
            is_training=True
        )
        
        print(f"✅ 数据集加载成功，样本数: {len(dataset)}")
        
        # 测试数据加载器 - 使用自定义collate函数
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # 禁用多进程
            pin_memory=False,
            collate_fn=custom_collate_fn  # 添加自定义collate函数
        )
        
        # 测试获取一个批次
        for batch in dataloader:
            print("✅ 数据批次获取成功")
            print(f"   序列形状: {batch['sequences'].shape}")
            print(f"   注意力掩码形状: {batch['attention_mask'].shape}")
            print(f"   标签: {batch['peptide_type']}")
            print(f"   标签范围: {batch['peptide_type'].min()}-{batch['peptide_type'].max()}")
            break
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_training_step():
    """测试单个训练步骤"""
    print("\n🏋️ 测试单个训练步骤...")
    
    device = torch.device('cpu')  # 先用CPU测试
    
    try:
        # 使用更简单的配置
        config = OmegaConf.create({
            "model": {
                "sequence_encoder": {
                    "pretrained_model": "facebook/esm2_t6_8M_UR50D",
                    "freeze_encoder": True,
                    "use_lora": False
                },
                "structure_encoder": {
                    "use_esmfold": False,
                    "hidden_dim": 256
                },
                "denoiser": {
                    "hidden_dim": 320,
                    "num_layers": 2,
                    "num_heads": 4,
                    "dropout": 0.1,
                    "use_cross_attention": False
                }
            },
            "diffusion": {
                "num_timesteps": 100,
                "noise_schedule": "linear",
                "beta_start": 0.0001,
                "beta_end": 0.02
            }
        })
        
        # 创建模型和优化器
        model = StructDiff(config.model).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        
        # 创建扩散过程
        diffusion = GaussianDiffusion(
            num_timesteps=config.diffusion.num_timesteps,
            noise_schedule=config.diffusion.noise_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end
        )
        
        # 模拟批次数据
        batch_size = 2
        seq_length = 20
        
        # 创建序列嵌入（模拟ESM输出）
        seq_embeddings = torch.randn(batch_size, seq_length, 320, device=device)
        attention_mask = torch.ones(batch_size, seq_length, device=device)
        conditions = {'peptide_type': torch.tensor([0, 1], device=device)}
        
        # 扩散前向过程
        timesteps = torch.randint(0, config.diffusion.num_timesteps, (batch_size,), device=device)
        noise = torch.randn_like(seq_embeddings)
        noisy_embeddings = diffusion.q_sample(seq_embeddings, timesteps, noise)
        
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        
        # 预测噪声
        predicted_noise = model.denoiser(
            noisy_embeddings=noisy_embeddings,
            timesteps=timesteps,
            attention_mask=attention_mask,
            structure_features=None,
            conditions=conditions
        )[0]  # 只取预测结果，忽略注意力权重
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        print(f"✅ 训练步骤成功，损失: {loss.item():.6f}")
        return True
        
    except Exception as e:
        print(f"❌ 训练步骤测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 开始StructDiff基础训练测试")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    tests = [
        ("模型初始化", test_basic_model),
        ("数据加载", test_data_loading),
        ("训练步骤", test_single_training_step),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 执行测试: {test_name}")
        success = test_func()
        results.append((test_name, success))
    
    # 总结
    print("\n" + "=" * 50)
    print("🎯 测试结果总结:")
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 所有测试通过！可以开始正式训练。")
    else:
        print("\n⚠️ 存在失败的测试，需要进一步修复。")
    
    return all_passed

if __name__ == "__main__":
    main() 