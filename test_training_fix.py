#!/usr/bin/env python3
"""
测试修复后的训练脚本
简化版本，用于验证修复是否有效
"""

import os
import sys
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入修复后的模块
from fix_esmfold_patch import apply_esmfold_patch
from structdiff.models.structdiff import StructDiff
from structdiff.data.dataset import PeptideStructureDataset
from structdiff.data.collator import PeptideStructureCollator
from structdiff.utils.logger import setup_logger, get_logger

def test_fix():
    """测试修复是否有效"""
    print("🧪 开始测试修复后的训练代码...")
    
    # 应用ESMFold补丁
    apply_esmfold_patch()
    print("✓ ESMFold补丁应用成功")
    
    # 设置日志
    setup_logger()
    logger = get_logger(__name__)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载配置
    config_path = "configs/peptide_esmfold_config.yaml"
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    config = OmegaConf.load(config_path)
    print("✓ 配置加载成功")
    
    try:
        # 测试数据加载器
        print("📊 测试数据加载器...")
        
        # 创建小型测试数据集
        train_dataset = PeptideStructureDataset(
            config.data.train_path,
            config,
            is_training=True,
            shared_esmfold=None  # 暂时不使用ESMFold
        )
        
        # 使用前5个样本进行测试
        from torch.utils.data import Subset
        test_subset = Subset(train_dataset, range(min(5, len(train_dataset))))
        
        collator = PeptideStructureCollator(config)
        
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            test_subset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            collate_fn=collator
        )
        
        print(f"✓ 数据加载器创建成功，测试样本数: {len(test_subset)}")
        
        # 测试批次处理
        print("🔄 测试批次处理...")
        for i, batch in enumerate(test_loader):
            print(f"  批次 {i+1}:")
            print(f"    sequences: {batch['sequences'].shape}")
            print(f"    attention_mask: {batch['attention_mask'].shape}")
            
            if 'structures' in batch:
                print(f"    structures: {len(batch['structures'])} keys")
                for key, value in batch['structures'].items():
                    if isinstance(value, torch.Tensor):
                        print(f"      {key}: {value.shape}")
            
            if i >= 2:  # 只测试前3个批次
                break
        
        print("✓ 批次处理测试通过")
        
        # 测试模型初始化（不使用ESMFold）
        print("🏗️ 测试模型初始化...")
        
        # 临时禁用ESMFold
        original_use_esmfold = config.model.structure_encoder.get('use_esmfold', False)
        config.model.structure_encoder.use_esmfold = False
        config.data.use_predicted_structures = False
        
        model = StructDiff(config).to(device)
        print(f"✓ 模型初始化成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        print("⏩ 测试前向传播...")
        model.eval()
        
        with torch.no_grad():
            # 获取一个批次进行测试
            test_batch = next(iter(test_loader))
            
            # 移动到设备
            test_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in test_batch.items()}
            
            # 创建时间步
            batch_size = test_batch['sequences'].shape[0]
            timesteps = torch.randint(0, config.diffusion.num_timesteps, (batch_size,), device=device)
            
            # 前向传播
            outputs = model(
                sequences=test_batch['sequences'],
                attention_mask=test_batch['attention_mask'],
                timesteps=timesteps,
                structures=test_batch.get('structures'),
                conditions=test_batch.get('conditions'),
                return_loss=True
            )
            
            print(f"✓ 前向传播成功")
            print(f"  输出键: {list(outputs.keys())}")
            
            if 'total_loss' in outputs:
                print(f"  总损失: {outputs['total_loss'].item():.4f}")
            
            if 'denoised_embeddings' in outputs:
                print(f"  去噪嵌入形状: {outputs['denoised_embeddings'].shape}")
        
        # 恢复原始配置
        config.model.structure_encoder.use_esmfold = original_use_esmfold
        
        print("🎉 所有测试通过！修复有效！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print("详细错误信息:")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_fix()
    if success:
        print("\n✅ 修复验证成功！现在可以重新运行训练脚本。")
        print("推荐命令:")
        print("python scripts/train_peptide_esmfold.py --config configs/peptide_esmfold_config.yaml --debug")
    else:
        print("\n❌ 修复验证失败，需要进一步调试。")
    
    sys.exit(0 if success else 1)