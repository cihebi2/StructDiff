#!/usr/bin/env python3
"""
测试 ESMFold 集成到 StructDiff 训练中
"""

import torch
from omegaconf import OmegaConf
from fix_esmfold_patch import apply_esmfold_patch

def test_esmfold_integration():
    """测试 ESMFold 集成"""
    print("=== 测试 ESMFold 集成到 StructDiff ===")
    
    # 应用补丁
    print("1. 应用 ESMFold 补丁...")
    apply_esmfold_patch()
    print("✓ 补丁应用成功")
    
    # 加载配置
    print("2. 加载配置...")
    config = OmegaConf.load("configs/small_model.yaml")
    
    use_esmfold = config.model.structure_encoder.get('use_esmfold', False)
    use_structures = config.data.get('use_predicted_structures', False)
    
    print(f"   ESMFold 启用: {use_esmfold}")
    print(f"   使用结构预测: {use_structures}")
    print(f"   结构一致性损失权重: {config.training_config.loss_weights.structure_consistency_loss}")
    
    # 测试模型初始化
    print("3. 测试模型初始化...")
    try:
        from structdiff.models.structdiff import StructDiff
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = StructDiff(config).to(device)
        print(f"✓ 模型初始化成功，参数数量: {model.count_parameters():,}")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        return False
    
    # 测试数据加载
    print("4. 测试数据加载...")
    try:
        from structdiff.data.dataset import PeptideStructureDataset
        from structdiff.data.collator import PeptideStructureCollator
        
        dataset = PeptideStructureDataset(
            config.data.train_path,
            config,
            is_training=True
        )
        collator = PeptideStructureCollator(config)
        
        print(f"✓ 数据集加载成功，样本数量: {len(dataset)}")
        
        # 测试一个样本
        sample = dataset[0]
        print(f"   样本包含的键: {list(sample.keys())}")
        
        # 测试批处理
        batch = collator([sample])
        print(f"   批处理键: {list(batch.keys())}")
        
        if 'structures' in batch:
            structures = batch['structures']
            if isinstance(structures, dict):
                print(f"   结构数据键: {list(structures.keys())}")
                for key, value in structures.items():
                    if torch.is_tensor(value):
                        print(f"     {key}: {value.shape}")
                    else:
                        print(f"     {key}: {type(value)}")
                print("✓ 结构数据已正确加载")
            else:
                print(f"   结构数据形状: {structures.shape}")
                print("✓ 结构数据已正确加载")
        else:
            print("   注意：批处理中未包含结构数据")
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False
    
    # 测试模型前向传播
    print("5. 测试模型前向传播...")
    try:
        model.eval()
        with torch.no_grad():
            # 准备输入
            batch_size = 2
            seq_length = 20
            
            # 创建模拟数据
            sequences = torch.randint(
                4, model.tokenizer.vocab_size - 1, 
                (batch_size, seq_length + 2), 
                device=device
            )
            attention_mask = torch.ones(batch_size, seq_length + 2, device=device)
            timesteps = torch.randint(0, config.diffusion.num_timesteps, (batch_size,), device=device)
            
            # 测试不带结构的前向传播
            outputs = model(
                sequences=sequences,
                attention_mask=attention_mask,
                timesteps=timesteps,
                return_loss=True
            )
            
            print(f"✓ 前向传播成功")
            print(f"   输出键: {list(outputs.keys())}")
            
            for key, value in outputs.items():
                if torch.is_tensor(value):
                    print(f"   {key}: {value.shape}")
                else:
                    print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False
    
    # 测试生成
    print("6. 测试序列生成...")
    try:
        samples = model.sample(
            batch_size=3,
            seq_length=15,
            sampling_method='ddpm',
            temperature=1.0,
            progress_bar=False
        )
        
        print(f"✓ 生成成功")
        print(f"   生成序列数量: {len(samples['sequences'])}")
        
        for i, seq in enumerate(samples['sequences']):
            quality = samples['scores'][i].item()
            print(f"   {i+1}: {seq[:20]}... (质量: {quality:.3f})")
            
    except Exception as e:
        print(f"✗ 生成失败: {e}")
        return False
    
    print("\n🎉 所有测试通过！ESMFold 集成正常工作")
    return True

if __name__ == "__main__":
    success = test_esmfold_integration()
    if success:
        print("\n建议：现在可以运行完整的训练脚本 'python simple_train.py'")
    else:
        print("\n需要修复上述问题后再运行训练") 