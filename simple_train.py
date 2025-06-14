# simple_train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import os
import sys
import gc

# 导入 ESMFold 补丁以修复兼容性问题
from fix_esmfold_patch import apply_esmfold_patch

from structdiff.models.structdiff import StructDiff
from structdiff.data.dataset import PeptideStructureDataset
from structdiff.data.collator import PeptideStructureCollator

def setup_esmfold_patch():
    """设置 ESMFold 补丁"""
    print("正在应用 ESMFold 兼容性补丁...")
    apply_esmfold_patch()
    print("✓ ESMFold 补丁应用成功")

def clear_memory():
    """清理 GPU 内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def move_to_device(obj, device):
    """递归地将对象移动到指定设备"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    else:
        return obj

def train_epoch(model, train_loader, optimizer, device, config):
    model.train()
    total_loss = 0
    total_diffusion_loss = 0
    total_structure_loss = 0
    successful_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch_raw in enumerate(pbar):
        try:
            # Move to device - 确保所有张量都正确移动到设备
            batch = move_to_device(batch_raw, device)
            
            # 检查batch的基本字段
            if 'sequences' not in batch or 'attention_mask' not in batch:
                print(f"Batch {batch_idx} missing required fields")
                continue
            
            # 打印调试信息
            print(f"\nBatch {batch_idx} debug info:")
            print(f"  sequences shape: {batch['sequences'].shape}")
            print(f"  attention_mask shape: {batch['attention_mask'].shape}")
            if 'structures' in batch:
                print(f"  structures keys: {list(batch['structures'].keys())}")
                for k, v in batch['structures'].items():
                    if hasattr(v, 'shape'):
                        print(f"    {k} shape: {v.shape}")
            
            # Sample timesteps
            batch_size = batch['sequences'].shape[0]
            timesteps = torch.randint(
                0, config.diffusion.num_timesteps, 
                (batch_size,), device=device
            )
            
            # Prepare structures if using ESMFold
            structures = None
            if config.data.get('use_predicted_structures', False) and 'structures' in batch:
                structures = batch['structures']
            
            # Forward pass
            print(f"  Calling model forward...")
            outputs = model(
                sequences=batch['sequences'],
                attention_mask=batch['attention_mask'],
                timesteps=timesteps,
                structures=structures,
                return_loss=True
            )
            
            loss = outputs['total_loss']
            diffusion_loss = outputs.get('diffusion_loss', torch.tensor(0.0))
            structure_loss = outputs.get('structure_loss', torch.tensor(0.0))
            
            # Scale loss for gradient accumulation
            loss = loss / config.training.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Clear memory periodically
                if (batch_idx + 1) % (config.training.gradient_accumulation_steps * 4) == 0:
                    clear_memory()
            
            # Update statistics
            total_loss += loss.item() * config.training.gradient_accumulation_steps
            total_diffusion_loss += diffusion_loss.item()
            total_structure_loss += structure_loss.item()
            successful_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'total': f"{loss.item() * config.training.gradient_accumulation_steps:.4f}",
                'diff': f"{diffusion_loss.item():.4f}",
                'struct': f"{structure_loss.item():.4f}",
                'mem': f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "CPU"
            })
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # 清理内存后继续
            optimizer.zero_grad()
            clear_memory()
            continue
    
    # Final gradient update if needed
    if len(train_loader) % config.training.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    if successful_batches > 0:
        avg_total_loss = total_loss / successful_batches
        avg_diffusion_loss = total_diffusion_loss / successful_batches
        avg_structure_loss = total_structure_loss / successful_batches
    else:
        avg_total_loss = avg_diffusion_loss = avg_structure_loss = 0.0
    
    return {
        'total_loss': avg_total_loss,
        'diffusion_loss': avg_diffusion_loss,
        'structure_loss': avg_structure_loss,
        'successful_batches': successful_batches
    }

def validate_model(model, val_loader, device, config):
    """验证模型性能"""
    model.eval()
    total_loss = 0
    total_diffusion_loss = 0
    total_structure_loss = 0
    successful_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch_raw in pbar:
            try:
                # Move to device - 确保所有张量都正确移动到设备
                batch = move_to_device(batch_raw, device)
                
                # 检查batch的基本字段
                if 'sequences' not in batch or 'attention_mask' not in batch:
                    continue
                
                # Sample timesteps
                batch_size = batch['sequences'].shape[0]
                timesteps = torch.randint(
                    0, config.diffusion.num_timesteps, 
                    (batch_size,), device=device
                )
                
                # Prepare structures
                structures = None
                if config.data.get('use_predicted_structures', False) and 'structures' in batch:
                    structures = batch['structures']
                
                # Forward pass
                outputs = model(
                    sequences=batch['sequences'],
                    attention_mask=batch['attention_mask'],
                    timesteps=timesteps,
                    structures=structures,
                    return_loss=True
                )
                
                loss = outputs['total_loss']
                diffusion_loss = outputs.get('diffusion_loss', torch.tensor(0.0))
                structure_loss = outputs.get('structure_loss', torch.tensor(0.0))
                
                total_loss += loss.item()
                total_diffusion_loss += diffusion_loss.item()
                total_structure_loss += structure_loss.item()
                successful_batches += 1
                
                pbar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'mem': f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "CPU"
                })
                
            except Exception as e:
                print(f"Validation error: {e}")
                clear_memory()
                continue
    
    if successful_batches > 0:
        return {
            'total_loss': total_loss / successful_batches,
            'diffusion_loss': total_diffusion_loss / successful_batches,
            'structure_loss': total_structure_loss / successful_batches
        }
    else:
        return {'total_loss': 0.0, 'diffusion_loss': 0.0, 'structure_loss': 0.0}

def main():
    # Setup ESMFold patch first
    setup_esmfold_patch()
    
    # Load minimal config
    config = OmegaConf.load("configs/minimal_test.yaml")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"Available Memory: {torch.cuda.memory_reserved(0) / 1e9:.1f}GB")
    
    # Clear initial memory
    clear_memory()
    
    # Check if ESMFold is enabled
    use_esmfold = config.model.structure_encoder.get('use_esmfold', False)
    use_structures = config.data.get('use_predicted_structures', False)
    
    print(f"ESMFold enabled: {use_esmfold}")
    print(f"Using predicted structures: {use_structures}")
    
    # 创建共享的 ESMFold 实例
    shared_esmfold = None
    if use_esmfold and use_structures:
        print("正在创建共享的 ESMFold 实例...")
        try:
            from structdiff.models.esmfold_wrapper import ESMFoldWrapper
            shared_esmfold = ESMFoldWrapper(device=device)
            if shared_esmfold.available:
                print("✓ 共享 ESMFold 实例创建成功")
                print(f"ESMFold 加载后 GPU 内存使用: {torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "")
            else:
                print("❌ 共享 ESMFold 实例创建失败")
                shared_esmfold = None
        except Exception as e:
            print(f"创建共享 ESMFold 实例失败: {e}")
            shared_esmfold = None
    
    if use_esmfold and use_structures and shared_esmfold and shared_esmfold.available:
        print("✓ 将使用共享的 ESMFold 进行结构预测")
    
    # Create model - 临时禁用ESMFold以避免重复加载
    print("正在初始化模型...")
    try:
        # 备份原始配置
        original_use_esmfold = config.model.structure_encoder.get('use_esmfold', False)
        
        # 如果已有共享实例，临时禁用模型内部的ESMFold加载
        if shared_esmfold and shared_esmfold.available:
            print("临时禁用模型内部ESMFold加载以避免内存不足...")
            config.model.structure_encoder.use_esmfold = False
        
        model = StructDiff(config).to(device)
        
        # 恢复配置并设置共享实例
        config.model.structure_encoder.use_esmfold = original_use_esmfold
        
        # 如果有共享的ESMFold实例，手动设置到模型中
        if shared_esmfold and shared_esmfold.available:
            print("正在将共享 ESMFold 实例设置到模型中...")
            if hasattr(model.structure_encoder, 'esmfold') or hasattr(model.structure_encoder, '_esmfold'):
                # 设置ESMFold实例
                model.structure_encoder.esmfold = shared_esmfold
                model.structure_encoder._esmfold = shared_esmfold
                # 确保ESMFold被标记为可用
                model.structure_encoder.use_esmfold = True
                print("✓ 共享 ESMFold 实例已设置到模型中")
            else:
                # 如果模型结构不同，尝试直接设置属性
                setattr(model.structure_encoder, 'esmfold', shared_esmfold)
                setattr(model.structure_encoder, 'use_esmfold', True)
                print("✓ 共享 ESMFold 实例已强制设置到模型中")
        
        print(f"模型参数数量: {model.count_parameters():,}")
        
        # Print memory usage after model creation
        if torch.cuda.is_available():
            print(f"模型加载后 GPU 内存使用: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        
    except Exception as e:
        print(f"模型初始化失败: {e}")
        # 如果失败，尝试完全禁用ESMFold
        print("尝试禁用ESMFold重新初始化模型...")
        try:
            config.model.structure_encoder.use_esmfold = False
            config.data.use_predicted_structures = False
            model = StructDiff(config).to(device)
            print("✓ 模型初始化成功（未使用ESMFold）")
            shared_esmfold = None  # 清除共享实例
        except Exception as e2:
            print(f"禁用ESMFold后仍然失败: {e2}")
            return
    
    # Create optimizer with reduced parameters
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay
    )
    
    # Create datasets with limited data and shared ESMFold
    print("正在加载数据集...")
    try:
        train_dataset = PeptideStructureDataset(
            config.data.train_path, 
            config,  # 使用原始配置
            is_training=True,
            shared_esmfold=shared_esmfold  # 直接传递共享实例
        )
        
        # Limit dataset size for testing
        max_samples = min(20, len(train_dataset))  # 减少到20个样本
        train_dataset.data = train_dataset.data.head(max_samples)
        
        # Create validation dataset if available
        val_dataset = None
        if os.path.exists(config.data.get('val_path', '')):
            val_dataset = PeptideStructureDataset(
                config.data.val_path,
                config,  # 使用原始配置
                is_training=False,
                shared_esmfold=shared_esmfold  # 直接传递共享实例
            )
            
            # Limit validation dataset
            max_val_samples = min(10, len(val_dataset))
            val_dataset.data = val_dataset.data.head(max_val_samples)
        
        print(f"训练数据集大小: {len(train_dataset)}")
        if val_dataset:
            print(f"验证数据集大小: {len(val_dataset)}")
            
        # 最终内存使用情况
        if torch.cuda.is_available():
            print(f"数据集加载后 GPU 内存使用: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
            
    except Exception as e:
        print(f"数据集加载失败: {e}")
        return
    
    # Create data loaders with smaller batch size
    collator = PeptideStructureCollator(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # 使用batch_size=1来避免堆叠问题
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # 在 Windows 上使用 0 避免问题
        pin_memory=False  # 禁用 pin_memory 节省内存
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=False
        )
    
    # Create learning rate scheduler
    scheduler = None
    if config.training.get('scheduler', None):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=config.training.num_epochs,
            eta_min=1e-6
        )
    
    # Training loop
    os.makedirs("checkpoints", exist_ok=True)
    best_val_loss = float('inf')
    
    print(f"\n开始训练 {config.training.num_epochs} 个 epochs...")
    print(f"批量大小: 1, 梯度累积步数: {config.training.gradient_accumulation_steps}")
    print(f"有效批量大小: {config.training.gradient_accumulation_steps}")
    
    for epoch in range(config.training.num_epochs):
        print(f"\n🚀 Epoch {epoch + 1}/{config.training.num_epochs}")
        
        try:
            # Train
            train_metrics = train_epoch(model, train_loader, optimizer, device, config)
            
            print(f"训练损失 - 总计: {train_metrics['total_loss']:.4f}, "
                  f"扩散: {train_metrics['diffusion_loss']:.4f}, "
                  f"结构: {train_metrics['structure_loss']:.4f}, "
                  f"成功批次: {train_metrics['successful_batches']}/{len(train_loader)}")
            
            # Validate
            if val_loader and (epoch + 1) % config.training.get('validate_every', 5) == 0:
                val_metrics = validate_model(model, val_loader, device, config)
                print(f"验证损失 - 总计: {val_metrics['total_loss']:.4f}, "
                      f"扩散: {val_metrics['diffusion_loss']:.4f}, "
                      f"结构: {val_metrics['structure_loss']:.4f}")
                
                # Save best model
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                        'val_loss': best_val_loss,
                        'train_metrics': train_metrics,
                        'val_metrics': val_metrics
                    }
                    torch.save(checkpoint, "checkpoints/best_model.pth")
                    print(f"💾 保存最佳模型 (验证损失: {best_val_loss:.4f})")
            
            # Update learning rate
            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print(f"学习率: {current_lr:.2e}")
            
            # Memory cleanup
            clear_memory()
            
        except Exception as e:
            print(f"Epoch {epoch + 1} 训练失败: {e}")
            clear_memory()
            continue
    
    print("\n🎉 训练完成！")
    
    # Try to generate a simple test sample
    print("\n正在测试生成功能...")
    model.eval()
    with torch.no_grad():
        try:
            samples = model.sample(
                batch_size=2,
                seq_length=10,
                sampling_method='ddpm',
                temperature=1.0,
                progress_bar=True
            )
            
            print("\n生成的测试序列:")
            for i, seq in enumerate(samples['sequences']):
                quality = samples['scores'][i].item()
                print(f"{i+1}: {seq} (质量: {quality:.3f})")
                
        except Exception as e:
            print(f"生成测试样本时出错: {e}")

if __name__ == "__main__":
    main()