# memory_optimized_train.py
"""
内存优化的训练脚本，专门处理ESMFold的内存管理
"""

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

def setup_memory_optimization():
    """设置内存优化"""
    if torch.cuda.is_available():
        # 启用内存片段管理
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        # 启用调试模式
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        torch.cuda.empty_cache()
    print("✓ 内存优化设置完成")

def aggressive_memory_cleanup():
    """激进的内存清理"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def check_memory_usage(stage=""):
    """检查内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[{stage}] GPU内存 - 已分配: {allocated:.1f}GB, 已保留: {reserved:.1f}GB, 总计: {max_memory:.1f}GB")
        return allocated, reserved, max_memory
    return 0, 0, 0

def create_lightweight_model(config, device, use_esmfold=False):
    """创建轻量级模型"""
    print("正在创建轻量级模型...")
    
    # 备份原始配置
    original_config = config.copy()
    
    # 禁用ESMFold
    config.model.structure_encoder.use_esmfold = use_esmfold
    config.data.use_predicted_structures = use_esmfold
    
    try:
        model = StructDiff(config)
        
        # 检查内存使用
        check_memory_usage("模型创建后")
        
        # 移动到设备
        model = model.to(device)
        check_memory_usage("移动到GPU后")
        
        print(f"✓ 模型创建成功，参数数量: {model.count_parameters():,}")
        return model, original_config
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        # 恢复原始配置
        config.update(original_config)
        return None, original_config

def load_esmfold_separately(device):
    """单独加载ESMFold"""
    print("正在单独加载 ESMFold...")
    
    try:
        from structdiff.models.esmfold_wrapper import ESMFoldWrapper
        
        # 清理内存
        aggressive_memory_cleanup()
        check_memory_usage("ESMFold加载前")
        
        esmfold = ESMFoldWrapper(device=device)
        
        if esmfold.available:
            check_memory_usage("ESMFold加载后")
            print("✓ ESMFold 加载成功")
            return esmfold
        else:
            print("❌ ESMFold 不可用")
            return None
            
    except Exception as e:
        print(f"❌ ESMFold 加载失败: {e}")
        return None

def smart_attach_esmfold(model, esmfold):
    """智能地将ESMFold附加到模型"""
    if not esmfold or not esmfold.available:
        return False
    
    try:
        print("正在将 ESMFold 附加到模型...")
        
        # 尝试多种方式设置ESMFold
        if hasattr(model, 'structure_encoder'):
            model.structure_encoder.esmfold = esmfold
            model.structure_encoder.use_esmfold = True
            
            # 确保模型知道ESMFold可用
            if hasattr(model.structure_encoder, '_esmfold'):
                model.structure_encoder._esmfold = esmfold
                
        print("✓ ESMFold 成功附加到模型")
        return True
        
    except Exception as e:
        print(f"❌ ESMFold 附加失败: {e}")
        return False

def create_minimal_dataset(config, shared_esmfold=None, max_samples=10):
    """创建最小数据集"""
    print(f"正在创建最小数据集（最多{max_samples}个样本）...")
    
    try:
        # 创建训练数据集
        train_dataset = PeptideStructureDataset(
            config.data.train_path, 
            config,
            is_training=True,
            shared_esmfold=shared_esmfold
        )
        
        # 限制数据集大小
        if len(train_dataset) > max_samples:
            train_dataset.data = train_dataset.data.head(max_samples)
        
        print(f"✓ 训练数据集大小: {len(train_dataset)}")
        
        # 创建验证数据集（如果存在）
        val_dataset = None
        if os.path.exists(config.data.get('val_path', '')):
            val_dataset = PeptideStructureDataset(
                config.data.val_path,
                config,
                is_training=False,
                shared_esmfold=shared_esmfold
            )
            
            max_val_samples = min(5, len(val_dataset))
            val_dataset.data = val_dataset.data.head(max_val_samples)
            print(f"✓ 验证数据集大小: {len(val_dataset)}")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return None, None

def safe_train_step(model, batch, optimizer, device, config):
    """安全的训练步骤"""
    try:
        # 移动数据到设备
        if isinstance(batch, dict):
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
        
        # 检查必要字段
        if 'sequences' not in batch or 'attention_mask' not in batch:
            return None, "缺少必要字段"
        
        # 生成时间步
        batch_size = batch['sequences'].shape[0]
        timesteps = torch.randint(
            0, config.diffusion.num_timesteps, 
            (batch_size,), device=device
        )
        
        # 前向传播
        outputs = model(
            sequences=batch['sequences'],
            attention_mask=batch['attention_mask'],
            timesteps=timesteps,
            structures=batch.get('structures', None),
            return_loss=True
        )
        
        loss = outputs['total_loss']
        
        # 反向传播
        loss.backward()
        
        return loss.item(), None
        
    except Exception as e:
        return None, str(e)

def main():
    """主函数"""
    print("=== 内存优化训练脚本 ===\n")
    
    # 1. 设置内存优化
    setup_memory_optimization()
    apply_esmfold_patch()
    
    # 2. 加载配置
    config = OmegaConf.load("configs/minimal_test.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    initial_allocated, initial_reserved, total_memory = check_memory_usage("初始状态")
    
    # 3. 策略选择：根据可用内存决定是否使用ESMFold
    use_esmfold = False
    esmfold_instance = None
    
    # 如果有足够内存（>16GB可用），尝试加载ESMFold
    available_memory = total_memory - initial_allocated
    if available_memory > 16.0:
        print(f"可用内存充足（{available_memory:.1f}GB），尝试加载ESMFold...")
        esmfold_instance = load_esmfold_separately(device)
        use_esmfold = esmfold_instance is not None
    else:
        print(f"可用内存不足（{available_memory:.1f}GB），跳过ESMFold")
    
    # 4. 创建轻量级模型
    model, original_config = create_lightweight_model(config, device, use_esmfold=False)
    
    if model is None:
        print("❌ 模型创建失败，退出")
        return
    
    # 5. 如果有ESMFold，尝试附加
    if use_esmfold and esmfold_instance:
        if not smart_attach_esmfold(model, esmfold_instance):
            print("⚠️ ESMFold附加失败，继续使用无ESMFold的模型")
            use_esmfold = False
    
    check_memory_usage("模型设置完成")
    
    # 6. 创建优化器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay
    )
    
    # 7. 创建数据集
    train_dataset, val_dataset = create_minimal_dataset(
        config, 
        shared_esmfold=esmfold_instance if use_esmfold else None,
        max_samples=15
    )
    
    if train_dataset is None:
        print("❌ 数据集创建失败，退出")
        return
    
    # 8. 创建数据加载器
    collator = PeptideStructureCollator(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False
    )
    
    check_memory_usage("数据加载器创建完成")
    
    # 9. 开始训练
    print(f"\n🚀 开始训练...")
    print(f"使用ESMFold: {use_esmfold}")
    print(f"训练样本: {len(train_dataset)}")
    
    model.train()
    successful_steps = 0
    total_loss = 0.0
    
    for epoch in range(min(3, config.training.num_epochs)):  # 限制epoch数
        print(f"\nEpoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # 定期清理内存
                if batch_idx % 5 == 0:
                    aggressive_memory_cleanup()
                
                # 执行训练步骤
                loss, error = safe_train_step(model, batch, optimizer, device, config)
                
                if loss is not None:
                    # 梯度裁剪和优化器步骤
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    total_loss += loss
                    successful_steps += 1
                    
                    print(f"  批次 {batch_idx}: 损失 {loss:.4f}")
                else:
                    print(f"  批次 {batch_idx}: 失败 - {error}")
                
                # 内存检查
                if batch_idx % 3 == 0:
                    check_memory_usage(f"Epoch{epoch+1}-Batch{batch_idx}")
                
            except Exception as e:
                print(f"  批次 {batch_idx}: 异常 - {e}")
                optimizer.zero_grad()
                aggressive_memory_cleanup()
                continue
        
        # Epoch结束清理
        aggressive_memory_cleanup()
    
    # 10. 训练结果
    if successful_steps > 0:
        avg_loss = total_loss / successful_steps
        print(f"\n🎉 训练完成！")
        print(f"成功步骤: {successful_steps}")
        print(f"平均损失: {avg_loss:.4f}")
    else:
        print(f"\n❌ 训练失败，没有成功的步骤")
    
    check_memory_usage("训练完成")

if __name__ == "__main__":
    main() 