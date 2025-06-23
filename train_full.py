# train_full.py - 完整的大规模StructDiff训练脚本
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from tqdm import tqdm
import os
import sys
import gc
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import random

# 导入 ESMFold 补丁以修复兼容性问题
from fix_esmfold_patch import apply_esmfold_patch

from structdiff.models.structdiff import StructDiff
from structdiff.data.dataset import PeptideStructureDataset
from structdiff.data.collator import PeptideStructureCollator

def setup_logging(output_dir, rank=0):
    """设置日志记录"""
    if rank == 0:
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置日志格式
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # 同时输出到文件和控制台
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger(__name__)
    else:
        # 非主进程只输出到控制台
        logging.basicConfig(level=logging.WARNING)
        return logging.getLogger(__name__)

def setup_distributed():
    """设置分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        
        return rank, world_size, gpu
    else:
        return 0, 1, 0

def setup_seed(seed):
    """设置随机种子以确保可重现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_esmfold_patch():
    """设置 ESMFold 补丁"""
    apply_esmfold_patch()

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

def save_checkpoint(model, optimizer, scheduler, epoch, loss, config, output_dir, 
                   is_best=False, rank=0, is_distributed=False):
    """保存检查点"""
    if rank == 0:
        checkpoint_dir = Path(output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取模型状态字典
        if is_distributed:
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存最新检查点
        latest_path = checkpoint_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = checkpoint_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
        
        # 保存定期检查点
        if epoch % 10 == 0:
            epoch_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, epoch_path)
        
        return str(latest_path)
    return None

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, rank=0, is_distributed=False):
    """加载检查点"""
    if os.path.exists(checkpoint_path):
        if rank == 0:
            print(f"正在加载检查点: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载模型状态
        if is_distributed:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器状态
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        
        if rank == 0:
            print(f"✓ 检查点加载成功，从 epoch {start_epoch} 开始")
        
        return start_epoch, best_loss
    else:
        return 0, float('inf')

def create_optimizer(model, config):
    """创建优化器"""
    if config.training.optimizer.name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.optimizer.lr,
            weight_decay=config.training.optimizer.weight_decay,
            betas=config.training.optimizer.get('betas', (0.9, 0.999)),
            eps=config.training.optimizer.get('eps', 1e-8)
        )
    elif config.training.optimizer.name.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.optimizer.lr,
            weight_decay=config.training.optimizer.weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器: {config.training.optimizer.name}")
    
    return optimizer

def create_scheduler(optimizer, config, total_steps):
    """创建学习率调度器"""
    if not config.training.get('scheduler', None):
        return None
    
    scheduler_name = config.training.scheduler.name.lower()
    
    if scheduler_name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.scheduler.get('min_lr', 1e-6)
        )
    elif scheduler_name == 'cosine_warmup':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.training.scheduler.get('warmup_epochs', 10),
            T_mult=config.training.scheduler.get('restart_mult', 2),
            eta_min=config.training.scheduler.get('min_lr', 1e-6)
        )
    elif scheduler_name == 'linear_warmup':
        from transformers import get_linear_schedule_with_warmup
        warmup_steps = int(total_steps * config.training.scheduler.get('warmup_ratio', 0.1))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    else:
        return None
    
    return scheduler

def train_epoch(model, train_loader, optimizer, scheduler, device, config, 
                epoch, writer=None, rank=0, world_size=1, logger=None):
    """训练一个epoch"""
    model.train()
    
    # 统计信息
    total_loss = 0.0
    total_diffusion_loss = 0.0
    total_structure_loss = 0.0
    successful_batches = 0
    total_batches = len(train_loader)
    
    # 分布式采样器设置
    if hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)
    
    # 进度条（仅主进程显示）
    if rank == 0:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} 训练中")
    else:
        pbar = train_loader
    
    optimizer.zero_grad()
    
    for batch_idx, batch_raw in enumerate(pbar):
        try:
            # 数据移动到设备
            batch = move_to_device(batch_raw, device)
            
            # 检查必要字段
            if 'sequences' not in batch or 'attention_mask' not in batch:
                if logger:
                    logger.warning(f"Batch {batch_idx} 缺少必要字段，跳过")
                continue
            
            # 采样时间步
            batch_size = batch['sequences'].shape[0]
            timesteps = torch.randint(
                0, config.diffusion.num_timesteps,
                (batch_size,), device=device
            )
            
            # 准备结构数据
            structures = None
            if config.data.get('use_predicted_structures', False) and 'structures' in batch:
                structures = batch['structures']
            
            # 前向传播
            outputs = model(
                sequences=batch['sequences'],
                attention_mask=batch['attention_mask'],
                timesteps=timesteps,
                structures=structures,
                return_loss=True
            )
            
            # 获得损失
            loss = outputs['total_loss']
            diffusion_loss = outputs.get('diffusion_loss', torch.tensor(0.0))
            structure_loss = outputs.get('structure_loss', torch.tensor(0.0))
            
            # 梯度累积缩放
            loss = loss / config.training.gradient_accumulation_steps
            
            # 反向传播
            loss.backward()
            
            # 梯度更新
            if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if config.training.get('max_grad_norm', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config.training.max_grad_norm
                    )
                
                # 优化器步骤
                optimizer.step()
                optimizer.zero_grad()
                
                # 调度器步骤（如果是基于步骤的）
                if scheduler and config.training.scheduler.get('step_based', False):
                    scheduler.step()
            
            # 统计更新
            actual_loss = loss.item() * config.training.gradient_accumulation_steps
            total_loss += actual_loss
            total_diffusion_loss += diffusion_loss.item()
            total_structure_loss += structure_loss.item()
            successful_batches += 1
            
            # 更新进度条（仅主进程）
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{actual_loss:.4f}",
                    'diff': f"{diffusion_loss.item():.4f}",
                    'struct': f"{structure_loss.item():.4f}",
                    'lr': f"{current_lr:.2e}",
                    'mem': f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "CPU"
                })
            
            # 记录到TensorBoard
            if writer and rank == 0:
                global_step = epoch * total_batches + batch_idx
                writer.add_scalar('Train/Loss_Total', actual_loss, global_step)
                writer.add_scalar('Train/Loss_Diffusion', diffusion_loss.item(), global_step)
                writer.add_scalar('Train/Loss_Structure', structure_loss.item(), global_step)
                writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('Train/GPU_Memory_GB', torch.cuda.memory_allocated() / 1e9, global_step)
            
            # 定期清理内存
            if batch_idx % 100 == 0:
                clear_memory()
                
        except Exception as e:
            if logger:
                logger.error(f"训练批次 {batch_idx} 出错: {e}")
            else:
                print(f"训练批次 {batch_idx} 出错: {e}")
            
            # 清理并继续
            optimizer.zero_grad()
            clear_memory()
            continue
    
    # 最终梯度更新（如果需要）
    if total_batches % config.training.gradient_accumulation_steps != 0:
        if config.training.get('max_grad_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    
    # 调度器步骤（如果是基于epoch的）
    if scheduler and not config.training.scheduler.get('step_based', False):
        scheduler.step()
    
    # 计算平均损失
    if successful_batches > 0:
        avg_metrics = {
            'total_loss': total_loss / successful_batches,
            'diffusion_loss': total_diffusion_loss / successful_batches,
            'structure_loss': total_structure_loss / successful_batches,
            'successful_batches': successful_batches,
            'total_batches': total_batches
        }
    else:
        avg_metrics = {
            'total_loss': 0.0,
            'diffusion_loss': 0.0,
            'structure_loss': 0.0,
            'successful_batches': 0,
            'total_batches': total_batches
        }
    
    return avg_metrics

def validate_model(model, val_loader, device, config, epoch, writer=None, rank=0, logger=None):
    """验证模型"""
    model.eval()
    
    total_loss = 0.0
    total_diffusion_loss = 0.0
    total_structure_loss = 0.0
    successful_batches = 0
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(val_loader, desc="验证中")
        else:
            pbar = val_loader
        
        for batch_idx, batch_raw in enumerate(pbar):
            try:
                # 数据移动到设备
                batch = move_to_device(batch_raw, device)
                
                # 检查必要字段
                if 'sequences' not in batch or 'attention_mask' not in batch:
                    continue
                
                # 采样时间步
                batch_size = batch['sequences'].shape[0]
                timesteps = torch.randint(
                    0, config.diffusion.num_timesteps,
                    (batch_size,), device=device
                )
                
                # 准备结构数据
                structures = None
                if config.data.get('use_predicted_structures', False) and 'structures' in batch:
                    structures = batch['structures']
                
                # 前向传播
                outputs = model(
                    sequences=batch['sequences'],
                    attention_mask=batch['attention_mask'],
                    timesteps=timesteps,
                    structures=structures,
                    return_loss=True
                )
                
                # 获得损失
                loss = outputs['total_loss']
                diffusion_loss = outputs.get('diffusion_loss', torch.tensor(0.0))
                structure_loss = outputs.get('structure_loss', torch.tensor(0.0))
                
                # 统计
                total_loss += loss.item()
                total_diffusion_loss += diffusion_loss.item()
                total_structure_loss += structure_loss.item()
                successful_batches += 1
                
                # 更新进度条
                if rank == 0:
                    pbar.set_postfix({
                        'val_loss': f"{loss.item():.4f}",
                        'mem': f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "CPU"
                    })
                
            except Exception as e:
                if logger:
                    logger.warning(f"验证批次 {batch_idx} 出错: {e}")
                clear_memory()
                continue
    
    # 计算平均损失
    if successful_batches > 0:
        avg_metrics = {
            'total_loss': total_loss / successful_batches,
            'diffusion_loss': total_diffusion_loss / successful_batches,
            'structure_loss': total_structure_loss / successful_batches
        }
    else:
        avg_metrics = {
            'total_loss': float('inf'),
            'diffusion_loss': float('inf'),
            'structure_loss': float('inf')
        }
    
    # 记录到TensorBoard
    if writer and rank == 0:
        writer.add_scalar('Val/Loss_Total', avg_metrics['total_loss'], epoch)
        writer.add_scalar('Val/Loss_Diffusion', avg_metrics['diffusion_loss'], epoch)
        writer.add_scalar('Val/Loss_Structure', avg_metrics['structure_loss'], epoch)
    
    return avg_metrics

def create_shared_esmfold(config, device, logger=None):
    """创建共享的ESMFold实例"""
    use_esmfold = config.model.structure_encoder.get('use_esmfold', False)
    use_structures = config.data.get('use_predicted_structures', False)
    
    shared_esmfold = None
    if use_esmfold and use_structures:
        if logger:
            logger.info("正在创建共享的 ESMFold 实例...")
        try:
            from structdiff.models.esmfold_wrapper import ESMFoldWrapper
            shared_esmfold = ESMFoldWrapper(device=device)
            if shared_esmfold.available:
                if logger:
                    logger.info("✓ 共享 ESMFold 实例创建成功")
                    if torch.cuda.is_available():
                        logger.info(f"ESMFold 加载后 GPU 内存使用: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
            else:
                if logger:
                    logger.warning("❌ 共享 ESMFold 实例创建失败")
                shared_esmfold = None
        except Exception as e:
            if logger:
                logger.error(f"创建共享 ESMFold 实例失败: {e}")
            shared_esmfold = None
    
    return shared_esmfold

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='StructDiff 大规模训练脚本')
    parser.add_argument('--config', type=str, default='configs/test_train.yaml',
                       help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='输出目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--local_rank', type=int, default=0,
                       help='本地进程排名')
    args = parser.parse_args()
    
    # 设置分布式训练
    rank, world_size, gpu = setup_distributed()
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(output_dir, rank)
    
    if rank == 0:
        logger.info(f"开始 StructDiff 大规模训练")
        logger.info(f"配置文件: {args.config}")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"使用设备: {device}")
        logger.info(f"分布式设置: rank={rank}, world_size={world_size}")
    
    # 应用ESMFold补丁
    setup_esmfold_patch()
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 设置随机种子
    if config.get('seed', None):
        setup_seed(config.seed)
    
    # 创建共享ESMFold实例
    shared_esmfold = create_shared_esmfold(config, device, logger)
    
    # 创建模型
    if rank == 0:
        logger.info("正在初始化模型...")
    
    try:
        # 临时禁用模型内部ESMFold加载以避免重复
        original_use_esmfold = config.model.structure_encoder.get('use_esmfold', False)
        if shared_esmfold and shared_esmfold.available:
            config.model.structure_encoder.use_esmfold = False
        
        model = StructDiff(config).to(device)
        
        # 恢复配置并设置共享实例
        config.model.structure_encoder.use_esmfold = original_use_esmfold
        if shared_esmfold and shared_esmfold.available:
            model.structure_encoder.esmfold = shared_esmfold
            model.structure_encoder.use_esmfold = True
        
        if rank == 0:
            logger.info(f"模型参数数量: {model.count_parameters():,}")
            if torch.cuda.is_available():
                logger.info(f"模型加载后 GPU 内存使用: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        return
    
    # 分布式包装
    is_distributed = world_size > 1
    if is_distributed:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
        if rank == 0:
            logger.info("✓ 模型已包装为分布式模型")
    
    # 创建数据集
    if rank == 0:
        logger.info("正在加载数据集...")
    
    try:
        train_dataset = PeptideStructureDataset(
            config.data.train_path,
            config,
            is_training=True,
            shared_esmfold=shared_esmfold
        )
        
        val_dataset = None
        if os.path.exists(config.data.get('val_path', '')):
            val_dataset = PeptideStructureDataset(
                config.data.val_path,
                config,
                is_training=False,
                shared_esmfold=shared_esmfold
            )
        
        if rank == 0:
            logger.info(f"训练数据集大小: {len(train_dataset):,}")
            if val_dataset:
                logger.info(f"验证数据集大小: {len(val_dataset):,}")
        
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        return
    
    # 创建数据加载器
    collator = PeptideStructureCollator(config)
    
    # 训练数据加载器
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=config.data.get('num_workers', 4),
        pin_memory=config.data.get('pin_memory', True),
        drop_last=True
    )
    
    # 验证数据加载器
    val_loader = None
    if val_dataset:
        if is_distributed:
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            val_sampler = None
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.get('val_batch_size', config.training.batch_size),
            shuffle=False,
            sampler=val_sampler,
            collate_fn=collator,
            num_workers=config.data.get('num_workers', 4),
            pin_memory=config.data.get('pin_memory', True)
        )
    
    # 创建优化器
    optimizer = create_optimizer(model, config)
    
    # 创建调度器
    total_steps = len(train_loader) * config.training.num_epochs // config.training.gradient_accumulation_steps
    scheduler = create_scheduler(optimizer, config, total_steps)
    
    # 设置TensorBoard（仅主进程）
    writer = None
    if rank == 0:
        log_dir = output_dir / "tensorboard"
        writer = SummaryWriter(log_dir)
        logger.info(f"TensorBoard 日志目录: {log_dir}")
    
    # 加载检查点（如果存在）
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, args.resume, rank, is_distributed
        )
    
    # 保存配置文件
    if rank == 0:
        config_save_path = output_dir / "config.yaml"
        OmegaConf.save(config, config_save_path)
        logger.info(f"配置文件已保存到: {config_save_path}")
    
    # 训练循环
    if rank == 0:
        logger.info(f"开始训练 {config.training.num_epochs} 个 epochs...")
        logger.info(f"批量大小: {config.training.batch_size}")
        logger.info(f"梯度累积步数: {config.training.gradient_accumulation_steps}")
        logger.info(f"有效批量大小: {config.training.batch_size * config.training.gradient_accumulation_steps * world_size}")
        logger.info(f"总训练步数: {total_steps}")
    
    for epoch in range(start_epoch, config.training.num_epochs):
        if rank == 0:
            logger.info(f"\n🚀 Epoch {epoch + 1}/{config.training.num_epochs}")
        
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, config,
            epoch, writer, rank, world_size, logger
        )
        
        if rank == 0:
            logger.info(
                f"训练损失 - 总计: {train_metrics['total_loss']:.4f}, "
                f"扩散: {train_metrics['diffusion_loss']:.4f}, "
                f"结构: {train_metrics['structure_loss']:.4f}, "
                f"成功批次: {train_metrics['successful_batches']}/{train_metrics['total_batches']}"
            )
        
        # 验证
        val_metrics = None
        if val_loader and (epoch + 1) % config.training.get('validate_every', 1) == 0:
            val_metrics = validate_model(model, val_loader, device, config, epoch, writer, rank, logger)
            
            if rank == 0:
                logger.info(
                    f"验证损失 - 总计: {val_metrics['total_loss']:.4f}, "
                    f"扩散: {val_metrics['diffusion_loss']:.4f}, "
                    f"结构: {val_metrics['structure_loss']:.4f}"
                )
        
        # 保存检查点
        is_best = val_metrics and val_metrics['total_loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['total_loss']
        
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            val_metrics['total_loss'] if val_metrics else train_metrics['total_loss'],
            config, output_dir, is_best, rank, is_distributed
        )
        
        # 定期清理内存
        clear_memory()
    
    # 关闭资源
    if writer:
        writer.close()
    
    if is_distributed:
        dist.destroy_process_group()
    
    if rank == 0:
        logger.info("🎉 训练完成！")

if __name__ == "__main__":
    main() 