#!/usr/bin/env python3
"""
Training script for StructDiff
"""

import os
import sys
import argparse
import logging
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import yaml
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入 ESMFold 补丁以修复兼容性问题
from fix_esmfold_patch import apply_esmfold_patch

from structdiff.models.structdiff import StructDiff
from structdiff.data.dataset import PeptideStructureDataset
from structdiff.data.collator import PeptideStructureCollator
from structdiff.utils.checkpoint import CheckpointManager
from structdiff.utils.ema import EMA
from structdiff.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Train StructDiff model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced data"
    )
    return parser.parse_args()


def compute_validation_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    config: Dict
) -> Dict[str, float]:
    """Compute validation metrics"""
    metrics = {}
    
    # Basic MSE loss
    mse_loss = F.mse_loss(predictions, targets).item()
    metrics['val_loss'] = mse_loss
    
    # Perplexity (approximation)
    metrics['perplexity'] = torch.exp(torch.tensor(mse_loss)).item()
    
    # You can add more sophisticated metrics here
    
    return metrics


def setup_shared_esmfold(config: Dict, device: torch.device):
    """创建共享的 ESMFold 实例"""
    shared_esmfold = None
    
    # Check if ESMFold is enabled
    use_esmfold = config.model.structure_encoder.get('use_esmfold', False)
    use_structures = config.data.get('use_predicted_structures', False)
    
    print(f"ESMFold enabled: {use_esmfold}")
    print(f"Using predicted structures: {use_structures}")
    
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
    
    return shared_esmfold


def setup_training(config: Dict, resume_path: Optional[str] = None):
    """Setup training components"""
    
    # Setup ESMFold patch first
    setup_esmfold_patch()
    
    # Create directories
    os.makedirs(config.experiment.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(config.experiment.output_dir, config.experiment.name, "checkpoints")
    log_dir = os.path.join(config.experiment.output_dir, config.experiment.name, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logger
    log_file = os.path.join(
        log_dir,
        f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    setup_logger(log_file)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        logger.info(f"Available Memory: {torch.cuda.memory_reserved(0) / 1e9:.1f}GB")
    
    # Clear initial memory
    clear_memory()
    
    # Setup shared ESMFold
    shared_esmfold = setup_shared_esmfold(config, device)
    
    # Create model
    logger.info("正在初始化模型...")
    try:
        model = StructDiff(config).to(device)
        
        # 如果有共享的ESMFold实例，替换模型中的实例
        if shared_esmfold and shared_esmfold.available:
            if hasattr(model.structure_encoder, 'esmfold'):
                logger.info("正在将模型的 ESMFold 实例替换为共享实例...")
                model.structure_encoder.esmfold = shared_esmfold
                clear_memory()  # 清理被替换的实例占用的内存
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Print memory usage after model creation
        if torch.cuda.is_available():
            logger.info(f"模型加载后 GPU 内存使用: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        raise
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay,
        betas=config.training.optimizer.betas
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs,
        eta_min=config.training.scheduler.min_lr
    )
    
    # Create EMA
    ema = None
    if config.training.use_ema:
        ema = EMA(model, decay=config.training.ema_decay, device=device)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    global_step = 0
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if ema and 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    return model, optimizer, scheduler, ema, device, start_epoch, global_step, checkpoint_dir, shared_esmfold


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ema: Optional[EMA],
    device: torch.device,
    config: Dict,
    epoch: int,
    global_step: int,
    writer: SummaryWriter
) -> int:
    """Train for one epoch"""
    model.train()
    
    # Setup mixed precision training
    scaler = GradScaler() if config.training.use_amp else None
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    # Accumulate losses
    loss_accumulator = {
        'total_loss': 0.0,
        'diffusion_loss': 0.0,
        'structure_loss': 0.0
    }
    accumulation_steps = 0
    successful_batches = 0
    
    for batch_idx, batch_raw in enumerate(pbar):
        try:
            # Move batch to device with improved handling
            batch = move_to_device(batch_raw, device)
            
            # 检查batch的基本字段
            if 'sequences' not in batch or 'attention_mask' not in batch:
                logger.warning(f"Batch {batch_idx} missing required fields")
                continue
            
            # Sample timesteps
            batch_size = batch['sequences'].shape[0]
            timesteps = torch.randint(
                0, config.diffusion.num_timesteps,
                (batch_size,), device=device
            )
            
            # Forward pass with mixed precision
            if config.training.use_amp and scaler is not None:
                with autocast():
                    outputs = model(
                        sequences=batch['sequences'],
                        attention_mask=batch['attention_mask'],
                        timesteps=timesteps,
                        structures=batch.get('structures'),
                        conditions=batch.get('conditions'),
                        return_loss=True
                    )
            else:
                outputs = model(
                    sequences=batch['sequences'],
                    attention_mask=batch['attention_mask'],
                    timesteps=timesteps,
                    structures=batch.get('structures'),
                    conditions=batch.get('conditions'),
                    return_loss=True
                )
            
            loss = outputs['total_loss']
            
            # Scale loss for gradient accumulation
            loss = loss / config.training.gradient_accumulation_steps
            
            # Backward pass
            if config.training.use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                if config.training.use_amp and scaler is not None:
                    # Unscale before gradient clipping
                    scaler.unscale_(optimizer)
                
                # Gradient clipping
                if hasattr(config.training, 'gradient_clip') and config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.training.gradient_clip
                    )
                
                # Optimizer step
                if config.training.use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Update EMA
                if ema is not None:
                    ema.update()
                
                # Clear memory periodically
                if (batch_idx + 1) % (config.training.gradient_accumulation_steps * 4) == 0:
                    clear_memory()
            
            # Accumulate losses
            for key in loss_accumulator:
                if key in outputs:
                    loss_accumulator[key] += outputs[key].item() * config.training.gradient_accumulation_steps
            accumulation_steps += 1
            successful_batches += 1
            
            # Logging
            if global_step % config.logging.log_every == 0 and accumulation_steps > 0:
                # Average losses
                avg_losses = {
                    k: v / accumulation_steps
                    for k, v in loss_accumulator.items()
                }
                
                # Write to tensorboard
                for key, value in avg_losses.items():
                    writer.add_scalar(f"train/{key}", value, global_step)
                
                writer.add_scalar(
                    "train/learning_rate",
                    scheduler.get_last_lr()[0],
                    global_step
                )
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{avg_losses['total_loss']:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                    'mem': f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "CPU"
                })
                
                # Reset accumulator
                loss_accumulator = {k: 0.0 for k in loss_accumulator}
                accumulation_steps = 0
                
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            # 清理内存后继续
            optimizer.zero_grad()
            clear_memory()
            continue
        
        global_step += 1
    
    # Update learning rate
    scheduler.step()
    
    logger.info(f"Epoch {epoch} completed. Successful batches: {successful_batches}/{len(train_loader)}")
    
    return global_step


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict,
    epoch: int,
    writer: SummaryWriter,
    ema: Optional[EMA] = None
) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    
    # Use EMA model if available
    if ema is not None:
        ema.apply_shadow()
    
    all_losses = []
    all_predictions = []
    all_targets = []
    successful_batches = 0
    
    with torch.no_grad():
        for batch_raw in tqdm(val_loader, desc="Validation"):
            try:
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
                
                # Forward pass
                if config.training.use_amp:
                    with autocast():
                        outputs = model(
                            sequences=batch['sequences'],
                            attention_mask=batch['attention_mask'],
                            timesteps=timesteps,
                            structures=batch.get('structures'),
                            conditions=batch.get('conditions'),
                            return_loss=True
                        )
                else:
                    outputs = model(
                        sequences=batch['sequences'],
                        attention_mask=batch['attention_mask'],
                        timesteps=timesteps,
                        structures=batch.get('structures'),
                        conditions=batch.get('conditions'),
                        return_loss=True
                    )
                
                all_losses.append(outputs['total_loss'].item())
                successful_batches += 1
                
                # Store predictions for metric computation
                all_predictions.append(outputs['denoised_embeddings'].cpu())
                all_targets.append(batch['sequences'].cpu())
                
            except Exception as e:
                logger.warning(f"Validation error: {e}")
                clear_memory()
                continue
    
    # Restore original weights if using EMA
    if ema is not None:
        ema.restore()
    
    if successful_batches == 0:
        logger.warning("No successful validation batches!")
        return {'val_loss': float('inf'), 'perplexity': float('inf')}
    
    # Compute metrics
    avg_loss = sum(all_losses) / len(all_losses)
    
    # Compute additional metrics if we have predictions
    metrics = {'val_loss': avg_loss}
    if all_predictions and all_targets:
        try:
            predictions_tensor = torch.cat(all_predictions)
            targets_tensor = torch.cat(all_targets)
            
            additional_metrics = compute_validation_metrics(
                predictions_tensor[:, :10, :10],  # Use subset for efficiency
                targets_tensor[:, :10].float().unsqueeze(-1).expand(-1, -1, 10),
                config
            )
            metrics.update(additional_metrics)
        except Exception as e:
            logger.warning(f"Could not compute additional metrics: {e}")
    
    # Log metrics
    for key, value in metrics.items():
        writer.add_scalar(f"val/{key}", value, epoch)
    
    logger.info(f"Validation metrics: {metrics}")
    logger.info(f"Successful validation batches: {successful_batches}/{len(val_loader)}")
    
    return metrics


def main():
    args = parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup Weights & Biases if requested
    if args.wandb or config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            name=config.experiment.name,
            config=OmegaConf.to_container(config, resolve=True),
            tags=config.wandb.tags
        )
    
    # Setup training
    model, optimizer, scheduler, ema, device, start_epoch, global_step, checkpoint_dir, shared_esmfold = setup_training(
        config, args.resume
    )
    
    # Create data loaders
    logger.info("正在加载数据集...")
    try:
        train_dataset = PeptideStructureDataset(
            config.data.train_path,
            config,
            is_training=True,
            shared_esmfold=shared_esmfold  # 传递共享ESMFold实例
        )
        val_dataset = PeptideStructureDataset(
            config.data.val_path,
            config,
            is_training=False,
            shared_esmfold=shared_esmfold  # 传递共享ESMFold实例
        )
        
        if args.debug:
            # Use subset for debugging
            train_dataset = torch.utils.data.Subset(train_dataset, range(min(100, len(train_dataset))))
            val_dataset = torch.utils.data.Subset(val_dataset, range(min(50, len(val_dataset))))
            logger.info("Debug mode: Using reduced datasets")
        
        logger.info(f"训练数据集大小: {len(train_dataset)}")
        logger.info(f"验证数据集大小: {len(val_dataset)}")
        
        # 最终内存使用情况
        if torch.cuda.is_available():
            logger.info(f"数据集加载后 GPU 内存使用: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
            
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        raise
    
    collator = PeptideStructureCollator(config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size * 2,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collator
    )
    
    # Setup tensorboard
    tensorboard_dir = os.path.join(config.experiment.output_dir, config.experiment.name, "tensorboard")
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir, max_checkpoints=config.training.max_checkpoints)
    
    # Training loop
    best_val_loss = float('inf')
    
    logger.info(f"\n开始训练 {config.training.num_epochs} 个 epochs...")
    logger.info(f"批量大小: {config.data.batch_size}, 梯度累积步数: {config.training.gradient_accumulation_steps}")
    logger.info(f"有效批量大小: {config.data.batch_size * config.training.gradient_accumulation_steps}")
    
    for epoch in range(start_epoch, config.training.num_epochs):
        logger.info(f"Starting epoch {epoch}")
        
        try:
            # Train
            global_step = train_epoch(
                model, train_loader, optimizer, scheduler, ema,
                device, config, epoch, global_step, writer
            )
            
            # Validate
            if (epoch + 1) % config.training.validate_every == 0 or epoch == config.training.num_epochs - 1:
                val_metrics = validate(
                    model, val_loader, device, config, epoch, writer, ema
                )
                
                # Save checkpoint
                is_best = val_metrics['val_loss'] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics['val_loss']
                    logger.info(f"💾 New best validation loss: {best_val_loss:.4f}")
                
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_metrics': val_metrics,
                    'config': config
                }
                
                if ema is not None:
                    checkpoint['ema_state_dict'] = ema.state_dict()
                
                checkpoint_manager.save_checkpoint(
                    checkpoint,
                    epoch,
                    is_best=is_best
                )
                
                # Log to wandb
                if args.wandb or config.wandb.enabled:
                    wandb.log(val_metrics, step=global_step)
                    
        except Exception as e:
            logger.error(f"Epoch {epoch} 训练失败: {e}")
            clear_memory()
            continue
    
    writer.close()
    logger.info("🎉 训练完成！")


if __name__ == "__main__":
    main()
