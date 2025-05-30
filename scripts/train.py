#!/usr/bin/env python3
"""
Training script for StructDiff
"""

import os
import sys
import argparse
import logging
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

from structdiff.models.structdiff import StructDiff
from structdiff.data.dataset import PeptideStructureDataset
from structdiff.data.collator import PeptideStructureCollator
from structdiff.utils.checkpoint import CheckpointManager
from structdiff.utils.ema import EMA
from structdiff.utils.logger import setup_logger, get_logger
from structdiff.metrics import compute_validation_metrics

logger = get_logger(__name__)


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


def setup_training(config: Dict, resume_path: Optional[str] = None):
    """Setup training components"""
    
    # Create directories
    os.makedirs(config.logging.log_dir, exist_ok=True)
    os.makedirs(config.logging.checkpoint_dir, exist_ok=True)
    
    # Setup logger
    log_file = os.path.join(
        config.logging.log_dir,
        f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    setup_logger(log_file)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = StructDiff(config).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.optimizer.weight_decay,
        betas=config.training.optimizer.betas
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.scheduler.T_max,
        eta_min=config.training.scheduler.eta_min
    )
    
    # Create EMA
    ema = EMA(model, decay=config.training.ema_decay)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    global_step = 0
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        ema.load_state_dict(checkpoint['ema_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        logger.info(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    return model, optimizer, scheduler, ema, device, start_epoch, global_step


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    ema: EMA,
    device: torch.device,
    config: Dict,
    epoch: int,
    global_step: int,
    writer: SummaryWriter
) -> int:
    """Train for one epoch"""
    model.train()
    
    # Setup mixed precision training
    scaler = GradScaler()
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    # Accumulate losses
    loss_accumulator = {
        'total_loss': 0.0,
        'sequence_loss': 0.0,
        'structure_loss': 0.0
    }
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # Sample timesteps
        batch_size = batch['sequences'].shape[0]
        timesteps = torch.randint(
            0, config.model.diffusion.num_timesteps,
            (batch_size,), device=device
        )
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(
                sequences=batch['sequences'],
                attention_mask=batch['attention_mask'],
                timesteps=timesteps,
                structures=batch.get('structures'),
                conditions=batch.get('conditions'),
                return_loss=True
            )
        
        loss = outputs['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.gradient_clip
        )
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Update EMA
        ema.update()
        
        # Update learning rate
        scheduler.step()
        
        # Accumulate losses
        for key in loss_accumulator:
            if key in outputs:
                loss_accumulator[key] += outputs[key].item()
        
        # Logging
        if global_step % config.logging.log_every == 0:
            # Average losses
            avg_losses = {
                k: v / config.logging.log_every
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
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Reset accumulator
            loss_accumulator = {k: 0.0 for k in loss_accumulator}
        
        global_step += 1
    
    return global_step


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict,
    epoch: int,
    writer: SummaryWriter
) -> Dict[str, float]:
    """Validate model"""
    model.eval()
    ema_model = model  # Could use EMA model here
    
    all_losses = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Sample timesteps
            batch_size = batch['sequences'].shape[0]
            timesteps = torch.randint(
                0, config.model.diffusion.num_timesteps,
                (batch_size,), device=device
            )
            
            # Forward pass
            with autocast():
                outputs = ema_model(
                    sequences=batch['sequences'],
                    attention_mask=batch['attention_mask'],
                    timesteps=timesteps,
                    structures=batch.get('structures'),
                    conditions=batch.get('conditions'),
                    return_loss=True
                )
            
            all_losses.append(outputs['total_loss'].item())
            
            # Store predictions for metric computation
            all_predictions.append(outputs['denoised_embeddings'].cpu())
            all_targets.append(batch['sequences'].cpu())
    
    # Compute metrics
    avg_loss = sum(all_losses) / len(all_losses)
    
    # Compute additional metrics
    metrics = compute_validation_metrics(
        torch.cat(all_predictions),
        torch.cat(all_targets),
        config
    )
    metrics['val_loss'] = avg_loss
    
    # Log metrics
    for key, value in metrics.items():
        writer.add_scalar(f"val/{key}", value, epoch)
    
    logger.info(f"Validation metrics: {metrics}")
    
    return metrics


def main():
    args = parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup Weights & Biases if requested
    if args.wandb:
        wandb.init(
            project="structdiff",
            config=OmegaConf.to_container(config, resolve=True)
        )
    
    # Setup training
    model, optimizer, scheduler, ema, device, start_epoch, global_step = setup_training(
        config, args.resume
    )
    
    # Create data loaders
    train_dataset = PeptideStructureDataset(
        config.data.train_path,
        config,
        is_training=True
    )
    val_dataset = PeptideStructureDataset(
        config.data.val_path,
        config,
        is_training=False
    )
    
    if args.debug:
        # Use subset for debugging
        train_dataset = torch.utils.data.Subset(train_dataset, range(100))
        val_dataset = torch.utils.data.Subset(val_dataset, range(50))
    
    collator = PeptideStructureCollator(config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator
    )
    
    # Setup tensorboard
    writer = SummaryWriter(
        log_dir=os.path.join(config.logging.log_dir, "tensorboard")
    )
    
    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(
        config.logging.checkpoint_dir,
        max_checkpoints=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.training.num_epochs):
        logger.info(f"Starting epoch {epoch}")
        
        # Train
        global_step = train_epoch(
            model, train_loader, optimizer, scheduler, ema,
            device, config, epoch, global_step, writer
        )
        
        # Validate
        if (epoch + 1) % config.logging.save_every == 0:
            val_metrics = validate(
                model, val_loader, device, config, epoch, writer
            )
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['val_loss']
            
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }
            
            checkpoint_manager.save_checkpoint(
                checkpoint,
                epoch,
                is_best=is_best
            )
            
            # Log to wandb
            if args.wandb:
                wandb.log(val_metrics, step=global_step)
    
    writer.close()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()