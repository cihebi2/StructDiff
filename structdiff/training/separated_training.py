#!/usr/bin/env python3
"""
CPL-Diff启发的分离式训练策略
基于CPL-Diff的两阶段训练设计，将去噪器训练和序列解码分离
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any, List
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from tqdm import tqdm

from ..models.structdiff import StructDiff
from ..diffusion.gaussian_diffusion import GaussianDiffusion
from ..utils.logger import get_logger
from ..utils.checkpoint import CheckpointManager
from ..utils.ema import EMA

logger = get_logger(__name__)


@dataclass
class SeparatedTrainingConfig:
    """分离式训练配置"""
    # 阶段1: 去噪器训练配置
    stage1_epochs: int = 200
    stage1_lr: float = 1e-4
    stage1_batch_size: int = 32
    stage1_gradient_clip: float = 1.0
    stage1_warmup_steps: int = 1000
    
    # 阶段2: 解码器训练配置  
    stage2_epochs: int = 100
    stage2_lr: float = 5e-5
    stage2_batch_size: int = 64
    stage2_gradient_clip: float = 0.5
    stage2_warmup_steps: int = 500
    
    # 共同配置
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9999
    save_every: int = 1000
    validate_every: int = 500
    log_every: int = 100
    
    # 长度控制
    use_length_control: bool = True
    max_length: int = 50
    min_length: int = 5
    
    # 条件控制
    use_cfg: bool = True
    cfg_dropout_prob: float = 0.1
    cfg_guidance_scale: float = 2.0
    
    # 数据路径
    data_dir: str = "./data/processed"
    output_dir: str = "./outputs/separated_training"
    checkpoint_dir: str = "./checkpoints/separated"


class SeparatedTrainingManager:
    """分离式训练管理器"""
    
    def __init__(self, 
                 config: SeparatedTrainingConfig,
                 model: StructDiff,
                 diffusion: GaussianDiffusion,
                 device: str = 'cuda'):
        self.config = config
        self.model = model
        self.diffusion = diffusion
        self.device = device
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.training_stats = {
            'stage1': {'losses': [], 'val_losses': []},
            'stage2': {'losses': [], 'val_losses': []},
        }
        
        # EMA
        if config.use_ema:
            self.ema = EMA(model, decay=config.ema_decay)
        else:
            self.ema = None
            
        logger.info(f"初始化分离式训练管理器，设备: {device}")
    
    def prepare_stage1_components(self) -> Tuple[nn.Module, torch.optim.Optimizer, Any]:
        """准备阶段1训练组件（去噪器训练）"""
        # 冻结序列编码器
        for param in self.model.sequence_encoder.parameters():
            param.requires_grad = False
        
        # 只训练去噪器和结构编码器
        trainable_params = []
        for name, param in self.model.named_parameters():
            if 'sequence_encoder' not in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
        
        logger.info(f"阶段1可训练参数数量: {sum(p.numel() for p in trainable_params)}")
        
        # 优化器
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.stage1_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.stage1_epochs,
            eta_min=1e-6
        )
        
        return self.model, optimizer, scheduler
    
    def prepare_stage2_components(self) -> Tuple[nn.Module, torch.optim.Optimizer, Any]:
        """准备阶段2训练组件（解码器训练）"""
        # 冻结去噪器和结构编码器
        for name, param in self.model.named_parameters():
            if 'sequence_decoder' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 只训练序列解码器
        decoder_params = [p for n, p in self.model.named_parameters() 
                         if 'sequence_decoder' in n and p.requires_grad]
        
        logger.info(f"阶段2可训练参数数量: {sum(p.numel() for p in decoder_params)}")
        
        # 优化器
        optimizer = torch.optim.AdamW(
            decoder_params,
            lr=self.config.stage2_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.stage2_epochs,
            eta_min=1e-6
        )
        
        return self.model, optimizer, scheduler
    
    def stage1_training_step(self, 
                           batch: Dict[str, torch.Tensor],
                           model: nn.Module,
                           optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """阶段1训练步骤：去噪器训练"""
        model.train()
        
        # 准备数据
        sequences = batch['sequences'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        structures = batch.get('structures', None)
        conditions = batch.get('conditions', None)
        
        # 获取固定的序列嵌入（不计算梯度）
        with torch.no_grad():
            seq_embeddings = model.sequence_encoder(sequences, attention_mask)
            if hasattr(seq_embeddings, 'last_hidden_state'):
                seq_embeddings = seq_embeddings.last_hidden_state
        
        # 随机时间步
        batch_size = sequences.shape[0]
        timesteps = torch.randint(
            0, self.diffusion.num_timesteps, 
            (batch_size,), device=self.device
        )
        
        # 添加噪声
        noise = torch.randn_like(seq_embeddings)
        noisy_embeddings = self.diffusion.q_sample(seq_embeddings, timesteps, noise)
        
        # 处理条件（支持CFG）
        if self.config.use_cfg and conditions is not None:
            # 随机丢弃部分条件
            dropout_mask = torch.rand(batch_size) < self.config.cfg_dropout_prob
            for key, value in conditions.items():
                if isinstance(value, torch.Tensor):
                    conditions[key] = value.clone()
                    conditions[key][dropout_mask] = -1  # 无条件标记
        
        # 去噪预测
        if hasattr(model, 'denoiser'):
            predicted_noise = model.denoiser(
                noisy_embeddings, timesteps, attention_mask,
                structure_features=structures, conditions=conditions
            )
        else:
            predicted_noise = model(
                noisy_embeddings, timesteps, attention_mask,
                structure_features=structures, conditions=conditions
            )
        
        # 计算损失（预测噪声）
        loss = F.mse_loss(predicted_noise, noise)
        
        # 反向传播
        optimizer.zero_grad()
        if self.config.use_amp:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config.stage1_gradient_clip
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config.stage1_gradient_clip
            )
            optimizer.step()
        
        # EMA更新
        if self.ema is not None:
            self.ema.update()
        
        return {
            'loss': loss.item(),
            'lr': optimizer.param_groups[0]['lr']
        }
    
    def stage2_training_step(self,
                           batch: Dict[str, torch.Tensor],
                           model: nn.Module,
                           optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """阶段2训练步骤：解码器训练"""
        model.train()
        
        # 准备数据
        sequences = batch['sequences'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # 获取干净的序列嵌入（固定编码器）
        with torch.no_grad():
            seq_embeddings = model.sequence_encoder(sequences, attention_mask)
            if hasattr(seq_embeddings, 'last_hidden_state'):
                seq_embeddings = seq_embeddings.last_hidden_state
        
        # 序列解码
        if hasattr(model, 'sequence_decoder'):
            logits = model.sequence_decoder(seq_embeddings, attention_mask)
        else:
            # 如果没有独立解码器，使用原始前向传播
            logits = model.decode_sequences(seq_embeddings, attention_mask)
        
        # 计算交叉熵损失
        vocab_size = logits.size(-1)
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = sequences.view(-1)
        
        # 忽略padding位置
        if attention_mask is not None:
            flat_mask = attention_mask.view(-1).bool()
            flat_logits = flat_logits[flat_mask]
            flat_targets = flat_targets[flat_mask]
        
        loss = F.cross_entropy(flat_logits, flat_targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.config.stage2_gradient_clip
        )
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'lr': optimizer.param_groups[0]['lr']
        }
    
    def validate_stage1(self, val_loader: DataLoader, model: nn.Module) -> Dict[str, float]:
        """阶段1验证"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequences'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                structures = batch.get('structures', None)
                conditions = batch.get('conditions', None)
                
                # 获取序列嵌入
                seq_embeddings = model.sequence_encoder(sequences, attention_mask)
                if hasattr(seq_embeddings, 'last_hidden_state'):
                    seq_embeddings = seq_embeddings.last_hidden_state
                
                # 随机时间步和噪声
                batch_size = sequences.shape[0]
                timesteps = torch.randint(
                    0, self.diffusion.num_timesteps,
                    (batch_size,), device=self.device
                )
                noise = torch.randn_like(seq_embeddings)
                noisy_embeddings = self.diffusion.q_sample(seq_embeddings, timesteps, noise)
                
                # 预测
                if hasattr(model, 'denoiser'):
                    predicted_noise = model.denoiser(
                        noisy_embeddings, timesteps, attention_mask,
                        structure_features=structures, conditions=conditions
                    )
                else:
                    predicted_noise = model(
                        noisy_embeddings, timesteps, attention_mask,
                        structure_features=structures, conditions=conditions
                    )
                
                loss = F.mse_loss(predicted_noise, noise)
                total_loss += loss.item()
                num_batches += 1
        
        return {'val_loss': total_loss / max(num_batches, 1)}
    
    def validate_stage2(self, val_loader: DataLoader, model: nn.Module) -> Dict[str, float]:
        """阶段2验证"""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequences'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 获取序列嵌入
                seq_embeddings = model.sequence_encoder(sequences, attention_mask)
                if hasattr(seq_embeddings, 'last_hidden_state'):
                    seq_embeddings = seq_embeddings.last_hidden_state
                
                # 解码
                if hasattr(model, 'sequence_decoder'):
                    logits = model.sequence_decoder(seq_embeddings, attention_mask)
                else:
                    logits = model.decode_sequences(seq_embeddings, attention_mask)
                
                # 计算损失
                vocab_size = logits.size(-1)
                flat_logits = logits.view(-1, vocab_size)
                flat_targets = sequences.view(-1)
                
                if attention_mask is not None:
                    flat_mask = attention_mask.view(-1).bool()
                    flat_logits = flat_logits[flat_mask]
                    flat_targets = flat_targets[flat_mask]
                
                loss = F.cross_entropy(flat_logits, flat_targets)
                total_loss += loss.item()
                num_batches += 1
        
        return {'val_loss': total_loss / max(num_batches, 1)}
    
    def train_stage1(self, 
                    train_loader: DataLoader, 
                    val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """执行阶段1训练：去噪器训练"""
        logger.info("🚀 开始阶段1训练：去噪器训练（固定ESM编码器）")
        
        model, optimizer, scheduler = self.prepare_stage1_components()
        
        # 使用EMA模型进行验证
        eval_model = self.ema.ema_model if self.ema else model
        
        stage1_stats = {'losses': [], 'val_losses': []}
        
        for epoch in range(self.config.stage1_epochs):
            # 训练循环
            model.train()
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f"阶段1 Epoch {epoch+1}/{self.config.stage1_epochs}")
            for step, batch in enumerate(pbar):
                step_stats = self.stage1_training_step(batch, model, optimizer)
                epoch_losses.append(step_stats['loss'])
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{step_stats['loss']:.4f}",
                    'lr': f"{step_stats['lr']:.2e}"
                })
                
                # 日志记录
                if step % self.config.log_every == 0:
                    logger.info(f"阶段1 Epoch {epoch+1}, Step {step}: "
                              f"Loss = {step_stats['loss']:.4f}, "
                              f"LR = {step_stats['lr']:.2e}")
                
                # 验证
                if val_loader and step % self.config.validate_every == 0:
                    val_stats = self.validate_stage1(val_loader, eval_model)
                    stage1_stats['val_losses'].append(val_stats['val_loss'])
                    logger.info(f"阶段1验证 - Val Loss: {val_stats['val_loss']:.4f}")
                
                # 保存检查点
                if step % self.config.save_every == 0:
                    self.checkpoint_manager.save_checkpoint(
                        model=eval_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=step,
                        stage='stage1',
                        stats=stage1_stats
                    )
            
            # 记录epoch统计
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            stage1_stats['losses'].append(epoch_loss)
            
            # 学习率调度
            scheduler.step()
            
            logger.info(f"阶段1 Epoch {epoch+1} 完成 - 平均Loss: {epoch_loss:.4f}")
        
        # 保存最终检查点
        self.checkpoint_manager.save_checkpoint(
            model=eval_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=self.config.stage1_epochs,
            step=0,
            stage='stage1_final',
            stats=stage1_stats
        )
        
        logger.info("✅ 阶段1训练完成")
        self.training_stats['stage1'] = stage1_stats
        return stage1_stats
    
    def train_stage2(self,
                    train_loader: DataLoader,
                    val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """执行阶段2训练：解码器训练"""
        logger.info("🚀 开始阶段2训练：解码器训练（固定去噪器）")
        
        model, optimizer, scheduler = self.prepare_stage2_components()
        
        stage2_stats = {'losses': [], 'val_losses': []}
        
        for epoch in range(self.config.stage2_epochs):
            # 训练循环
            model.train()
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f"阶段2 Epoch {epoch+1}/{self.config.stage2_epochs}")
            for step, batch in enumerate(pbar):
                step_stats = self.stage2_training_step(batch, model, optimizer)
                epoch_losses.append(step_stats['loss'])
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{step_stats['loss']:.4f}",
                    'lr': f"{step_stats['lr']:.2e}"
                })
                
                # 日志记录
                if step % self.config.log_every == 0:
                    logger.info(f"阶段2 Epoch {epoch+1}, Step {step}: "
                              f"Loss = {step_stats['loss']:.4f}, "
                              f"LR = {step_stats['lr']:.2e}")
                
                # 验证
                if val_loader and step % self.config.validate_every == 0:
                    val_stats = self.validate_stage2(val_loader, model)
                    stage2_stats['val_losses'].append(val_stats['val_loss'])
                    logger.info(f"阶段2验证 - Val Loss: {val_stats['val_loss']:.4f}")
                
                # 保存检查点
                if step % self.config.save_every == 0:
                    self.checkpoint_manager.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step=step,
                        stage='stage2',
                        stats=stage2_stats
                    )
            
            # 记录epoch统计
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            stage2_stats['losses'].append(epoch_loss)
            
            # 学习率调度
            scheduler.step()
            
            logger.info(f"阶段2 Epoch {epoch+1} 完成 - 平均Loss: {epoch_loss:.4f}")
        
        # 保存最终检查点
        self.checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=self.config.stage2_epochs,
            step=0,
            stage='stage2_final',
            stats=stage2_stats
        )
        
        logger.info("✅ 阶段2训练完成")
        self.training_stats['stage2'] = stage2_stats
        return stage2_stats
    
    def run_complete_training(self,
                            train_loader: DataLoader,
                            val_loader: Optional[DataLoader] = None) -> Dict[str, Any]:
        """运行完整的分离式训练"""
        logger.info("🎯 开始CPL-Diff启发的分离式训练策略")
        
        # 保存配置
        config_path = Path(self.config.output_dir) / "training_config.json"
        with open(config_path, 'w') as f:
            # 将dataclass转换为dict
            config_dict = {
                field.name: getattr(self.config, field.name)
                for field in self.config.__dataclass_fields__.values()
            }
            json.dump(config_dict, f, indent=2)
        
        # 阶段1：去噪器训练
        stage1_stats = self.train_stage1(train_loader, val_loader)
        
        # 阶段2：解码器训练
        stage2_stats = self.train_stage2(train_loader, val_loader)
        
        # 保存完整训练统计
        final_stats = {
            'stage1': stage1_stats,
            'stage2': stage2_stats,
            'config': config_dict
        }
        
        stats_path = Path(self.config.output_dir) / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        logger.info("🎉 分离式训练完成！")
        logger.info(f"统计数据保存到: {stats_path}")
        logger.info(f"检查点保存到: {self.config.checkpoint_dir}")
        
        return final_stats
    
    def load_stage1_checkpoint(self, checkpoint_path: str):
        """加载阶段1检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"已加载阶段1检查点: {checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        summary = {
            'stage1': {
                'final_loss': self.training_stats['stage1']['losses'][-1] if self.training_stats['stage1']['losses'] else None,
                'best_val_loss': min(self.training_stats['stage1']['val_losses']) if self.training_stats['stage1']['val_losses'] else None,
                'total_epochs': len(self.training_stats['stage1']['losses'])
            },
            'stage2': {
                'final_loss': self.training_stats['stage2']['losses'][-1] if self.training_stats['stage2']['losses'] else None,
                'best_val_loss': min(self.training_stats['stage2']['val_losses']) if self.training_stats['stage2']['val_losses'] else None,
                'total_epochs': len(self.training_stats['stage2']['losses'])
            }
        }
        return summary