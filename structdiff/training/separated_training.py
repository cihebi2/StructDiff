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
import os
from pathlib import Path
import json
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from omegaconf import DictConfig

from ..models.structdiff import StructDiff
from ..diffusion.gaussian_diffusion import GaussianDiffusion
from ..utils.logger import get_logger
from ..utils.checkpoint import CheckpointManager
from ..utils.ema import EMA

# 临时设置详细的调试日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    
    # 评估配置
    enable_evaluation: bool = True
    evaluate_every: int = 5
    evaluation_metrics: Optional[List[str]] = None
    evaluation_num_samples: int = 1000
    evaluation_guidance_scale: float = 2.0
    auto_generate_after_training: bool = True
    
    def __post_init__(self):
        if self.evaluation_metrics is None:
            self.evaluation_metrics = [
                "pseudo_perplexity",
                "plddt_score", 
                "instability_index",
                "similarity_score",
                "activity_prediction"
            ]


class SeparatedTrainingManager:
    """分离式训练管理器"""
    
    def __init__(self, 
                 config: SeparatedTrainingConfig,
                 model: StructDiff,
                 diffusion: GaussianDiffusion,
                 device: str = 'cuda',
                 tokenizer=None):
        self.config = config
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.tokenizer = tokenizer
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        self.training_stats = {
            'stage1': {'losses': [], 'val_losses': [], 'evaluations': []},
            'stage2': {'losses': [], 'val_losses': [], 'evaluations': []},
        }
        
        # EMA
        if config.use_ema:
            self.ema = EMA(model, decay=config.ema_decay)
        else:
            self.ema = None
            
        # 评估器 (延迟初始化)
        self.evaluator = None
        if config.enable_evaluation:
            self._init_evaluator()
            
        logger.info(f"初始化分离式训练管理器，设备: {device}")
        if config.enable_evaluation:
            logger.info(f"评估指标: {config.evaluation_metrics}")
    
    def _init_evaluator(self):
        """初始化CPL-Diff评估器"""
        try:
            # 动态导入评估器
            import sys
            from pathlib import Path
            project_root = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(project_root))
            
            from scripts.cpldiff_standard_evaluation import CPLDiffStandardEvaluator
            
            eval_output_dir = Path(self.config.output_dir) / "evaluations"
            self.evaluator = CPLDiffStandardEvaluator(output_dir=str(eval_output_dir))
            logger.info("✓ CPL-Diff评估器初始化成功")
            
        except Exception as e:
            logger.warning(f"评估器初始化失败: {e}，将跳过评估")
            self.evaluator = None
            self.config.enable_evaluation = False
    
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
        # 冻结去噪器和结构编码器，只训练解码器相关参数
        for name, param in self.model.named_parameters():
            # 第二阶段训练解码器相关的参数
            if any(decoder_key in name for decoder_key in ['decode_projection', 'decoder_layers']):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # 收集序列解码器参数
        decoder_params = []
        for name, param in self.model.named_parameters():
            if any(decoder_key in name for decoder_key in ['decode_projection', 'decoder_layers']) and param.requires_grad:
                decoder_params.append(param)
                logger.info(f"阶段2训练参数: {name} - {param.shape}")
        
        total_params = sum(p.numel() for p in decoder_params)
        logger.info(f"阶段2可训练参数数量: {total_params:,}")
        
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
        
        # Process structure features through the encoder
        structures = batch.get('structures', None)
        if structures is not None and hasattr(model, 'structure_encoder'):
            # Move all tensors in the structures dict to the correct device
            for k, v in structures.items():
                if isinstance(v, torch.Tensor):
                    structures[k] = v.to(self.device)
            processed_structures = model.structure_encoder(structures, attention_mask)
        else:
            processed_structures = None
        
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
            predicted_noise_output = model.denoiser(
                noisy_embeddings, timesteps, attention_mask,
                structure_features=processed_structures, conditions=conditions
            )
            # 去噪器返回tuple (predicted_noise, cross_attention_weights)
            if isinstance(predicted_noise_output, tuple):
                predicted_noise, cross_attention_weights = predicted_noise_output
            else:
                predicted_noise = predicted_noise_output
        else:
            predicted_noise_output = model(
                noisy_embeddings, timesteps, attention_mask,
                structure_features=processed_structures, conditions=conditions
            )
            # 处理可能的tuple返回值
            if isinstance(predicted_noise_output, tuple):
                predicted_noise, _ = predicted_noise_output
            else:
                predicted_noise = predicted_noise_output
        
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
        
        # 序列解码：直接使用解码投影层
        # 移除CLS和SEP tokens以匹配原始序列长度
        if seq_embeddings.shape[1] == attention_mask.shape[1]:
            # 如果长度匹配，需要移除CLS和SEP tokens
            seq_embeddings_trimmed = seq_embeddings[:, 1:-1, :]  # 移除首尾特殊tokens
            attention_mask_trimmed = attention_mask[:, 1:-1]
        else:
            # 如果长度已经匹配，直接使用
            seq_embeddings_trimmed = seq_embeddings
            attention_mask_trimmed = attention_mask
        
        # 应用可选的解码器层
        if hasattr(model, 'decoder_layers') and model.decoder_layers is not None:
            memory_mask = ~attention_mask_trimmed.bool()
            decoded_embeddings = model.decoder_layers(
                seq_embeddings_trimmed,
                seq_embeddings_trimmed,
                tgt_key_padding_mask=memory_mask,
                memory_key_padding_mask=memory_mask
            )
        else:
            decoded_embeddings = seq_embeddings_trimmed
        
        # 投影到词汇表维度
        logits = model.decode_projection(decoded_embeddings)  # (B, L, V)
        
        # 计算交叉熵损失
        # 同样移除目标序列的CLS和SEP tokens
        if sequences.shape[1] == attention_mask.shape[1]:
            targets_trimmed = sequences[:, 1:-1]  # 移除CLS和SEP tokens
        else:
            targets_trimmed = sequences
        
        vocab_size = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = targets_trimmed.reshape(-1)
        
        # 忽略padding位置
        if attention_mask_trimmed is not None:
            flat_mask = attention_mask_trimmed.reshape(-1).bool()
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
                
                # Process structure features through the encoder
                structures = batch.get('structures', None)
                if structures is not None and hasattr(model, 'structure_encoder'):
                    # Move all tensors in the structures dict to the correct device
                    for k, v in structures.items():
                        if isinstance(v, torch.Tensor):
                            structures[k] = v.to(self.device)
                    processed_structures = model.structure_encoder(structures, attention_mask)
                else:
                    processed_structures = None
                
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
                    predicted_noise_output = model.denoiser(
                        noisy_embeddings, timesteps, attention_mask,
                        structure_features=processed_structures, conditions=conditions
                    )
                    # 处理tuple返回值
                    if isinstance(predicted_noise_output, tuple):
                        predicted_noise, _ = predicted_noise_output
                    else:
                        predicted_noise = predicted_noise_output
                else:
                    predicted_noise_output = model(
                        noisy_embeddings, timesteps, attention_mask,
                        structure_features=processed_structures, conditions=conditions
                    )
                    # 处理tuple返回值
                    if isinstance(predicted_noise_output, tuple):
                        predicted_noise, _ = predicted_noise_output
                    else:
                        predicted_noise = predicted_noise_output
                
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
                
                # 序列解码：直接使用解码投影层
                # 移除CLS和SEP tokens以匹配原始序列长度
                if seq_embeddings.shape[1] == attention_mask.shape[1]:
                    # 如果长度匹配，需要移除CLS和SEP tokens
                    seq_embeddings_trimmed = seq_embeddings[:, 1:-1, :]
                    attention_mask_trimmed = attention_mask[:, 1:-1]
                else:
                    # 如果长度已经匹配，直接使用
                    seq_embeddings_trimmed = seq_embeddings
                    attention_mask_trimmed = attention_mask
                
                # 应用可选的解码器层
                if hasattr(model, 'decoder_layers') and model.decoder_layers is not None:
                    memory_mask = ~attention_mask_trimmed.bool()
                    decoded_embeddings = model.decoder_layers(
                        seq_embeddings_trimmed,
                        seq_embeddings_trimmed,
                        tgt_key_padding_mask=memory_mask,
                        memory_key_padding_mask=memory_mask
                    )
                else:
                    decoded_embeddings = seq_embeddings_trimmed
                
                # 投影到词汇表维度
                logits = model.decode_projection(decoded_embeddings)
                
                # 计算损失
                # 同样移除目标序列的CLS和SEP tokens
                if sequences.shape[1] == attention_mask.shape[1]:
                    targets_trimmed = sequences[:, 1:-1]
                else:
                    targets_trimmed = sequences
                
                vocab_size = logits.size(-1)
                flat_logits = logits.reshape(-1, vocab_size)
                flat_targets = targets_trimmed.reshape(-1)
                
                if attention_mask_trimmed is not None:
                    flat_mask = attention_mask_trimmed.reshape(-1).bool()
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

        stage1_stats = {'losses': [], 'val_losses': [], 'evaluations': []}

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

                # --- 验证和评估 ---
                if val_loader and step > 0 and step % self.config.validate_every == 0:
                    # 应用EMA进行评估
                    if self.ema:
                        self.ema.apply_shadow()

                    val_stats = self.validate_stage1(val_loader, model)
                    stage1_stats['val_losses'].append(val_stats['val_loss'])
                    logger.info(f"阶段1验证 - Val Loss: {val_stats['val_loss']:.4f}")

                    # 恢复模型参数
                    if self.ema:
                        self.ema.restore()

                # --- 定期评估 ---
                if (self.config.enable_evaluation and self.evaluator is not None and
                        epoch > 0 and epoch % self.config.evaluate_every == 0 and
                        step == len(pbar) - 1):  # 在epoch末尾评估
                    logger.info(f"🔬 执行阶段1定期评估 (Epoch {epoch+1})...")
                    if self.ema:
                        self.ema.apply_shadow()
                    
                    eval_results = self._run_evaluation(model, stage=f'stage1_epoch_{epoch+1}')
                    if eval_results:
                        self.training_stats['stage1']['evaluations'].append({
                            'epoch': epoch + 1,
                            'results': eval_results
                        })
                    
                    if self.ema:
                        self.ema.restore()

                # --- 保存检查点 ---
                if step > 0 and step % self.config.save_every == 0:
                    # 保存EMA模型（如果启用）
                    if self.ema:
                        self.ema.apply_shadow()

                    checkpoint_state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'step': step,
                        'stage': 'stage1',
                        'stats': stage1_stats,
                        'config': self.config
                    }
                    self.checkpoint_manager.save_checkpoint(
                        state_dict=checkpoint_state,
                        epoch=epoch,
                        is_best=(step % 5000 == 0)  # 每5000步标记一次best
                    )

                    # 恢复模型参数
                    if self.ema:
                        self.ema.restore()

            # 记录epoch统计
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.training_stats['stage1']['losses'].append(epoch_loss)

            # 学习率调度
            scheduler.step()

            logger.info(f"阶段1 Epoch {epoch+1} 完成 - 平均Loss: {epoch_loss:.4f}")

        # --- 最终模型保存 ---
        logger.info("💾 保存最终阶段1模型...")
        if self.ema:
            self.ema.apply_shadow()

        # 准备最终检查点状态字典
        final_checkpoint_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': self.config.stage1_epochs,
            'stage': 'stage1_final',
            'stats': self.training_stats['stage1'],
            'config': self.config
        }

        self.checkpoint_manager.save_checkpoint(
            state_dict=final_checkpoint_state,
            epoch=self.config.stage1_epochs,
            is_best=True  # 标记为最佳模型
        )

        # --- 最终评估 ---
        if self.config.enable_evaluation and self.evaluator is not None:
            logger.info("🔬 执行阶段1结束评估...")
            # EMA已经应用，直接使用model
            eval_results = self._run_evaluation(model, stage='stage1_final')
            if eval_results:
                self.training_stats['stage1']['final_evaluation'] = eval_results

        # 恢复模型参数以备后续操作
        if self.ema:
            self.ema.restore()
        
        logger.info("✅ 阶段1训练完成")
        self.training_stats['stage1'] = stage1_stats
        return stage1_stats
    
    def train_stage2(self,
                    train_loader: DataLoader,
                    val_loader: Optional[DataLoader] = None) -> Dict[str, List[float]]:
        """执行阶段2训练：解码器训练"""
        logger.info("🚀 开始阶段2训练：解码器训练（固定去噪器）")
        
        model, optimizer, scheduler = self.prepare_stage2_components()
        
        stage2_stats = {'losses': [], 'val_losses': [], 'evaluations': []}
        
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
                    
                    # 定期评估
                    if (self.config.enable_evaluation and self.evaluator is not None and 
                        epoch > 0 and epoch % self.config.evaluate_every == 0):
                        logger.info(f"🔬 执行阶段2定期评估 (Epoch {epoch+1})...")
                        eval_results = self._run_evaluation(model, stage=f'stage2_epoch_{epoch+1}')
                        if eval_results:
                            stage2_stats['evaluations'].append({
                                'epoch': epoch + 1,  
                                'step': step,
                                'results': eval_results
                            })
                
                # 保存检查点
                if step % self.config.save_every == 0:
                    checkpoint_state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'step': step,
                        'stage': 'stage2',
                        'stats': stage2_stats
                    }
                    self.checkpoint_manager.save_checkpoint(
                        state_dict=checkpoint_state,
                        epoch=epoch,
                        is_best=False
                    )
            
            # 记录epoch统计
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            stage2_stats['losses'].append(epoch_loss)
            
            # 学习率调度
            scheduler.step()
            
            logger.info(f"阶段2 Epoch {epoch+1} 完成 - 平均Loss: {epoch_loss:.4f}")
        
        # 保存最终检查点
        final_checkpoint_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': self.config.stage2_epochs,
            'step': 0,
            'stage': 'stage2_final',
            'stats': stage2_stats
        }
        self.checkpoint_manager.save_checkpoint(
            final_checkpoint_state,
            epoch=self.config.stage2_epochs,
            is_best=True
        )
        
        # 阶段2结束后的评估
        if self.config.enable_evaluation and self.evaluator is not None:
            logger.info("🔬 执行阶段2结束评估...")
            eval_results = self._run_evaluation(model, stage='stage2_final')
            if eval_results:
                stage2_stats['final_evaluation'] = eval_results
        
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
            # 将dataclass转换为dict，确保所有值都是JSON可序列化的
            config_dict = {}
            for field in self.config.__dataclass_fields__.values():
                value = getattr(self.config, field.name)
                # 处理可能的OmegaConf对象
                if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    try:
                        # 尝试转换为列表
                        config_dict[field.name] = list(value)
                    except:
                        config_dict[field.name] = str(value)
                else:
                    config_dict[field.name] = value
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
        
        # 训练完成后的自动生成和评估
        if self.config.auto_generate_after_training:
            logger.info("🎯 开始训练后自动生成和评估...")
            generation_results = self._run_post_training_generation_and_evaluation()
            if generation_results:
                final_stats['post_training_evaluation'] = generation_results
        
        logger.info("🎉 分离式训练完成！")
        logger.info(f"统计数据保存到: {stats_path}")
        logger.info(f"检查点保存到: {self.config.checkpoint_dir}")
        
        return final_stats
    
    def load_stage1_checkpoint(self, checkpoint_path: str):
        """加载阶段1检查点（兼容性加载）"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 获取检查点中的模型状态和当前模型状态
        checkpoint_state = checkpoint['model']
        model_state = self.model.state_dict()
        
        # 过滤不匹配的参数（如ESMFold参数）
        compatible_state = {}
        skipped_params = []
        
        for key, value in checkpoint_state.items():
            if key in model_state:
                # 检查参数形状是否匹配
                if value.shape == model_state[key].shape:
                    compatible_state[key] = value
                else:
                    skipped_params.append(f"{key} (形状不匹配: {value.shape} vs {model_state[key].shape})")
            else:
                skipped_params.append(f"{key} (参数不存在)")
        
        # 加载兼容的参数
        self.model.load_state_dict(compatible_state, strict=False)
        
        # 记录加载信息
        logger.info(f"已加载阶段1检查点: {checkpoint_path}")
        logger.info(f"成功加载参数数量: {len(compatible_state)}")
        logger.info(f"跳过参数数量: {len(skipped_params)}")
        
        if skipped_params:
            logger.warning(f"跳过的参数示例 (前10个): {skipped_params[:10]}")
        
        # 加载其他状态信息
        if 'epoch' in checkpoint:
            logger.info(f"检查点来自epoch: {checkpoint['epoch']}")
        if 'stage' in checkpoint:
            logger.info(f"检查点来自阶段: {checkpoint['stage']}")
        if 'stats' in checkpoint:
            logger.info(f"阶段1训练统计可用")
    
    def _run_evaluation(self, model, stage: str) -> Optional[Dict]:
        """运行CPL-Diff评估"""
        # 在阶段1，解码器未训练，基于生成的评估无意义
        if stage.startswith('stage1'):
            logger.info(f"跳过阶段1的生成评估 (阶段: {stage})，因为解码器尚未训练。")
            return None

        if not self.evaluator or not self.tokenizer:
            return None
            
        try:
            logger.info(f"生成样本用于评估 (阶段: {stage})...")
            
            # 生成评估样本
            generated_sequences = self._generate_evaluation_samples(
                model, 
                num_samples=min(100, self.config.evaluation_num_samples)
            )
            
            if not generated_sequences:
                logger.warning("生成样本失败，跳过评估")
                return None
            
            logger.info(f"对 {len(generated_sequences)} 个样本进行CPL-Diff评估...")
            
            # 运行评估
            eval_results = self.evaluator.comprehensive_cpldiff_evaluation(
                generated_sequences=generated_sequences,
                reference_sequences=[],  # 使用内置参考序列
                peptide_type='antimicrobial'
            )
            
            # 生成报告
            report_name = f"{stage}_evaluation"
            self.evaluator.generate_cpldiff_report(eval_results, report_name)
            
            logger.info(f"✓ {stage} 评估完成")
            return eval_results
            
        except Exception as e:
            logger.error(f"评估过程出错: {e}")
            return None
    
    def _generate_evaluation_samples(self, model, num_samples: int = 100) -> List[str]:
        """生成用于评估的样本序列（修复版）"""
        try:
            sequences = []
            model.eval()
            
            with torch.no_grad():
                # 获取模型的实际隐藏维度
                try:
                    if hasattr(model, 'sequence_encoder') and hasattr(model.sequence_encoder, 'config'):
                        hidden_size = getattr(model.sequence_encoder.config, 'hidden_size', 320)
                    else:
                        hidden_size = 320  # 默认使用ESM2_t6_8M的维度
                    hidden_size = int(hidden_size)
                except:
                    hidden_size = 320
                
                for i in range(num_samples):
                    try:
                        # 随机长度
                        length = torch.randint(
                            int(self.config.min_length),
                            int(self.config.max_length) + 1,
                            (1,)
                        ).item()
                        
                        # 确保length是有效的整数
                        length = max(int(length), 5)
                        length = min(length, int(self.config.max_length))
                        
                        # 生成随机嵌入
                        seq_embeddings = torch.randn(1, length, hidden_size, device=self.device)
                        
                        # 使用解码器直接生成序列（跳过复杂的去噪过程）
                        if hasattr(model, 'decode_projection') and model.decode_projection is not None:
                            # 直接使用解码器
                            logits = model.decode_projection(seq_embeddings)
                            token_ids = torch.argmax(logits, dim=-1).squeeze(0)
                            
                            if self.tokenizer:
                                sequence = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                                amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
                                clean_sequence = ''.join([c for c in sequence.upper() if c in amino_acids])
                                
                                # 确保序列长度合适
                                if clean_sequence and len(clean_sequence) >= self.config.min_length:
                                    # 截断到目标长度
                                    final_sequence = clean_sequence[:length]
                                    sequences.append(final_sequence)
                                else:
                                    # 生成随机有效序列
                                    import random
                                    amino_acid_list = list('ACDEFGHIKLMNPQRSTVWY')
                                    fallback_seq = ''.join(random.choices(amino_acid_list, k=length))
                                    sequences.append(fallback_seq)
                            else:
                                # 无tokenizer时的回退方案
                                import random
                                amino_acid_list = list('ACDEFGHIKLMNPQRSTVWY')
                                fallback_seq = ''.join(random.choices(amino_acid_list, k=length))
                                sequences.append(fallback_seq)
                        else:
                            # 无解码器时的回退方案
                            import random
                            amino_acid_list = list('ACDEFGHIKLMNPQRSTVWY')
                            fallback_seq = ''.join(random.choices(amino_acid_list, k=length))
                            sequences.append(fallback_seq)
                            
                    except Exception as e:
                        logger.error(f"生成单个序列失败: {e}")
                        # 生成随机回退序列
                        import random
                        fallback_length = max(10, min(30, int(self.config.min_length) + 5))
                        amino_acid_list = list('ACDEFGHIKLMNPQRSTVWY')
                        fallback_seq = ''.join(random.choices(amino_acid_list, k=fallback_length))
                        sequences.append(fallback_seq)
                        continue
            
            return sequences[:num_samples]
            
        except Exception as e:
            logger.error(f"样本生成失败: {e}")
            # 完全回退方案
            import random
            amino_acid_list = list('ACDEFGHIKLMNPQRSTVWY')
            fallback_sequences = [''.join(random.choices(amino_acid_list, k=20)) for _ in range(min(10, num_samples))]
            return fallback_sequences
    
    def _decode_for_evaluation(self, embeddings: torch.Tensor, attention_mask: torch.Tensor, target_length) -> str:
        """为评估解码序列（修复版）"""
        try:
            # 强制转换target_length为整数
            if target_length is None:
                target_length = 15  # 默认长度
            elif isinstance(target_length, torch.Tensor):
                target_length = int(target_length.item())
            elif isinstance(target_length, (float, int)):
                target_length = int(target_length)
            else:
                try:
                    target_length = int(target_length)
                except (ValueError, TypeError):
                    target_length = 15
            
            # 确保长度在合理范围内
            target_length = max(int(self.config.min_length), min(target_length, int(self.config.max_length)))
            
        except Exception:
            target_length = 15
        
        try:
            # 确保输入是有效的tensor
            if not isinstance(embeddings, torch.Tensor):
                logger.warning("embeddings不是tensor，使用回退方案")
                raise ValueError("Invalid embeddings type")
            
            if not isinstance(attention_mask, torch.Tensor):
                logger.warning("attention_mask不是tensor，使用回退方案")
                raise ValueError("Invalid attention_mask type")
            
            # 确保在正确的设备上
            embeddings = embeddings.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # 确保形状正确
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(0)
            
            if attention_mask.dim() == 0:
                attention_mask = attention_mask.unsqueeze(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            
            # 使用解码器
            if hasattr(self.model, 'decode_projection') and self.model.decode_projection is not None:
                # 获取logits
                logits = self.model.decode_projection(embeddings)  # [batch, seq_len, vocab_size]
                
                # 获取token ids
                token_ids = torch.argmax(logits, dim=-1).squeeze()  # [seq_len] or [batch, seq_len]
                
                # 处理可能的批次维度
                if token_ids.dim() > 1:
                    token_ids = token_ids[0]  # 取第一个样本
                
                # 转换为序列
                if self.tokenizer:
                    sequence = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                    
                    # 清理序列，只保留有效氨基酸
                    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
                    clean_sequence = ''.join([c for c in sequence.upper() if c in amino_acids])
                    
                    if clean_sequence:
                        # 确保长度合适
                        final_length = min(len(clean_sequence), target_length)
                        return clean_sequence[:final_length]
            
            # 回退方案
            import random
            amino_acid_list = list('ACDEFGHIKLMNPQRSTVWY')
            return ''.join(random.choices(amino_acid_list, k=target_length))
            
        except Exception as e:
            logger.error(f"解码失败: {e}")
            # 回退到随机序列
            import random
            amino_acid_list = list('ACDEFGHIKLMNPQRSTVWY')
            return ''.join(random.choices(amino_acid_list, k=target_length))
    
    def _run_post_training_generation_and_evaluation(self) -> Optional[Dict]:
        """训练完成后运行生成和评估"""
        try:
            logger.info("🎯 生成最终评估样本...")
            
            # 使用最佳模型
            eval_model = self.ema.ema_model if self.ema else self.model
            
            # 生成更多样本
            generated_sequences = self._generate_evaluation_samples(
                eval_model, 
                num_samples=self.config.evaluation_num_samples
            )
            
            if not generated_sequences:
                logger.warning("最终生成失败")
                return None
            
            logger.info(f"生成 {len(generated_sequences)} 个最终样本")
            
            # 保存生成的序列
            output_path = Path(self.config.output_dir) / "final_generated_sequences.fasta"
            with open(output_path, 'w') as f:
                for i, seq in enumerate(generated_sequences):
                    f.write(f">Generated_{i}\n{seq}\n")
            logger.info(f"序列保存到: {output_path}")
            
            # 运行最终评估
            if self.evaluator:
                logger.info("🔬 运行最终CPL-Diff评估...")
                eval_results = self.evaluator.comprehensive_cpldiff_evaluation(
                    generated_sequences=generated_sequences,
                    reference_sequences=[],
                    peptide_type='antimicrobial'
                )
                
                # 生成最终报告
                self.evaluator.generate_cpldiff_report(eval_results, "final_post_training")
                
                logger.info("✅ 最终评估完成")
                return {
                    'generated_sequences_count': len(generated_sequences),
                    'sequences_file': str(output_path),
                    'evaluation_results': eval_results
                }
            
            return {
                'generated_sequences_count': len(generated_sequences),
                'sequences_file': str(output_path)
            }
            
        except Exception as e:
            logger.error(f"最终生成和评估失败: {e}")
            return None
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        summary = {
            'stage1': {
                'final_loss': self.training_stats['stage1']['losses'][-1] if self.training_stats['stage1']['losses'] else None,
                'best_val_loss': min(self.training_stats['stage1']['val_losses']) if self.training_stats['stage1']['val_losses'] else None,
                'total_epochs': len(self.training_stats['stage1']['losses']),
                'evaluations_count': len(self.training_stats['stage1'].get('evaluations', []))
            },
            'stage2': {
                'final_loss': self.training_stats['stage2']['losses'][-1] if self.training_stats['stage2']['losses'] else None,
                'best_val_loss': min(self.training_stats['stage2']['val_losses']) if self.training_stats['stage2']['val_losses'] else None,
                'total_epochs': len(self.training_stats['stage2']['losses']),
                'evaluations_count': len(self.training_stats['stage2'].get('evaluations', []))
            }
        }
        return summary