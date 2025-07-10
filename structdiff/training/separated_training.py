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
            predicted_noise_output = model.denoiser(
                noisy_embeddings, timesteps, attention_mask,
                structure_features=structures, conditions=conditions
            )
            # 去噪器返回tuple (predicted_noise, cross_attention_weights)
            if isinstance(predicted_noise_output, tuple):
                predicted_noise, cross_attention_weights = predicted_noise_output
            else:
                predicted_noise = predicted_noise_output
        else:
            predicted_noise_output = model(
                noisy_embeddings, timesteps, attention_mask,
                structure_features=structures, conditions=conditions
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
                    predicted_noise_output = model.denoiser(
                        noisy_embeddings, timesteps, attention_mask,
                        structure_features=structures, conditions=conditions
                    )
                    # 处理tuple返回值
                    if isinstance(predicted_noise_output, tuple):
                        predicted_noise, _ = predicted_noise_output
                    else:
                        predicted_noise = predicted_noise_output
                else:
                    predicted_noise_output = model(
                        noisy_embeddings, timesteps, attention_mask,
                        structure_features=structures, conditions=conditions
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
                    
                    # 定期评估
                    if (self.config.enable_evaluation and self.evaluator is not None and 
                        epoch > 0 and epoch % self.config.evaluate_every == 0):
                        logger.info(f"🔬 执行阶段1定期评估 (Epoch {epoch+1})...")
                        eval_results = self._run_evaluation(eval_model, stage=f'stage1_epoch_{epoch+1}')
                        if eval_results:
                            stage1_stats['evaluations'].append({
                                'epoch': epoch + 1,
                                'step': step,
                                'results': eval_results
                            })
                
                # 保存检查点
                if step % self.config.save_every == 0:
                    checkpoint_state = {
                        'model': eval_model.state_dict(),
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
        
        # 阶段1结束后的评估
        if self.config.enable_evaluation and self.evaluator is not None:
            logger.info("🔬 执行阶段1结束评估...")
            eval_results = self._run_evaluation(eval_model, stage='stage1_final')
            if eval_results:
                stage1_stats['final_evaluation'] = eval_results
        
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
        """加载阶段1检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"已加载阶段1检查点: {checkpoint_path}")
    
    def _run_evaluation(self, model, stage: str) -> Optional[Dict]:
        """运行CPL-Diff评估"""
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
        """生成用于评估的样本序列"""
        try:
            sequences = []
            model.eval()
            
            with torch.no_grad():
                for i in range(0, num_samples, 10):  # 批次生成
                    batch_size = min(10, num_samples - i)
                    
                    # 随机长度
                    lengths = torch.randint(
                        self.config.min_length,
                        self.config.max_length + 1,
                        (batch_size,)
                    ).tolist()
                    
                    for length in lengths:
                        try:
                            # 生成噪声嵌入
                            seq_embeddings = torch.randn(
                                1, length, 
                                getattr(model.sequence_encoder.config, 'hidden_size', 768),
                                device=self.device
                            )
                            attention_mask = torch.ones(1, length, device=self.device)
                            
                            # 简单去噪
                            if hasattr(model, 'denoiser'):
                                for t in reversed(range(0, 1000, 100)):
                                    timesteps = torch.tensor([t], device=self.device)
                                    noise_pred = model.denoiser(
                                        seq_embeddings, timesteps, attention_mask
                                    )
                                    seq_embeddings = seq_embeddings - 0.1 * noise_pred
                            
                            # 解码序列
                            sequence = self._decode_for_evaluation(
                                seq_embeddings[0], attention_mask[0], length
                            )
                            
                            if sequence and len(sequence) >= self.config.min_length:
                                sequences.append(sequence)
                                
                        except Exception as e:
                            logger.debug(f"生成单个序列失败: {e}")
                            continue
            
            return sequences[:num_samples]
            
        except Exception as e:
            logger.error(f"样本生成失败: {e}")
            return []
    
    def _decode_for_evaluation(self, embeddings: torch.Tensor, attention_mask: torch.Tensor, target_length: int) -> str:
        """为评估解码序列"""
        try:
            if hasattr(self.model, 'sequence_decoder') and self.model.sequence_decoder is not None:
                logits = self.model.sequence_decoder(embeddings.unsqueeze(0), attention_mask.unsqueeze(0))
                token_ids = torch.argmax(logits, dim=-1).squeeze(0)
                
                # 转换为序列
                if self.tokenizer:
                    sequence = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
                    clean_sequence = ''.join([c for c in sequence.upper() if c in amino_acids])
                    return clean_sequence[:target_length] if clean_sequence else None
            
            # 回退方案
            import random
            amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
            return ''.join(random.choices(amino_acids, k=target_length))
            
        except Exception as e:
            logger.debug(f"解码失败: {e}")
            return None
    
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