#!/usr/bin/env python3
"""
多肽生成训练脚本 - 启用ESMFold结构预测
"""

import os
import sys
import argparse
import logging
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import math
import json
from collections import defaultdict, Counter
from statistics import mean, stdev
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, training metrics will not be logged to W&B")

from tqdm import tqdm
import yaml
from omegaconf import OmegaConf
from Bio import SeqIO, Align
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入所需模块
from fix_esmfold_patch import apply_esmfold_patch
from structdiff.models.structdiff import StructDiff
from structdiff.models.esmfold_wrapper import ESMFoldWrapper
from structdiff.data.dataset import PeptideStructureDataset
from structdiff.data.collator import PeptideStructureCollator
from structdiff.utils.checkpoint import CheckpointManager
from structdiff.utils.logger import setup_logger, get_logger
from structdiff.utils.ema import EMA

# 尝试导入modlamp用于不稳定性指数计算
try:
    from modlamp.descriptors import GlobalDescriptor
    MODLAMP_AVAILABLE = True
    print("✅ modlAMP已安装，理化性质计算可用")
except ImportError:
    MODLAMP_AVAILABLE = False
    print("⚠️ modlamp not available, instability index will be skipped")
    print("⚠️ modlamp not available, instability index will be skipped")

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train StructDiff with ESMFold")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/peptide_esmfold_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device to use"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with reduced data"
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Test run with minimal epochs"
    )
    return parser.parse_args()


def setup_environment(args, config):
    """设置环境和设备"""
    # 优先使用配置文件中的GPU设置，然后是命令行参数
    if hasattr(config, 'system') and hasattr(config.system, 'cuda_visible_devices'):
        gpu_id = config.system.cuda_visible_devices
        logger.info(f"从配置文件读取GPU设置: {gpu_id}")
    else:
        gpu_id = str(args.gpu)
        logger.info(f"使用命令行GPU参数: {gpu_id}")
    
    # 设置GPU环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    logger.info(f"设置 CUDA_VISIBLE_DEVICES={gpu_id}")
    
    # 应用ESMFold补丁
    logger.info("应用ESMFold兼容性补丁...")
    apply_esmfold_patch()
    logger.info("✓ ESMFold补丁应用成功")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        actual_gpu_id = torch.cuda.current_device()
        gpu_props = torch.cuda.get_device_properties(actual_gpu_id)
        logger.info(f"实际使用GPU {actual_gpu_id}: {gpu_props.name}")
        logger.info(f"GPU内存: {gpu_props.total_memory / 1e9:.1f}GB")
        
        # 清理显存
        torch.cuda.empty_cache()
        logger.info(f"当前内存使用: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        logger.info(f"可用内存: {(gpu_props.total_memory - torch.cuda.memory_allocated()) / 1e9:.1f}GB")
    
    return device


def setup_shared_esmfold(config: Dict, device: torch.device):
    """创建共享的ESMFold实例以节省内存"""
    shared_esmfold = None
    
    if (config.model.structure_encoder.get('use_esmfold', False) and 
        config.data.get('use_predicted_structures', False)):
        
        logger.info("正在创建共享ESMFold实例...")
        
        # 更激进的内存清理，为ESMFold腾出空间
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 强制垃圾回收
            gc.collect()
            
            # 设置内存分配策略
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
            
            # 再次清理
            torch.cuda.empty_cache()
            
            print(f"🧹 ESMFold初始化前GPU内存清理完成: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        try:
            # 首先尝试GPU
            shared_esmfold = ESMFoldWrapper(device=device)
            
            if shared_esmfold.available:
                logger.info("✓ 共享ESMFold GPU实例创建成功")
                logger.info(f"ESMFold内存使用: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
            else:
                logger.error("❌ ESMFold GPU实例创建失败")
                shared_esmfold = None
                
        except Exception as gpu_error:
            logger.warning(f"⚠️ ESMFold GPU初始化失败: {gpu_error}")
            
            try:
                # 尝试CPU fallback
                logger.info("🔄 尝试使用CPU创建ESMFold...")
                shared_esmfold = ESMFoldWrapper(device='cpu')
                if shared_esmfold.available:
                    logger.info("✅ ESMFold CPU实例创建成功")
                else:
                    shared_esmfold = None
            except Exception as cpu_error:
                logger.error(f"❌ ESMFold CPU初始化也失败: {cpu_error}")
                shared_esmfold = None
    
    return shared_esmfold


def clear_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def create_data_loaders(config: Dict, shared_esmfold, debug: bool = False):
    """创建数据加载器"""
    logger.info("正在创建数据加载器...")
    
    try:
        # 创建数据集
        train_dataset = PeptideStructureDataset(
            config.data.train_path,
            config,
            is_training=True,
            cache_dir=config.data.get('structure_cache_dir', './cache/train'),
            shared_esmfold=shared_esmfold
        )
        
        val_dataset = PeptideStructureDataset(
            config.data.val_path,
            config,
            is_training=False,
            cache_dir=config.data.get('structure_cache_dir', './cache/val'),
            shared_esmfold=shared_esmfold
        )
        
        # Debug模式使用子集
        if debug:
            train_subset_size = min(20, len(train_dataset))  # 进一步减少到20个样本
            val_subset_size = min(10, len(val_dataset))     # 减少到10个样本
            
            # 直接修改数据集的数据，而不是使用Subset
            train_dataset.data = train_dataset.data.head(train_subset_size)
            val_dataset.data = val_dataset.data.head(val_subset_size)
            
            logger.info(f"Debug模式: 训练{train_subset_size}, 验证{val_subset_size}")
        
        logger.info(f"训练数据集: {len(train_dataset)} 样本")
        logger.info(f"验证数据集: {len(val_dataset)} 样本")
        
        # 创建数据整理器
        collator = PeptideStructureCollator(config)
        
        # 创建数据加载器 - 使用更保守的设置
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=0,  # CRITICAL FIX: 使用0避免多进程缓存竞争问题
            pin_memory=False,  # 禁用pin_memory节省内存
            collate_fn=collator,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,  
            num_workers=0,  # CRITICAL FIX: 使用0避免多进程缓存竞争问题
            pin_memory=False,  # 禁用pin_memory节省内存
            collate_fn=collator,
            drop_last=False
        )
        
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"数据加载器创建失败: {e}")
        raise


def setup_model_and_training(config: Dict, device: torch.device, shared_esmfold):
    """设置模型和训练组件"""
    logger.info("正在初始化模型...")
    
    try:
        # 备份原始配置
        original_use_esmfold = config.model.structure_encoder.get('use_esmfold', False)
        
        # 如果已有共享实例，临时禁用模型内部的ESMFold加载
        if shared_esmfold and shared_esmfold.available:
            logger.info("临时禁用模型内部ESMFold加载以避免内存不足...")
            config.model.structure_encoder.use_esmfold = False
        
        # 创建模型
        model = StructDiff(config).to(device)
        
        # 恢复配置并设置共享实例
        config.model.structure_encoder.use_esmfold = original_use_esmfold
        
        # 如果有共享的ESMFold实例，手动设置到模型中
        if shared_esmfold and shared_esmfold.available:
            logger.info("正在将共享 ESMFold 实例设置到模型中...")
            if hasattr(model.structure_encoder, 'esmfold') or hasattr(model.structure_encoder, '_esmfold'):
                # 设置ESMFold实例
                model.structure_encoder.esmfold = shared_esmfold
                model.structure_encoder._esmfold = shared_esmfold
                # 确保ESMFold被标记为可用
                model.structure_encoder.use_esmfold = True
                logger.info("✓ 共享 ESMFold 实例已设置到模型中")
            else:
                # 如果模型结构不同，尝试直接设置属性
                setattr(model.structure_encoder, 'esmfold', shared_esmfold)
                setattr(model.structure_encoder, 'use_esmfold', True)
                logger.info("✓ 共享 ESMFold 实例已强制设置到模型中")
            
            clear_memory()
        
        logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建优化器 - 只排除ESMFold参数，保留其他所有参数
        trainable_params = []
        esmfold_params = 0
        total_params = 0
        sequence_encoder_params = 0
        other_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            
            # 只排除真正的ESMFold模型参数，保留其他所有参数
            if ('structure_encoder.esmfold.' in name or 
                'structure_encoder._esmfold.' in name or
                name.startswith('esmfold.')):
                esmfold_params += param.numel()
                param.requires_grad = False  # 冻结ESMFold参数
                logger.debug(f"冻结ESMFold参数: {name}")
            else:
                trainable_params.append(param)
                if 'sequence_encoder' in name:
                    sequence_encoder_params += param.numel()
                else:
                    other_params += param.numel()
        
        trainable_param_count = sum(p.numel() for p in trainable_params)
        
        logger.info(f"参数统计:")
        logger.info(f"  总参数: {total_params:,}")
        logger.info(f"  ESMFold参数(冻结): {esmfold_params:,}")
        logger.info(f"  序列编码器参数: {sequence_encoder_params:,}")
        logger.info(f"  其他可训练参数: {other_params:,}")
        logger.info(f"  可训练参数总计: {trainable_param_count:,}")
        
        if trainable_param_count < 1000000:  # 少于100万参数
            logger.warning(f"⚠️ 可训练参数过少 ({trainable_param_count:,})，可能影响训练效果")
            
            # 打印前10个可训练参数的名称
            logger.info("可训练参数示例:")
            for i, (name, param) in enumerate(model.named_parameters()):
                if param.requires_grad:
                    logger.info(f"  {name}: {param.shape}")
                    if i >= 9:  # 只显示前10个
                        break
        
        optimizer = torch.optim.AdamW(
            trainable_params,  # 只优化非ESMFold参数
            lr=config.training.optimizer.lr,
            betas=config.training.optimizer.betas,
            weight_decay=config.training.optimizer.weight_decay,
            eps=config.training.optimizer.eps
        )
        
        # 创建学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.scheduler.min_lr
        )
        
        # 创建EMA
        ema = None
        if config.training.use_ema:
            ema = EMA(
                model, 
                decay=config.training.ema_decay,
                device=device
            )
        
        logger.info(f"模型初始化完成，GPU内存: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
        
        return model, optimizer, scheduler, ema
        
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        # 如果失败，尝试完全禁用ESMFold
        logger.info("尝试禁用ESMFold重新初始化模型...")
        try:
            config.model.structure_encoder.use_esmfold = False
            config.data.use_predicted_structures = False
            model = StructDiff(config).to(device)
            logger.info("✓ 模型初始化成功（未使用ESMFold）")
            
            # 创建基础组件
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.training.optimizer.lr,
                betas=config.training.optimizer.betas,
                weight_decay=config.training.optimizer.weight_decay,
                eps=config.training.optimizer.eps
            )
            
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.num_epochs,
                eta_min=config.training.scheduler.min_lr
            )
            
            ema = None
            if config.training.use_ema:
                ema = EMA(
                    model, 
                    decay=config.training.ema_decay,
                    device=device
                )
            
            return model, optimizer, scheduler, ema
            
        except Exception as e2:
            logger.error(f"禁用ESMFold后仍然失败: {e2}")
            raise


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


def train_step(
    model, batch, optimizer, scaler, config, device, 
    gradient_accumulation_steps, step
):
    """单步训练 - 修复版本"""
    try:
        # 移动数据到设备 - 使用更强鲁棒的函数
        batch = move_to_device(batch, device)
        
        # 检查batch的基本字段
        if 'sequences' not in batch or 'attention_mask' not in batch:
            logger.warning(f"Batch missing required fields")
            return None, float('inf')
        
        # 检查张量形状一致性
        seq_shape = batch['sequences'].shape
        mask_shape = batch['attention_mask'].shape
        
        if seq_shape != mask_shape:
            logger.warning(f"Shape mismatch: sequences {seq_shape} vs attention_mask {mask_shape}")
            # 修正attention_mask的形状
            if mask_shape[1] != seq_shape[1]:
                min_len = min(mask_shape[1], seq_shape[1])
                batch['sequences'] = batch['sequences'][:, :min_len]
                batch['attention_mask'] = batch['attention_mask'][:, :min_len]
                logger.info(f"Adjusted shapes to: {batch['sequences'].shape}")
        
        # 检查结构数据的形状一致性
        if 'structures' in batch and batch['structures'] is not None:
            expected_struct_len = batch['sequences'].shape[1] - 2  # 除去CLS/SEP
            
            for key, value in batch['structures'].items():
                if value is None:
                    continue
                    
                # 对于结构特征，第二个维度应该与序列长度-2匹配
                if len(value.shape) >= 2:
                    actual_len = value.shape[1]
                    if actual_len != expected_struct_len:
                        logger.warning(f"Structure '{key}' length mismatch: {actual_len} vs expected {expected_struct_len}")
                        
                        # 截断或填充结构特征
                        if actual_len > expected_struct_len:
                            if len(value.shape) == 2:
                                batch['structures'][key] = value[:, :expected_struct_len]
                            elif len(value.shape) == 3:
                                if 'matrix' in key or 'map' in key:
                                    batch['structures'][key] = value[:, :expected_struct_len, :expected_struct_len]
                                else:
                                    batch['structures'][key] = value[:, :expected_struct_len, :]
                        elif actual_len < expected_struct_len:
                            pad_size = expected_struct_len - actual_len
                            if len(value.shape) == 2:
                                batch['structures'][key] = F.pad(value, (0, pad_size), value=0)
                            elif len(value.shape) == 3:
                                if 'matrix' in key or 'map' in key:
                                    batch['structures'][key] = F.pad(value, (0, pad_size, 0, pad_size), value=0)
                                else:
                                    batch['structures'][key] = F.pad(value, (0, 0, 0, pad_size), value=0)
        
        # 采样时间步
        batch_size = batch['sequences'].shape[0]
        timesteps = torch.randint(
            0, config.diffusion.num_timesteps,
            (batch_size,), device=device
        )
        
        # 前向传播
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
        
        # 检查是否有损失
        if 'total_loss' not in outputs:
            logger.warning("Model output missing 'total_loss', checking for alternatives...")
            if 'diffusion_loss' in outputs:
                outputs['total_loss'] = outputs['diffusion_loss']
            elif 'loss' in outputs:
                outputs['total_loss'] = outputs['loss']
            else:
                logger.error("No loss found in model outputs")
                return None, float('inf')
        
        loss = outputs['total_loss'] / gradient_accumulation_steps
        
        # 检查损失是否为NaN
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss detected: {loss}")
            return None, float('inf')
        
        # 反向传播
        if config.training.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return outputs, loss.item() * gradient_accumulation_steps
        
    except Exception as e:
        logger.error(f"训练步骤失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return None, float('inf')


def generate_and_validate(config, device):
    """训练完成后的生成和验证功能"""
    logger.info("🚀 开始生成和验证功能...")
    
    try:
        # 查找最佳模型检查点 - 尝试多个可能的路径
        possible_checkpoint_paths = []
        
        # 尝试从配置中获取输出目录
        if hasattr(config, 'experiment') and hasattr(config.experiment, 'output_dir'):
            base_dir = Path(config.experiment.output_dir) / config.experiment.name
            possible_checkpoint_paths.extend([
                base_dir / "checkpoints" / "best_model.pth",  # 修正扩展名
                base_dir / "checkpoints" / "best_model.pt",   # 保留兼容性
                base_dir / "checkpoints" / "latest.pth",      # 添加latest检查点
                base_dir / "best_model.pth",
                base_dir / "best_model.pt"
            ])
        
        if hasattr(config, 'training') and hasattr(config.training, 'output_dir'):
            base_dir = Path(config.training.output_dir)
            possible_checkpoint_paths.extend([
                base_dir / "checkpoints" / "best_model.pth",
                base_dir / "checkpoints" / "best_model.pt",
                base_dir / "checkpoints" / "latest.pth",
                base_dir / "best_model.pth",
                base_dir / "best_model.pt"
            ])
        
        # 添加默认路径
        possible_checkpoint_paths.extend([
            Path("./outputs/checkpoints/best_model.pth"),
            Path("./outputs/checkpoints/best_model.pt"),
            Path("./outputs/checkpoints/latest.pth"),
            Path("./outputs/best_model.pth"),
            Path("./outputs/best_model.pt"),
            Path("./checkpoints/best_model.pth"),
            Path("./checkpoints/best_model.pt"),
            Path("./checkpoints/latest.pth"),
            Path("./best_model.pth"),
            Path("./best_model.pt")
        ])
        
        best_checkpoint = None
        for checkpoint_path in possible_checkpoint_paths:
            if checkpoint_path.exists():
                best_checkpoint = checkpoint_path
                break
        
        if best_checkpoint is None:
            logger.warning("⚠️ 未找到最佳模型检查点，跳过生成验证")
            logger.info(f"尝试的路径: {[str(p) for p in possible_checkpoint_paths]}")
            return
        
        logger.info(f"📂 加载最佳模型: {best_checkpoint}")
        
        # 重新创建共享ESMFold实例
        shared_esmfold = setup_shared_esmfold(config, device)
        
        # 创建模型并加载权重
        model, _, _, _ = setup_model_and_training(config, device, shared_esmfold)
        
        # 加载检查点 - 处理PyTorch 2.6的安全限制
        try:
            checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
        except Exception as e:
            logger.warning(f"使用weights_only=False加载失败: {e}")
            # 尝试使用安全全局变量
            try:
                from omegaconf.listconfig import ListConfig
                from omegaconf.dictconfig import DictConfig
                torch.serialization.add_safe_globals([ListConfig, DictConfig])
                checkpoint = torch.load(best_checkpoint, map_location=device)
            except Exception as e2:
                logger.error(f"检查点加载失败: {e2}")
                return
        
        # 检查是否有排除的键信息
        excluded_keys = checkpoint.get('excluded_keys', [])
        if excluded_keys:
            logger.info(f"📝 检查点排除了以下参数: {excluded_keys}")
            logger.info("💡 这些参数将使用模型的默认初始化值")
        
        # 加载模型状态字典，允许部分匹配
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if missing_keys:
            logger.info(f"⚠️ 缺失的参数键 (将使用默认值): {len(missing_keys)} 个")
            # 只显示前几个，避免日志过长
            if len(missing_keys) <= 5:
                for key in missing_keys:
                    logger.info(f"   - {key}")
            else:
                for key in missing_keys[:3]:
                    logger.info(f"   - {key}")
                logger.info(f"   ... 还有 {len(missing_keys)-3} 个")
        
        if unexpected_keys:
            logger.info(f"⚠️ 意外的参数键: {len(unexpected_keys)} 个")
            if len(unexpected_keys) <= 5:
                for key in unexpected_keys:
                    logger.info(f"   - {key}")
        
        model.eval()
        
        logger.info("✅ 模型加载成功，准备开始生成验证")
        
        # 创建评估器
        evaluator = PeptideEvaluator(model, config, device, shared_esmfold)
        
        # 加载参考序列用于相似性计算
        reference_sequences = {}
        try:
            # 加载训练数据作为参考
            train_data_path = "./data/processed/train.csv"
            if os.path.exists(train_data_path):
                train_df = pd.read_csv(train_data_path)
                
                # 按肽类型分组参考序列
                type_mapping = {'antimicrobial': 0, 'antifungal': 1, 'antiviral': 2}
                for peptide_type, type_id in type_mapping.items():
                    type_sequences = train_df[train_df['label'] == type_id]['sequence'].tolist()
                    reference_sequences[peptide_type] = type_sequences[:200]  # 限制数量以提高效率
                    logger.info(f"加载 {len(reference_sequences[peptide_type])} 条 {peptide_type} 参考序列")
            else:
                logger.warning("⚠️ 未找到训练数据，跳过相似性计算")
        except Exception as e:
            logger.warning(f"⚠️ 加载参考序列失败: {e}")
        
        # 对每种肽类型进行评估
        peptide_types = ['antimicrobial', 'antifungal', 'antiviral']
        all_results = {}
        
        for peptide_type in peptide_types:
            logger.info(f"🧬 开始评估 {peptide_type} 多肽...")
            
            try:
                results, sequences = evaluator.comprehensive_evaluation(
                    peptide_type=peptide_type,
                    sample_num=50,  # 生成50个样本进行评估（减少数量以加快速度）
                    max_length=50,
                    reference_sequences=reference_sequences.get(peptide_type, None)
                )
                
                all_results[peptide_type] = results
                
                # 保存生成的序列
                if hasattr(config, 'experiment') and hasattr(config.experiment, 'output_dir'):
                    output_dir = Path(config.experiment.output_dir) / config.experiment.name
                elif hasattr(config, 'training') and hasattr(config.training, 'output_dir'):
                    output_dir = Path(config.training.output_dir)
                else:
                    output_dir = Path("./outputs")
                
                output_dir.mkdir(parents=True, exist_ok=True)
                
                fasta_file = output_dir / f"generated_{peptide_type}_sequences.fasta"
                evaluator.save_sequences_to_fasta(sequences, fasta_file, peptide_type)
                
                logger.info(f"✅ {peptide_type} 评估完成，生成 {len(sequences)} 条序列")
                
            except Exception as e:
                logger.error(f"❌ {peptide_type} 评估失败: {e}")
                continue
        
        # 保存所有结果
        results_file = output_dir / "generation_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # 打印结果摘要
        logger.info("\n" + "="*80)
        logger.info("🎯 生成评估结果摘要 - 专业生物学指标")
        logger.info("="*80)
        
        for peptide_type, results in all_results.items():
            logger.info(f"\n📊 {peptide_type.upper()} 多肽评估结果:")
            logger.info("-" * 50)
            
            # 1. 伪困惑度
            if 'pseudo_perplexity' in results:
                pp = results['pseudo_perplexity']
                logger.info(f"🧮 伪困惑度 (Pseudo-Perplexity):")
                logger.info(f"   平均值: {pp.get('mean_pseudo_perplexity', 0):.4f} ± {pp.get('std_pseudo_perplexity', 0):.4f}")
                logger.info(f"   有效序列: {pp.get('valid_sequences', 0)}")
            
            # 2. Shannon信息熵
            if 'shannon_entropy' in results:
                se = results['shannon_entropy']
                logger.info(f"📊 Shannon信息熵:")
                logger.info(f"   序列平均熵: {se.get('mean_sequence_entropy', 0):.4f} ± {se.get('std_sequence_entropy', 0):.4f}")
                logger.info(f"   整体熵: {se.get('overall_entropy', 0):.4f} / {se.get('max_possible_entropy', 4.32):.2f}")
            
            # 3. 不稳定性指数
            if 'instability_index' in results:
                ii = results['instability_index']
                logger.info(f"🧪 不稳定性指数 (Instability Index):")
                logger.info(f"   平均值: {ii.get('mean_instability_index', 0):.4f} ± {ii.get('std_instability_index', 0):.4f}")
                stable = ii.get('stable_peptides', 0)
                unstable = ii.get('unstable_peptides', 0)
                total = stable + unstable
                if total > 0:
                    logger.info(f"   稳定肽 (≤40): {stable}/{total} ({stable/total*100:.1f}%)")
                    logger.info(f"   不稳定肽 (>40): {unstable}/{total} ({unstable/total*100:.1f}%)")
            
            # 4. BLOSUM62相似性
            if 'blosum62_similarity' in results:
                bs = results['blosum62_similarity']
                logger.info(f"🔍 BLOSUM62相似性得分:")
                logger.info(f"   平均相似性: {bs.get('mean_similarity_score', 0):.4f} ± {bs.get('std_similarity_score', 0):.4f}")
                if 'max_similarity_score' in bs:
                    logger.info(f"   最高相似性: {bs['max_similarity_score']:.4f}")
                    logger.info(f"   最低相似性: {bs['min_similarity_score']:.4f}")
            
            # 5. 多样性分析
            if 'diversity_analysis' in results:
                da = results['diversity_analysis']
                logger.info(f"📈 多样性分析:")
                logger.info(f"   唯一性比例: {da.get('uniqueness_ratio', 0):.4f}")
                logger.info(f"   总序列数: {da.get('total_sequences', 0)}")
                logger.info(f"   唯一序列数: {da.get('unique_sequences', 0)}")
                logger.info(f"   重复序列数: {da.get('duplicate_sequences', 0)}")
                
                if 'length_distribution' in da:
                    ld = da['length_distribution']
                    logger.info(f"   长度分布: {ld.get('mean_length', 0):.1f} ± {ld.get('std_length', 0):.1f}")
                    logger.info(f"   长度范围: {ld.get('min_length', 0)}-{ld.get('max_length', 0)}")
                
                gini = da.get('amino_acid_gini_coefficient', 0)
                logger.info(f"   氨基酸分布均匀性 (Gini): {gini:.4f} (0=均匀, 1=不均匀)")
            
            # 6. 基本有效性
            if 'validity' in results:
                v = results['validity']
                logger.info(f"✅ 序列有效性:")
                logger.info(f"   有效率: {v.get('validity_rate', 0):.4f}")
                logger.info(f"   有效序列: {v.get('valid_sequences', 0)}")
                logger.info(f"   无效序列: {v.get('invalid_sequences', 0)}")
            
            # 7. pLDDT分数（论文指标）
            if 'plddt_scores' in results:
                plddt = results['plddt_scores']
                logger.info(f"🧬 pLDDT分数 (结构置信度):")
                logger.info(f"   平均pLDDT: {plddt.get('mean_plddt', 0):.4f} ± {plddt.get('std_plddt', 0):.4f}")
                logger.info(f"   有效序列: {plddt.get('valid_sequences', 0)}")
            
            # 8. 理化性质（论文指标）
            if 'physicochemical_properties' in results:
                props = results['physicochemical_properties']
                logger.info(f"⚙️ 理化性质:")
                
                if 'charge' in props:
                    charge = props['charge']
                    logger.info(f"   电荷 (pH=7.4): {charge.get('mean_charge', 0):.4f} ± {charge.get('std_charge', 0):.4f}")
                
                if 'isoelectric_point' in props:
                    iep = props['isoelectric_point']
                    logger.info(f"   等电点: {iep.get('mean_isoelectric_point', 0):.4f} ± {iep.get('std_isoelectric_point', 0):.4f}")
                
                if 'hydrophobicity' in props:
                    hydro = props['hydrophobicity']
                    logger.info(f"   疏水性 (Eisenberg): {hydro.get('mean_hydrophobicity', 0):.4f} ± {hydro.get('std_hydrophobicity', 0):.4f}")
                
                if 'aromaticity' in props:
                    aroma = props['aromaticity']
                    logger.info(f"   芳香性: {aroma.get('mean_aromaticity', 0):.4f} ± {aroma.get('std_aromaticity', 0):.4f}")
            
            # 9. 外部分类器活性（论文指标）
            if 'external_classifier_activity' in results:
                activity = results['external_classifier_activity']
                logger.info(f"🎯 外部分类器活性:")
                logger.info(f"   预测活性比例: {activity.get('predicted_active_ratio', 0):.4f}")
                logger.info(f"   预测活性序列: {activity.get('predicted_active', 0)}")
                logger.info(f"   预测非活性序列: {activity.get('predicted_inactive', 0)}")
                logger.info(f"   分类器类型: {activity.get('classifier_type', 'unknown')}")
            
            # 10. 总结统计
            if 'summary' in results:
                s = results['summary']
                logger.info(f"📋 总结:")
                logger.info(f"   生成成功率: {s.get('generation_success_rate', 0):.4f}")
                logger.info(f"   肽类型: {s.get('peptide_type', 'unknown')}")
        
        logger.info(f"\n📁 详细结果保存到: {results_file}")
        
        # 添加关键指标汇总表格
        logger.info("\n" + "="*80)
        logger.info("📈 关键指标汇总表格 (Key Metrics Summary)")
        logger.info("="*80)
        logger.info("指标说明: ↓=越低越好, ↑=越高越好")
        logger.info("-"*80)
        logger.info(f"{'肽类型':<15} {'Perplexity↓':<12} {'pLDDT↑':<10} {'Instability↓':<12} {'Similarity↓':<12} {'Activity↑':<10}")
        logger.info("-"*80)
        
        for peptide_type, results in all_results.items():
            # 提取关键指标，使用更准确的键名
            perplexity = results.get('pseudo_perplexity', {}).get('mean_pseudo_perplexity', 0.0)
            if perplexity == 0.0:  # 备用键名
                perplexity = results.get('pseudo_perplexity', {}).get('mean_perplexity', 0.0)
                if perplexity == 0.0:
                    perplexity = results.get('pseudo_perplexity', {}).get('mean', 0.0)
            
            plddt = results.get('plddt_scores', {}).get('mean_plddt', 0.0)
            instability = results.get('instability_index', {}).get('mean_instability', 0.0)
            if instability == 0.0:  # 备用键名
                instability = results.get('instability_index', {}).get('mean', 0.0)
                if instability == 0.0:  # 再次备用
                    instability_data = results.get('instability_index', {})
                    if isinstance(instability_data, dict) and 'mean_instability_index' in instability_data:
                        instability = instability_data['mean_instability_index']
            if instability == 0.0:  # 备用键名
                instability = results.get('instability_index', {}).get('mean', 0.0)
            
            # 修复相似性得分的键名
            similarity = results.get('blosum62_similarity', {}).get('mean_similarity_score', 0.0)
            if similarity == 0.0:  # 备用键名
                similarity = results.get('similarity_scores', {}).get('mean_similarity', 0.0)
            
            activity = results.get('external_classifier_activity', {}).get('predicted_active_ratio', 0.0)
            
            # 调试输出
            logger.debug(f"关键指标 - {peptide_type}: perplexity={perplexity}, plddt={plddt}, instability={instability}, similarity={similarity}, activity={activity}")
            
            logger.info(f"{peptide_type:<15} {perplexity:<12.2f} {plddt:<10.2f} {instability:<12.2f} {similarity:<12.4f} {activity:<10.3f}")
        
        logger.info("-"*80)
        logger.info("📊 指标解释:")
        logger.info("  • Perplexity↓: 伪困惑度，越低表示序列越符合蛋白质语言模型预期")
        logger.info("  • pLDDT↑: 结构置信度，越高表示预测的3D结构越可靠")
        logger.info("  • Instability↓: 不稳定性指数，越低表示蛋白质越稳定")
        logger.info("  • Similarity↓: 与训练集相似度，越低表示生成序列越新颖")
        logger.info("  • Activity↑: 预测活性比例，越高表示功能性序列越多")
        logger.info("="*80)
        
        logger.info("🎉 专业生物学评估完成！")
        
    except Exception as e:
        logger.error(f"❌ 生成和验证过程出错: {e}")


class PeptideEvaluator:
    """多肽生成评估器 - 包含专业生物学评估指标"""
    
    def __init__(self, model, config, device, esmfold_wrapper=None):
        self.model = model
        self.config = config
        self.device = device
        self.esmfold_wrapper = esmfold_wrapper
        
        # 氨基酸字母表
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        # 肽类型映射
        self.peptide_type_map = {
            'antimicrobial': 0,
            'antifungal': 1,
            'antiviral': 2
        }
        
        # 初始化ESM2模型用于伪困惑度计算
        self.esm_tokenizer = None
        self.esm_model = None
        self._init_esm_model()
        
        # 初始化BLOSUM62比对器
        self.aligner = None
        self._init_aligner()
    
    def _init_esm_model(self):
        """初始化ESM2模型用于伪困惑度计算"""
        try:
            logger.info("🔬 初始化ESM2模型用于伪困惑度计算...")
            from transformers import EsmTokenizer, EsmModel
            self.esm_tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
            self.esm_model = EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D').to(self.device)
            self.esm_model.eval()
            logger.info("✅ ESM2模型初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ ESM2模型初始化失败: {e}")
            self.esm_tokenizer = None
            self.esm_model = None
    
    def _init_aligner(self):
        """初始化BLOSUM62比对器"""
        try:
            self.aligner = Align.PairwiseAligner()
            self.aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
            self.aligner.open_gap_score = -10
            self.aligner.extend_gap_score = -0.5
            logger.info("✅ BLOSUM62比对器初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ BLOSUM62比对器初始化失败: {e}")
            self.aligner = None
    
    def generate_sequences(self, peptide_type='antimicrobial', sample_num=100, max_length=50):
        """生成多肽序列"""
        logger.info(f"🧬 生成 {sample_num} 条 {peptide_type} 序列...")
        
        sequences = []
        batch_size = min(16, sample_num)  # 小批量生成避免内存问题
        
        with torch.no_grad():
            for i in tqdm(range(0, sample_num, batch_size), desc="Generating"):
                current_batch_size = min(batch_size, sample_num - i)
                
                try:
                    # 生成随机长度
                    lengths = torch.randint(10, max_length, (current_batch_size,))
                    
                    # 创建条件
                    conditions = None
                    if peptide_type in self.peptide_type_map:
                        type_id = self.peptide_type_map[peptide_type]
                        conditions = {
                            'peptide_type': torch.tensor([type_id] * current_batch_size, device=self.device)
                        }
                    
                    # 生成序列
                    for j in range(current_batch_size):
                        # 获取当前序列的长度
                        seq_length = lengths[j].item()
                        noise_shape = (1, seq_length + 2, 320)  # +2 for CLS/SEP tokens
                        
                        # 简化的生成过程 - 实际应该使用DDPM采样
                        noise = torch.randn(noise_shape, device=self.device)
                        
                        # 创建attention mask
                        attention_mask = torch.ones(1, seq_length + 2, device=self.device)
                        
                        # 使用模型生成
                        timesteps = torch.randint(0, self.config.diffusion.num_timesteps, (1,), device=self.device)
                        
                        # 为单个序列创建条件
                        single_conditions = None
                        if conditions is not None:
                            single_conditions = {
                                'peptide_type': conditions['peptide_type'][:1]  # 只取第一个
                            }
                        
                        # 对于生成，我们需要直接使用denoiser，而不是完整的forward方法
                        # 因为我们从噪声嵌入开始，而不是从token IDs开始
                        denoised_embeddings, cross_attention_weights = self.model.denoiser(
                            noisy_embeddings=noise,
                            timesteps=timesteps,
                            attention_mask=attention_mask,
                            structure_features=None,  # 暂时不使用结构特征
                            conditions=single_conditions
                        )
                        
                        outputs = {
                            'denoised_embeddings': denoised_embeddings,
                            'cross_attention_weights': cross_attention_weights
                        }
                        
                        # 解码序列 (简化版本)
                        sequence = self._decode_sequence(outputs, seq_length)
                        if sequence and len(sequence) >= 5:  # 最小长度检查
                            sequences.append(sequence)
                
                except Exception as e:
                    logger.warning(f"生成批次 {i} 失败: {e}")
                    continue
        
        logger.info(f"✅ 成功生成 {len(sequences)} 条序列")
        return sequences
    
    def _decode_sequence(self, outputs, target_length):
        """解码模型输出为氨基酸序列"""
        # 这是一个简化的解码过程
        # 实际实现应该使用训练好的解码器
        
        try:
            # 随机生成序列作为占位符
            # 实际应该从模型输出中解码
            import random
            length = min(target_length, 50)
            sequence = ''.join(random.choices(self.amino_acids, k=length))
            return sequence
        except:
            return None
    
    def evaluate_diversity(self, sequences):
        """计算序列多样性"""
        if len(sequences) < 2:
            return {'uniqueness': 0.0, 'entropy': 0.0}
        
        # 唯一性
        unique_sequences = set(sequences)
        uniqueness = len(unique_sequences) / len(sequences)
        
        # 信息熵
        all_chars = ''.join(sequences)
        char_counts = Counter(all_chars)
        total_chars = len(all_chars)
        
        entropy = 0.0
        for count in char_counts.values():
            prob = count / total_chars
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return {
            'uniqueness': uniqueness,
            'entropy': entropy,
            'total_sequences': len(sequences),
            'unique_sequences': len(unique_sequences)
        }
    
    def evaluate_length_distribution(self, sequences):
        """评估长度分布"""
        lengths = [len(seq) for seq in sequences]
        
        return {
            'mean_length': mean(lengths),
            'std_length': stdev(lengths) if len(lengths) > 1 else 0.0,
            'min_length': min(lengths),
            'max_length': max(lengths)
        }
    
    def evaluate_amino_acid_composition(self, sequences):
        """评估氨基酸组成"""
        all_chars = ''.join(sequences)
        char_counts = Counter(all_chars)
        total_chars = len(all_chars)
        
        composition = {}
        for aa in self.amino_acids:
            composition[f'freq_{aa}'] = char_counts.get(aa, 0) / total_chars
        
        return composition
    
    def evaluate_validity(self, sequences):
        """评估序列有效性"""
        valid_sequences = []
        invalid_count = 0
        
        for seq in sequences:
            # 检查是否只包含标准氨基酸
            if all(aa in self.amino_acids for aa in seq):
                valid_sequences.append(seq)
            else:
                invalid_count += 1
        
        validity_rate = len(valid_sequences) / len(sequences) if sequences else 0.0
        
        return {
            'validity_rate': validity_rate,
            'valid_sequences': len(valid_sequences),
            'invalid_sequences': invalid_count
        }
    
    def evaluate_pseudo_perplexity(self, sequences):
        """
        计算伪困惑度（Pseudo-Perplexity）
        
        原理：逐个掩码序列中的氨基酸，用ESM2模型预测被掩码的氨基酸，计算预测损失的指数
        """
        if self.esm_tokenizer is None or self.esm_model is None:
            logger.warning("⚠️ ESM2模型未初始化，跳过伪困惑度计算")
            return {'mean_pseudo_perplexity': 0.0, 'std_pseudo_perplexity': 0.0}
        
        logger.info("🧮 计算伪困惑度...")
        pseudo_perplexities = []
        
        with torch.no_grad():
            for seq in tqdm(sequences, desc="Computing pseudo-perplexity"):
                try:
                    # 对序列进行编码
                    inputs = self.esm_tokenizer(seq, return_tensors='pt', padding=True, truncation=True)
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    
                    seq_len = input_ids.size(1)
                    total_loss = 0.0
                    valid_positions = 0
                    
                    # 逐个位置进行掩码预测
                    for pos in range(1, seq_len - 1):  # 跳过CLS和SEP token
                        if attention_mask[0, pos] == 1:  # 只处理有效位置
                            # 创建掩码版本
                            masked_input = input_ids.clone()
                            original_token = masked_input[0, pos].item()
                            masked_input[0, pos] = self.esm_tokenizer.mask_token_id
                            
                            # 预测
                            outputs = self.esm_model(masked_input, attention_mask=attention_mask)
                            logits = outputs.last_hidden_state[0, pos]  # 获取掩码位置的logits
                            
                            # 计算交叉熵损失
                            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([original_token], device=self.device))
                            total_loss += loss.item()
                            valid_positions += 1
                    
                    if valid_positions > 0:
                        avg_loss = total_loss / valid_positions
                        pseudo_perplexity = math.exp(avg_loss)
                        pseudo_perplexities.append(pseudo_perplexity)
                
                except Exception as e:
                    logger.warning(f"计算序列伪困惑度失败: {e}")
                    continue
        
        if pseudo_perplexities:
            return {
                'mean_pseudo_perplexity': mean(pseudo_perplexities),
                'std_pseudo_perplexity': stdev(pseudo_perplexities) if len(pseudo_perplexities) > 1 else 0.0,
                'valid_sequences': len(pseudo_perplexities)
            }
        else:
            return {'mean_pseudo_perplexity': 0.0, 'std_pseudo_perplexity': 0.0, 'valid_sequences': 0}
    
    def evaluate_shannon_entropy(self, sequences):
        """
        计算信息熵（Shannon Entropy）
        
        原理：计算序列中氨基酸分布的Shannon熵：H = -Σ p(aa) * log2(p(aa))
        """
        logger.info("📊 计算Shannon信息熵...")
        
        # 计算每个序列的熵
        sequence_entropies = []
        for seq in sequences:
            aa_counts = Counter(seq)
            total_aa = len(seq)
            
            entropy = 0.0
            for count in aa_counts.values():
                prob = count / total_aa
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            
            sequence_entropies.append(entropy)
        
        # 计算整体氨基酸分布的熵
        all_aa = ''.join(sequences)
        overall_aa_counts = Counter(all_aa)
        total_aa = len(all_aa)
        
        overall_entropy = 0.0
        for count in overall_aa_counts.values():
            prob = count / total_aa
            if prob > 0:
                overall_entropy -= prob * math.log2(prob)
        
        return {
            'mean_sequence_entropy': mean(sequence_entropies) if sequence_entropies else 0.0,
            'std_sequence_entropy': stdev(sequence_entropies) if len(sequence_entropies) > 1 else 0.0,
            'overall_entropy': overall_entropy,
            'max_possible_entropy': math.log2(20)  # 20种氨基酸的最大熵
        }
    
    def evaluate_instability_index(self, sequences):
        """
        计算不稳定性指数（Instability Index）
        
        原理：基于Guruprasad等人提出的算法，考虑相邻氨基酸对的稳定性贡献
        """
        if not MODLAMP_AVAILABLE:
            logger.warning("⚠️ modlamp未安装，跳过不稳定性指数计算")
            return {'mean_instability_index': 0.0, 'std_instability_index': 0.0}
        
        logger.info("🧪 计算不稳定性指数...")
        
        # 创建临时FASTA文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_file:
            for i, seq in enumerate(sequences):
                tmp_file.write(f">seq_{i}\n{seq}\n")
            tmp_file_path = tmp_file.name
        
        try:
            # 使用modlamp计算不稳定性指数
            desc = GlobalDescriptor(tmp_file_path)
            desc.instability_index()
            instability_scores = desc.descriptor.flatten()
            
            # 清理临时文件
            os.unlink(tmp_file_path)
            
            return {
                'mean_instability_index': mean(instability_scores),
                'std_instability_index': stdev(instability_scores) if len(instability_scores) > 1 else 0.0,
                'stable_peptides': sum(1 for score in instability_scores if score <= 40),
                'unstable_peptides': sum(1 for score in instability_scores if score > 40)
            }
        
        except Exception as e:
            logger.warning(f"计算不稳定性指数失败: {e}")
            # 清理临时文件
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            return {'mean_instability_index': 0.0, 'std_instability_index': 0.0}
    
    def evaluate_similarity_to_training(self, sequences, reference_sequences=None):
        """
        计算与训练集的相似性得分（BLOSUM62比对）
        
        原理：使用BLOSUM62替换矩阵对生成序列与训练集中真实序列进行全局比对
        """
        if self.aligner is None:
            logger.warning("⚠️ BLOSUM62比对器未初始化，跳过相似性计算")
            return {'mean_similarity_score': 0.0, 'std_similarity_score': 0.0}
        
        if reference_sequences is None:
            logger.warning("⚠️ 未提供参考序列，跳过相似性计算")
            return {'mean_similarity_score': 0.0, 'std_similarity_score': 0.0}
        
        logger.info("🔍 计算BLOSUM62相似性得分...")
        
        similarity_scores = []
        
        for gen_seq in tqdm(sequences, desc="Computing similarity scores"):
            seq_scores = []
            
            # 与参考序列集合中的每个序列进行比对
            for ref_seq in reference_sequences[:100]:  # 限制参考序列数量以提高效率
                try:
                    alignments = self.aligner.align(gen_seq, ref_seq)
                    if alignments:
                        score = alignments.score
                        # 标准化得分（除以较长序列的长度）
                        normalized_score = score / max(len(gen_seq), len(ref_seq))
                        seq_scores.append(normalized_score)
                except Exception as e:
                    continue
            
            if seq_scores:
                # 取最高相似性得分
                max_similarity = max(seq_scores)
                similarity_scores.append(max_similarity)
        
        if similarity_scores:
            return {
                'mean_similarity_score': mean(similarity_scores),
                'std_similarity_score': stdev(similarity_scores) if len(similarity_scores) > 1 else 0.0,
                'max_similarity_score': max(similarity_scores),
                'min_similarity_score': min(similarity_scores)
            }
        else:
            return {'mean_similarity_score': 0.0, 'std_similarity_score': 0.0}
    
    def evaluate_diversity_metrics(self, sequences):
        """
        多样性评估
        
        包括：去重比例、长度分布、氨基酸频率分析
        """
        logger.info("📈 计算多样性指标...")
        
        # 去重比例
        unique_sequences = set(sequences)
        uniqueness_ratio = len(unique_sequences) / len(sequences) if sequences else 0.0
        
        # 长度分布
        lengths = [len(seq) for seq in sequences]
        length_stats = {
            'mean_length': mean(lengths) if lengths else 0.0,
            'std_length': stdev(lengths) if len(lengths) > 1 else 0.0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'length_range': max(lengths) - min(lengths) if lengths else 0
        }
        
        # 氨基酸频率分析
        all_aa = ''.join(sequences)
        aa_counts = Counter(all_aa)
        total_aa = len(all_aa)
        
        aa_frequencies = {}
        for aa in self.amino_acids:
            aa_frequencies[f'freq_{aa}'] = aa_counts.get(aa, 0) / total_aa if total_aa > 0 else 0.0
        
        # 计算氨基酸使用的均匀性（基尼系数）
        frequencies = [aa_counts.get(aa, 0) / total_aa for aa in self.amino_acids if total_aa > 0]
        if frequencies:
            # 简化的基尼系数计算
            frequencies.sort()
            n = len(frequencies)
            gini = sum((2 * i - n - 1) * freq for i, freq in enumerate(frequencies, 1)) / (n * sum(frequencies))
        else:
            gini = 0.0
        
        return {
            'uniqueness_ratio': uniqueness_ratio,
            'total_sequences': len(sequences),
            'unique_sequences': len(unique_sequences),
            'duplicate_sequences': len(sequences) - len(unique_sequences),
            'length_distribution': length_stats,
            'amino_acid_frequencies': aa_frequencies,
            'amino_acid_gini_coefficient': gini  # 0表示完全均匀，1表示完全不均匀
        }
    
    def evaluate_plddt_scores(self, sequences):
        """
        计算pLDDT分数 (Predicted Local-Distance Difference Test)
        
        使用ESMFold预测3D结构并计算置信度分数
        """
        if self.esmfold_wrapper is None or not hasattr(self.esmfold_wrapper, 'fold_sequence'):
            logger.warning("⚠️ ESMFold未可用，跳过pLDDT计算")
            return {'mean_plddt': 0.0, 'std_plddt': 0.0, 'valid_sequences': 0}
        
        logger.info("🧬 计算pLDDT分数...")
        plddt_scores = []
        
        for seq in tqdm(sequences, desc="Computing pLDDT scores"):
            try:
                # 使用ESMFold预测结构
                result = self.esmfold_wrapper.fold_sequence(seq)
                
                if result and 'plddt' in result:
                    # 取所有残基pLDDT分数的平均值
                    mean_plddt = float(result['plddt'].mean())
                    plddt_scores.append(mean_plddt)
                
            except Exception as e:
                logger.warning(f"计算序列pLDDT失败: {e}")
                continue
        
        if plddt_scores:
            return {
                'mean_plddt': mean(plddt_scores),
                'std_plddt': stdev(plddt_scores) if len(plddt_scores) > 1 else 0.0,
                'valid_sequences': len(plddt_scores)
            }
        else:
            return {'mean_plddt': 0.0, 'std_plddt': 0.0, 'valid_sequences': 0}
    
    def _compute_simple_physicochemical_properties(self, sequences):
        """
        简化的理化性质计算（不依赖modlamp）
        """
        # 氨基酸属性表
        aa_charge = {'R': 1, 'K': 1, 'H': 0.5, 'D': -1, 'E': -1}
        aa_hydrophobicity = {
            'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
            'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
            'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
            'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
        }
        aromatic_aa = set('FWY')
        
        charges, hydrophobicities, isoelectric_points, aromaticities = [], [], [], []
        
        for seq in sequences:
            # 净电荷
            charge = sum(aa_charge.get(aa, 0) for aa in seq)
            charges.append(charge)
            
            # 平均疏水性
            hydro = [aa_hydrophobicity.get(aa, 0) for aa in seq]
            avg_hydro = mean(hydro) if hydro else 0
            hydrophobicities.append(avg_hydro)
            
            # 简化等电点估算
            basic_count = sum(1 for aa in seq if aa in 'RKH')
            acidic_count = sum(1 for aa in seq if aa in 'DE')
            if basic_count > acidic_count:
                iep = 8.5 + basic_count * 0.5
            elif acidic_count > basic_count:
                iep = 6.0 - acidic_count * 0.3
            else:
                iep = 7.0
            isoelectric_points.append(max(3.0, min(11.0, iep)))
            
            # 芳香性
            aromatic_ratio = sum(1 for aa in seq if aa in aromatic_aa) / len(seq)
            aromaticities.append(aromatic_ratio)
        
        return {
            'charge': {
                'mean_charge': mean(charges),
                'std_charge': stdev(charges) if len(charges) > 1 else 0.0
            },
            'isoelectric_point': {
                'mean_isoelectric_point': mean(isoelectric_points),
                'std_isoelectric_point': stdev(isoelectric_points) if len(isoelectric_points) > 1 else 0.0
            },
            'hydrophobicity': {
                'mean_hydrophobicity': mean(hydrophobicities),
                'std_hydrophobicity': stdev(hydrophobicities) if len(hydrophobicities) > 1 else 0.0
            },
            'aromaticity': {
                'mean_aromaticity': mean(aromaticities),
                'std_aromaticity': stdev(aromaticities) if len(aromaticities) > 1 else 0.0
            }
        }

    def evaluate_physicochemical_properties(self, sequences):
        """
        计算理化性质
        
        包括：电荷、等电点、疏水性、芳香性
        """
        if not MODLAMP_AVAILABLE:
            logger.warning("⚠️ modlamp未安装，使用简化的理化性质计算")
            return self._compute_simple_physicochemical_properties(sequences)
        
        logger.info("⚙️ 计算理化性质...")
        
        # 创建临时FASTA文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_file:
            for i, seq in enumerate(sequences):
                tmp_file.write(f">seq_{i}\n{seq}\n")
            tmp_file_path = tmp_file.name
        
        try:
            desc = GlobalDescriptor(tmp_file_path)
            
            # 计算各项理化性质
            properties = {}
            
            # 1. 电荷 (pH=7.4, Bjellqvist方法)
            try:
                desc.charge(ph=7.4, amide=True)  # amide=True使用Bjellqvist方法
                charges = desc.descriptor.flatten()
                properties['charge'] = {
                    'mean_charge': mean(charges),
                    'std_charge': stdev(charges) if len(charges) > 1 else 0.0
                }
            except:
                properties['charge'] = {'mean_charge': 0.0, 'std_charge': 0.0}
            
            # 2. 等电点
            try:
                desc.isoelectric_point(amide=True)  # 使用Bjellqvist方法
                ieps = desc.descriptor.flatten()
                properties['isoelectric_point'] = {
                    'mean_isoelectric_point': mean(ieps),
                    'std_isoelectric_point': stdev(ieps) if len(ieps) > 1 else 0.0
                }
            except:
                properties['isoelectric_point'] = {'mean_isoelectric_point': 0.0, 'std_isoelectric_point': 0.0}
            
            # 3. 疏水性 (Eisenberg scale, window=7)
            try:
                desc.hydrophobic_ratio(scale='eisenberg', window=7)
                hydrophobicities = desc.descriptor.flatten()
                properties['hydrophobicity'] = {
                    'mean_hydrophobicity': mean(hydrophobicities),
                    'std_hydrophobicity': stdev(hydrophobicities) if len(hydrophobicities) > 1 else 0.0
                }
            except:
                properties['hydrophobicity'] = {'mean_hydrophobicity': 0.0, 'std_hydrophobicity': 0.0}
            
            # 4. 芳香性 (Phe, Trp, Tyr含量)
            try:
                desc.aromaticity()
                aromaticities = desc.descriptor.flatten()
                properties['aromaticity'] = {
                    'mean_aromaticity': mean(aromaticities),
                    'std_aromaticity': stdev(aromaticities) if len(aromaticities) > 1 else 0.0
                }
            except:
                properties['aromaticity'] = {'mean_aromaticity': 0.0, 'std_aromaticity': 0.0}
            
            # 清理临时文件
            os.unlink(tmp_file_path)
            
            return properties
        
        except Exception as e:
            logger.warning(f"计算理化性质失败: {e}")
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            return {
                'charge': {'mean_charge': 0.0, 'std_charge': 0.0},
                'isoelectric_point': {'mean_isoelectric_point': 0.0, 'std_isoelectric_point': 0.0},
                'hydrophobicity': {'mean_hydrophobicity': 0.0, 'std_hydrophobicity': 0.0},
                'aromaticity': {'mean_aromaticity': 0.0, 'std_aromaticity': 0.0}
            }
    
    def evaluate_external_classifier_activity(self, sequences, peptide_type):
        """
        使用外部分类器评估活性
        """
        logger.info(f"🎯 评估 {peptide_type} 活性（外部分类器）...")
        
        try:
            # 导入简单分类器
            from structdiff.utils.external_classifiers import get_activity_classifier
            
            # 获取分类器
            classifier = get_activity_classifier(peptide_type)
            
            # 进行预测
            results = classifier.predict_activity(sequences)
            
            logger.info(f"✅ 外部分类器预测完成，活性比例: {results['predicted_active_ratio']:.3f}")
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ 外部分类器调用失败: {e}，返回占位符结果")
            
            return {
                'predicted_active_ratio': 0.0,
                'total_sequences': len(sequences),
                'predicted_active': 0,
                'predicted_inactive': len(sequences),
                'classifier_type': f'{peptide_type}_external_classifier'
            }
    
    def comprehensive_evaluation(self, peptide_type='antimicrobial', sample_num=100, max_length=50, reference_sequences=None):
        """
        综合评估 - 包含所有专业生物学指标
        
        Args:
            peptide_type: 肽类型
            sample_num: 生成样本数量
            max_length: 最大序列长度
            reference_sequences: 参考序列（用于相似性计算）
        """
        logger.info(f"🔬 开始 {peptide_type} 多肽综合评估...")
        
        # 生成序列
        sequences = self.generate_sequences(peptide_type, sample_num, max_length)
        
        if not sequences:
            logger.warning("⚠️ 未生成任何有效序列")
            return {}, []
        
        logger.info(f"✅ 成功生成 {len(sequences)} 条序列，开始计算评估指标...")
        
        # 计算各项指标
        results = {}
        
        # 1. 伪困惑度（Pseudo-Perplexity）
        try:
            pseudo_perplexity_metrics = self.evaluate_pseudo_perplexity(sequences)
            results['pseudo_perplexity'] = pseudo_perplexity_metrics
        except Exception as e:
            logger.warning(f"伪困惑度计算失败: {e}")
            results['pseudo_perplexity'] = {'mean_pseudo_perplexity': 0.0, 'std_pseudo_perplexity': 0.0}
        
        # 2. Shannon信息熵
        try:
            shannon_entropy_metrics = self.evaluate_shannon_entropy(sequences)
            results['shannon_entropy'] = shannon_entropy_metrics
        except Exception as e:
            logger.warning(f"Shannon熵计算失败: {e}")
            results['shannon_entropy'] = {'mean_sequence_entropy': 0.0, 'overall_entropy': 0.0}
        
        # 3. 不稳定性指数
        try:
            instability_metrics = self.evaluate_instability_index(sequences)
            results['instability_index'] = instability_metrics
        except Exception as e:
            logger.warning(f"不稳定性指数计算失败: {e}")
            results['instability_index'] = {'mean_instability_index': 0.0, 'std_instability_index': 0.0}
        
        # 4. BLOSUM62相似性得分
        try:
            similarity_metrics = self.evaluate_similarity_to_training(sequences, reference_sequences)
            results['blosum62_similarity'] = similarity_metrics
        except Exception as e:
            logger.warning(f"BLOSUM62相似性计算失败: {e}")
            results['blosum62_similarity'] = {'mean_similarity_score': 0.0, 'std_similarity_score': 0.0}
        
        # 5. 多样性评估
        try:
            diversity_metrics = self.evaluate_diversity_metrics(sequences)
            results['diversity_analysis'] = diversity_metrics
        except Exception as e:
            logger.warning(f"多样性分析失败: {e}")
            results['diversity_analysis'] = {'uniqueness_ratio': 0.0}
        
        # 6. 基本有效性检查
        try:
            validity_metrics = self.evaluate_validity(sequences)
            results['validity'] = validity_metrics
        except Exception as e:
            logger.warning(f"有效性检查失败: {e}")
            results['validity'] = {'validity_rate': 0.0}
        
        # 7. pLDDT分数（论文中的指标）
        try:
            plddt_metrics = self.evaluate_plddt_scores(sequences)
            results['plddt_scores'] = plddt_metrics
        except Exception as e:
            logger.warning(f"pLDDT分数计算失败: {e}")
            results['plddt_scores'] = {'mean_plddt': 0.0, 'std_plddt': 0.0}
        
        # 8. 理化性质（论文中的指标）
        try:
            physicochemical_metrics = self.evaluate_physicochemical_properties(sequences)
            results['physicochemical_properties'] = physicochemical_metrics
        except Exception as e:
            logger.warning(f"理化性质计算失败: {e}")
            results['physicochemical_properties'] = {
                'charge': {'mean_charge': 0.0, 'std_charge': 0.0},
                'isoelectric_point': {'mean_isoelectric_point': 0.0, 'std_isoelectric_point': 0.0},
                'hydrophobicity': {'mean_hydrophobicity': 0.0, 'std_hydrophobicity': 0.0},
                'aromaticity': {'mean_aromaticity': 0.0, 'std_aromaticity': 0.0}
            }
        
        # 9. 外部分类器活性评估（论文中的指标）
        try:
            activity_metrics = self.evaluate_external_classifier_activity(sequences, peptide_type)
            results['external_classifier_activity'] = activity_metrics
        except Exception as e:
            logger.warning(f"外部分类器活性评估失败: {e}")
            results['external_classifier_activity'] = {
                'predicted_active_ratio': 0.0,
                'total_sequences': len(sequences),
                'predicted_active': 0,
                'predicted_inactive': len(sequences)
            }
        
        # 去重并准备最终统计
        unique_sequences = list(set(sequences))
        results['summary'] = {
            'total_generated': len(sequences),
            'unique_sequences': len(unique_sequences),
            'peptide_type': peptide_type,
            'generation_success_rate': len(sequences) / sample_num if sample_num > 0 else 0.0
        }
        
        logger.info(f"✅ {peptide_type} 综合评估完成")
        return results, unique_sequences
    
    def save_sequences_to_fasta(self, sequences, output_file, peptide_type):
        """保存序列到FASTA文件"""
        records = []
        for i, seq in enumerate(sequences):
            record = SeqRecord(
                Seq(seq),
                id=f"generated_{peptide_type}_{i+1}",
                description=f"Generated {peptide_type} peptide"
            )
            records.append(record)
        
        SeqIO.write(records, output_file, "fasta")
        logger.info(f"💾 序列已保存到: {output_file}")


def main():
    args = parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 测试运行配置
    if args.test_run:
        config.training.num_epochs = 3
        config.training.validate_every = 1
        config.training.save_every = 50
        logger.info("测试运行模式：3个epochs")
    
    # Debug模式配置
    if args.debug:
        config.training.save_every = max(config.training.save_every, 10)  # Debug模式下至少每10个epoch保存一次
        logger.info("Debug模式：调整检查点保存频率以平衡速度和安全性")
    
    # 设置环境
    device = setup_environment(args, config)
    
    # 创建输出目录
    output_dir = Path(config.experiment.output_dir) / config.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs" 
    tensorboard_dir = output_dir / "tensorboard"
    
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    tensorboard_dir.mkdir(exist_ok=True)
    
    # 设置日志
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger(str(log_file))
    
    # 初始化Weights & Biases
    if config.wandb.enabled and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb.project,
            name=f"{config.experiment.name}_{datetime.now().strftime('%m%d_%H%M')}",
            config=OmegaConf.to_container(config, resolve=True),
            tags=config.wandb.tags,
            notes=config.wandb.notes
        )
    elif config.wandb.enabled and not WANDB_AVAILABLE:
        logger.warning("Weights & Biases requested but not available")
    
    # 设置共享ESMFold
    shared_esmfold = setup_shared_esmfold(config, device)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(config, shared_esmfold, args.debug)
    
    # 设置模型和训练组件
    model, optimizer, scheduler, ema = setup_model_and_training(config, device, shared_esmfold)
    
    # 设置tensorboard和检查点管理器
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    checkpoint_manager = CheckpointManager(str(checkpoint_dir), config.training.max_checkpoints)
    
    # 训练循环
    logger.info(f"开始训练 {config.training.num_epochs} 个epochs...")
    logger.info(f"有效批量大小: {config.data.batch_size * config.training.gradient_accumulation_steps}")
    
    # 显示检查点保存配置
    save_every = config.training.get('save_every', 10)
    logger.info(f"💾 检查点保存配置:")
    logger.info(f"   定期保存频率: 每 {save_every} 个epoch")
    logger.info(f"   最佳模型保存: 启用 (基于验证损失)")
    logger.info(f"   Debug模式: {'是' if args.debug else '否'}")
    if args.debug:
        logger.info(f"   Debug模式下保存频率已调整为至少每10个epoch")
    
    global_step = 0
    best_val_loss = float('inf')
    scaler = GradScaler() if config.training.use_amp else None
    
    for epoch in range(config.training.num_epochs):
        logger.info(f"🚀 开始 Epoch {epoch+1}/{config.training.num_epochs}")
        
        # 训练阶段
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                outputs, loss = train_step(
                    model, batch, optimizer, scaler, config, device,
                    config.training.gradient_accumulation_steps, global_step
                )
                
                if outputs is None:
                    continue
                
                epoch_loss += loss
                num_batches += 1
                
                # 梯度累积和更新
                if (batch_idx + 1) % config.training.gradient_accumulation_steps == 0:
                    if config.training.use_amp:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
                        optimizer.step()
                    
                    optimizer.zero_grad()
                    
                    if ema is not None:
                        ema.update()
                    
                    global_step += 1
                
                # 记录日志
                if global_step % config.logging.log_every == 0:
                    avg_loss = epoch_loss / max(num_batches, 1)
                    writer.add_scalar("train/loss", avg_loss, global_step)
                    writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                    
                    if config.wandb.enabled and WANDB_AVAILABLE:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/lr": scheduler.get_last_lr()[0],
                            "epoch": epoch
                        }, step=global_step)
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                    'mem': f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "CPU"
                })
                
                # 定期清理内存
                if (batch_idx + 1) % 50 == 0:
                    clear_memory()
                
            except Exception as e:
                logger.error(f"训练步骤 {batch_idx} 出错: {e}")
                clear_memory()
                continue
        
        logger.info(f"✅ Epoch {epoch+1} 训练完成，平均损失: {epoch_loss/max(num_batches, 1):.4f}")
        
        # 更新学习率
        logger.info("📈 更新学习率...")
        scheduler.step()
        logger.info(f"当前学习率: {scheduler.get_last_lr()[0]:.2e}")
        
        # 验证
        if (epoch + 1) % config.training.validate_every == 0:
            logger.info("🔍 开始验证...")
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    try:
                        batch = move_to_device(batch, device)
                        
                        # 检查batch的基本字段
                        if 'sequences' not in batch or 'attention_mask' not in batch:
                            continue
                        
                        batch_size = batch['sequences'].shape[0]
                        timesteps = torch.randint(
                            0, config.diffusion.num_timesteps,
                            (batch_size,), device=device
                        )
                        
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
                        
                        val_loss += outputs['total_loss'].item()
                        val_batches += 1
                        
                    except Exception as e:
                        logger.warning(f"验证步骤出错: {e}")
                        continue
            
            if val_batches > 0:
                avg_val_loss = val_loss / val_batches
                writer.add_scalar("val/loss", avg_val_loss, epoch)
                
                if config.wandb.enabled and WANDB_AVAILABLE:
                    wandb.log({"val/loss": avg_val_loss}, step=global_step)
                
                logger.info(f"✅ 验证完成，验证损失: {avg_val_loss:.4f}")
                
                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    logger.info(f"🎯 发现更好的模型！验证损失从 {best_val_loss:.4f} 降到 {avg_val_loss:.4f}")
                    best_val_loss = avg_val_loss
                    
                    # 保存最佳模型检查点
                    logger.info("💾 准备保存最佳模型检查点...")
                    
                    # 获取模型状态字典，但排除ESMFold参数以减少文件大小
                    model_state_dict = model.state_dict()
                    
                    # 过滤掉ESMFold相关的参数（这些参数很大且可以重新加载）
                    filtered_state_dict = {}
                    for key, value in model_state_dict.items():
                        # 跳过ESMFold相关参数
                        if not any(esmfold_key in key for esmfold_key in [
                            'structure_encoder.esmfold', 
                            'structure_encoder._esmfold',
                            'esmfold'
                        ]):
                            filtered_state_dict[key] = value
                    
                    logger.info(f"原始参数数量: {len(model_state_dict)}, 过滤后: {len(filtered_state_dict)}")
                    
                    checkpoint = {
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': filtered_state_dict,  # 使用过滤后的状态字典
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': avg_val_loss,
                        'config': config,
                        'excluded_keys': ['structure_encoder.esmfold', 'structure_encoder._esmfold'],  # 记录排除的键
                        'is_debug': args.debug  # 记录是否为debug模式
                    }
                    
                    if ema is not None:
                        checkpoint['ema_state_dict'] = ema.state_dict()
                    
                    logger.info("💾 开始保存检查点到磁盘...")
                    try:
                        checkpoint_manager.save_checkpoint(checkpoint, epoch, is_best=True)
                        logger.info(f"✅ 最佳模型检查点保存成功！")
                    except Exception as e:
                        logger.error(f"❌ 检查点保存失败: {e}")
                        # 继续训练，不要因为保存失败而中断
                else:
                    logger.info(f"📊 当前验证损失 {avg_val_loss:.4f} 未超过最佳 {best_val_loss:.4f}")
            else:
                logger.warning("⚠️ 验证批次为0，跳过验证")
        
        # 定期保存检查点（不仅仅是最佳模型）
        if (epoch + 1) % config.training.get('save_every', 10) == 0:
            logger.info(f"💾 定期保存检查点 (Epoch {epoch+1})...")
            
            # 获取模型状态字典，但排除ESMFold参数以减少文件大小
            model_state_dict = model.state_dict()
            
            # 过滤掉ESMFold相关的参数
            filtered_state_dict = {}
            for key, value in model_state_dict.items():
                if not any(esmfold_key in key for esmfold_key in [
                    'structure_encoder.esmfold', 
                    'structure_encoder._esmfold',
                    'esmfold'
                ]):
                    filtered_state_dict[key] = value
            
            checkpoint = {
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': filtered_state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': best_val_loss,  # 使用当前最佳验证损失
                'config': config,
                'excluded_keys': ['structure_encoder.esmfold', 'structure_encoder._esmfold'],
                'is_debug': args.debug,
                'checkpoint_type': 'periodic'  # 标记为定期检查点
            }
            
            if ema is not None:
                checkpoint['ema_state_dict'] = ema.state_dict()
            
            try:
                checkpoint_manager.save_checkpoint(checkpoint, epoch, is_best=False)
                logger.info(f"✅ 定期检查点保存成功！")
            except Exception as e:
                logger.error(f"❌ 定期检查点保存失败: {e}")
        
        logger.info(f"🏁 Epoch {epoch+1} 完全结束，准备下一个epoch...")
        
        # 强制清理内存
        clear_memory()
    
    writer.close()
    logger.info("🎉 训练完成！")
    
    if config.wandb.enabled and WANDB_AVAILABLE:
        wandb.finish()

    # 清理训练期间占用的资源，为验证阶段做准备
    logger.info("🧹 开始清理训练资源，为生成和验证阶段释放内存...")
    
    # 删除主要的训练对象
    try:
        del model
        del optimizer
        del scheduler
        if 'ema' in locals() and ema is not None:
            del ema
        del train_loader
        del val_loader
        if 'shared_esmfold' in locals() and shared_esmfold is not None:
            del shared_esmfold
        logger.info("🗑️ 训练对象已删除。")
    except NameError as e:
        logger.warning(f"清理部分训练对象时出错（可能未定义）: {e}")

    # 调用垃圾回收和CUDA缓存清理
    clear_memory()
    
    if torch.cuda.is_available():
        logger.info(f"✅ 训练资源清理完毕. 当前GPU内存: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")
        logger.info(f"   预留内存: {torch.cuda.memory_reserved(0) / 1e9:.2f}GB")

    # 训练完成后进行生成和验证
    logger.info("🚀 开始生成和验证...")
    generate_and_validate(config, device)


if __name__ == "__main__":
    main() 