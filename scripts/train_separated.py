#!/usr/bin/env python3
"""
分离式训练脚本
基于CPL-Diff的两阶段训练策略：
1. 阶段1：固定ESM编码器，训练去噪器
2. 阶段2：固定去噪器，训练序列解码器
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import yaml
from omegaconf import OmegaConf

# 添加项目路径到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 添加structdiff包路径
structdiff_path = project_root / "structdiff"
if str(structdiff_path.parent) not in sys.path:
    sys.path.insert(0, str(structdiff_path.parent))

from structdiff.models.structdiff import StructDiff
from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
from structdiff.data.dataset import PeptideStructureDataset
from structdiff.training.separated_training import SeparatedTrainingManager, SeparatedTrainingConfig
from structdiff.training.length_controller import create_length_controller_from_data, LengthAwareDataCollator
from structdiff.utils.logger import setup_logger, get_logger
from structdiff.utils.config import load_config

# 全局日志记录器
logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="分离式训练脚本")
    
    # 基础配置
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/separated_training_production.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/processed",
        help="数据目录"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/separated_training",
        help="输出目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="设备 (cuda/cpu/auto)"
    )
    
    # 训练配置覆盖
    parser.add_argument("--stage1-epochs", type=int, help="阶段1训练轮数")
    parser.add_argument("--stage2-epochs", type=int, help="阶段2训练轮数")
    parser.add_argument("--stage1-lr", type=float, help="阶段1学习率")
    parser.add_argument("--stage2-lr", type=float, help="阶段2学习率")
    parser.add_argument("--batch-size", type=int, help="批次大小")
    
    # 功能开关
    parser.add_argument("--use-cfg", action="store_true", help="使用分类器自由引导")
    parser.add_argument("--use-length-control", action="store_true", help="使用长度控制")
    parser.add_argument("--use-amp", action="store_true", help="使用混合精度训练")
    parser.add_argument("--use-ema", action="store_true", help="使用EMA")
    
    # 阶段控制
    parser.add_argument("--stage", type=str, choices=['1', '2', 'both'], default='both',
                       help="训练阶段 (1: 只训练阶段1, 2: 只训练阶段2, both: 完整训练)")
    parser.add_argument("--stage1-checkpoint", type=str, help="阶段1检查点路径（用于阶段2训练）")
    
    # 调试选项
    parser.add_argument("--debug", action="store_true", help="调试模式")
    parser.add_argument("--dry-run", action="store_true", help="干运行（不实际训练）")
    
    return parser.parse_args()


def setup_device(config: Dict, device_arg: str = "auto") -> str:
    """设置设备和GPU资源分配"""
    # 优先使用配置文件中的设备设置
    if hasattr(config, 'resources') and hasattr(config.resources, 'device'):
        device = config.resources.device
    elif device_arg != "auto":
        device = device_arg
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA不可用，回退到CPU")
        device = "cpu"
    
    # 设置可见GPU
    if hasattr(config, 'resources') and hasattr(config.resources, 'available_gpus'):
        available_gpus = config.resources.available_gpus
        # 设置CUDA_VISIBLE_DEVICES环境变量
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, available_gpus))
        logger.info(f"设置可见GPU: {available_gpus}")
    
    logger.info(f"使用设备: {device}")
    
    # 如果配置了阶段特定的GPU分配，记录信息
    if hasattr(config, 'resources'):
        if hasattr(config.resources, 'stage1_gpus'):
            logger.info(f"阶段1 GPU分配: {config.resources.stage1_gpus}")
        if hasattr(config.resources, 'stage2_gpus'):
            logger.info(f"阶段2 GPU分配: {config.resources.stage2_gpus}")
    
    return device


def create_model_and_diffusion(config: Dict) -> tuple:
    """创建模型和扩散过程"""
    # 创建模型
    model = StructDiff(config.model)
    
    # 创建扩散过程
    diffusion = GaussianDiffusion(
        num_timesteps=config.diffusion.num_timesteps,
        noise_schedule=config.diffusion.noise_schedule,
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end
    )
    
    return model, diffusion


def create_data_loaders(config: Dict, 
                       training_config: SeparatedTrainingConfig,
                       tokenizer) -> tuple:
    """创建数据加载器"""
    
    # 创建长度控制器（如果需要）
    length_controller = None
    if training_config.use_length_control:
        train_data_path = Path(training_config.data_dir) / "train.csv"
        if train_data_path.exists():
            length_controller = create_length_controller_from_data(
                str(train_data_path),
                min_length=training_config.min_length,
                max_length=training_config.max_length
            )
            logger.info("✓ 创建长度控制器成功")
        else:
            logger.warning(f"训练数据文件不存在: {train_data_path}")
    
    # 创建数据整理器
    collator = LengthAwareDataCollator(
        length_controller=length_controller,
        tokenizer=tokenizer,
        use_length_control=training_config.use_length_control
    )
    
    # 获取数据加载配置
    num_workers = config.data.get('num_workers', 2)
    pin_memory = config.data.get('pin_memory', True)
    
    # 检查结构特征设置
    use_structures = config.data.get('use_predicted_structures', False)
    structure_cache_dir = config.data.get('structure_cache_dir', './cache')
    
    if use_structures:
        logger.info(f"✓ 启用结构特征，缓存目录: {structure_cache_dir}")
        if not Path(structure_cache_dir).exists():
            logger.warning(f"结构缓存目录不存在: {structure_cache_dir}")
    else:
        logger.info("结构特征已禁用")
    
    # 训练数据集
    train_dataset = PeptideStructureDataset(
        data_path=str(Path(training_config.data_dir) / "train.csv"),
        config=config,
        is_training=True,
        cache_dir=str(Path(structure_cache_dir) / "train") if use_structures else None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.stage1_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator
    )
    
    # 验证数据集
    val_dataset = None
    val_loader = None
    val_data_path = Path(training_config.data_dir) / "val.csv"
    if val_data_path.exists():
        val_dataset = PeptideStructureDataset(
            data_path=str(val_data_path),
            config=config,
            is_training=False,
            cache_dir=str(Path(structure_cache_dir) / "val") if use_structures else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config.stage1_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collator
        )
        logger.info("✓ 创建验证数据集成功")
    else:
        logger.warning("验证数据集不存在，将跳过验证")
    
    logger.info(f"训练数据集大小: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"验证数据集大小: {len(val_dataset)}")
    
    return train_loader, val_loader


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logger(
        level=logging.DEBUG if args.debug else logging.INFO,
        log_file=Path(args.output_dir) / "training.log"
    )
    
    logger.info("🚀 开始分离式训练")
    logger.info(f"参数: {vars(args)}")
    
    # 设置设备（在加载配置之后）
    
    # 加载配置
    if Path(args.config).exists():
        config = load_config(args.config)
        logger.info(f"加载配置文件: {args.config}")
    else:
        # 使用默认配置
        config = OmegaConf.create({
            "model": {
                "type": "StructDiff",
                "sequence_encoder": {
                    "pretrained_model": "facebook/esm2_t6_8M_UR50D",
                    "freeze_encoder": False
                },
                "denoiser": {
                    "hidden_dim": 768,
                    "num_layers": 12,
                    "num_heads": 12
                }
            },
            "diffusion": {
                "num_timesteps": 1000,
                "noise_schedule": "sqrt",
                "beta_start": 0.0001,
                "beta_end": 0.02
            },
            "data": {
                "max_length": 50,
                "min_length": 5,
                "use_predicted_structures": False
            }
        })
        logger.warning("使用默认配置")
    
    # 设置设备（在配置加载之后）
    device = setup_device(config, args.device)
    
    # 从主配置中提取训练和评估相关的配置
    train_params = config.get('separated_training', {})
    stage1_params = train_params.get('stage1', {})
    stage2_params = train_params.get('stage2', {})
    evaluation_config = config.get('evaluation', {})
    enhancements_config = config.get('training_enhancements', {})

    # 创建训练配置，优先使用YAML中的值，然后才是dataclass的默认值
    training_config = SeparatedTrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_dir=str(Path(args.output_dir) / "checkpoints"),
        
        # 阶段1
        stage1_epochs=stage1_params.get('epochs', 200),
        stage1_lr=stage1_params.get('learning_rate', 1e-4),
        stage1_batch_size=stage1_params.get('batch_size', 32),
        stage1_gradient_clip=stage1_params.get('gradient_clip', 1.0),
        stage1_warmup_steps=stage1_params.get('warmup_steps', 1000),

        # 阶段2
        stage2_epochs=stage2_params.get('epochs', 100),
        stage2_lr=stage2_params.get('learning_rate', 5e-5),
        stage2_batch_size=stage2_params.get('batch_size', 64),
        stage2_gradient_clip=stage2_params.get('gradient_clip', 0.5),
        stage2_warmup_steps=stage2_params.get('warmup_steps', 500),

        # 功能开关 (命令行优先)
        use_cfg=args.use_cfg or config.get('classifier_free_guidance', {}).get('enabled', True),
        use_length_control=args.use_length_control or config.get('length_control', {}).get('enabled', True),
        use_amp=args.use_amp or enhancements_config.get('use_amp', True),
        use_ema=args.use_ema or enhancements_config.get('use_ema', True),
        ema_decay=enhancements_config.get('ema_decay', 0.9999),

        # 评估配置
        enable_evaluation=evaluation_config.get('enabled', True),
        evaluate_every=evaluation_config.get('evaluate_every', 5),
        evaluation_metrics=evaluation_config.get('metrics', None),
        evaluation_num_samples=evaluation_config.get('generation', {}).get('num_samples', 1000),
        evaluation_guidance_scale=evaluation_config.get('generation', {}).get('guidance_scale', 2.0),
        auto_generate_after_training=True  # 默认启用
    )
    
    # 命令行参数覆盖配置
    if args.stage1_epochs:
        training_config.stage1_epochs = args.stage1_epochs
    if args.stage2_epochs:
        training_config.stage2_epochs = args.stage2_epochs
    if args.stage1_lr:
        training_config.stage1_lr = args.stage1_lr
    if args.stage2_lr:
        training_config.stage2_lr = args.stage2_lr
    if args.batch_size:
        training_config.stage1_batch_size = args.batch_size
        training_config.stage2_batch_size = args.batch_size
    
    logger.info(f"训练配置: {training_config}")
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.dry_run:
        logger.info("🔍 干运行模式，不进行实际训练")
        return
    
    try:
        # 创建分词器
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.sequence_encoder.pretrained_model
        )
        logger.info("✓ 创建分词器成功")
        
        # 创建模型和扩散过程
        model, diffusion = create_model_and_diffusion(config)
        model = model.to(device)
        logger.info("✓ 创建模型成功")
        
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(config, training_config, tokenizer)
        logger.info("✓ 创建数据加载器成功")
        
        # 创建训练管理器
        trainer = SeparatedTrainingManager(
            config=training_config,
            model=model,
            diffusion=diffusion,
            device=device,
            tokenizer=tokenizer
        )
        logger.info("✓ 创建训练管理器成功")
        
        # 执行训练
        if args.stage == 'both':
            # 完整的两阶段训练
            logger.info("🎯 开始完整的两阶段训练")
            final_stats = trainer.run_complete_training(train_loader, val_loader)
            
        elif args.stage == '1':
            # 只训练阶段1
            logger.info("🎯 只执行阶段1训练")
            stage1_stats = trainer.train_stage1(train_loader, val_loader)
            final_stats = {'stage1': stage1_stats}
            
        elif args.stage == '2':
            # 只训练阶段2
            if args.stage1_checkpoint:
                logger.info(f"🎯 加载阶段1检查点并执行阶段2训练: {args.stage1_checkpoint}")
                trainer.load_stage1_checkpoint(args.stage1_checkpoint)
            else:
                logger.warning("阶段2训练需要阶段1检查点，但未提供")
                
            stage2_stats = trainer.train_stage2(train_loader, val_loader)
            final_stats = {'stage2': stage2_stats}
        
        # 输出训练摘要
        summary = trainer.get_training_summary()
        logger.info("🎉 训练完成！")
        logger.info("训练摘要:")
        for stage, stats in summary.items():
            logger.info(f"  {stage}:")
            for key, value in stats.items():
                logger.info(f"    {key}: {value}")
        
        # 保存最终摘要
        summary_path = Path(args.output_dir) / "training_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"训练摘要保存到: {summary_path}")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()