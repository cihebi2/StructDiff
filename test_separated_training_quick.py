#!/usr/bin/env python3
"""
快速测试分离式训练功能
使用最小配置验证基本训练流程
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf

# 设置环境
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 使用GPU 2

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from structdiff.training.separated_training import SeparatedTrainingManager, SeparatedTrainingConfig
from structdiff.models.structdiff import StructDiff
from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
from structdiff.data.dataset import PeptideStructureDataset
from structdiff.training.length_controller import LengthAwareDataCollator
from structdiff.utils.logger import setup_logger, get_logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

setup_logger(level="INFO")
logger = get_logger(__name__)

def create_minimal_config():
    """创建最小配置"""
    return OmegaConf.create({
        "model": {
            "type": "StructDiff",
            "sequence_encoder": {
                "pretrained_model": "facebook/esm2_t6_8M_UR50D",
                "freeze_encoder": False,
                "use_lora": False  # 禁用LoRA简化
            },
            "structure_encoder": {
                "type": "multi_scale",
                "hidden_dim": 128,  # 减小维度
                "use_esmfold": False  # 禁用ESMFold
            },
            "denoiser": {
                "hidden_dim": 320,
                "num_layers": 4,    # 减少层数
                "num_heads": 4,     # 减少头数
                "dropout": 0.1,
                "use_cross_attention": False  # 简化交叉注意力
            },
            "sequence_decoder": {
                "hidden_dim": 320,
                "num_layers": 2,    # 减少解码器层数
                "vocab_size": 33,
                "dropout": 0.1
            }
        },
        "diffusion": {
            "num_timesteps": 100,  # 减少时间步
            "noise_schedule": "sqrt",
            "beta_start": 0.0001,
            "beta_end": 0.02
        }
    })

def create_minimal_training_config():
    """创建最小训练配置"""
    return SeparatedTrainingConfig(
        # 阶段1配置
        stage1_epochs=2,           # 很少的epochs用于测试
        stage1_lr=1e-4,
        stage1_batch_size=2,       # 最小批次
        stage1_gradient_clip=1.0,
        stage1_warmup_steps=10,
        
        # 阶段2配置
        stage2_epochs=1,
        stage2_lr=5e-5,
        stage2_batch_size=2,
        stage2_gradient_clip=0.5,
        stage2_warmup_steps=5,
        
        # 其他配置
        use_amp=False,             # 禁用混合精度避免复杂性
        use_ema=False,             # 禁用EMA
        save_every=1000,           # 不保存检查点
        validate_every=50,         # 减少验证频率
        log_every=5,               # 增加日志频率
        
        # 禁用高级功能
        use_length_control=False,
        use_cfg=False,
        
        # 路径
        data_dir="./data/processed",
        output_dir="./outputs/test_separated",
        checkpoint_dir="./outputs/test_separated/checkpoints",
        
        # 禁用评估
        enable_evaluation=False
    )

def create_simple_data_loader(config, training_config, tokenizer, split='train'):
    """创建简化的数据加载器"""
    dataset = PeptideStructureDataset(
        data_path=f"./data/processed/{split}.csv",
        config=config,
        is_training=(split == 'train')
    )
    
    # 简化的collator，不使用长度控制
    def simple_collate_fn(batch):
        sequences = []
        attention_masks = []
        
        for item in batch:
            sequences.append(item['sequences'])
            attention_masks.append(item['attention_mask'])
        
        return {
            'sequences': torch.stack(sequences),
            'attention_mask': torch.stack(attention_masks),
            'structures': None,  # 简化，不使用结构特征
            'conditions': None   # 简化，不使用条件
        }
    
    data_loader = DataLoader(
        dataset,
        batch_size=training_config.stage1_batch_size,
        shuffle=(split == 'train'),
        num_workers=0,  # 禁用多进程
        collate_fn=simple_collate_fn
    )
    
    return data_loader

def test_separated_training():
    """测试分离式训练"""
    logger.info("🚀 开始快速分离式训练测试")
    
    # 检查GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    try:
        # 创建配置
        config = create_minimal_config()
        training_config = create_minimal_training_config()
        
        logger.info("✓ 配置创建成功")
        
        # 创建分词器
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.sequence_encoder.pretrained_model
        )
        logger.info("✓ 分词器创建成功")
        
        # 创建模型
        model = StructDiff(config.model)
        model = model.to(device)
        logger.info(f"✓ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建扩散过程
        diffusion = GaussianDiffusion(
            num_timesteps=config.diffusion.num_timesteps,
            noise_schedule=config.diffusion.noise_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end
        )
        logger.info("✓ 扩散过程创建成功")
        
        # 创建数据加载器
        train_loader = create_simple_data_loader(config, training_config, tokenizer, 'train')
        val_loader = create_simple_data_loader(config, training_config, tokenizer, 'val')
        logger.info(f"✓ 数据加载器创建成功，训练样本数: {len(train_loader.dataset)}")
        
        # 创建训练管理器
        trainer = SeparatedTrainingManager(
            config=training_config,
            model=model,
            diffusion=diffusion,
            device=str(device),
            tokenizer=tokenizer
        )
        logger.info("✓ 训练管理器创建成功")
        
        # 测试一个训练批次
        logger.info("🧪 测试单个训练批次...")
        
        # 准备阶段1组件
        model, optimizer, scheduler = trainer.prepare_stage1_components()
        
        # 获取一个批次
        train_iter = iter(train_loader)
        batch = next(train_iter)
        
        # 执行一个训练步骤
        step_stats = trainer.stage1_training_step(batch, model, optimizer)
        logger.info(f"✓ 阶段1训练步骤成功，损失: {step_stats['loss']:.4f}")
        
        # 测试验证
        val_stats = trainer.validate_stage1(val_loader, model)
        logger.info(f"✓ 阶段1验证成功，验证损失: {val_stats['val_loss']:.4f}")
        
        # 快速测试阶段2准备
        logger.info("🧪 测试阶段2组件准备...")
        model, optimizer2, scheduler2 = trainer.prepare_stage2_components()
        logger.info("✓ 阶段2组件准备成功")
        
        # 如果一切正常，运行少量epochs的完整训练
        logger.info("🚀 开始快速完整训练测试...")
        final_stats = trainer.run_complete_training(train_loader, val_loader)
        
        logger.info("🎉 分离式训练测试完成！")
        logger.info("训练统计:")
        for stage, stats in final_stats.items():
            logger.info(f"  {stage}:")
            if 'losses' in stats and stats['losses']:
                logger.info(f"    最终损失: {stats['losses'][-1]:.4f}")
            if 'val_losses' in stats and stats['val_losses']:
                logger.info(f"    最终验证损失: {stats['val_losses'][-1]:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_separated_training()
    if success:
        logger.info("✅ 分离式训练功能验证成功！")
        print("\n" + "="*60)
        print("🎉 分离式训练系统已就绪！")
        print("💡 现在可以使用完整配置进行生产训练")
        print("="*60)
    else:
        logger.error("❌ 分离式训练功能验证失败")
        sys.exit(1) 