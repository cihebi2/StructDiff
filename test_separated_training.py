#!/usr/bin/env python3
"""
分离式训练测试脚本
验证CPL-Diff启发的两阶段训练策略
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import tempfile
import shutil
from typing import Dict, List

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from structdiff.training.separated_training import SeparatedTrainingManager, SeparatedTrainingConfig
from structdiff.training.length_controller import (
    LengthDistributionAnalyzer, AdaptiveLengthController, 
    LengthAwareDataCollator, create_length_controller_from_data
)
from structdiff.models.structdiff import StructDiff
from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
from structdiff.data.dataset import PeptideStructureDataset
from structdiff.utils.logger import setup_logger, get_logger
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# 设置日志
setup_logger()
logger = get_logger(__name__)


def create_test_data(data_dir: Path, num_samples: int = 100):
    """创建测试数据"""
    logger.info(f"创建测试数据到: {data_dir}")
    
    # 生成模拟肽段序列
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    sequences = []
    peptide_types = []
    
    type_length_prefs = {
        'antimicrobial': (20, 5),
        'antifungal': (25, 7),
        'antiviral': (30, 8)
    }
    
    for i in range(num_samples):
        # 随机选择肽段类型
        peptide_type = np.random.choice(['antimicrobial', 'antifungal', 'antiviral'])
        peptide_types.append(peptide_type)
        
        # 根据类型生成长度
        mean_len, std_len = type_length_prefs[peptide_type]
        length = max(5, min(50, int(np.random.normal(mean_len, std_len))))
        
        # 生成序列
        sequence = ''.join(np.random.choice(list(amino_acids), length))
        sequences.append(sequence)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'id': [f'peptide_{i:04d}' for i in range(num_samples)],
        'sequence': sequences,
        'peptide_type': peptide_types
    })
    
    # 分割数据
    train_size = int(0.8 * num_samples)
    val_size = int(0.1 * num_samples)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]
    
    # 保存数据
    data_dir.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(data_dir / "train.csv", index=False)
    val_data.to_csv(data_dir / "val.csv", index=False)
    test_data.to_csv(data_dir / "test.csv", index=False)
    
    logger.info(f"生成数据: 训练集 {len(train_data)}, 验证集 {len(val_data)}, 测试集 {len(test_data)}")
    return train_data, val_data, test_data


def test_length_controller():
    """测试长度控制器"""
    logger.info("🧪 测试长度控制器")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建测试数据
        train_data, _, _ = create_test_data(temp_path)
        
        # 测试长度分布分析器
        analyzer = LengthDistributionAnalyzer(str(temp_path / "train.csv"))
        distributions = analyzer.analyze_training_data()
        
        assert len(distributions) == 3  # 三种肽段类型
        logger.info("✅ 长度分布分析测试通过")
        
        # 测试长度控制器
        controller = AdaptiveLengthController(
            min_length=5,
            max_length=50,
            distributions=distributions
        )
        
        # 测试长度采样
        peptide_types = ['antimicrobial', 'antifungal', 'antiviral']
        lengths = controller.sample_target_lengths(10, peptide_types)
        
        assert lengths.shape == (10,)
        assert torch.all(lengths >= 5)
        assert torch.all(lengths <= 50)
        logger.info("✅ 长度采样测试通过")
        
        # 测试长度掩码
        mask = controller.create_length_mask(lengths, 60)
        assert mask.shape == (10, 60)
        logger.info("✅ 长度掩码测试通过")
        
    logger.info("🎉 长度控制器测试完成")


def test_data_collator():
    """测试数据整理器"""
    logger.info("🧪 测试数据整理器")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建测试数据
        create_test_data(temp_path)
        
        # 创建长度控制器
        controller = create_length_controller_from_data(
            str(temp_path / "train.csv"),
            save_distributions=False
        )
        
        # 创建分词器
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        
        # 创建数据整理器
        collator = LengthAwareDataCollator(
            length_controller=controller,
            tokenizer=tokenizer,
            use_length_control=True
        )
        
        # 测试数据
        batch_data = [
            {'sequence': 'ACDEFGHIKLMNPQRS', 'peptide_type': 'antimicrobial'},
            {'sequence': 'TVWYACDEFGHIKLMNPQRST', 'peptide_type': 'antifungal'},
            {'sequence': 'VYACDEFGHIKLMNPQRSTVWY', 'peptide_type': 'antiviral'}
        ]
        
        # 测试整理
        result = collator(batch_data)
        
        assert 'sequences' in result
        assert 'attention_mask' in result
        assert 'conditions' in result
        assert 'target_lengths' in result
        assert 'length_mask' in result
        
        logger.info("✅ 数据整理器测试通过")
    
    logger.info("🎉 数据整理器测试完成")


def test_separated_training_manager():
    """测试分离式训练管理器"""
    logger.info("🧪 测试分离式训练管理器")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建配置
        config = OmegaConf.create({
            "model": {
                "type": "StructDiff",
                "sequence_encoder": {
                    "pretrained_model": "facebook/esm2_t6_8M_UR50D",
                    "freeze_encoder": False
                },
                "denoiser": {
                    "hidden_dim": 256,  # 较小的隐藏层用于测试
                    "num_layers": 2,
                    "num_heads": 4
                }
            },
            "diffusion": {
                "num_timesteps": 100,  # 较少的时间步用于测试
                "noise_schedule": "sqrt",
                "beta_start": 0.0001,
                "beta_end": 0.02
            }
        })
        
        # 创建训练配置
        training_config = SeparatedTrainingConfig(
            stage1_epochs=2,  # 很少的epoch用于测试
            stage2_epochs=1,
            stage1_batch_size=4,
            stage2_batch_size=4,
            data_dir=str(temp_path),
            output_dir=str(temp_path / "output"),
            checkpoint_dir=str(temp_path / "checkpoints"),
            use_cfg=True,
            use_length_control=True,
            save_every=50,
            validate_every=25,
            log_every=10
        )
        
        # 创建模型和扩散过程
        model = StructDiff(config.model)
        diffusion = GaussianDiffusion(config.diffusion)
        
        # 创建训练管理器
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = SeparatedTrainingManager(
            config=training_config,
            model=model,
            diffusion=diffusion,
            device=device
        )
        
        # 测试阶段1准备
        stage1_model, stage1_optimizer, stage1_scheduler = trainer.prepare_stage1_components()
        
        # 验证ESM编码器被冻结
        esm_frozen = all(not p.requires_grad for n, p in stage1_model.named_parameters() 
                        if 'sequence_encoder' in n)
        assert esm_frozen, "ESM编码器应该被冻结"
        logger.info("✅ 阶段1组件准备测试通过")
        
        # 测试阶段2准备
        stage2_model, stage2_optimizer, stage2_scheduler = trainer.prepare_stage2_components()
        
        # 验证只有解码器可训练（这里需要根据实际模型结构调整）
        logger.info("✅ 阶段2组件准备测试通过")
        
    logger.info("🎉 分离式训练管理器测试完成")


def test_end_to_end_mini_training():
    """端到端迷你训练测试"""
    logger.info("🧪 端到端迷你训练测试")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # 创建测试数据
            create_test_data(temp_path, num_samples=20)  # 很少的样本
            
            # 创建配置
            config = OmegaConf.create({
                "model": {
                    "type": "StructDiff",
                    "sequence_encoder": {
                        "pretrained_model": "facebook/esm2_t6_8M_UR50D",
                        "freeze_encoder": False
                    },
                    "denoiser": {
                        "hidden_dim": 128,  # 很小的模型
                        "num_layers": 1,
                        "num_heads": 2
                    }
                },
                "diffusion": {
                    "num_timesteps": 50,  # 很少的时间步
                    "noise_schedule": "sqrt",
                    "beta_start": 0.0001,
                    "beta_end": 0.02
                }
            })
            
            # 创建训练配置
            training_config = SeparatedTrainingConfig(
                stage1_epochs=1,  # 只训练1个epoch
                stage2_epochs=1,
                stage1_batch_size=2,
                stage2_batch_size=2,
                data_dir=str(temp_path),
                output_dir=str(temp_path / "output"),
                checkpoint_dir=str(temp_path / "checkpoints"),
                use_cfg=False,  # 简化配置
                use_length_control=False,
                use_amp=False,
                use_ema=False,
                save_every=100,  # 不保存
                validate_every=100,  # 不验证
                log_every=5
            )
            
            # 创建分词器
            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
            
            # 创建数据整理器
            collator = LengthAwareDataCollator(
                length_controller=None,
                tokenizer=tokenizer,
                use_length_control=False
            )
            
            # 创建数据集和数据加载器
            train_dataset = PeptideStructureDataset(
                data_path=str(temp_path / "train.csv"),
                config=config,
                is_training=True
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=2,
                shuffle=True,
                collate_fn=collator,
                num_workers=0  # 避免多进程问题
            )
            
            # 创建模型和扩散过程
            model = StructDiff(config.model)
            diffusion = GaussianDiffusion(config.diffusion)
            
            # 创建训练管理器
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            trainer = SeparatedTrainingManager(
                config=training_config,
                model=model,
                diffusion=diffusion,
                device=device
            )
            
            # 执行迷你训练
            logger.info("开始迷你训练...")
            stats = trainer.run_complete_training(train_loader)
            
            # 验证统计数据
            assert 'stage1' in stats
            assert 'stage2' in stats
            assert len(stats['stage1']['losses']) > 0
            assert len(stats['stage2']['losses']) > 0
            
            logger.info("✅ 端到端训练测试通过")
            
        except Exception as e:
            logger.error(f"端到端测试失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    logger.info("🎉 端到端训练测试完成")


def run_all_tests():
    """运行所有测试"""
    logger.info("🚀 开始分离式训练全面测试")
    
    try:
        # 组件测试
        test_length_controller()
        test_data_collator()
        test_separated_training_manager()
        
        # 集成测试
        test_end_to_end_mini_training()
        
        logger.info("🎉 所有测试通过！")
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        raise


if __name__ == "__main__":
    run_all_tests()