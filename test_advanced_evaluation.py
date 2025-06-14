#!/usr/bin/env python3
"""
测试专业生物学评估指标
包括伪困惑度、Shannon熵、不稳定性指数、BLOSUM62相似性等
"""

import os
import sys
import torch
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.train_peptide_esmfold import PeptideEvaluator
from structdiff.models.structdiff import StructDiff
from omegaconf import OmegaConf

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_config():
    """创建测试配置"""
    config = OmegaConf.create({
        'model': {
            'sequence_encoder': {
                'pretrained_model': 'facebook/esm2_t6_8M_UR50D',
                'freeze_encoder': True
            },
            'structure_encoder': {
                'pretrained_model': 'facebook/esmfold_v1',
                'freeze_encoder': True
            },
            'denoiser': {
                'hidden_dim': 256,
                'num_layers': 6,
                'num_heads': 8,
                'dropout': 0.1
            },
            'diffusion': {
                'num_timesteps': 1000,
                'beta_schedule': 'linear',
                'beta_start': 0.0001,
                'beta_end': 0.02
            }
        },
        'data': {
            'max_length': 50,
            'vocab_size': 21  # 20 amino acids + padding
        }
    })
    return config

def test_evaluation_metrics():
    """测试评估指标"""
    logger.info("🧪 开始测试专业生物学评估指标...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建配置和模型
    config = create_test_config()
    
    # 创建简单的模型用于测试
    model = StructDiff(config)
    model.to(device)
    model.eval()
    
    # 创建评估器
    evaluator = PeptideEvaluator(model, config, device)
    
    # 测试序列
    test_sequences = [
        "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",  # 抗菌肽
        "FLPIIAKFFSKVM",  # 抗真菌肽
        "GLLSKLWKKVFKAFKKFLKK",  # 抗病毒肽
        "ACDEFGHIKLMNPQRSTVWY",  # 包含所有氨基酸
        "AAAAAAAAAAAAAAAAAAAA",  # 单一氨基酸
        "KWKLFKKIEKVGQNIR",  # 短序列
        "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAKWKLFKKIEKVGQNIR"  # 长序列
    ]
    
    logger.info(f"测试序列数量: {len(test_sequences)}")
    
    # 1. 测试伪困惑度
    logger.info("\n🧮 测试伪困惑度计算...")
    try:
        pp_results = evaluator.evaluate_pseudo_perplexity(test_sequences)
        logger.info(f"伪困惑度结果: {pp_results}")
    except Exception as e:
        logger.warning(f"伪困惑度测试失败: {e}")
    
    # 2. 测试Shannon信息熵
    logger.info("\n📊 测试Shannon信息熵计算...")
    try:
        entropy_results = evaluator.evaluate_shannon_entropy(test_sequences)
        logger.info(f"Shannon熵结果: {entropy_results}")
    except Exception as e:
        logger.warning(f"Shannon熵测试失败: {e}")
    
    # 3. 测试不稳定性指数
    logger.info("\n🧪 测试不稳定性指数计算...")
    try:
        instability_results = evaluator.evaluate_instability_index(test_sequences)
        logger.info(f"不稳定性指数结果: {instability_results}")
    except Exception as e:
        logger.warning(f"不稳定性指数测试失败: {e}")
    
    # 4. 测试BLOSUM62相似性
    logger.info("\n🔍 测试BLOSUM62相似性计算...")
    try:
        # 创建参考序列
        reference_sequences = [
            "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
            "FLPIIAKFFSKVM",
            "GLLSKLWKKVFKAFKKFLKK"
        ]
        similarity_results = evaluator.evaluate_similarity_to_training(
            test_sequences, reference_sequences
        )
        logger.info(f"BLOSUM62相似性结果: {similarity_results}")
    except Exception as e:
        logger.warning(f"BLOSUM62相似性测试失败: {e}")
    
    # 5. 测试多样性指标
    logger.info("\n📈 测试多样性指标计算...")
    try:
        diversity_results = evaluator.evaluate_diversity_metrics(test_sequences)
        logger.info(f"多样性指标结果: {diversity_results}")
    except Exception as e:
        logger.warning(f"多样性指标测试失败: {e}")
    
    # 6. 测试综合评估
    logger.info("\n🔬 测试综合评估...")
    try:
        # 创建参考序列
        reference_sequences = [
            "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
            "FLPIIAKFFSKVM",
            "GLLSKLWKKVFKAFKKFLKK",
            "ACDEFGHIKLMNPQRSTVWY"
        ]
        
        # 由于我们没有训练好的模型，这里只测试评估部分
        # 直接使用测试序列模拟生成结果
        logger.info("模拟综合评估过程...")
        
        # 手动创建评估结果
        results = {}
        
        # 伪困惑度
        try:
            results['pseudo_perplexity'] = evaluator.evaluate_pseudo_perplexity(test_sequences)
        except:
            results['pseudo_perplexity'] = {'mean_pseudo_perplexity': 0.0, 'std_pseudo_perplexity': 0.0}
        
        # Shannon熵
        try:
            results['shannon_entropy'] = evaluator.evaluate_shannon_entropy(test_sequences)
        except:
            results['shannon_entropy'] = {'mean_sequence_entropy': 0.0, 'overall_entropy': 0.0}
        
        # 不稳定性指数
        try:
            results['instability_index'] = evaluator.evaluate_instability_index(test_sequences)
        except:
            results['instability_index'] = {'mean_instability_index': 0.0, 'std_instability_index': 0.0}
        
        # BLOSUM62相似性
        try:
            results['blosum62_similarity'] = evaluator.evaluate_similarity_to_training(
                test_sequences, reference_sequences
            )
        except:
            results['blosum62_similarity'] = {'mean_similarity_score': 0.0, 'std_similarity_score': 0.0}
        
        # 多样性分析
        try:
            results['diversity_analysis'] = evaluator.evaluate_diversity_metrics(test_sequences)
        except:
            results['diversity_analysis'] = {'uniqueness_ratio': 0.0}
        
        # 有效性
        try:
            results['validity'] = evaluator.evaluate_validity(test_sequences)
        except:
            results['validity'] = {'validity_rate': 0.0}
        
        # 总结
        results['summary'] = {
            'total_generated': len(test_sequences),
            'unique_sequences': len(set(test_sequences)),
            'peptide_type': 'test',
            'generation_success_rate': 1.0
        }
        
        # 打印结果
        logger.info("\n" + "="*80)
        logger.info("🎯 测试评估结果摘要")
        logger.info("="*80)
        
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
        
        logger.info("🎉 专业生物学评估指标测试完成！")
        
    except Exception as e:
        logger.error(f"综合评估测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evaluation_metrics() 