#!/usr/bin/env python3
"""
测试生成和验证功能的简化脚本
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import json
from collections import Counter
import math
from statistics import mean, stdev
import random

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from structdiff.models.structdiff import StructDiff
from structdiff.models.esmfold_wrapper import ESMFoldWrapper
from structdiff.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


class SimplePeptideEvaluator:
    """简化的多肽生成评估器"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        # 肽类型映射
        self.peptide_type_map = {
            'antimicrobial': 0,
            'antifungal': 1,
            'antiviral': 2
        }
    
    def generate_sequences(self, peptide_type='antimicrobial', sample_num=50, max_length=30):
        """生成多肽序列（简化版本）"""
        logger.info(f"🧬 生成 {sample_num} 条 {peptide_type} 序列...")
        
        sequences = []
        
        # 简化的生成过程 - 随机生成序列作为演示
        for i in range(sample_num):
            # 随机长度
            length = random.randint(10, max_length)
            
            # 根据肽类型调整氨基酸偏好
            if peptide_type == 'antimicrobial':
                # 抗菌肽通常富含阳离子氨基酸
                weighted_aa = 'KKRRHHACDEFGHIKLMNPQRSTVWY'
            elif peptide_type == 'antifungal':
                # 抗真菌肽通常富含疏水性氨基酸
                weighted_aa = 'AILVFWACDEFGHIKLMNPQRSTVWY'
            else:  # antiviral
                # 抗病毒肽通常富含芳香族氨基酸
                weighted_aa = 'FYWACDEFGHIKLMNPQRSTVWY'
            
            # 生成序列
            sequence = ''.join(random.choices(weighted_aa, k=length))
            sequences.append(sequence)
        
        logger.info(f"✅ 成功生成 {len(sequences)} 条序列")
        return sequences
    
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
    
    def comprehensive_evaluation(self, peptide_type='antimicrobial', sample_num=50):
        """综合评估"""
        logger.info(f"🔬 开始 {peptide_type} 多肽综合评估...")
        
        # 生成序列
        sequences = self.generate_sequences(peptide_type, sample_num)
        
        if not sequences:
            logger.warning("⚠️ 未生成任何有效序列")
            return {}, []
        
        # 计算各项指标
        results = {}
        
        # 多样性指标
        diversity_metrics = self.evaluate_diversity(sequences)
        results['diversity'] = diversity_metrics
        
        # 长度分布
        length_metrics = self.evaluate_length_distribution(sequences)
        results['length_distribution'] = length_metrics
        
        # 氨基酸组成
        composition_metrics = self.evaluate_amino_acid_composition(sequences)
        results['amino_acid_composition'] = composition_metrics
        
        # 有效性
        validity_metrics = self.evaluate_validity(sequences)
        results['validity'] = validity_metrics
        
        # 去重
        unique_sequences = list(set(sequences))
        results['final_stats'] = {
            'total_generated': len(sequences),
            'unique_sequences': len(unique_sequences),
            'peptide_type': peptide_type
        }
        
        logger.info(f"✅ {peptide_type} 评估完成")
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
    """主函数"""
    setup_logger()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建评估器
    evaluator = SimplePeptideEvaluator(device)
    
    # 创建输出目录
    output_dir = Path("./test_generation_results")
    output_dir.mkdir(exist_ok=True)
    
    # 对每种肽类型进行评估
    peptide_types = ['antimicrobial', 'antifungal', 'antiviral']
    all_results = {}
    
    for peptide_type in peptide_types:
        logger.info(f"🧬 开始评估 {peptide_type} 多肽...")
        
        try:
            results, sequences = evaluator.comprehensive_evaluation(
                peptide_type=peptide_type,
                sample_num=50  # 生成50个样本进行测试
            )
            
            all_results[peptide_type] = results
            
            # 保存生成的序列
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
    logger.info("\n" + "="*60)
    logger.info("🎯 生成评估结果摘要")
    logger.info("="*60)
    
    for peptide_type, results in all_results.items():
        logger.info(f"\n📊 {peptide_type.upper()} 多肽:")
        for metric, values in results.items():
            if isinstance(values, dict) and 'mean' in values:
                logger.info(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f}")
            elif isinstance(values, dict):
                for sub_metric, sub_value in values.items():
                    if isinstance(sub_value, float):
                        logger.info(f"  {metric}.{sub_metric}: {sub_value:.4f}")
                    else:
                        logger.info(f"  {metric}.{sub_metric}: {sub_value}")
            else:
                logger.info(f"  {metric}: {values}")
    
    logger.info(f"\n📁 详细结果保存到: {results_file}")
    logger.info("🎉 生成和验证测试完成！")


if __name__ == "__main__":
    main()