"""
外部分类器接口
提供多肽活性预测的基础框架
"""

import torch
import numpy as np
from typing import List, Dict
import random
from ..utils.logger import get_logger

logger = get_logger(__name__)

class SimpleActivityClassifier:
    """
    简单的活性分类器
    基于序列特征进行基础的活性预测
    """
    
    def __init__(self, peptide_type: str):
        self.peptide_type = peptide_type
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        # 不同肽类型的特征权重（基于文献）
        self.feature_weights = {
            'antimicrobial': {
                'K': 0.8, 'R': 0.8, 'H': 0.6,  # 阳离子氨基酸
                'F': 0.4, 'W': 0.5, 'Y': 0.3,  # 芳香族氨基酸
                'L': 0.3, 'I': 0.3, 'V': 0.2,  # 疏水氨基酸
                'length_optimal': (10, 30),     # 最佳长度范围
                'charge_threshold': 2.0         # 净电荷阈值
            },
            'antifungal': {
                'F': 0.7, 'W': 0.8, 'Y': 0.6,  # 芳香族氨基酸更重要
                'K': 0.6, 'R': 0.6, 'H': 0.4,  # 阳离子氨基酸
                'C': 0.5,                       # 半胱氨酸（二硫键）
                'length_optimal': (15, 40),
                'hydrophobicity_threshold': 0.3
            },
            'antiviral': {
                'W': 0.9, 'F': 0.7, 'Y': 0.8,  # 芳香族氨基酸最重要
                'K': 0.5, 'R': 0.5,            # 阳离子氨基酸
                'G': 0.3, 'P': 0.4,            # 结构灵活性
                'length_optimal': (20, 50),
                'aromaticity_threshold': 0.15
            }
        }
    
    def predict_activity(self, sequences: List[str]) -> Dict:
        """
        预测序列活性
        
        Args:
            sequences: 氨基酸序列列表
            
        Returns:
            包含预测结果的字典
        """
        logger.info(f"🎯 使用简单分类器预测 {self.peptide_type} 活性...")
        
        if self.peptide_type not in self.feature_weights:
            logger.warning(f"未知肽类型: {self.peptide_type}")
            return self._get_random_predictions(sequences)
        
        weights = self.feature_weights[self.peptide_type]
        active_count = 0
        predictions = []
        
        for seq in sequences:
            score = self._calculate_activity_score(seq, weights)
            is_active = score > 0.5  # 阈值
            predictions.append(is_active)
            if is_active:
                active_count += 1
        
        return {
            'predicted_active_ratio': active_count / len(sequences),
            'total_sequences': len(sequences),
            'predicted_active': active_count,
            'predicted_inactive': len(sequences) - active_count,
            'classifier_type': f'{self.peptide_type}_simple_classifier',
            'predictions': predictions
        }
    
    def _calculate_activity_score(self, sequence: str, weights: Dict) -> float:
        """计算活性得分"""
        score = 0.0
        seq_len = len(sequence)
        
        if seq_len == 0:
            return 0.0
        
        # 1. 氨基酸组成得分
        aa_counts = {aa: sequence.count(aa) for aa in self.amino_acids}
        for aa, weight in weights.items():
            if aa in self.amino_acids:
                frequency = aa_counts[aa] / seq_len
                score += frequency * weight
        
        # 2. 长度得分
        if 'length_optimal' in weights:
            min_len, max_len = weights['length_optimal']
            if min_len <= seq_len <= max_len:
                score += 0.2
            else:
                # 长度偏离惩罚
                deviation = min(abs(seq_len - min_len), abs(seq_len - max_len))
                score -= deviation * 0.01
        
        # 3. 特定属性得分
        if self.peptide_type == 'antimicrobial':
            # 净电荷
            charge = self._calculate_net_charge(sequence)
            if charge >= weights.get('charge_threshold', 2.0):
                score += 0.3
        
        elif self.peptide_type == 'antifungal':
            # 疏水性
            hydrophobicity = self._calculate_hydrophobicity(sequence)
            if hydrophobicity >= weights.get('hydrophobicity_threshold', 0.3):
                score += 0.2
        
        elif self.peptide_type == 'antiviral':
            # 芳香性
            aromaticity = self._calculate_aromaticity(sequence)
            if aromaticity >= weights.get('aromaticity_threshold', 0.15):
                score += 0.3
        
        # 归一化到 [0, 1]
        return min(max(score, 0.0), 1.0)
    
    def _calculate_net_charge(self, sequence: str) -> float:
        """计算净电荷（pH=7.4）"""
        positive = sequence.count('K') + sequence.count('R') + sequence.count('H') * 0.1
        negative = sequence.count('D') + sequence.count('E')
        return positive - negative
    
    def _calculate_hydrophobicity(self, sequence: str) -> float:
        """计算疏水性（Eisenberg scale）"""
        hydrophobic_scale = {
            'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
            'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
            'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
            'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
        }
        
        if not sequence:
            return 0.0
        
        total_hydrophobicity = sum(hydrophobic_scale.get(aa, 0.0) for aa in sequence)
        return total_hydrophobicity / len(sequence)
    
    def _calculate_aromaticity(self, sequence: str) -> float:
        """计算芳香性（Phe, Trp, Tyr含量）"""
        if not sequence:
            return 0.0
        
        aromatic_count = sequence.count('F') + sequence.count('W') + sequence.count('Y')
        return aromatic_count / len(sequence)
    
    def _get_random_predictions(self, sequences: List[str]) -> Dict:
        """随机预测（备用方案）"""
        # 基于序列长度的简单启发式
        active_count = 0
        predictions = []
        
        for seq in sequences:
            # 简单规则：长度在合理范围内的序列更可能有活性
            is_active = 10 <= len(seq) <= 50 and random.random() > 0.7
            predictions.append(is_active)
            if is_active:
                active_count += 1
        
        return {
            'predicted_active_ratio': active_count / len(sequences),
            'total_sequences': len(sequences),
            'predicted_active': active_count,
            'predicted_inactive': len(sequences) - active_count,
            'classifier_type': f'{self.peptide_type}_random_classifier',
            'predictions': predictions
        }


def get_activity_classifier(peptide_type: str) -> SimpleActivityClassifier:
    """
    获取指定类型的活性分类器
    
    Args:
        peptide_type: 肽类型 ('antimicrobial', 'antifungal', 'antiviral')
        
    Returns:
        活性分类器实例
    """
    return SimpleActivityClassifier(peptide_type) 