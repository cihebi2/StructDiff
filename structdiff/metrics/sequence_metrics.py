import numpy as np
from typing import List, Dict
from collections import Counter
import re

def compute_sequence_metrics(generated_sequences: List[str], reference_sequences: List[str]) -> Dict[str, float]:
    """
    计算生成序列的评估指标
    
    Args:
        generated_sequences: 生成的序列列表
        reference_sequences: 参考序列列表
        
    Returns:
        包含各种指标的字典
    """
    
    metrics = {}
    
    # 1. 基本序列统计
    metrics.update(_compute_basic_stats(generated_sequences))
    
    # 2. 序列多样性指标
    metrics.update(_compute_diversity_metrics(generated_sequences))
    
    # 3. 与参考序列的相似性指标
    metrics.update(_compute_similarity_metrics(generated_sequences, reference_sequences))
    
    # 4. 生物学相关指标
    metrics.update(_compute_biological_metrics(generated_sequences))
    
    # 5. 语言学指标（n-gram 相似性）
    metrics.update(_compute_linguistic_metrics(generated_sequences, reference_sequences))
    
    return metrics

def _compute_basic_stats(sequences: List[str]) -> Dict[str, float]:
    """计算基本序列统计信息"""
    lengths = [len(seq) for seq in sequences]
    
    return {
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'min_length': np.min(lengths),
        'max_length': np.max(lengths),
        'num_sequences': len(sequences)
    }

def _compute_diversity_metrics(sequences: List[str]) -> Dict[str, float]:
    """计算序列多样性指标"""
    if not sequences:
        return {'diversity': 0.0, 'uniqueness': 0.0}
    
    # 唯一序列比例
    unique_sequences = set(sequences)
    uniqueness = len(unique_sequences) / len(sequences)
    
    # 氨基酸多样性（香农熵）
    all_aas = ''.join(sequences)
    aa_counts = Counter(all_aas)
    total_aas = len(all_aas)
    
    if total_aas == 0:
        diversity = 0.0
    else:
        diversity = -sum((count/total_aas) * np.log2(count/total_aas) for count in aa_counts.values())
    
    # 序列内多样性（平均每个序列的氨基酸种类数）
    intra_diversity = np.mean([len(set(seq)) for seq in sequences])
    
    return {
        'uniqueness': uniqueness,
        'aa_diversity': diversity,
        'intra_seq_diversity': intra_diversity
    }

def _compute_similarity_metrics(generated_sequences: List[str], reference_sequences: List[str]) -> Dict[str, float]:
    """计算与参考序列的相似性指标"""
    if not reference_sequences:
        return {'avg_similarity': 0.0, 'max_similarity': 0.0}
    
    similarities = []
    
    for gen_seq in generated_sequences:
        seq_similarities = []
        for ref_seq in reference_sequences:
            # 计算编辑距离相似性
            similarity = 1 - (levenshtein_distance(gen_seq, ref_seq) / max(len(gen_seq), len(ref_seq)))
            seq_similarities.append(similarity)
        
        # 取与最相似参考序列的相似度
        similarities.append(max(seq_similarities))
    
    return {
        'avg_similarity': np.mean(similarities),
        'max_similarity': np.max(similarities),
        'min_similarity': np.min(similarities)
    }

def _compute_biological_metrics(sequences: List[str]) -> Dict[str, float]:
    """计算生物学相关指标"""
    
    # 氨基酸组成分析
    aa_properties = {
        'hydrophobic': set('AILMFPWYV'),
        'hydrophilic': set('NQST'),
        'positive': set('RHK'),
        'negative': set('DE'),
        'aromatic': set('FWY'),
        'small': set('AGST'),
        'large': set('FWYE')
    }
    
    property_ratios = {}
    
    for prop_name, prop_aas in aa_properties.items():
        ratios = []
        for seq in sequences:
            if len(seq) == 0:
                ratios.append(0.0)
            else:
                ratio = sum(1 for aa in seq if aa in prop_aas) / len(seq)
                ratios.append(ratio)
        property_ratios[f'{prop_name}_ratio'] = np.mean(ratios)
    
    # 重复序列检测
    repeat_scores = []
    for seq in sequences:
        # 检测连续重复氨基酸
        repeat_score = _compute_repeat_score(seq)
        repeat_scores.append(repeat_score)
    
    property_ratios['avg_repeat_score'] = np.mean(repeat_scores)
    
    return property_ratios

def _compute_linguistic_metrics(generated_sequences: List[str], reference_sequences: List[str]) -> Dict[str, float]:
    """计算语言学指标（n-gram 相似性）"""
    
    metrics = {}
    
    for n in [2, 3]:  # 2-gram 和 3-gram
        gen_ngrams = set()
        ref_ngrams = set()
        
        # 收集生成序列的 n-grams
        for seq in generated_sequences:
            for i in range(len(seq) - n + 1):
                gen_ngrams.add(seq[i:i+n])
        
        # 收集参考序列的 n-grams
        for seq in reference_sequences:
            for i in range(len(seq) - n + 1):
                ref_ngrams.add(seq[i:i+n])
        
        # 计算 Jaccard 相似性
        if len(gen_ngrams) == 0 and len(ref_ngrams) == 0:
            jaccard = 1.0
        elif len(gen_ngrams) == 0 or len(ref_ngrams) == 0:
            jaccard = 0.0
        else:
            intersection = len(gen_ngrams & ref_ngrams)
            union = len(gen_ngrams | ref_ngrams)
            jaccard = intersection / union if union > 0 else 0.0
        
        metrics[f'{n}gram_jaccard'] = jaccard
    
    return metrics

def _compute_repeat_score(sequence: str) -> float:
    """计算序列的重复分数（越高表示重复越多）"""
    if len(sequence) <= 1:
        return 0.0
    
    repeat_count = 0
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i-1]:
            repeat_count += 1
    
    return repeat_count / (len(sequence) - 1)

def levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串的编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

# 辅助函数：计算序列的理化性质
def compute_sequence_properties(sequence: str) -> Dict[str, float]:
    """计算单个序列的理化性质"""
    
    # 氨基酸理化性质表
    aa_properties = {
        'A': {'mw': 89.1, 'pI': 6.11, 'hydro': 1.8},
        'C': {'mw': 121.2, 'pI': 5.05, 'hydro': 2.5},
        'D': {'mw': 133.1, 'pI': 2.85, 'hydro': -3.5},
        'E': {'mw': 147.1, 'pI': 4.45, 'hydro': -3.5},
        'F': {'mw': 165.2, 'pI': 5.49, 'hydro': 2.8},
        'G': {'mw': 75.1, 'pI': 6.06, 'hydro': -0.4},
        'H': {'mw': 155.2, 'pI': 7.60, 'hydro': -3.2},
        'I': {'mw': 131.2, 'pI': 6.05, 'hydro': 4.5},
        'K': {'mw': 146.2, 'pI': 9.60, 'hydro': -3.9},
        'L': {'mw': 131.2, 'pI': 6.01, 'hydro': 3.8},
        'M': {'mw': 149.2, 'pI': 5.74, 'hydro': 1.9},
        'N': {'mw': 132.1, 'pI': 5.43, 'hydro': -3.5},
        'P': {'mw': 115.1, 'pI': 6.30, 'hydro': -1.6},
        'Q': {'mw': 146.2, 'pI': 5.65, 'hydro': -3.5},
        'R': {'mw': 174.2, 'pI': 10.76, 'hydro': -4.5},
        'S': {'mw': 105.1, 'pI': 5.68, 'hydro': -0.8},
        'T': {'mw': 119.1, 'pI': 5.60, 'hydro': -0.7},
        'V': {'mw': 117.1, 'pI': 6.00, 'hydro': 4.2},
        'W': {'mw': 204.2, 'pI': 5.89, 'hydro': -0.9},
        'Y': {'mw': 181.2, 'pI': 5.64, 'hydro': -1.3}
    }
    
    if not sequence:
        return {'molecular_weight': 0.0, 'avg_hydrophobicity': 0.0}
    
    # 计算分子量
    mw = sum(aa_properties.get(aa, {'mw': 0})['mw'] for aa in sequence)
    
    # 计算平均疏水性
    hydro_values = [aa_properties.get(aa, {'hydro': 0})['hydro'] for aa in sequence]
    avg_hydro = np.mean(hydro_values) if hydro_values else 0.0
    
    return {
        'molecular_weight': mw,
        'avg_hydrophobicity': avg_hydro
    }
