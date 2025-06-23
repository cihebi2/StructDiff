#!/usr/bin/env python3
"""
多肽生成模型评估套件
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class PeptideEvaluationSuite:
    """多肽生成模型评估套件"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 氨基酸属性
        self.aa_properties = {
            'hydrophobic': set('AILMFWYV'),
            'polar': set('NQST'),
            'charged_positive': set('KRH'),
            'charged_negative': set('DE'),
            'aromatic': set('FWY'),
            'small': set('AGSP'),
            'large': set('FWYH')
        }
        
        # 已知抗菌肽的氨基酸偏好
        self.amp_preferences = {
            'antimicrobial': {
                'preferred': set('KRLWF'),  # 阳离子，疏水
                'avoided': set('DE'),       # 阴离子
                'optimal_length_range': (8, 40),
                'optimal_charge_range': (2, 8)
            },
            'antifungal': {
                'preferred': set('KRFWYC'),
                'avoided': set('DE'),
                'optimal_length_range': (10, 50),
                'optimal_charge_range': (1, 6)
            },
            'antiviral': {
                'preferred': set('KRFWY'),
                'avoided': set('E'),
                'optimal_length_range': (12, 35),
                'optimal_charge_range': (3, 7)
            }
        }
    
    def evaluate_sequence_quality(self, sequences: List[str]) -> Dict:
        """评估序列质量"""
        results = {
            'total_sequences': len(sequences),
            'valid_sequences': 0,
            'average_length': 0,
            'length_distribution': {},
            'amino_acid_composition': {},
            'hydrophobicity_scores': [],
            'charge_scores': [],
            'complexity_scores': []
        }
        
        valid_sequences = []
        valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
        
        for seq in sequences:
            # 检查序列有效性
            if all(aa in valid_aa for aa in seq) and len(seq) > 0:
                valid_sequences.append(seq)
                results['valid_sequences'] += 1
        
        if not valid_sequences:
            return results
        
        # 长度统计
        lengths = [len(seq) for seq in valid_sequences]
        results['average_length'] = np.mean(lengths)
        results['length_distribution'] = dict(Counter(lengths))
        
        # 氨基酸组成
        all_aas = ''.join(valid_sequences)
        aa_counts = Counter(all_aas)
        total_aas = len(all_aas)
        results['amino_acid_composition'] = {
            aa: count/total_aas for aa, count in aa_counts.items()
        }
        
        # 生物化学属性分析
        for seq in valid_sequences:
            # 疏水性得分
            hydrophobic_count = sum(1 for aa in seq if aa in self.aa_properties['hydrophobic'])
            hydrophobicity = hydrophobic_count / len(seq)
            results['hydrophobicity_scores'].append(hydrophobicity)
            
            # 电荷得分
            positive_charge = sum(1 for aa in seq if aa in self.aa_properties['charged_positive'])
            negative_charge = sum(1 for aa in seq if aa in self.aa_properties['charged_negative'])
            net_charge = positive_charge - negative_charge
            results['charge_scores'].append(net_charge)
            
            # 复杂性得分 (基于氨基酸多样性)
            aa_diversity = len(set(seq)) / 20  # 20种氨基酸
            results['complexity_scores'].append(aa_diversity)
        
        return results
    
    def evaluate_condition_specificity(self, sequences_by_condition: Dict[str, List[str]]) -> Dict:
        """评估条件特异性"""
        results = {}
        
        for condition, sequences in sequences_by_condition.items():
            if condition in self.amp_preferences:
                prefs = self.amp_preferences[condition]
                
                condition_results = {
                    'sequences_count': len(sequences),
                    'preferred_aa_ratio': [],
                    'avoided_aa_ratio': [],
                    'length_compliance': 0,
                    'charge_compliance': 0
                }
                
                for seq in sequences:
                    if not seq:
                        continue
                    
                    # 优选氨基酸比例
                    preferred_count = sum(1 for aa in seq if aa in prefs['preferred'])
                    preferred_ratio = preferred_count / len(seq)
                    condition_results['preferred_aa_ratio'].append(preferred_ratio)
                    
                    # 避免氨基酸比例
                    avoided_count = sum(1 for aa in seq if aa in prefs['avoided'])
                    avoided_ratio = avoided_count / len(seq)
                    condition_results['avoided_aa_ratio'].append(avoided_ratio)
                    
                    # 长度合规性
                    min_len, max_len = prefs['optimal_length_range']
                    if min_len <= len(seq) <= max_len:
                        condition_results['length_compliance'] += 1
                    
                    # 电荷合规性
                    net_charge = self._calculate_net_charge(seq)
                    min_charge, max_charge = prefs['optimal_charge_range']
                    if min_charge <= net_charge <= max_charge:
                        condition_results['charge_compliance'] += 1
                
                # 计算合规率
                total_seqs = len(sequences) if sequences else 1
                condition_results['length_compliance'] /= total_seqs
                condition_results['charge_compliance'] /= total_seqs
                
                results[condition] = condition_results
        
        return results
    
    def evaluate_diversity(self, sequences: List[str]) -> Dict:
        """评估序列多样性"""
        if len(sequences) < 2:
            return {'diversity_score': 0, 'unique_sequences': len(sequences)}
        
        # 去重
        unique_sequences = list(set(sequences))
        uniqueness_ratio = len(unique_sequences) / len(sequences)
        
        # 计算编辑距离多样性
        edit_distances = []
        for i in range(min(100, len(unique_sequences))):  # 采样100个序列避免计算量过大
            for j in range(i+1, min(100, len(unique_sequences))):
                distance = self._edit_distance(unique_sequences[i], unique_sequences[j])
                edit_distances.append(distance)
        
        diversity_score = np.mean(edit_distances) if edit_distances else 0
        
        return {
            'diversity_score': diversity_score,
            'unique_sequences': len(unique_sequences),
            'uniqueness_ratio': uniqueness_ratio,
            'average_edit_distance': diversity_score
        }
    
    def evaluate_novelty(self, generated_sequences: List[str], reference_sequences: List[str]) -> Dict:
        """评估序列新颖性"""
        reference_set = set(reference_sequences)
        
        novel_sequences = []
        similar_sequences = []
        
        for seq in generated_sequences:
            if seq in reference_set:
                similar_sequences.append(seq)
            else:
                # 检查是否与已知序列高度相似
                is_similar = False
                for ref_seq in reference_set:
                    if self._sequence_similarity(seq, ref_seq) > 0.8:
                        is_similar = True
                        break
                
                if is_similar:
                    similar_sequences.append(seq)
                else:
                    novel_sequences.append(seq)
        
        total_generated = len(generated_sequences)
        novelty_ratio = len(novel_sequences) / total_generated if total_generated > 0 else 0
        
        return {
            'total_generated': total_generated,
            'novel_sequences': len(novel_sequences),
            'similar_sequences': len(similar_sequences),
            'novelty_ratio': novelty_ratio
        }
    
    def _calculate_net_charge(self, sequence: str) -> int:
        """计算序列净电荷"""
        positive = sum(1 for aa in sequence if aa in self.aa_properties['charged_positive'])
        negative = sum(1 for aa in sequence if aa in self.aa_properties['charged_negative'])
        return positive - negative
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
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
    
    def _sequence_similarity(self, s1: str, s2: str) -> float:
        """计算序列相似性 (基于编辑距离)"""
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        edit_dist = self._edit_distance(s1, s2)
        return 1 - (edit_dist / max_len)
    
    def generate_report(self, evaluation_results: Dict, output_name: str = "evaluation_report"):
        """生成评估报告"""
        report_path = self.output_dir / f"{output_name}.json"
        
        # 保存JSON报告
        with open(report_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # 生成可视化报告
        self._generate_visualizations(evaluation_results, output_name)
        
        # 生成文本摘要
        self._generate_text_summary(evaluation_results, output_name)
        
        print(f"📊 评估报告已生成: {report_path}")
    
    def _generate_visualizations(self, results: Dict, output_name: str):
        """生成可视化图表"""
        try:
            # 序列长度分布
            if 'sequence_quality' in results and 'length_distribution' in results['sequence_quality']:
                plt.figure(figsize=(10, 6))
                length_dist = results['sequence_quality']['length_distribution']
                lengths = list(length_dist.keys())
                counts = list(length_dist.values())
                
                plt.bar(lengths, counts)
                plt.xlabel('序列长度')
                plt.ylabel('序列数量')
                plt.title('生成序列长度分布')
                plt.savefig(self.output_dir / f"{output_name}_length_distribution.png")
                plt.close()
            
            # 氨基酸组成
            if 'sequence_quality' in results and 'amino_acid_composition' in results['sequence_quality']:
                plt.figure(figsize=(12, 6))
                aa_comp = results['sequence_quality']['amino_acid_composition']
                aas = list(aa_comp.keys())
                freqs = list(aa_comp.values())
                
                plt.bar(aas, freqs)
                plt.xlabel('氨基酸')
                plt.ylabel('频率')
                plt.title('氨基酸组成分布')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.output_dir / f"{output_name}_aa_composition.png")
                plt.close()
            
        except Exception as e:
            print(f"⚠️ 可视化生成失败: {e}")
    
    def _generate_text_summary(self, results: Dict, output_name: str):
        """生成文本摘要"""
        summary_path = self.output_dir / f"{output_name}_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("多肽生成模型评估摘要\n")
            f.write("=" * 40 + "\n\n")
            
            # 序列质量摘要
            if 'sequence_quality' in results:
                sq = results['sequence_quality']
                f.write("序列质量评估:\n")
                f.write(f"  总序列数: {sq.get('total_sequences', 0)}\n")
                f.write(f"  有效序列数: {sq.get('valid_sequences', 0)}\n")
                f.write(f"  平均长度: {sq.get('average_length', 0):.1f}\n")
                
                if 'hydrophobicity_scores' in sq and sq['hydrophobicity_scores']:
                    avg_hydro = np.mean(sq['hydrophobicity_scores'])
                    f.write(f"  平均疏水性: {avg_hydro:.3f}\n")
                
                if 'charge_scores' in sq and sq['charge_scores']:
                    avg_charge = np.mean(sq['charge_scores'])
                    f.write(f"  平均净电荷: {avg_charge:.1f}\n")
                f.write("\n")
            
            # 条件特异性摘要
            if 'condition_specificity' in results:
                f.write("条件特异性评估:\n")
                for condition, data in results['condition_specificity'].items():
                    f.write(f"  {condition}:\n")
                    f.write(f"    序列数: {data.get('sequences_count', 0)}\n")
                    f.write(f"    长度合规率: {data.get('length_compliance', 0):.3f}\n")
                    f.write(f"    电荷合规率: {data.get('charge_compliance', 0):.3f}\n")
                f.write("\n")
            
            # 多样性摘要
            if 'diversity' in results:
                div = results['diversity']
                f.write("多样性评估:\n")
                f.write(f"  独特序列比例: {div.get('uniqueness_ratio', 0):.3f}\n")
                f.write(f"  多样性得分: {div.get('diversity_score', 0):.3f}\n")
                f.write("\n")
            
            # 新颖性摘要
            if 'novelty' in results:
                nov = results['novelty']
                f.write("新颖性评估:\n")
                f.write(f"  新颖序列比例: {nov.get('novelty_ratio', 0):.3f}\n")
                f.write(f"  新颖序列数: {nov.get('novel_sequences', 0)}\n")


def main():
    """主函数 - 示例用法"""
    evaluator = PeptideEvaluationSuite()
    
    # 示例数据
    generated_sequences = [
        "KRWWKWIRWKK",
        "FRLKWFKRLLK", 
        "KLRFKKLRWFK",
        "GILDTILKILR",
        "KLAKLRWKLKL"
    ]
    
    sequences_by_condition = {
        "antimicrobial": ["KRWWKWIRWKK", "FRLKWFKRLLK"],
        "antifungal": ["KLRFKKLRWFK", "GILDTILKILR"], 
        "antiviral": ["KLAKLRWKLKL"]
    }
    
    reference_sequences = [
        "KRWWKWIRWKK",  # 已知序列
        "MAGAININ1PEPTIDE"
    ]
    
    # 运行评估
    results = {}
    results['sequence_quality'] = evaluator.evaluate_sequence_quality(generated_sequences)
    results['condition_specificity'] = evaluator.evaluate_condition_specificity(sequences_by_condition)
    results['diversity'] = evaluator.evaluate_diversity(generated_sequences)
    results['novelty'] = evaluator.evaluate_novelty(generated_sequences, reference_sequences)
    
    # 生成报告
    evaluator.generate_report(results, "demo_evaluation")
    
    print("✅ 评估完成!")


if __name__ == "__main__":
    main()