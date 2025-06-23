#!/usr/bin/env python3
"""
轻量级多肽生成模型评估套件 - 集成CPL-Diff评估指标
仅使用Python标准库，对外部依赖进行优雅降级处理
"""

import os
import sys
import json
import math
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, stdev
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class LightweightPeptideEvaluationSuite:
    """
    轻量级多肽生成模型评估套件
    集成CPL-Diff关键评估指标，对依赖进行优雅降级
    """
    
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
        
        # 多肽类型偏好
        self.amp_preferences = {
            'antimicrobial': {
                'preferred': set('KRLWF'),
                'avoided': set('DE'),
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
        
        # 检查可用的外部库
        self.available_libs = self._check_dependencies()
        
        print("🔬 轻量级评估套件初始化完成")
        print(f"📦 可用外部库: {list(self.available_libs.keys())}")
    
    def _check_dependencies(self) -> Dict[str, bool]:
        """检查外部依赖的可用性"""
        deps = {}
        
        # 检查各种依赖
        libraries = [
            'transformers', 'torch', 'Bio', 'scipy', 
            'matplotlib', 'seaborn', 'modlamp', 'numpy', 'pandas'
        ]
        
        for lib in libraries:
            try:
                __import__(lib)
                deps[lib] = True
            except ImportError:
                deps[lib] = False
        
        return deps
    
    def evaluate_information_entropy(self, sequences: List[str]) -> Dict:
        """
        评估信息熵 (Information Entropy) - 纯Python实现
        衡量序列的氨基酸组成多样性
        """
        print("📊 计算信息熵...")
        
        entropies = []
        for seq in sequences:
            if not seq:
                continue
                
            # 统计氨基酸频率
            aa_counts = Counter(seq)
            length = len(seq)
            
            if length == 0:
                continue
            
            # 计算香农熵
            entropy = 0
            for count in aa_counts.values():
                prob = count / length
                if prob > 0:  # 避免log(0)
                    entropy -= prob * math.log2(prob)
            
            entropies.append(entropy)
        
        if entropies:
            result = {
                'mean_entropy': mean(entropies),
                'std_entropy': stdev(entropies) if len(entropies) > 1 else 0.0,
                'min_entropy': min(entropies),
                'max_entropy': max(entropies),
                'entropy_distribution': entropies[:100]  # 保存前100个用于分析
            }
            print(f"✅ 信息熵计算完成: {result['mean_entropy']:.3f}±{result['std_entropy']:.3f}")
            return result
        else:
            return {
                'mean_entropy': float('nan'),
                'std_entropy': float('nan'),
                'min_entropy': float('nan'),
                'max_entropy': float('nan'),
                'error': 'No valid sequences'
            }
    
    def evaluate_pseudo_perplexity_fallback(self, sequences: List[str]) -> Dict:
        """
        伪困惑度的简化实现 - 当ESM模型不可用时的备选方案
        基于氨基酸频率和转移概率的启发式计算
        """
        print("🧮 计算简化伪困惑度（无需ESM模型）...")
        
        # 标准氨基酸频率（来自自然蛋白质统计）
        natural_aa_freq = {
            'A': 0.0825, 'R': 0.0553, 'N': 0.0406, 'D': 0.0546, 'C': 0.0137,
            'Q': 0.0393, 'E': 0.0675, 'G': 0.0707, 'H': 0.0227, 'I': 0.0596,
            'L': 0.0966, 'K': 0.0584, 'M': 0.0242, 'F': 0.0386, 'P': 0.0470,
            'S': 0.0656, 'T': 0.0534, 'W': 0.0108, 'Y': 0.0292, 'V': 0.0686
        }
        
        pseudo_perplexities = []
        
        for seq in sequences:
            if not seq or len(seq) < 2:
                continue
            
            try:
                total_log_prob = 0
                valid_positions = 0
                
                # 计算基于自然频率的伪概率
                for aa in seq:
                    if aa in natural_aa_freq:
                        # 使用负对数概率
                        log_prob = -math.log(natural_aa_freq[aa])
                        total_log_prob += log_prob
                        valid_positions += 1
                
                if valid_positions > 0:
                    avg_log_prob = total_log_prob / valid_positions
                    pseudo_perplexity = math.exp(avg_log_prob)
                    pseudo_perplexities.append(pseudo_perplexity)
                    
            except Exception as e:
                print(f"⚠️ 计算序列 '{seq[:20]}...' 的简化伪困惑度失败: {e}")
                continue
        
        if pseudo_perplexities:
            result = {
                'mean_pseudo_perplexity': mean(pseudo_perplexities),
                'std_pseudo_perplexity': stdev(pseudo_perplexities) if len(pseudo_perplexities) > 1 else 0.0,
                'valid_sequences': len(pseudo_perplexities),
                'method': 'fallback_natural_frequency',
                'perplexity_distribution': pseudo_perplexities[:100]
            }
            print(f"✅ 简化伪困惑度计算完成: {result['mean_pseudo_perplexity']:.3f}±{result['std_pseudo_perplexity']:.3f}")
            return result
        else:
            return {
                'mean_pseudo_perplexity': float('nan'),
                'std_pseudo_perplexity': float('nan'),
                'valid_sequences': 0,
                'method': 'fallback_natural_frequency',
                'error': 'No valid sequences processed'
            }
    
    def evaluate_simple_similarity(self, generated_sequences: List[str], reference_sequences: List[str]) -> Dict:
        """
        简化相似性评估 - 使用编辑距离而不是BLOSUM62
        """
        print("🔍 计算简化序列相似性...")
        
        similarity_scores = []
        identical_matches = 0
        
        for gen_seq in generated_sequences:
            if not gen_seq:
                continue
            
            min_edit_distance = float('inf')
            
            # 与所有参考序列比较（限制数量以提高速度）
            for ref_seq in reference_sequences[:500]:
                if not ref_seq:
                    continue
                
                # 计算编辑距离
                edit_dist = self._edit_distance(gen_seq, ref_seq)
                min_edit_distance = min(min_edit_distance, edit_dist)
                
                # 检查完全匹配
                if gen_seq == ref_seq:
                    identical_matches += 1
            
            if min_edit_distance != float('inf'):
                # 转换为相似性得分 (0-1, 1表示完全相同)
                max_len = max(len(gen_seq), len(reference_sequences[0]) if reference_sequences else len(gen_seq))
                similarity = 1 - (min_edit_distance / max_len) if max_len > 0 else 0
                similarity_scores.append(similarity)
        
        if similarity_scores:
            # 计算新颖性比例（低相似性的序列比例）
            novelty_threshold = 0.8  # 相似性低于0.8认为是新颖的
            novel_sequences = [s for s in similarity_scores if s < novelty_threshold]
            novelty_ratio = len(novel_sequences) / len(similarity_scores)
            
            result = {
                'mean_similarity': mean(similarity_scores),
                'std_similarity': stdev(similarity_scores) if len(similarity_scores) > 1 else 0.0,
                'min_similarity': min(similarity_scores),
                'max_similarity': max(similarity_scores),
                'novelty_ratio': novelty_ratio,
                'identical_matches': identical_matches,
                'similarity_threshold': novelty_threshold,
                'total_comparisons': len(similarity_scores),
                'method': 'edit_distance_based'
            }
            print(f"✅ 简化相似性计算完成: {result['mean_similarity']:.3f}±{result['std_similarity']:.3f}, 新颖性: {novelty_ratio:.3f}")
            return result
        else:
            return {
                'mean_similarity': float('nan'),
                'std_similarity': float('nan'),
                'novelty_ratio': float('nan'),
                'method': 'edit_distance_based',
                'error': 'No valid comparisons'
            }
    
    def evaluate_hydropathy_index(self, sequences: List[str]) -> Dict:
        """
        评估疏水性指数 - 替代modlamp的不稳定性指数
        使用Kyte-Doolittle疏水性标度
        """
        print("💧 计算疏水性指数...")
        
        # Kyte-Doolittle疏水性标度
        kd_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        hydropathy_scores = []
        
        for seq in sequences:
            if not seq:
                continue
            
            total_score = 0
            valid_residues = 0
            
            for aa in seq:
                if aa in kd_scale:
                    total_score += kd_scale[aa]
                    valid_residues += 1
            
            if valid_residues > 0:
                avg_hydropathy = total_score / valid_residues
                hydropathy_scores.append(avg_hydropathy)
        
        if hydropathy_scores:
            result = {
                'mean_hydropathy': mean(hydropathy_scores),
                'std_hydropathy': stdev(hydropathy_scores) if len(hydropathy_scores) > 1 else 0.0,
                'min_hydropathy': min(hydropathy_scores),
                'max_hydropathy': max(hydropathy_scores),
                'hydrophobic_peptides_ratio': len([s for s in hydropathy_scores if s > 0]) / len(hydropathy_scores)
            }
            print(f"✅ 疏水性指数计算完成: {result['mean_hydropathy']:.3f}±{result['std_hydropathy']:.3f}")
            return result
        else:
            return {
                'mean_hydropathy': float('nan'),
                'std_hydropathy': float('nan'),
                'error': 'No valid sequences'
            }
    
    def evaluate_sequence_quality(self, sequences: List[str]) -> Dict:
        """评估基础序列质量"""
        print("🔍 评估基础序列质量...")
        
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
        results['average_length'] = mean(lengths)
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
        
        print(f"✅ 基础质量评估完成: {results['valid_sequences']}/{results['total_sequences']} 有效序列")
        return results
    
    def evaluate_condition_specificity(self, sequences_by_condition: Dict[str, List[str]]) -> Dict:
        """评估条件特异性"""
        print("🎯 评估条件特异性...")
        
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
        
        print(f"✅ 条件特异性评估完成: {len(results)} 个条件")
        return results
    
    def evaluate_diversity(self, sequences: List[str]) -> Dict:
        """评估序列多样性"""
        print("🌈 评估序列多样性...")
        
        if len(sequences) < 2:
            return {'diversity_score': 0, 'unique_sequences': len(sequences)}
        
        # 去重
        unique_sequences = list(set(sequences))
        uniqueness_ratio = len(unique_sequences) / len(sequences)
        
        # 计算编辑距离多样性（采样以避免计算量过大）
        edit_distances = []
        sample_size = min(50, len(unique_sequences))  # 减少采样数量
        
        for i in range(sample_size):
            for j in range(i+1, sample_size):
                distance = self._edit_distance(unique_sequences[i], unique_sequences[j])
                edit_distances.append(distance)
        
        diversity_score = mean(edit_distances) if edit_distances else 0
        
        result = {
            'diversity_score': diversity_score,
            'unique_sequences': len(unique_sequences),
            'uniqueness_ratio': uniqueness_ratio,
            'average_edit_distance': diversity_score,
            'sample_size_used': sample_size
        }
        
        print(f"✅ 多样性评估完成: 独特性={uniqueness_ratio:.3f}, 多样性={diversity_score:.2f}")
        return result
    
    def comprehensive_evaluation(self, 
                                generated_sequences: List[str], 
                                reference_sequences: Optional[List[str]] = None,
                                peptide_type: str = 'antimicrobial') -> Dict:
        """
        轻量级综合评估 - 不依赖外部库
        """
        print(f"🚀 开始轻量级综合评估 - 肽类型: {peptide_type}")
        print(f"📊 生成序列数量: {len(generated_sequences)}")
        if reference_sequences:
            print(f"📚 参考序列数量: {len(reference_sequences)}")
        
        results = {
            'metadata': {
                'peptide_type': peptide_type,
                'generated_count': len(generated_sequences),
                'reference_count': len(reference_sequences) if reference_sequences else 0,
                'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'method': 'lightweight_evaluation',
                'available_libraries': self.available_libs
            }
        }
        
        # 1. 基础序列质量评估
        print("\n🔸 基础序列质量评估...")
        results['basic_quality'] = self.evaluate_sequence_quality(generated_sequences)
        
        # 2. 轻量级CPL-Diff评估指标
        print("\n🔸 轻量级CPL-Diff评估指标...")
        
        # 信息熵（纯Python实现）
        results['information_entropy'] = self.evaluate_information_entropy(generated_sequences)
        
        # 简化伪困惑度（无需ESM模型）
        results['pseudo_perplexity'] = self.evaluate_pseudo_perplexity_fallback(generated_sequences)
        
        # 疏水性指数（替代不稳定性指数）
        results['hydropathy_index'] = self.evaluate_hydropathy_index(generated_sequences)
        
        # 3. 与参考序列的比较评估
        if reference_sequences:
            print("\n🔸 参考序列比较评估...")
            
            # 简化相似性分析
            results['similarity_analysis'] = self.evaluate_simple_similarity(
                generated_sequences, reference_sequences
            )
            
            # 多样性评估
            results['diversity_analysis'] = self.evaluate_diversity(generated_sequences)
            
            # 长度分布比较（基础统计）
            gen_lengths = [len(seq) for seq in generated_sequences if seq]
            ref_lengths = [len(seq) for seq in reference_sequences if seq]
            results['length_distribution'] = self._compare_length_distributions(gen_lengths, ref_lengths)
        
        # 4. 条件特异性评估
        if peptide_type in self.amp_preferences:
            print(f"\n🔸 {peptide_type}特异性评估...")
            sequences_by_condition = {peptide_type: generated_sequences}
            results['condition_specificity'] = self.evaluate_condition_specificity(sequences_by_condition)
        
        print("\n✅ 轻量级综合评估完成!")
        return results
    
    def _compare_length_distributions(self, gen_lengths: List[int], ref_lengths: List[int]) -> Dict:
        """基础长度分布比较"""
        gen_stats = {
            'mean': mean(gen_lengths) if gen_lengths else 0,
            'std': stdev(gen_lengths) if len(gen_lengths) > 1 else 0.0,
            'min': min(gen_lengths) if gen_lengths else 0,
            'max': max(gen_lengths) if gen_lengths else 0
        }
        
        ref_stats = {
            'mean': mean(ref_lengths) if ref_lengths else 0,
            'std': stdev(ref_lengths) if len(ref_lengths) > 1 else 0.0,
            'min': min(ref_lengths) if ref_lengths else 0,
            'max': max(ref_lengths) if ref_lengths else 0
        }
        
        # 简单的分布相似性指标
        mean_diff = abs(gen_stats['mean'] - ref_stats['mean'])
        std_diff = abs(gen_stats['std'] - ref_stats['std'])
        
        return {
            'generated_stats': gen_stats,
            'reference_stats': ref_stats,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'method': 'basic_statistics'
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
    
    def generate_report(self, evaluation_results: Dict, output_name: str = "lightweight_evaluation_report"):
        """生成轻量级评估报告"""
        report_path = self.output_dir / f"{output_name}.json"
        
        # 保存JSON报告
        with open(report_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # 生成文本摘要
        self._generate_text_summary(evaluation_results, output_name)
        
        print(f"📊 轻量级评估报告已生成: {report_path}")
    
    def _generate_text_summary(self, results: Dict, output_name: str):
        """生成文本摘要"""
        summary_path = self.output_dir / f"{output_name}_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("轻量级多肽生成模型评估摘要\n")
            f.write("=" * 50 + "\n\n")
            
            # 元数据
            if 'metadata' in results:
                meta = results['metadata']
                f.write("评估信息:\n")
                f.write(f"  肽类型: {meta.get('peptide_type', 'N/A')}\n")
                f.write(f"  生成序列数: {meta.get('generated_count', 0)}\n")
                f.write(f"  参考序列数: {meta.get('reference_count', 0)}\n")
                f.write(f"  评估时间: {meta.get('evaluation_timestamp', 'N/A')}\n")
                f.write(f"  评估方法: {meta.get('method', 'N/A')}\n\n")
            
            # CPL-Diff指标
            f.write("CPL-Diff评估指标 (轻量级实现):\n")
            f.write("-" * 35 + "\n")
            
            # 信息熵
            if 'information_entropy' in results:
                ie = results['information_entropy']
                if 'error' not in ie:
                    f.write(f"  信息熵: {ie.get('mean_entropy', 'N/A'):.3f}±{ie.get('std_entropy', 0):.3f}\n")
                    f.write(f"  熵范围: {ie.get('min_entropy', 'N/A'):.3f} - {ie.get('max_entropy', 'N/A'):.3f}\n")
                else:
                    f.write(f"  信息熵: 计算失败 ({ie['error']})\n")
            
            # 简化伪困惑度
            if 'pseudo_perplexity' in results:
                pp = results['pseudo_perplexity']
                if 'error' not in pp:
                    f.write(f"  简化伪困惑度: {pp.get('mean_pseudo_perplexity', 'N/A'):.3f}±{pp.get('std_pseudo_perplexity', 0):.3f}\n")
                    f.write(f"  有效序列数: {pp.get('valid_sequences', 0)}\n")
                    f.write(f"  计算方法: {pp.get('method', 'N/A')}\n")
                else:
                    f.write(f"  简化伪困惑度: 计算失败 ({pp['error']})\n")
            
            # 疏水性指数
            if 'hydropathy_index' in results:
                hi = results['hydropathy_index']
                if 'error' not in hi:
                    f.write(f"  疏水性指数: {hi.get('mean_hydropathy', 'N/A'):.3f}±{hi.get('std_hydropathy', 0):.3f}\n")
                    f.write(f"  疏水性肽比例: {hi.get('hydrophobic_peptides_ratio', 'N/A'):.3f}\n")
                else:
                    f.write(f"  疏水性指数: 计算失败 ({hi['error']})\n")
            
            f.write("\n")
            
            # 基础序列质量
            if 'basic_quality' in results:
                sq = results['basic_quality']
                f.write("基础序列质量:\n")
                f.write("-" * 20 + "\n")
                f.write(f"  总序列数: {sq.get('total_sequences', 0)}\n")
                f.write(f"  有效序列数: {sq.get('valid_sequences', 0)}\n")
                f.write(f"  平均长度: {sq.get('average_length', 0):.1f}\n")
                
                if 'hydrophobicity_scores' in sq and sq['hydrophobicity_scores']:
                    avg_hydro = mean(sq['hydrophobicity_scores'])
                    f.write(f"  平均疏水性: {avg_hydro:.3f}\n")
                
                if 'charge_scores' in sq and sq['charge_scores']:
                    avg_charge = mean(sq['charge_scores'])
                    f.write(f"  平均净电荷: {avg_charge:.1f}\n")
                f.write("\n")
            
            # 相似性分析
            if 'similarity_analysis' in results:
                sa = results['similarity_analysis']
                if 'error' not in sa:
                    f.write("相似性分析 (基于编辑距离):\n")
                    f.write("-" * 30 + "\n")
                    f.write(f"  平均相似性: {sa.get('mean_similarity', 'N/A'):.3f}±{sa.get('std_similarity', 0):.3f}\n")
                    f.write(f"  新颖性比例: {sa.get('novelty_ratio', 'N/A'):.3f}\n")
                    f.write(f"  完全匹配数: {sa.get('identical_matches', 0)}\n")
                    f.write(f"  计算方法: {sa.get('method', 'N/A')}\n\n")
            
            # 长度分布分析
            if 'length_distribution' in results:
                ld = results['length_distribution']
                f.write("长度分布分析:\n")
                f.write("-" * 20 + "\n")
                gen_stats = ld.get('generated_stats', {})
                ref_stats = ld.get('reference_stats', {})
                f.write(f"  生成序列长度: {gen_stats.get('mean', 0):.1f}±{gen_stats.get('std', 0):.1f}\n")
                f.write(f"  参考序列长度: {ref_stats.get('mean', 0):.1f}±{ref_stats.get('std', 0):.1f}\n")
                f.write(f"  均值差异: {ld.get('mean_difference', 0):.1f}\n\n")
            
            # 多样性分析
            if 'diversity_analysis' in results:
                div = results['diversity_analysis']
                f.write("多样性评估:\n")
                f.write("-" * 15 + "\n")
                f.write(f"  独特序列比例: {div.get('uniqueness_ratio', 0):.3f}\n")
                f.write(f"  多样性得分: {div.get('diversity_score', 0):.3f}\n")
                f.write(f"  采样大小: {div.get('sample_size_used', 0)}\n\n")
            
            # 条件特异性
            if 'condition_specificity' in results:
                f.write("条件特异性评估:\n")
                f.write("-" * 25 + "\n")
                for condition, data in results['condition_specificity'].items():
                    f.write(f"  {condition}:\n")
                    f.write(f"    序列数: {data.get('sequences_count', 0)}\n")
                    f.write(f"    长度合规率: {data.get('length_compliance', 0):.3f}\n")
                    f.write(f"    电荷合规率: {data.get('charge_compliance', 0):.3f}\n")
                f.write("\n")
            
            f.write("注意: 这是轻量级评估，某些指标使用了简化实现。\n")
            f.write("要获得完整评估，请安装所需依赖并使用enhanced_evaluation_suite.py")


def main():
    """主函数 - 轻量级评估演示"""
    evaluator = LightweightPeptideEvaluationSuite()
    
    # 示例数据
    generated_sequences = [
        "KRWWKWIRWKK",
        "FRLKWFKRLLK", 
        "KLRFKKLRWFK",
        "GILDTILKILR",
        "KLAKLRWKLKL",
        "KWKLFKKIEK",
        "GLFDVIKKV",
        "RWWRRRWWRR",
        "KLKLLLLLKL",
        "AIKGKFAKFK"
    ]
    
    reference_sequences = [
        "MAGAININ1PEPTIDE",
        "KRWWKWIRWKK",  # 已知序列
        "CECROPINPEPTIDE",
        "DEFENSINPEPTIDE",
        "MELITTINPEPTIDE",
        "BOMBININPEPTIDE"
    ]
    
    # 运行轻量级综合评估
    print("🚀 开始轻量级评估演示...")
    results = evaluator.comprehensive_evaluation(
        generated_sequences=generated_sequences,
        reference_sequences=reference_sequences,
        peptide_type='antimicrobial'
    )
    
    # 生成报告
    evaluator.generate_report(results, "lightweight_evaluation_demo")
    
    print("\n✅ 轻量级评估演示完成!")
    print("📊 查看生成的报告文件:")
    print("   - lightweight_evaluation_demo.json")
    print("   - lightweight_evaluation_demo_summary.txt")


if __name__ == "__main__":
    main()