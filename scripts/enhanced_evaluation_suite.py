#!/usr/bin/env python3
"""
增强的多肽生成模型评估套件 - 集成CPL-Diff评估指标
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean, stdev

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class EnhancedPeptideEvaluationSuite:
    """
    增强的多肽生成模型评估套件
    集成了StructDiff原有评估指标和CPL-Diff的关键评估方法
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
        
        # 初始化模型（延迟加载）
        self.esm_model = None
        self.esm_tokenizer = None
        self.aligner = None
        
        print("🔬 增强评估套件初始化完成")
    
    def _init_esm_model(self):
        """延迟初始化ESM模型"""
        if self.esm_model is None:
            try:
                print("📥 正在加载ESM-2模型进行伪困惑度计算...")
                from transformers import AutoTokenizer, AutoModelForMaskedLM
                import torch
                
                model_name = 'facebook/esm2_t6_8M_UR50D'
                self.esm_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.esm_model = AutoModelForMaskedLM.from_pretrained(model_name)
                
                # 移动到合适的设备
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.esm_model = self.esm_model.to(device)
                self.esm_model.eval()
                
                print(f"✅ ESM-2模型加载完成，设备: {device}")
                
            except ImportError as e:
                print(f"⚠️ 无法导入transformers库: {e}")
                print("   请安装: pip install transformers torch")
                self.esm_model = None
            except Exception as e:
                print(f"⚠️ ESM-2模型加载失败: {e}")
                self.esm_model = None
    
    def _init_aligner(self):
        """初始化BLOSUM62比对器"""
        if self.aligner is None:
            try:
                from Bio import Align
                self.aligner = Align.PairwiseAligner()
                self.aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
                self.aligner.open_gap_score = -10
                self.aligner.extend_gap_score = -0.5
                print("✅ BLOSUM62比对器初始化完成")
            except ImportError as e:
                print(f"⚠️ 无法导入BioPython: {e}")
                print("   请安装: pip install biopython")
                self.aligner = None
            except Exception as e:
                print(f"⚠️ 比对器初始化失败: {e}")
                self.aligner = None
    
    def evaluate_pseudo_perplexity(self, sequences: List[str]) -> Dict:
        """
        评估伪困惑度 (Pseudo-Perplexity) - 从CPL-Diff借鉴
        衡量序列的生物学合理性
        """
        print("🧮 计算伪困惑度...")
        self._init_esm_model()
        
        if self.esm_model is None:
            print("❌ ESM模型未可用，跳过伪困惑度计算")
            return {
                'mean_pseudo_perplexity': float('nan'),
                'std_pseudo_perplexity': float('nan'),
                'valid_sequences': 0,
                'error': 'ESM model not available'
            }
        
        import torch
        import torch.nn.functional as F
        
        device = next(self.esm_model.parameters()).device
        pseudo_perplexities = []
        valid_sequences = 0
        
        for seq in sequences:
            if not seq or len(seq) < 2:
                continue
                
            try:
                # 编码序列
                inputs = self.esm_tokenizer(
                    seq, 
                    return_tensors='pt', 
                    add_special_tokens=True,
                    max_length=512,
                    truncation=True
                )
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                
                if input_ids.shape[1] < 3:  # 至少需要CLS, 一个氨基酸, SEP
                    continue
                
                total_loss = 0
                valid_positions = 0
                
                # 逐位掩码预测 (跳过CLS和SEP标记)
                for pos in range(1, input_ids.shape[1] - 1):
                    if attention_mask[0, pos] == 0:  # 跳过padding
                        continue
                        
                    # 创建掩码输入
                    masked_input = input_ids.clone()
                    original_token = masked_input[0, pos].item()
                    masked_input[0, pos] = self.esm_tokenizer.mask_token_id
                    
                    # 预测
                    with torch.no_grad():
                        outputs = self.esm_model(masked_input, attention_mask=attention_mask)
                        logits = outputs.logits[0, pos]  # 获取掩码位置的logits
                        
                        # 计算交叉熵损失
                        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([original_token], device=device))
                        total_loss += loss.item()
                        valid_positions += 1
                
                if valid_positions > 0:
                    avg_loss = total_loss / valid_positions
                    pseudo_perplexity = math.exp(avg_loss)
                    pseudo_perplexities.append(pseudo_perplexity)
                    valid_sequences += 1
                
            except Exception as e:
                print(f"⚠️ 计算序列 '{seq[:20]}...' 的伪困惑度失败: {e}")
                continue
        
        if pseudo_perplexities:
            result = {
                'mean_pseudo_perplexity': mean(pseudo_perplexities),
                'std_pseudo_perplexity': stdev(pseudo_perplexities) if len(pseudo_perplexities) > 1 else 0.0,
                'valid_sequences': valid_sequences,
                'perplexity_distribution': pseudo_perplexities[:100]  # 保存前100个用于分析
            }
            print(f"✅ 伪困惑度计算完成: {result['mean_pseudo_perplexity']:.3f}±{result['std_pseudo_perplexity']:.3f}")
            return result
        else:
            return {
                'mean_pseudo_perplexity': float('nan'),
                'std_pseudo_perplexity': float('nan'),
                'valid_sequences': 0,
                'error': 'No valid sequences processed'
            }
    
    def evaluate_information_entropy(self, sequences: List[str]) -> Dict:
        """
        评估信息熵 (Information Entropy) - 从CPL-Diff借鉴
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
    
    def evaluate_instability_index(self, sequences: List[str]) -> Dict:
        """
        评估不稳定性指数 (Instability Index) - 从CPL-Diff借鉴
        衡量肽序列的结构稳定性
        """
        print("⚖️ 计算不稳定性指数...")
        
        try:
            from modlamp.descriptors import GlobalDescriptor
            
            # 创建临时fasta文件
            temp_fasta = self.output_dir / "temp_sequences.fasta"
            with open(temp_fasta, 'w') as f:
                for i, seq in enumerate(sequences):
                    if seq:  # 跳过空序列
                        f.write(f">seq_{i}\n{seq}\n")
            
            # 计算不稳定性指数
            desc = GlobalDescriptor(str(temp_fasta))
            desc.instability_index()
            instability_scores = desc.descriptor.squeeze()
            
            # 清理临时文件
            temp_fasta.unlink()
            
            if len(instability_scores) > 0:
                # 确保是数组
                if isinstance(instability_scores, (int, float)):
                    instability_scores = [instability_scores]
                
                result = {
                    'mean_instability': mean(instability_scores),
                    'std_instability': stdev(instability_scores) if len(instability_scores) > 1 else 0.0,
                    'min_instability': min(instability_scores),
                    'max_instability': max(instability_scores),
                    'stable_peptides_ratio': len([s for s in instability_scores if s < 40]) / len(instability_scores)
                }
                print(f"✅ 不稳定性指数计算完成: {result['mean_instability']:.2f}±{result['std_instability']:.2f}")
                return result
            else:
                return {'error': 'No instability scores computed'}
                
        except ImportError:
            print("⚠️ modlamp库未安装，跳过不稳定性指数计算")
            print("   请安装: pip install modlamp")
            return {'error': 'modlamp not available'}
        except Exception as e:
            print(f"⚠️ 不稳定性指数计算失败: {e}")
            return {'error': str(e)}
    
    def evaluate_similarity_scores(self, generated_sequences: List[str], reference_sequences: List[str]) -> Dict:
        """
        评估序列相似性得分 - 从CPL-Diff借鉴
        使用BLOSUM62矩阵计算与参考序列的相似性
        """
        print("🔍 计算序列相似性得分...")
        self._init_aligner()
        
        if self.aligner is None:
            print("❌ BLOSUM62比对器未可用，跳过相似性计算")
            return {
                'mean_similarity': float('nan'),
                'std_similarity': float('nan'),
                'novelty_ratio': float('nan'),
                'error': 'Aligner not available'
            }
        
        similarity_scores = []
        high_similarity_threshold = 50  # 可调节的相似性阈值
        
        for gen_seq in generated_sequences:
            if not gen_seq:
                continue
                
            max_similarity = 0
            try:
                for ref_seq in reference_sequences[:1000]:  # 限制参考序列数量以提高速度
                    if not ref_seq:
                        continue
                    
                    # 计算比对得分
                    alignments = self.aligner.align(gen_seq, ref_seq)
                    if alignments:
                        score = alignments.score
                        max_similarity = max(max_similarity, score)
                
                similarity_scores.append(max_similarity)
                
            except Exception as e:
                print(f"⚠️ 计算序列相似性失败: {e}")
                continue
        
        if similarity_scores:
            # 计算新颖性比例（低相似性的序列比例）
            novel_sequences = [s for s in similarity_scores if s < high_similarity_threshold]
            novelty_ratio = len(novel_sequences) / len(similarity_scores)
            
            result = {
                'mean_similarity': mean(similarity_scores),
                'std_similarity': stdev(similarity_scores) if len(similarity_scores) > 1 else 0.0,
                'min_similarity': min(similarity_scores),
                'max_similarity': max(similarity_scores),
                'novelty_ratio': novelty_ratio,
                'similarity_threshold': high_similarity_threshold,
                'total_comparisons': len(similarity_scores)
            }
            print(f"✅ 相似性计算完成: {result['mean_similarity']:.2f}±{result['std_similarity']:.2f}, 新颖性: {novelty_ratio:.3f}")
            return result
        else:
            return {
                'mean_similarity': float('nan'),
                'std_similarity': float('nan'),
                'novelty_ratio': float('nan'),
                'error': 'No valid comparisons'
            }
    
    def evaluate_length_distribution(self, generated_lengths: List[int], reference_lengths: List[int]) -> Dict:
        """
        评估长度分布一致性 - 从CPL-Diff借鉴
        比较生成序列与参考序列的长度分布
        """
        print("📏 评估长度分布一致性...")
        
        try:
            from scipy.stats import ks_2samp, wasserstein_distance
            
            # Kolmogorov-Smirnov检验
            ks_stat, p_value = ks_2samp(generated_lengths, reference_lengths)
            
            # Wasserstein距离（Earth Mover's Distance）
            wasserstein_dist = wasserstein_distance(generated_lengths, reference_lengths)
            
            # 基本统计
            gen_stats = {
                'mean': mean(generated_lengths),
                'std': stdev(generated_lengths) if len(generated_lengths) > 1 else 0.0,
                'min': min(generated_lengths),
                'max': max(generated_lengths)
            }
            
            ref_stats = {
                'mean': mean(reference_lengths),
                'std': stdev(reference_lengths) if len(reference_lengths) > 1 else 0.0,
                'min': min(reference_lengths),
                'max': max(reference_lengths)
            }
            
            result = {
                'ks_statistic': ks_stat,
                'ks_p_value': p_value,
                'wasserstein_distance': wasserstein_dist,
                'generated_stats': gen_stats,
                'reference_stats': ref_stats,
                'distribution_match': p_value > 0.05  # 显著性水平0.05
            }
            
            print(f"✅ 长度分布评估完成: KS={ks_stat:.3f}, p={p_value:.3f}, Wasserstein={wasserstein_dist:.2f}")
            return result
            
        except ImportError:
            print("⚠️ scipy库未安装，跳过统计检验")
            print("   请安装: pip install scipy")
            
            # 基本统计作为fallback
            gen_stats = {
                'mean': mean(generated_lengths),
                'std': stdev(generated_lengths) if len(generated_lengths) > 1 else 0.0
            }
            ref_stats = {
                'mean': mean(reference_lengths),
                'std': stdev(reference_lengths) if len(reference_lengths) > 1 else 0.0
            }
            
            return {
                'generated_stats': gen_stats,
                'reference_stats': ref_stats,
                'error': 'scipy not available'
            }
        except Exception as e:
            print(f"⚠️ 长度分布评估失败: {e}")
            return {'error': str(e)}
    
    def comprehensive_evaluation(self, 
                                generated_sequences: List[str], 
                                reference_sequences: Optional[List[str]] = None,
                                peptide_type: str = 'antimicrobial') -> Dict:
        """
        综合评估 - 集成所有评估指标
        """
        print(f"🚀 开始综合评估 - 肽类型: {peptide_type}")
        print(f"📊 生成序列数量: {len(generated_sequences)}")
        if reference_sequences:
            print(f"📚 参考序列数量: {len(reference_sequences)}")
        
        results = {
            'metadata': {
                'peptide_type': peptide_type,
                'generated_count': len(generated_sequences),
                'reference_count': len(reference_sequences) if reference_sequences else 0,
                'evaluation_timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        # 1. 基础序列质量评估（原有指标）
        print("\n🔸 基础序列质量评估...")
        results['basic_quality'] = self.evaluate_sequence_quality(generated_sequences)
        
        # 2. CPL-Diff评估指标
        print("\n🔸 CPL-Diff评估指标...")
        
        # 伪困惑度
        results['pseudo_perplexity'] = self.evaluate_pseudo_perplexity(generated_sequences)
        
        # 信息熵
        results['information_entropy'] = self.evaluate_information_entropy(generated_sequences)
        
        # 不稳定性指数
        results['instability_index'] = self.evaluate_instability_index(generated_sequences)
        
        # 3. 与参考序列的比较评估
        if reference_sequences:
            print("\n🔸 参考序列比较评估...")
            
            # 相似性得分
            results['similarity_analysis'] = self.evaluate_similarity_scores(
                generated_sequences, reference_sequences
            )
            
            # 长度分布比较
            gen_lengths = [len(seq) for seq in generated_sequences if seq]
            ref_lengths = [len(seq) for seq in reference_sequences if seq]
            results['length_distribution'] = self.evaluate_length_distribution(gen_lengths, ref_lengths)
            
            # 多样性评估
            results['diversity_analysis'] = self.evaluate_diversity(generated_sequences)
            
            # 新颖性评估
            results['novelty_analysis'] = self.evaluate_novelty(generated_sequences, reference_sequences)
        
        # 4. 条件特异性评估（如果指定了肽类型）
        if peptide_type in self.amp_preferences:
            print(f"\n🔸 {peptide_type}特异性评估...")
            sequences_by_condition = {peptide_type: generated_sequences}
            results['condition_specificity'] = self.evaluate_condition_specificity(sequences_by_condition)
        
        print("\n✅ 综合评估完成!")
        return results
    
    # 保留原有的评估方法（从原evaluation_suite.py继承）
    def evaluate_sequence_quality(self, sequences: List[str]) -> Dict:
        """评估序列质量（原有方法）"""
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
        """评估条件特异性（原有方法）"""
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
        """评估序列多样性（原有方法）"""
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
        """评估序列新颖性（原有方法）"""
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
    
    def generate_report(self, evaluation_results: Dict, output_name: str = "enhanced_evaluation_report"):
        """生成增强评估报告"""
        report_path = self.output_dir / f"{output_name}.json"
        
        # 保存JSON报告
        with open(report_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        # 生成可视化报告
        self._generate_visualizations(evaluation_results, output_name)
        
        # 生成文本摘要
        self._generate_text_summary(evaluation_results, output_name)
        
        print(f"📊 增强评估报告已生成: {report_path}")
    
    def _generate_visualizations(self, results: Dict, output_name: str):
        """生成可视化图表"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Enhanced Peptide Generation Evaluation', fontsize=16)
            
            # 1. 伪困惑度分布
            if 'pseudo_perplexity' in results and 'perplexity_distribution' in results['pseudo_perplexity']:
                perplexities = results['pseudo_perplexity']['perplexity_distribution']
                axes[0, 0].hist(perplexities, bins=20, alpha=0.7, color='blue')
                axes[0, 0].set_title('Pseudo-Perplexity Distribution')
                axes[0, 0].set_xlabel('Pseudo-Perplexity')
                axes[0, 0].set_ylabel('Frequency')
            
            # 2. 信息熵分布
            if 'information_entropy' in results and 'entropy_distribution' in results['information_entropy']:
                entropies = results['information_entropy']['entropy_distribution']
                axes[0, 1].hist(entropies, bins=20, alpha=0.7, color='green')
                axes[0, 1].set_title('Information Entropy Distribution')
                axes[0, 1].set_xlabel('Entropy')
                axes[0, 1].set_ylabel('Frequency')
            
            # 3. 长度分布比较
            if 'length_distribution' in results:
                gen_stats = results['length_distribution'].get('generated_stats', {})
                ref_stats = results['length_distribution'].get('reference_stats', {})
                
                if gen_stats and ref_stats:
                    categories = ['Mean', 'Std', 'Min', 'Max']
                    gen_values = [gen_stats.get('mean', 0), gen_stats.get('std', 0), 
                                 gen_stats.get('min', 0), gen_stats.get('max', 0)]
                    ref_values = [ref_stats.get('mean', 0), ref_stats.get('std', 0),
                                 ref_stats.get('min', 0), ref_stats.get('max', 0)]
                    
                    x = np.arange(len(categories))
                    width = 0.35
                    
                    axes[0, 2].bar(x - width/2, gen_values, width, label='Generated', alpha=0.7)
                    axes[0, 2].bar(x + width/2, ref_values, width, label='Reference', alpha=0.7)
                    axes[0, 2].set_title('Length Statistics Comparison')
                    axes[0, 2].set_xlabel('Statistics')
                    axes[0, 2].set_ylabel('Length')
                    axes[0, 2].set_xticks(x)
                    axes[0, 2].set_xticklabels(categories)
                    axes[0, 2].legend()
            
            # 4. 氨基酸组成
            if 'basic_quality' in results and 'amino_acid_composition' in results['basic_quality']:
                aa_comp = results['basic_quality']['amino_acid_composition']
                if aa_comp:
                    aas = list(aa_comp.keys())
                    freqs = list(aa_comp.values())
                    axes[1, 0].bar(aas, freqs, alpha=0.7, color='orange')
                    axes[1, 0].set_title('Amino Acid Composition')
                    axes[1, 0].set_xlabel('Amino Acid')
                    axes[1, 0].set_ylabel('Frequency')
                    axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 5. 评估指标总结
            metrics_data = []
            if 'pseudo_perplexity' in results:
                pp = results['pseudo_perplexity']
                if 'mean_pseudo_perplexity' in pp and not math.isnan(pp['mean_pseudo_perplexity']):
                    metrics_data.append(('Pseudo-Perplexity', pp['mean_pseudo_perplexity']))
            
            if 'information_entropy' in results:
                ie = results['information_entropy']
                if 'mean_entropy' in ie and not math.isnan(ie['mean_entropy']):
                    metrics_data.append(('Information Entropy', ie['mean_entropy']))
            
            if 'instability_index' in results:
                ii = results['instability_index']
                if 'mean_instability' in ii and not math.isnan(ii['mean_instability']):
                    metrics_data.append(('Instability Index', ii['mean_instability']))
            
            if metrics_data:
                metrics, values = zip(*metrics_data)
                axes[1, 1].barh(metrics, values, alpha=0.7, color='purple')
                axes[1, 1].set_title('Key Evaluation Metrics')
                axes[1, 1].set_xlabel('Value')
            
            # 6. 条件特异性（如果有）
            if 'condition_specificity' in results:
                cs = results['condition_specificity']
                for condition, data in cs.items():
                    if 'length_compliance' in data and 'charge_compliance' in data:
                        categories = ['Length Compliance', 'Charge Compliance']
                        values = [data['length_compliance'], data['charge_compliance']]
                        axes[1, 2].bar(categories, values, alpha=0.7, color='red')
                        axes[1, 2].set_title(f'{condition.title()} Specificity')
                        axes[1, 2].set_ylabel('Compliance Ratio')
                        break
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{output_name}_visualization.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 可视化图表已保存: {output_name}_visualization.png")
            
        except Exception as e:
            print(f"⚠️ 可视化生成失败: {e}")
    
    def _generate_text_summary(self, results: Dict, output_name: str):
        """生成增强文本摘要"""
        summary_path = self.output_dir / f"{output_name}_summary.txt"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("增强多肽生成模型评估摘要 (Enhanced Evaluation)\n")
            f.write("=" * 60 + "\n\n")
            
            # 元数据
            if 'metadata' in results:
                meta = results['metadata']
                f.write("评估信息:\n")
                f.write(f"  肽类型: {meta.get('peptide_type', 'N/A')}\n")
                f.write(f"  生成序列数: {meta.get('generated_count', 0)}\n")
                f.write(f"  参考序列数: {meta.get('reference_count', 0)}\n")
                f.write(f"  评估时间: {meta.get('evaluation_timestamp', 'N/A')}\n\n")
            
            # CPL-Diff评估指标
            f.write("CPL-Diff评估指标:\n")
            f.write("-" * 30 + "\n")
            
            # 伪困惑度
            if 'pseudo_perplexity' in results:
                pp = results['pseudo_perplexity']
                if 'error' not in pp:
                    f.write(f"  伪困惑度: {pp.get('mean_pseudo_perplexity', 'N/A'):.3f}±{pp.get('std_pseudo_perplexity', 0):.3f}\n")
                    f.write(f"  有效序列数: {pp.get('valid_sequences', 0)}\n")
                else:
                    f.write(f"  伪困惑度: 计算失败 ({pp['error']})\n")
            
            # 信息熵
            if 'information_entropy' in results:
                ie = results['information_entropy']
                if 'error' not in ie:
                    f.write(f"  信息熵: {ie.get('mean_entropy', 'N/A'):.3f}±{ie.get('std_entropy', 0):.3f}\n")
                    f.write(f"  熵范围: {ie.get('min_entropy', 'N/A'):.3f} - {ie.get('max_entropy', 'N/A'):.3f}\n")
                else:
                    f.write(f"  信息熵: 计算失败 ({ie['error']})\n")
            
            # 不稳定性指数
            if 'instability_index' in results:
                ii = results['instability_index']
                if 'error' not in ii:
                    f.write(f"  不稳定性指数: {ii.get('mean_instability', 'N/A'):.2f}±{ii.get('std_instability', 0):.2f}\n")
                    f.write(f"  稳定肽比例: {ii.get('stable_peptides_ratio', 'N/A'):.3f}\n")
                else:
                    f.write(f"  不稳定性指数: 计算失败 ({ii['error']})\n")
            
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
                    avg_hydro = np.mean(sq['hydrophobicity_scores'])
                    f.write(f"  平均疏水性: {avg_hydro:.3f}\n")
                
                if 'charge_scores' in sq and sq['charge_scores']:
                    avg_charge = np.mean(sq['charge_scores'])
                    f.write(f"  平均净电荷: {avg_charge:.1f}\n")
                f.write("\n")
            
            # 相似性分析
            if 'similarity_analysis' in results:
                sa = results['similarity_analysis']
                if 'error' not in sa:
                    f.write("相似性分析:\n")
                    f.write("-" * 15 + "\n")
                    f.write(f"  平均相似性得分: {sa.get('mean_similarity', 'N/A'):.2f}±{sa.get('std_similarity', 0):.2f}\n")
                    f.write(f"  新颖性比例: {sa.get('novelty_ratio', 'N/A'):.3f}\n")
                    f.write(f"  相似性阈值: {sa.get('similarity_threshold', 'N/A')}\n\n")
            
            # 长度分布分析
            if 'length_distribution' in results:
                ld = results['length_distribution']
                if 'error' not in ld:
                    f.write("长度分布分析:\n")
                    f.write("-" * 20 + "\n")
                    if 'ks_p_value' in ld:
                        f.write(f"  KS检验p值: {ld['ks_p_value']:.6f}\n")
                        f.write(f"  分布匹配: {'是' if ld.get('distribution_match', False) else '否'}\n")
                    if 'wasserstein_distance' in ld:
                        f.write(f"  Wasserstein距离: {ld['wasserstein_distance']:.3f}\n")
                    f.write("\n")
            
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
            
            # 多样性和新颖性
            if 'diversity_analysis' in results:
                div = results['diversity_analysis']
                f.write("多样性评估:\n")
                f.write("-" * 15 + "\n")
                f.write(f"  独特序列比例: {div.get('uniqueness_ratio', 0):.3f}\n")
                f.write(f"  多样性得分: {div.get('diversity_score', 0):.3f}\n\n")


def main():
    """主函数 - 示例用法"""
    evaluator = EnhancedPeptideEvaluationSuite()
    
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
        "KLKLLLLLKL"
    ]
    
    reference_sequences = [
        "MAGAININ1PEPTIDE",
        "KRWWKWIRWKK",  # 已知序列
        "CECROPIN",
        "DEFENSIN",
        "MELITTIN"
    ]
    
    # 运行综合评估
    print("🚀 开始增强评估演示...")
    results = evaluator.comprehensive_evaluation(
        generated_sequences=generated_sequences,
        reference_sequences=reference_sequences,
        peptide_type='antimicrobial'
    )
    
    # 生成报告
    evaluator.generate_report(results, "enhanced_evaluation_demo")
    
    print("\n✅ 增强评估演示完成!")
    print("📊 查看生成的报告文件:")
    print("   - enhanced_evaluation_demo.json")
    print("   - enhanced_evaluation_demo_summary.txt") 
    print("   - enhanced_evaluation_demo_visualization.png")


if __name__ == "__main__":
    main()