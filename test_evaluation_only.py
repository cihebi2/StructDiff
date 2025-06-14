#!/usr/bin/env python3
"""
简化的评估指标测试脚本
只测试评估功能，不需要完整的StructDiff模型
"""

import os
import sys
import torch
import logging
import math
from pathlib import Path
from collections import Counter
from statistics import mean, stdev
import tempfile

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 尝试导入所需的库
try:
    from transformers import EsmTokenizer, EsmModel
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    print("⚠️ transformers not available, pseudo-perplexity will be skipped")

try:
    from Bio import Align
    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False
    print("⚠️ Biopython not available, BLOSUM62 similarity will be skipped")

try:
    from modlamp.descriptors import GlobalDescriptor
    MODLAMP_AVAILABLE = True
except ImportError:
    MODLAMP_AVAILABLE = False
    print("⚠️ modlamp not available, instability index will be skipped")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleEvaluator:
    """简化的评估器，只包含评估功能"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        
        # 初始化ESM2模型
        self.esm_tokenizer = None
        self.esm_model = None
        if ESM_AVAILABLE:
            self._init_esm_model()
        
        # 初始化BLOSUM62比对器
        self.aligner = None
        if BIO_AVAILABLE:
            self._init_aligner()
    
    def _init_esm_model(self):
        """初始化ESM2模型"""
        try:
            logger.info("🔬 初始化ESM2模型...")
            self.esm_tokenizer = EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
            self.esm_model = EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D').to(self.device)
            self.esm_model.eval()
            logger.info("✅ ESM2模型初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ ESM2模型初始化失败: {e}")
            self.esm_tokenizer = None
            self.esm_model = None
    
    def _init_aligner(self):
        """初始化BLOSUM62比对器"""
        try:
            self.aligner = Align.PairwiseAligner()
            self.aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")
            self.aligner.open_gap_score = -10
            self.aligner.extend_gap_score = -0.5
            logger.info("✅ BLOSUM62比对器初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ BLOSUM62比对器初始化失败: {e}")
            self.aligner = None
    
    def evaluate_pseudo_perplexity(self, sequences):
        """计算伪困惑度"""
        if not ESM_AVAILABLE or self.esm_tokenizer is None or self.esm_model is None:
            logger.warning("⚠️ ESM2模型未可用，跳过伪困惑度计算")
            return {'mean_pseudo_perplexity': 0.0, 'std_pseudo_perplexity': 0.0}
        
        logger.info("🧮 计算伪困惑度...")
        pseudo_perplexities = []
        
        with torch.no_grad():
            for seq in sequences:
                try:
                    # 对序列进行编码
                    inputs = self.esm_tokenizer(seq, return_tensors='pt', padding=True, truncation=True)
                    input_ids = inputs['input_ids'].to(self.device)
                    attention_mask = inputs['attention_mask'].to(self.device)
                    
                    seq_len = input_ids.size(1)
                    total_loss = 0.0
                    valid_positions = 0
                    
                    # 逐个位置进行掩码预测
                    for pos in range(1, seq_len - 1):  # 跳过CLS和SEP token
                        if attention_mask[0, pos] == 1:  # 只处理有效位置
                            # 创建掩码版本
                            masked_input = input_ids.clone()
                            original_token = masked_input[0, pos].item()
                            masked_input[0, pos] = self.esm_tokenizer.mask_token_id
                            
                            # 预测
                            outputs = self.esm_model(masked_input, attention_mask=attention_mask)
                            logits = outputs.last_hidden_state[0, pos]
                            
                            # 计算交叉熵损失
                            loss = torch.nn.functional.cross_entropy(
                                logits.unsqueeze(0), 
                                torch.tensor([original_token], device=self.device)
                            )
                            total_loss += loss.item()
                            valid_positions += 1
                    
                    if valid_positions > 0:
                        avg_loss = total_loss / valid_positions
                        pseudo_perplexity = math.exp(avg_loss)
                        pseudo_perplexities.append(pseudo_perplexity)
                
                except Exception as e:
                    logger.warning(f"计算序列伪困惑度失败: {e}")
                    continue
        
        if pseudo_perplexities:
            return {
                'mean_pseudo_perplexity': mean(pseudo_perplexities),
                'std_pseudo_perplexity': stdev(pseudo_perplexities) if len(pseudo_perplexities) > 1 else 0.0,
                'valid_sequences': len(pseudo_perplexities)
            }
        else:
            return {'mean_pseudo_perplexity': 0.0, 'std_pseudo_perplexity': 0.0, 'valid_sequences': 0}
    
    def evaluate_shannon_entropy(self, sequences):
        """计算Shannon信息熵"""
        logger.info("📊 计算Shannon信息熵...")
        
        # 计算每个序列的熵
        sequence_entropies = []
        for seq in sequences:
            aa_counts = Counter(seq)
            total_aa = len(seq)
            
            entropy = 0.0
            for count in aa_counts.values():
                prob = count / total_aa
                if prob > 0:
                    entropy -= prob * math.log2(prob)
            
            sequence_entropies.append(entropy)
        
        # 计算整体氨基酸分布的熵
        all_aa = ''.join(sequences)
        overall_aa_counts = Counter(all_aa)
        total_aa = len(all_aa)
        
        overall_entropy = 0.0
        for count in overall_aa_counts.values():
            prob = count / total_aa
            if prob > 0:
                overall_entropy -= prob * math.log2(prob)
        
        return {
            'mean_sequence_entropy': mean(sequence_entropies) if sequence_entropies else 0.0,
            'std_sequence_entropy': stdev(sequence_entropies) if len(sequence_entropies) > 1 else 0.0,
            'overall_entropy': overall_entropy,
            'max_possible_entropy': math.log2(20)
        }
    
    def evaluate_instability_index(self, sequences):
        """计算不稳定性指数"""
        if not MODLAMP_AVAILABLE:
            logger.warning("⚠️ modlamp未安装，跳过不稳定性指数计算")
            return {'mean_instability_index': 0.0, 'std_instability_index': 0.0}
        
        logger.info("🧪 计算不稳定性指数...")
        
        # 创建临时FASTA文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_file:
            for i, seq in enumerate(sequences):
                tmp_file.write(f">seq_{i}\n{seq}\n")
            tmp_file_path = tmp_file.name
        
        try:
            # 使用modlamp计算不稳定性指数
            desc = GlobalDescriptor(tmp_file_path)
            desc.instability_index()
            instability_scores = desc.descriptor.flatten()
            
            # 清理临时文件
            os.unlink(tmp_file_path)
            
            return {
                'mean_instability_index': mean(instability_scores),
                'std_instability_index': stdev(instability_scores) if len(instability_scores) > 1 else 0.0,
                'stable_peptides': sum(1 for score in instability_scores if score <= 40),
                'unstable_peptides': sum(1 for score in instability_scores if score > 40)
            }
        
        except Exception as e:
            logger.warning(f"计算不稳定性指数失败: {e}")
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            return {'mean_instability_index': 0.0, 'std_instability_index': 0.0}
    
    def evaluate_similarity_to_training(self, sequences, reference_sequences):
        """计算BLOSUM62相似性得分"""
        if not BIO_AVAILABLE or self.aligner is None:
            logger.warning("⚠️ BLOSUM62比对器未可用，跳过相似性计算")
            return {'mean_similarity_score': 0.0, 'std_similarity_score': 0.0}
        
        if not reference_sequences:
            logger.warning("⚠️ 未提供参考序列，跳过相似性计算")
            return {'mean_similarity_score': 0.0, 'std_similarity_score': 0.0}
        
        logger.info("🔍 计算BLOSUM62相似性得分...")
        
        similarity_scores = []
        
        for gen_seq in sequences:
            seq_scores = []
            
            # 与参考序列集合中的每个序列进行比对
            for ref_seq in reference_sequences[:10]:  # 限制参考序列数量
                try:
                    alignments = self.aligner.align(gen_seq, ref_seq)
                    if alignments:
                        score = alignments.score
                        # 标准化得分
                        normalized_score = score / max(len(gen_seq), len(ref_seq))
                        seq_scores.append(normalized_score)
                except Exception as e:
                    continue
            
            if seq_scores:
                max_similarity = max(seq_scores)
                similarity_scores.append(max_similarity)
        
        if similarity_scores:
            return {
                'mean_similarity_score': mean(similarity_scores),
                'std_similarity_score': stdev(similarity_scores) if len(similarity_scores) > 1 else 0.0,
                'max_similarity_score': max(similarity_scores),
                'min_similarity_score': min(similarity_scores)
            }
        else:
            return {'mean_similarity_score': 0.0, 'std_similarity_score': 0.0}
    
    def evaluate_diversity_metrics(self, sequences):
        """多样性评估"""
        logger.info("📈 计算多样性指标...")
        
        # 去重比例
        unique_sequences = set(sequences)
        uniqueness_ratio = len(unique_sequences) / len(sequences) if sequences else 0.0
        
        # 长度分布
        lengths = [len(seq) for seq in sequences]
        length_stats = {
            'mean_length': mean(lengths) if lengths else 0.0,
            'std_length': stdev(lengths) if len(lengths) > 1 else 0.0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'length_range': max(lengths) - min(lengths) if lengths else 0
        }
        
        # 氨基酸频率分析
        all_aa = ''.join(sequences)
        aa_counts = Counter(all_aa)
        total_aa = len(all_aa)
        
        aa_frequencies = {}
        for aa in self.amino_acids:
            aa_frequencies[f'freq_{aa}'] = aa_counts.get(aa, 0) / total_aa if total_aa > 0 else 0.0
        
        # 计算氨基酸使用的均匀性（基尼系数）
        frequencies = [aa_counts.get(aa, 0) / total_aa for aa in self.amino_acids if total_aa > 0]
        if frequencies:
            frequencies.sort()
            n = len(frequencies)
            gini = sum((2 * i - n - 1) * freq for i, freq in enumerate(frequencies, 1)) / (n * sum(frequencies))
        else:
            gini = 0.0
        
        return {
            'uniqueness_ratio': uniqueness_ratio,
            'total_sequences': len(sequences),
            'unique_sequences': len(unique_sequences),
            'duplicate_sequences': len(sequences) - len(unique_sequences),
            'length_distribution': length_stats,
            'amino_acid_frequencies': aa_frequencies,
            'amino_acid_gini_coefficient': gini
        }

def test_evaluation_metrics():
    """测试评估指标"""
    logger.info("🧪 开始测试专业生物学评估指标...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建评估器
    evaluator = SimpleEvaluator(device)
    
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
    
    # 参考序列
    reference_sequences = [
        "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
        "FLPIIAKFFSKVM",
        "GLLSKLWKKVFKAFKKFLKK",
        "ACDEFGHIKLMNPQRSTVWY"
    ]
    
    # 运行所有评估
    results = {}
    
    # 1. 伪困惑度
    try:
        results['pseudo_perplexity'] = evaluator.evaluate_pseudo_perplexity(test_sequences)
    except Exception as e:
        logger.warning(f"伪困惑度计算失败: {e}")
        results['pseudo_perplexity'] = {'mean_pseudo_perplexity': 0.0, 'std_pseudo_perplexity': 0.0}
    
    # 2. Shannon熵
    try:
        results['shannon_entropy'] = evaluator.evaluate_shannon_entropy(test_sequences)
    except Exception as e:
        logger.warning(f"Shannon熵计算失败: {e}")
        results['shannon_entropy'] = {'mean_sequence_entropy': 0.0, 'overall_entropy': 0.0}
    
    # 3. 不稳定性指数
    try:
        results['instability_index'] = evaluator.evaluate_instability_index(test_sequences)
    except Exception as e:
        logger.warning(f"不稳定性指数计算失败: {e}")
        results['instability_index'] = {'mean_instability_index': 0.0, 'std_instability_index': 0.0}
    
    # 4. BLOSUM62相似性
    try:
        results['blosum62_similarity'] = evaluator.evaluate_similarity_to_training(
            test_sequences, reference_sequences
        )
    except Exception as e:
        logger.warning(f"BLOSUM62相似性计算失败: {e}")
        results['blosum62_similarity'] = {'mean_similarity_score': 0.0, 'std_similarity_score': 0.0}
    
    # 5. 多样性分析
    try:
        results['diversity_analysis'] = evaluator.evaluate_diversity_metrics(test_sequences)
    except Exception as e:
        logger.warning(f"多样性分析失败: {e}")
        results['diversity_analysis'] = {'uniqueness_ratio': 0.0}
    
    # 打印结果
    logger.info("\n" + "="*80)
    logger.info("🎯 专业生物学评估指标测试结果")
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
    
    logger.info("🎉 专业生物学评估指标测试完成！")

if __name__ == "__main__":
    test_evaluation_metrics() 