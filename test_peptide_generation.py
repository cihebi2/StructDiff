#!/usr/bin/env python3
"""
使用训练好的StructDiff模型生成肽段样本
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import json
from collections import Counter
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from structdiff.models.structdiff import StructDiff
from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
from structdiff.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


class PeptideGenerator:
    """使用训练好的StructDiff模型生成肽段"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device)
        self.model_path = model_path
        
        # 创建配置
        self.config = self._create_config()
        
        # 加载模型
        self.model = self._load_model()
        
        # 创建扩散过程
        self.diffusion = GaussianDiffusion(
            num_timesteps=self.config.diffusion.num_timesteps,
            noise_schedule=self.config.diffusion.noise_schedule,
            beta_start=0.0001,
            beta_end=0.02
        )
        
        # 肽段类型映射
        self.peptide_type_map = {
            'antimicrobial': 0,
            'antifungal': 1,
            'antiviral': 2
        }
        
        logger.info("✅ PeptideGenerator初始化完成")
    
    def _create_config(self):
        """创建模型配置"""
        config = OmegaConf.create({
            'sequence_encoder': {
                'pretrained_model': 'facebook/esm2_t6_8M_UR50D',
                'freeze_encoder': False,
                'use_lora': True,
                'lora_rank': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.1
            },
            'structure_encoder': {
                'hidden_dim': 256,
                'num_layers': 3,
                'use_esmfold': False,
                'use_coordinates': False,
                'use_distances': False,
                'use_angles': False,
                'use_secondary_structure': True
            },
            'denoiser': {
                'hidden_dim': 320,
                'num_layers': 6,
                'num_heads': 8,
                'dropout': 0.1,
                'use_cross_attention': True,
                'use_cfg': True,
                'cfg_dropout': 0.1
            },
            'diffusion': {
                'num_timesteps': 1000,
                'noise_schedule': 'cosine',
                'model_mean_type': 'epsilon',
                'model_var_type': 'fixed_small',
                'loss_type': 'mse'
            },
            'data': {
                'max_length': 512
            }
        })
        return config
    
    def _load_model(self):
        """加载训练好的模型"""
        logger.info(f"🔄 从 {self.model_path} 加载模型...")
        
        # 创建模型
        model = StructDiff(self.config)
        
        # 加载检查点
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # 处理不同的检查点格式
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # 加载状态字典
            model.load_state_dict(state_dict, strict=False)
            logger.info("✅ 模型加载成功")
        else:
            logger.warning(f"⚠️ 模型文件不存在: {self.model_path}")
            logger.warning("将使用随机初始化的模型")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def generate_sequences(
        self,
        peptide_type: str = 'antimicrobial',
        num_samples: int = 20,
        max_length: int = 30,
        min_length: int = 10,
        guidance_scale: float = 2.0,
        num_inference_steps: int = 50,
        temperature: float = 1.0
    ):
        """生成肽段序列"""
        logger.info(f"🧬 生成 {num_samples} 个 {peptide_type} 肽段...")
        
        # 准备条件
        peptide_type_id = self.peptide_type_map.get(peptide_type, 0)
        conditions = {
            'peptide_type': torch.full((num_samples,), peptide_type_id, 
                                     device=self.device, dtype=torch.long)
        }
        
        # 随机选择序列长度
        lengths = np.random.randint(min_length, max_length + 1, size=num_samples)
        
        generated_sequences = []
        
        for i in range(num_samples):
            try:
                # 生成单个序列
                seq_length = lengths[i]
                sequences = self._sample_single_sequence(
                    seq_length=seq_length,
                    peptide_type_id=peptide_type_id,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    temperature=temperature
                )
                
                if sequences:
                    generated_sequences.extend(sequences)
                    
                if (i + 1) % 5 == 0:
                    logger.info(f"  进度: {i + 1}/{num_samples}")
                    
            except Exception as e:
                logger.warning(f"生成第 {i+1} 个序列时出错: {e}")
                continue
        
        logger.info(f"✅ 成功生成 {len(generated_sequences)} 个序列")
        return generated_sequences
    
    def _sample_single_sequence(
        self,
        seq_length: int,
        peptide_type_id: int,
        guidance_scale: float = 2.0,
        num_inference_steps: int = 50,
        temperature: float = 1.0
    ):
        """生成单个序列"""
        batch_size = 1
        device = self.device
        
        # 创建随机噪声作为起点
        # 注意：需要考虑CLS和SEP token
        total_length = seq_length + 2
        noise_shape = (batch_size, total_length, self.model.seq_hidden_dim)
        x_t = torch.randn(noise_shape, device=device) * temperature
        
        # 创建注意力掩码
        attention_mask = torch.ones(batch_size, total_length, device=device)
        
        # 创建条件
        conditions = {
            'peptide_type': torch.tensor([peptide_type_id], device=device, dtype=torch.long)
        }
        
        # 逐步去噪
        timesteps = torch.linspace(
            self.diffusion.num_timesteps - 1, 0, 
            num_inference_steps, dtype=torch.long, device=device
        )
        
        x = x_t
        for t in timesteps:
            # 创建时间步张量
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 使用classifier-free guidance
            if guidance_scale > 1.0:
                # 无条件预测
                with torch.no_grad():
                    uncond_output = self.model(
                        sequences=torch.zeros(batch_size, total_length, device=device, dtype=torch.long),
                        attention_mask=attention_mask,
                        timesteps=t_batch,
                        conditions=None,
                        return_loss=False
                    )
                    uncond_pred = uncond_output['denoised_embeddings']
                
                # 有条件预测
                with torch.no_grad():
                    cond_output = self.model(
                        sequences=torch.zeros(batch_size, total_length, device=device, dtype=torch.long),
                        attention_mask=attention_mask,
                        timesteps=t_batch,
                        conditions=conditions,
                        return_loss=False
                    )
                    cond_pred = cond_output['denoised_embeddings']
                
                # 应用guidance
                noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                # 直接预测
                with torch.no_grad():
                    output = self.model(
                        sequences=torch.zeros(batch_size, total_length, device=device, dtype=torch.long),
                        attention_mask=attention_mask,
                        timesteps=t_batch,
                        conditions=conditions,
                        return_loss=False
                    )
                    noise_pred = output['denoised_embeddings']
            
            # 更新x（这里简化了扩散过程的更新步骤）
            alpha_t = self.diffusion.alphas_cumprod[t]
            alpha_t_prev = self.diffusion.alphas_cumprod[max(0, t-1)] if t > 0 else torch.tensor(1.0, device=device)
            
            # 计算前一步的样本
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
            
            if t > 0:
                # 添加噪声用于下一步
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_t_prev) * pred_x0 + torch.sqrt(1 - alpha_t_prev) * noise
            else:
                x = pred_x0
        
        # 解码为序列
        sequences = self.model._decode_embeddings(x, attention_mask)
        
        return sequences
    
    def analyze_sequences(self, sequences, peptide_type):
        """分析生成的序列"""
        if not sequences:
            return {}
        
        analysis = {
            'total_sequences': len(sequences),
            'peptide_type': peptide_type,
            'unique_sequences': len(set(sequences)),
            'average_length': np.mean([len(seq) for seq in sequences]),
            'length_std': np.std([len(seq) for seq in sequences]),
            'min_length': min(len(seq) for seq in sequences),
            'max_length': max(len(seq) for seq in sequences)
        }
        
        # 氨基酸频率分析
        all_chars = ''.join(sequences)
        char_counts = Counter(all_chars)
        total_chars = len(all_chars)
        
        aa_freq = {}
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            aa_freq[f'freq_{aa}'] = char_counts.get(aa, 0) / total_chars if total_chars > 0 else 0
        
        analysis.update(aa_freq)
        
        # 计算多样性指标
        analysis['uniqueness_ratio'] = len(set(sequences)) / len(sequences)
        
        # 计算序列复杂度（信息熵）
        if total_chars > 0:
            entropy = 0
            for count in char_counts.values():
                prob = count / total_chars
                if prob > 0:
                    entropy -= prob * np.log2(prob)
            analysis['sequence_entropy'] = entropy
        else:
            analysis['sequence_entropy'] = 0
        
        return analysis
    
    def save_sequences(self, sequences, output_file, peptide_type):
        """保存序列到FASTA文件"""
        if not sequences:
            logger.warning("没有序列可保存")
            return
        
        records = []
        for i, seq in enumerate(sequences):
            record = SeqRecord(
                Seq(seq),
                id=f"generated_{peptide_type}_{i+1:03d}",
                description=f"Generated {peptide_type} peptide (length: {len(seq)})"
            )
            records.append(record)
        
        SeqIO.write(records, output_file, "fasta")
        logger.info(f"💾 序列已保存到: {output_file}")


def main():
    """主函数"""
    setup_logger()
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 模型路径
    model_path = "./outputs/structdiff_fixed/best_model.pt"
    
    if not os.path.exists(model_path):
        logger.error(f"❌ 模型文件不存在: {model_path}")
        logger.info("请先运行训练脚本生成模型")
        return
    
    # 创建生成器
    try:
        generator = PeptideGenerator(model_path, device)
    except Exception as e:
        logger.error(f"❌ 创建生成器失败: {e}")
        return
    
    # 创建输出目录
    output_dir = Path("./test_generation_results")
    output_dir.mkdir(exist_ok=True)
    
    # 生成参数
    generation_params = {
        'num_samples': 20,
        'max_length': 30,
        'min_length': 10,
        'guidance_scale': 2.0,
        'num_inference_steps': 50,
        'temperature': 1.0
    }
    
    # 对每种肽类型进行生成
    peptide_types = ['antimicrobial', 'antifungal', 'antiviral']
    all_results = {}
    
    for peptide_type in peptide_types:
        logger.info(f"\n🧬 开始生成 {peptide_type} 肽段...")
        
        try:
            # 生成序列
            sequences = generator.generate_sequences(
                peptide_type=peptide_type,
                **generation_params
            )
            
            if sequences:
                # 分析序列
                analysis = generator.analyze_sequences(sequences, peptide_type)
                all_results[peptide_type] = analysis
                
                # 保存序列
                fasta_file = output_dir / f"generated_{peptide_type}_sequences.fasta"
                generator.save_sequences(sequences, fasta_file, peptide_type)
                
                # 打印一些示例序列
                logger.info(f"✅ {peptide_type} 生成完成，共 {len(sequences)} 个序列")
                logger.info("示例序列:")
                for i, seq in enumerate(sequences[:5]):
                    logger.info(f"  {i+1}. {seq} (长度: {len(seq)})")
                
                if len(sequences) > 5:
                    logger.info(f"  ... 还有 {len(sequences)-5} 个序列")
            else:
                logger.warning(f"⚠️ {peptide_type} 没有生成任何序列")
                
        except Exception as e:
            logger.error(f"❌ {peptide_type} 生成失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存分析结果
    if all_results:
        results_file = output_dir / "generation_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # 打印结果摘要
        logger.info("\n" + "="*60)
        logger.info("🎯 生成结果摘要")
        logger.info("="*60)
        
        for peptide_type, analysis in all_results.items():
            logger.info(f"\n📊 {peptide_type.upper()} 肽段:")
            logger.info(f"  总序列数: {analysis['total_sequences']}")
            logger.info(f"  唯一序列: {analysis['unique_sequences']}")
            logger.info(f"  平均长度: {analysis['average_length']:.1f} ± {analysis['length_std']:.1f}")
            logger.info(f"  长度范围: {analysis['min_length']}-{analysis['max_length']}")
            logger.info(f"  唯一性比率: {analysis['uniqueness_ratio']:.3f}")
            logger.info(f"  序列熵: {analysis['sequence_entropy']:.3f}")
        
        logger.info(f"\n📁 详细结果保存到: {results_file}")
    
    logger.info("\n🎉 肽段生成测试完成！")


if __name__ == "__main__":
    main() 