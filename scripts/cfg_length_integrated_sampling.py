#!/usr/bin/env python3
"""
CFG + 长度采样器集成采样脚本
展示如何使用Classifier-Free Guidance和长度分布采样器进行高质量肽段生成
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import argparse
import yaml
from dataclasses import dataclass
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from structdiff.models.classifier_free_guidance import CFGConfig, ClassifierFreeGuidance
from structdiff.sampling.length_sampler import (
    LengthSamplerConfig, 
    AdaptiveLengthSampler,
    LengthConstrainedSampler,
    get_default_length_config
)
from structdiff.models.denoise import StructureAwareDenoiser
from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion


@dataclass
class IntegratedSamplingConfig:
    """集成采样配置"""
    # CFG配置
    cfg_guidance_scale: float = 2.5
    cfg_adaptive_guidance: bool = True
    cfg_multi_level_guidance: bool = False
    
    # 长度采样配置
    length_distribution: str = "normal"  # normal, uniform, gamma, beta
    length_mean: float = 25.0
    length_std: float = 8.0
    min_length: int = 8
    max_length: int = 50
    
    # 生成配置
    batch_size: int = 16
    num_samples: int = 100
    num_inference_steps: int = 50
    
    # 条件配置
    peptide_types: List[str] = None
    temperature: float = 1.0
    
    # 输出配置
    output_file: str = "cfg_length_generated_peptides.fasta"
    save_intermediates: bool = False


class CFGLengthIntegratedSampler:
    """
    CFG + 长度采样器集成采样器
    结合分类器自由引导和自适应长度控制
    """
    
    def __init__(self, 
                 denoiser: StructureAwareDenoiser,
                 diffusion: GaussianDiffusion,
                 cfg_config: CFGConfig,
                 length_config: LengthSamplerConfig,
                 device: str = 'cuda'):
        
        self.denoiser = denoiser
        self.diffusion = diffusion
        self.device = device
        
        # CFG组件
        self.cfg = ClassifierFreeGuidance(cfg_config)
        self.cfg_config = cfg_config
        
        # 长度采样组件
        self.length_sampler = AdaptiveLengthSampler(length_config)
        self.length_constrainer = LengthConstrainedSampler(self.length_sampler)
        self.length_config = length_config
        
        # 移动到设备
        self.denoiser = self.denoiser.to(device)
        self.length_sampler = self.length_sampler.to(device)
    
    def sample_with_cfg_and_length(self, 
                                  config: IntegratedSamplingConfig) -> Dict[str, List]:
        """
        使用CFG和长度控制进行采样
        
        Args:
            config: 集成采样配置
            
        Returns:
            生成结果字典
        """
        print(f"🚀 开始CFG+长度控制集成采样")
        print(f"📊 配置: CFG引导强度={config.cfg_guidance_scale}, 长度分布={config.length_distribution}")
        
        results = {
            'sequences': [],
            'lengths': [],
            'peptide_types': [],
            'guidance_scales': [],
            'sampling_metadata': []
        }
        
        # 处理肽类型
        if config.peptide_types is None:
            config.peptide_types = ['antimicrobial', 'antifungal', 'antiviral']
        
        total_batches = (config.num_samples + config.batch_size - 1) // config.batch_size
        
        for batch_idx in range(total_batches):
            current_batch_size = min(config.batch_size, config.num_samples - batch_idx * config.batch_size)
            
            print(f"📦 批次 {batch_idx + 1}/{total_batches}, 大小: {current_batch_size}")
            
            # 生成条件
            conditions = self._generate_batch_conditions(current_batch_size, config)
            
            # 采样长度
            target_lengths = self._sample_batch_lengths(current_batch_size, conditions, config)
            
            # CFG引导采样
            batch_sequences = self._sample_batch_sequences(
                current_batch_size, conditions, target_lengths, config
            )
            
            # 收集结果
            self._collect_batch_results(
                batch_sequences, target_lengths, conditions, config, results
            )
        
        print(f"✅ 完成采样，共生成 {len(results['sequences'])} 个序列")
        return results
    
    def _generate_batch_conditions(self, 
                                  batch_size: int, 
                                  config: IntegratedSamplingConfig) -> Dict[str, torch.Tensor]:
        """生成批次条件"""
        # 随机选择肽类型
        type_mapping = {'antimicrobial': 0, 'antifungal': 1, 'antiviral': 2}
        
        peptide_types = []
        for _ in range(batch_size):
            peptide_type_name = np.random.choice(config.peptide_types)
            peptide_types.append(type_mapping[peptide_type_name])
        
        conditions = {
            'peptide_type': torch.tensor(peptide_types, device=self.device),
            'peptide_type_names': [list(type_mapping.keys())[i] for i in peptide_types]
        }
        
        return conditions
    
    def _sample_batch_lengths(self, 
                             batch_size: int,
                             conditions: Dict[str, torch.Tensor],
                             config: IntegratedSamplingConfig) -> torch.Tensor:
        """采样批次长度"""
        # 根据配置调整长度采样器
        if config.length_distribution == "normal":
            self.length_config.normal_mean = config.length_mean
            self.length_config.normal_std = config.length_std
        
        self.length_config.min_length = config.min_length
        self.length_config.max_length = config.max_length
        
        # 采样长度
        target_lengths = self.length_sampler.sample_lengths(
            batch_size=batch_size,
            conditions=conditions,
            distribution_type=config.length_distribution,
            temperature=config.temperature,
            device=self.device
        )
        
        return target_lengths
    
    def _sample_batch_sequences(self,
                               batch_size: int,
                               conditions: Dict[str, torch.Tensor],
                               target_lengths: torch.Tensor,
                               config: IntegratedSamplingConfig) -> torch.Tensor:
        """采样批次序列"""
        max_length = int(target_lengths.max().item())
        
        # 创建长度掩码
        attention_mask = self.length_constrainer.create_length_mask(
            target_lengths, max_length, self.device
        )
        
        # 初始化噪声
        noise_shape = (batch_size, max_length, self.denoiser.hidden_dim)
        x_T = torch.randn(noise_shape, device=self.device)
        
        # CFG采样过程
        x_t = x_T
        timesteps = torch.linspace(
            self.diffusion.num_timesteps - 1, 0, 
            config.num_inference_steps, 
            dtype=torch.long, 
            device=self.device
        )
        
        for step_idx, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)
            
            # 自适应引导强度
            if config.cfg_adaptive_guidance:
                guidance_scale = self.cfg.adaptive_guidance_scale(
                    step_idx, len(timesteps), config.cfg_guidance_scale
                )
            else:
                guidance_scale = config.cfg_guidance_scale
            
            # CFG去噪步骤
            if config.cfg_multi_level_guidance:
                # 多级引导
                guidance_scales = {
                    'peptide_type': guidance_scale,
                    'length': guidance_scale * 0.5,  # 长度引导强度较低
                }
                
                noise_pred = self.cfg.multi_level_guidance(
                    lambda x, time, cond: self.denoiser(
                        x, time, attention_mask, conditions=cond
                    )[0],
                    x_t, t_batch, conditions, guidance_scales
                )
            else:
                # 标准CFG
                noise_pred = self.cfg.guided_denoising(
                    lambda x, time, cond: self.denoiser(
                        x, time, attention_mask, conditions=cond
                    )[0],
                    x_t, t_batch, conditions, guidance_scale
                )
            
            # 扩散反向步骤
            x_t = self.diffusion.p_sample(noise_pred, x_t, t_batch)
            
            # 应用长度约束
            x_t = x_t * attention_mask.unsqueeze(-1)
        
        # 后处理：转换为序列ID
        sequences = self._convert_embeddings_to_sequences(x_t, target_lengths)
        
        return sequences
    
    def _convert_embeddings_to_sequences(self, 
                                       embeddings: torch.Tensor,
                                       target_lengths: torch.Tensor) -> torch.Tensor:
        """将嵌入转换为序列ID（简化实现）"""
        batch_size, max_length, hidden_dim = embeddings.shape
        
        # 这里应该使用实际的词汇表映射
        # 简化实现：通过最近邻查找映射到氨基酸ID
        
        # 20个氨基酸的随机映射（实际应该使用训练好的映射）
        amino_acids = list(range(20))  # 0-19 对应20种氨基酸
        
        # 简化：根据嵌入的均值映射到氨基酸
        sequence_ids = torch.zeros(batch_size, max_length, dtype=torch.long, device=embeddings.device)
        
        for i in range(batch_size):
            length = int(target_lengths[i].item())
            # 简化映射：使用嵌入均值的哈希
            for j in range(length):
                emb_hash = int(torch.sum(embeddings[i, j]).item()) % 20
                sequence_ids[i, j] = emb_hash
        
        return sequence_ids
    
    def _collect_batch_results(self,
                              sequences: torch.Tensor,
                              lengths: torch.Tensor,
                              conditions: Dict[str, torch.Tensor],
                              config: IntegratedSamplingConfig,
                              results: Dict[str, List]):
        """收集批次结果"""
        batch_size = sequences.shape[0]
        
        for i in range(batch_size):
            length = int(lengths[i].item())
            sequence = sequences[i, :length].cpu().numpy()
            
            # 转换为氨基酸字符串
            aa_sequence = self._ids_to_amino_acids(sequence)
            
            results['sequences'].append(aa_sequence)
            results['lengths'].append(length)
            results['peptide_types'].append(conditions['peptide_type_names'][i])
            results['guidance_scales'].append(config.cfg_guidance_scale)
            results['sampling_metadata'].append({
                'length_distribution': config.length_distribution,
                'temperature': config.temperature,
                'cfg_adaptive': config.cfg_adaptive_guidance
            })
    
    def _ids_to_amino_acids(self, sequence_ids: np.ndarray) -> str:
        """将序列ID转换为氨基酸字符串"""
        # 20种氨基酸的映射
        id_to_aa = {
            0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C',
            5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
            10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P',
            15: 'S', 16: 'T', 17: 'W', 18: 'Y', 19: 'V'
        }
        
        return ''.join([id_to_aa.get(int(id_val), 'X') for id_val in sequence_ids])
    
    def save_results(self, results: Dict[str, List], output_file: str):
        """保存生成结果"""
        print(f"💾 保存结果到 {output_file}")
        
        with open(output_file, 'w') as f:
            for i, sequence in enumerate(results['sequences']):
                peptide_type = results['peptide_types'][i]
                length = results['lengths'][i]
                guidance_scale = results['guidance_scales'][i]
                
                header = f">peptide_{i:04d}|{peptide_type}|len={length}|cfg={guidance_scale:.1f}"
                f.write(f"{header}\n{sequence}\n")
        
        # 保存元数据
        metadata_file = output_file.replace('.fasta', '_metadata.yaml')
        metadata = {
            'total_sequences': len(results['sequences']),
            'peptide_type_distribution': self._get_type_distribution(results['peptide_types']),
            'length_statistics': self._get_length_statistics(results['lengths']),
            'average_guidance_scale': np.mean(results['guidance_scales'])
        }
        
        with open(metadata_file, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        
        print(f"📊 元数据保存到 {metadata_file}")
    
    def _get_type_distribution(self, peptide_types: List[str]) -> Dict[str, int]:
        """获取肽类型分布"""
        distribution = {}
        for ptype in peptide_types:
            distribution[ptype] = distribution.get(ptype, 0) + 1
        return distribution
    
    def _get_length_statistics(self, lengths: List[int]) -> Dict[str, float]:
        """获取长度统计"""
        lengths_array = np.array(lengths)
        return {
            'mean': float(np.mean(lengths_array)),
            'std': float(np.std(lengths_array)),
            'min': int(np.min(lengths_array)),
            'max': int(np.max(lengths_array)),
            'median': float(np.median(lengths_array))
        }


def create_demo_models(device: str = 'cuda') -> Tuple[StructureAwareDenoiser, GaussianDiffusion]:
    """创建演示用的模型（简化版本）"""
    # 简化的去噪器配置
    denoiser_config = {
        'hidden_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'dropout': 0.1,
        'use_cross_attention': True
    }
    
    # 创建CFG配置
    cfg_config = CFGConfig(
        dropout_prob=0.1,
        guidance_scale=2.5,
        adaptive_guidance=True
    )
    
    # 创建去噪器
    denoiser = StructureAwareDenoiser(
        seq_hidden_dim=256,
        struct_hidden_dim=256,
        denoiser_config=denoiser_config,
        cfg_config=cfg_config
    )
    
    # 创建扩散过程
    diffusion = GaussianDiffusion(
        num_timesteps=1000,
        noise_schedule="cosine",
        beta_start=0.0001,
        beta_end=0.02
    )
    
    return denoiser, diffusion


def main():
    """主函数：CFG+长度采样演示"""
    parser = argparse.ArgumentParser(description="CFG + 长度采样器集成演示")
    parser.add_argument("--num_samples", type=int, default=50, help="生成序列数量")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--guidance_scale", type=float, default=2.5, help="CFG引导强度")
    parser.add_argument("--length_mean", type=float, default=25.0, help="长度分布均值")
    parser.add_argument("--length_std", type=float, default=8.0, help="长度分布标准差")
    parser.add_argument("--output", type=str, default="cfg_length_demo.fasta", help="输出文件")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    
    args = parser.parse_args()
    
    print("🔬 CFG + 长度采样器集成演示")
    print("=" * 50)
    
    # 创建模型
    print("📦 初始化模型...")
    denoiser, diffusion = create_demo_models(args.device)
    
    # 创建配置
    cfg_config = CFGConfig(
        dropout_prob=0.1,
        guidance_scale=args.guidance_scale,
        adaptive_guidance=True,
        multi_level_guidance=False
    )
    
    length_config = get_default_length_config()
    length_config.normal_mean = args.length_mean
    length_config.normal_std = args.length_std
    
    sampling_config = IntegratedSamplingConfig(
        cfg_guidance_scale=args.guidance_scale,
        length_mean=args.length_mean,
        length_std=args.length_std,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        output_file=args.output,
        peptide_types=['antimicrobial', 'antifungal', 'antiviral']
    )
    
    # 创建集成采样器
    print("🔧 创建集成采样器...")
    sampler = CFGLengthIntegratedSampler(
        denoiser=denoiser,
        diffusion=diffusion,
        cfg_config=cfg_config,
        length_config=length_config,
        device=args.device
    )
    
    # 执行采样
    print("🎯 开始采样...")
    results = sampler.sample_with_cfg_and_length(sampling_config)
    
    # 保存结果
    sampler.save_results(results, args.output)
    
    # 显示统计
    print("\n📊 生成统计:")
    print(f"  总序列数: {len(results['sequences'])}")
    print(f"  平均长度: {np.mean(results['lengths']):.1f}±{np.std(results['lengths']):.1f}")
    print(f"  长度范围: {min(results['lengths'])}-{max(results['lengths'])}")
    
    type_dist = sampler._get_type_distribution(results['peptide_types'])
    print(f"  类型分布: {type_dist}")
    
    print(f"\n✅ 演示完成！结果保存在 {args.output}")


if __name__ == "__main__":
    main()