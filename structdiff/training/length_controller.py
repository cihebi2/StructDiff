#!/usr/bin/env python3
"""
CPL-Diff启发的长度控制器
实现精确的长度分布采样和控制机制
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import Counter
import json
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class LengthDistributionAnalyzer:
    """长度分布分析器"""
    
    def __init__(self, data_path: str):
        """
        初始化长度分布分析器
        
        Args:
            data_path: 训练数据路径
        """
        self.data_path = data_path
        self.length_distributions = {}
        self.overall_distribution = None
        
    def analyze_training_data(self) -> Dict[str, np.ndarray]:
        """分析训练数据的长度分布"""
        logger.info(f"分析训练数据长度分布: {self.data_path}")
        
        # 读取数据
        if self.data_path.endswith('.csv'):
            data = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"不支持的数据格式: {self.data_path}")
        
        # 计算序列长度
        if 'sequence' in data.columns:
            data['length'] = data['sequence'].str.len()
        else:
            raise ValueError("数据中缺少'sequence'列")
        
        # 整体长度分布
        all_lengths = data['length'].values
        self.overall_distribution = self._compute_distribution(all_lengths)
        
        # 按肽段类型分组分析
        if 'peptide_type' in data.columns:
            for peptide_type in data['peptide_type'].unique():
                if pd.notna(peptide_type):
                    type_data = data[data['peptide_type'] == peptide_type]
                    type_lengths = type_data['length'].values
                    self.length_distributions[peptide_type] = self._compute_distribution(type_lengths)
                    
                    logger.info(f"肽段类型 '{peptide_type}': "
                              f"平均长度 {np.mean(type_lengths):.1f} ± {np.std(type_lengths):.1f}, "
                              f"范围 [{np.min(type_lengths)}, {np.max(type_lengths)}]")
        
        # 如果没有类型信息，使用整体分布
        if not self.length_distributions:
            self.length_distributions['general'] = self.overall_distribution
        
        logger.info(f"分析完成，发现 {len(self.length_distributions)} 种肽段类型")
        return self.length_distributions
    
    def _compute_distribution(self, lengths: np.ndarray) -> np.ndarray:
        """计算长度分布概率"""
        min_len, max_len = np.min(lengths), np.max(lengths)
        length_counts = Counter(lengths)
        
        # 创建完整的长度范围概率分布
        probs = np.zeros(max_len - min_len + 1)
        total_count = len(lengths)
        
        for length, count in length_counts.items():
            idx = length - min_len
            probs[idx] = count / total_count
        
        return probs
    
    def save_distributions(self, save_path: str):
        """保存长度分布"""
        distributions_data = {
            'overall': self.overall_distribution.tolist() if self.overall_distribution is not None else None,
            'by_type': {k: v.tolist() for k, v in self.length_distributions.items()},
        }
        
        with open(save_path, 'w') as f:
            json.dump(distributions_data, f, indent=2)
        
        logger.info(f"长度分布保存到: {save_path}")
    
    def load_distributions(self, load_path: str):
        """加载长度分布"""
        with open(load_path, 'r') as f:
            distributions_data = json.load(f)
        
        if distributions_data['overall']:
            self.overall_distribution = np.array(distributions_data['overall'])
        
        self.length_distributions = {
            k: np.array(v) for k, v in distributions_data['by_type'].items()
        }
        
        logger.info(f"从 {load_path} 加载长度分布")


class AdaptiveLengthController:
    """自适应长度控制器"""
    
    def __init__(self, 
                 min_length: int = 5,
                 max_length: int = 50,
                 distributions: Optional[Dict[str, np.ndarray]] = None):
        """
        初始化长度控制器
        
        Args:
            min_length: 最小长度
            max_length: 最大长度  
            distributions: 预计算的长度分布
        """
        self.min_length = min_length
        self.max_length = max_length
        self.distributions = distributions or {}
        
        # 默认分布（正态分布）
        self.default_mean = (min_length + max_length) / 2
        self.default_std = (max_length - min_length) / 6
        
        logger.info(f"初始化长度控制器: 范围 [{min_length}, {max_length}]")
    
    def sample_target_lengths(self, 
                             batch_size: int,
                             peptide_types: Optional[List[str]] = None,
                             device: str = 'cpu') -> torch.Tensor:
        """
        采样目标长度
        
        Args:
            batch_size: 批次大小
            peptide_types: 肽段类型列表
            device: 设备
            
        Returns:
            目标长度张量
        """
        if peptide_types is None:
            # 使用默认分布采样
            lengths = np.random.normal(self.default_mean, self.default_std, batch_size)
            lengths = np.clip(lengths, self.min_length, self.max_length)
            lengths = np.round(lengths).astype(int)
        else:
            # 根据肽段类型采样
            lengths = []
            for peptide_type in peptide_types:
                if peptide_type in self.distributions:
                    # 使用特定类型分布
                    type_dist = self.distributions[peptide_type]
                    length_range = np.arange(len(type_dist)) + self.min_length
                    sampled_length = np.random.choice(length_range, p=type_dist)
                else:
                    # 使用默认分布
                    length = np.random.normal(self.default_mean, self.default_std)
                    sampled_length = int(np.clip(np.round(length), self.min_length, self.max_length))
                
                lengths.append(sampled_length)
            
            lengths = np.array(lengths)
        
        return torch.tensor(lengths, dtype=torch.long, device=device)
    
    def create_length_mask(self, 
                          target_lengths: torch.Tensor,
                          max_sequence_length: int) -> torch.Tensor:
        """
        创建长度控制掩码
        
        Args:
            target_lengths: 目标长度
            max_sequence_length: 最大序列长度
            
        Returns:
            长度掩码
        """
        batch_size = target_lengths.size(0)
        device = target_lengths.device
        
        # 创建掩码
        mask = torch.zeros(batch_size, max_sequence_length, dtype=torch.bool, device=device)
        
        for i, length in enumerate(target_lengths):
            mask[i, :length] = True
        
        return mask
    
    def apply_length_penalty(self,
                           predicted_sequences: torch.Tensor,
                           target_lengths: torch.Tensor,
                           penalty_weight: float = 1.0) -> torch.Tensor:
        """
        应用长度惩罚
        
        Args:
            predicted_sequences: 预测序列
            target_lengths: 目标长度
            penalty_weight: 惩罚权重
            
        Returns:
            长度惩罚损失
        """
        # 计算实际序列长度（非padding位置）
        if predicted_sequences.dim() == 3:  # [batch, seq_len, vocab]
            # 假设padding token是0
            actual_lengths = (predicted_sequences.argmax(dim=-1) != 0).sum(dim=1).float()
        else:  # [batch, seq_len]
            actual_lengths = (predicted_sequences != 0).sum(dim=1).float()
        
        target_lengths = target_lengths.float()
        
        # 计算长度差异惩罚
        length_diff = torch.abs(actual_lengths - target_lengths)
        penalty = penalty_weight * torch.mean(length_diff)
        
        return penalty
    
    def get_length_statistics(self, peptide_type: str = None) -> Dict[str, float]:
        """获取长度统计信息"""
        if peptide_type and peptide_type in self.distributions:
            dist = self.distributions[peptide_type]
            length_range = np.arange(len(dist)) + self.min_length
            
            mean_length = np.sum(length_range * dist)
            var_length = np.sum((length_range - mean_length) ** 2 * dist)
            std_length = np.sqrt(var_length)
            
            return {
                'mean': float(mean_length),
                'std': float(std_length),
                'min': self.min_length,
                'max': self.min_length + len(dist) - 1
            }
        else:
            return {
                'mean': self.default_mean,
                'std': self.default_std,
                'min': self.min_length,
                'max': self.max_length
            }


class LengthAwareDataCollator:
    """长度感知的数据整理器"""
    
    def __init__(self, 
                 length_controller: AdaptiveLengthController,
                 tokenizer,
                 use_length_control: bool = True):
        """
        初始化数据整理器
        
        Args:
            length_controller: 长度控制器
            tokenizer: 分词器
            use_length_control: 是否使用长度控制
        """
        self.length_controller = length_controller
        self.tokenizer = tokenizer
        self.use_length_control = use_length_control
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        整理批次数据
        
        Args:
            batch: 批次数据
            
        Returns:
            整理后的批次数据
        """
        # 基础数据整理
        sequences = [item['sequence'] for item in batch]
        peptide_types = [item.get('peptide_type', 'general') for item in batch]
        
        # 编码序列
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'sequences': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
        }
        
        # 添加条件信息
        if any(item.get('peptide_type') for item in batch):
            # 编码肽段类型
            type_to_id = {'antimicrobial': 0, 'antifungal': 1, 'antiviral': 2, 'general': 3}
            peptide_type_ids = [type_to_id.get(pt, 3) for pt in peptide_types]
            result['conditions'] = {
                'peptide_type': torch.tensor(peptide_type_ids, dtype=torch.long)
            }
        
        # 长度控制
        if self.use_length_control:
            # 采样目标长度
            target_lengths = self.length_controller.sample_target_lengths(
                batch_size=len(batch),
                peptide_types=peptide_types
            )
            
            # 创建长度掩码
            max_len = encoded['input_ids'].size(1)
            length_mask = self.length_controller.create_length_mask(target_lengths, max_len)
            
            result['target_lengths'] = target_lengths
            result['length_mask'] = length_mask
        
        # 添加结构信息（如果有）
        if 'structure' in batch[0]:
            structures = [item['structure'] for item in batch]
            result['structures'] = torch.stack(structures) if structures[0] is not None else None
        
        return result


def create_length_controller_from_data(data_path: str,
                                     min_length: int = 5,
                                     max_length: int = 50,
                                     save_distributions: bool = True) -> AdaptiveLengthController:
    """
    从训练数据创建长度控制器
    
    Args:
        data_path: 训练数据路径
        min_length: 最小长度
        max_length: 最大长度
        save_distributions: 是否保存分布
        
    Returns:
        长度控制器
    """
    # 分析数据
    analyzer = LengthDistributionAnalyzer(data_path)
    distributions = analyzer.analyze_training_data()
    
    # 保存分布（可选）
    if save_distributions:
        save_path = Path(data_path).parent / "length_distributions.json"
        analyzer.save_distributions(str(save_path))
    
    # 创建控制器
    controller = AdaptiveLengthController(
        min_length=min_length,
        max_length=max_length,
        distributions=distributions
    )
    
    return controller