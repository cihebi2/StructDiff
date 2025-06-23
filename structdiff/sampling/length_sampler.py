#!/usr/bin/env python3
"""
Length Distribution Sampler for StructDiff
长度分布采样器，支持多种分布类型和自适应长度控制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math


@dataclass
class LengthSamplerConfig:
    """长度采样器配置"""
    # 基础配置
    min_length: int = 5
    max_length: int = 50
    default_length: int = 25
    
    # 分布类型
    distribution_type: str = "normal"  # normal, uniform, gamma, beta, custom
    
    # 正态分布参数
    normal_mean: float = 25.0
    normal_std: float = 5.0
    
    # Gamma分布参数
    gamma_shape: float = 2.0
    gamma_scale: float = 12.5
    
    # Beta分布参数（需要重新缩放到[min_length, max_length]）
    beta_alpha: float = 2.0
    beta_beta: float = 2.0
    
    # 自定义分布（长度->概率映射）
    custom_distribution: Dict[int, float] = field(default_factory=dict)
    
    # 高级配置
    use_adaptive_sampling: bool = True
    adaptive_temperature: float = 1.0
    length_embedding_dim: int = 64
    
    # 条件长度控制
    condition_dependent: bool = True
    peptide_type_length_prefs: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'antimicrobial': (20.0, 8.0),    # (mean, std)
        'antifungal': (25.0, 10.0),
        'antiviral': (30.0, 12.0),
        'general': (25.0, 5.0)
    })


class BaseLengthDistribution(ABC):
    """长度分布基类"""
    
    @abstractmethod
    def sample(self, batch_size: int, **kwargs) -> torch.Tensor:
        """采样长度"""
        pass
    
    @abstractmethod
    def log_prob(self, lengths: torch.Tensor) -> torch.Tensor:
        """计算对数概率"""
        pass


class NormalLengthDistribution(BaseLengthDistribution):
    """正态分布长度采样"""
    
    def __init__(self, mean: float, std: float, min_len: int, max_len: int):
        self.mean = mean
        self.std = std
        self.min_len = min_len
        self.max_len = max_len
        self.distribution = torch.distributions.Normal(mean, std)
    
    def sample(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        """采样并截断到有效范围"""
        samples = self.distribution.sample((batch_size,)).to(device)
        # 截断并转换为整数
        clipped = torch.clamp(samples, self.min_len, self.max_len)
        return torch.round(clipped).long()
    
    def log_prob(self, lengths: torch.Tensor) -> torch.Tensor:
        """计算对数概率（连续近似）"""
        return self.distribution.log_prob(lengths.float())


class UniformLengthDistribution(BaseLengthDistribution):
    """均匀分布长度采样"""
    
    def __init__(self, min_len: int, max_len: int):
        self.min_len = min_len
        self.max_len = max_len
        self.distribution = torch.distributions.Uniform(min_len, max_len + 1)
    
    def sample(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        samples = self.distribution.sample((batch_size,)).to(device)
        return torch.floor(samples).long()
    
    def log_prob(self, lengths: torch.Tensor) -> torch.Tensor:
        uniform_prob = 1.0 / (self.max_len - self.min_len + 1)
        return torch.full_like(lengths, math.log(uniform_prob), dtype=torch.float)


class GammaLengthDistribution(BaseLengthDistribution):
    """Gamma分布长度采样"""
    
    def __init__(self, shape: float, scale: float, min_len: int, max_len: int):
        self.shape = shape
        self.scale = scale
        self.min_len = min_len
        self.max_len = max_len
        self.distribution = torch.distributions.Gamma(shape, 1.0/scale)
    
    def sample(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        samples = self.distribution.sample((batch_size,)).to(device)
        clipped = torch.clamp(samples, self.min_len, self.max_len)
        return torch.round(clipped).long()
    
    def log_prob(self, lengths: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(lengths.float())


class BetaLengthDistribution(BaseLengthDistribution):
    """Beta分布长度采样（重新缩放到目标范围）"""
    
    def __init__(self, alpha: float, beta: float, min_len: int, max_len: int):
        self.alpha = alpha
        self.beta = beta
        self.min_len = min_len
        self.max_len = max_len
        self.distribution = torch.distributions.Beta(alpha, beta)
        self.range = max_len - min_len
    
    def sample(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        # Beta分布采样[0,1]，然后缩放到[min_len, max_len]
        beta_samples = self.distribution.sample((batch_size,)).to(device)
        scaled = self.min_len + beta_samples * self.range
        return torch.round(scaled).long()
    
    def log_prob(self, lengths: torch.Tensor) -> torch.Tensor:
        # 将长度转换回[0,1]范围计算概率
        normalized = (lengths.float() - self.min_len) / self.range
        return self.distribution.log_prob(normalized) - math.log(self.range)


class CustomLengthDistribution(BaseLengthDistribution):
    """自定义离散分布"""
    
    def __init__(self, length_probs: Dict[int, float]):
        self.length_probs = length_probs
        lengths = list(length_probs.keys())
        probs = list(length_probs.values())
        
        # 归一化概率
        total_prob = sum(probs)
        normalized_probs = [p/total_prob for p in probs]
        
        self.lengths = torch.tensor(lengths)
        self.probs = torch.tensor(normalized_probs)
        self.categorical = torch.distributions.Categorical(self.probs)
    
    def sample(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        indices = self.categorical.sample((batch_size,))
        return self.lengths[indices].to(device)
    
    def log_prob(self, lengths: torch.Tensor) -> torch.Tensor:
        # 找到每个长度对应的概率
        log_probs = torch.zeros_like(lengths, dtype=torch.float)
        for i, length in enumerate(lengths):
            idx = (self.lengths == length).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                log_probs[i] = torch.log(self.probs[idx[0]])
            else:
                log_probs[i] = -float('inf')  # 不在分布中的长度
        return log_probs


class AdaptiveLengthSampler(nn.Module):
    """
    自适应长度采样器
    根据条件和上下文动态调整长度分布
    """
    
    def __init__(self, config: LengthSamplerConfig):
        super().__init__()
        self.config = config
        
        # 长度嵌入
        self.length_embedding = nn.Embedding(
            config.max_length + 1,
            config.length_embedding_dim
        )
        
        # 条件依赖的长度预测器
        if config.condition_dependent:
            self.condition_length_predictor = nn.Sequential(
                nn.Linear(config.length_embedding_dim + 64, 128),  # 64为条件嵌入维度
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, config.max_length - config.min_length + 1),
                nn.Softmax(dim=-1)
            )
        
        # 基础分布
        self.base_distributions = self._create_distributions()
    
    def _create_distributions(self) -> Dict[str, BaseLengthDistribution]:
        """创建基础分布"""
        distributions = {}
        
        # 正态分布
        distributions['normal'] = NormalLengthDistribution(
            self.config.normal_mean,
            self.config.normal_std,
            self.config.min_length,
            self.config.max_length
        )
        
        # 均匀分布
        distributions['uniform'] = UniformLengthDistribution(
            self.config.min_length,
            self.config.max_length
        )
        
        # Gamma分布
        distributions['gamma'] = GammaLengthDistribution(
            self.config.gamma_shape,
            self.config.gamma_scale,
            self.config.min_length,
            self.config.max_length
        )
        
        # Beta分布
        distributions['beta'] = BetaLengthDistribution(
            self.config.beta_alpha,
            self.config.beta_beta,
            self.config.min_length,
            self.config.max_length
        )
        
        # 自定义分布
        if self.config.custom_distribution:
            distributions['custom'] = CustomLengthDistribution(
                self.config.custom_distribution
            )
        
        return distributions
    
    def sample_lengths(self, 
                      batch_size: int,
                      conditions: Optional[Dict[str, torch.Tensor]] = None,
                      distribution_type: Optional[str] = None,
                      temperature: float = 1.0,
                      device: str = 'cpu') -> torch.Tensor:
        """
        采样序列长度
        
        Args:
            batch_size: 批次大小
            conditions: 条件信息
            distribution_type: 分布类型（覆盖配置）
            temperature: 采样温度
            device: 设备
            
        Returns:
            采样的长度 [batch_size]
        """
        if distribution_type is None:
            distribution_type = self.config.distribution_type
        
        if self.config.use_adaptive_sampling and conditions is not None:
            return self._adaptive_sample(batch_size, conditions, temperature, device)
        else:
            return self._basic_sample(batch_size, distribution_type, temperature, device)
    
    def _basic_sample(self, 
                     batch_size: int,
                     distribution_type: str,
                     temperature: float,
                     device: str) -> torch.Tensor:
        """基础采样"""
        distribution = self.base_distributions[distribution_type]
        lengths = distribution.sample(batch_size, device)
        
        if temperature != 1.0:
            # 应用温度调节（对离散分布的近似）
            lengths = self._apply_temperature(lengths, temperature)
        
        return lengths
    
    def _adaptive_sample(self,
                        batch_size: int,
                        conditions: Dict[str, torch.Tensor],
                        temperature: float,
                        device: str) -> torch.Tensor:
        """自适应采样"""
        if not self.config.condition_dependent:
            return self._basic_sample(batch_size, self.config.distribution_type, temperature, device)
        
        # 获取条件嵌入
        condition_embeddings = self._get_condition_embeddings(conditions, device)
        
        # 预测长度分布
        length_logits = self._predict_length_distribution(condition_embeddings)
        
        # 应用温度
        if temperature != 1.0:
            length_logits = length_logits / temperature
        
        # 采样
        length_probs = F.softmax(length_logits, dim=-1)
        length_indices = torch.multinomial(length_probs, 1).squeeze(-1)
        
        # 转换为实际长度
        lengths = length_indices + self.config.min_length
        
        return lengths
    
    def _get_condition_embeddings(self, 
                                 conditions: Dict[str, torch.Tensor],
                                 device: str) -> torch.Tensor:
        """获取条件嵌入"""
        batch_size = next(iter(conditions.values())).shape[0]
        
        # 简化实现：使用多肽类型
        if 'peptide_type' in conditions:
            peptide_types = conditions['peptide_type']
            # 将类型ID转换为嵌入（这里简化处理）
            type_embeddings = torch.randn(batch_size, 64, device=device)
            return type_embeddings
        else:
            # 默认嵌入
            return torch.zeros(batch_size, 64, device=device)
    
    def _predict_length_distribution(self, condition_embeddings: torch.Tensor) -> torch.Tensor:
        """预测长度分布"""
        # 获取长度范围的平均嵌入
        length_range = torch.arange(
            self.config.min_length, 
            self.config.max_length + 1,
            device=condition_embeddings.device
        )
        avg_length_embedding = self.length_embedding(length_range).mean(dim=0, keepdim=True)
        avg_length_embedding = avg_length_embedding.expand(condition_embeddings.shape[0], -1)
        
        # 结合条件和长度信息
        combined_embedding = torch.cat([condition_embeddings, avg_length_embedding], dim=-1)
        
        # 预测分布
        return self.condition_length_predictor(combined_embedding)
    
    def _apply_temperature(self, lengths: torch.Tensor, temperature: float) -> torch.Tensor:
        """应用温度调节（离散分布的近似方法）"""
        if temperature == 1.0:
            return lengths
        
        # 对长度添加噪声模拟温度效果
        noise = torch.randn_like(lengths.float()) * (temperature - 1.0)
        noisy_lengths = lengths.float() + noise
        
        # 截断并转换回整数
        clipped = torch.clamp(noisy_lengths, self.config.min_length, self.config.max_length)
        return torch.round(clipped).long()
    
    def get_length_probabilities(self, 
                                conditions: Optional[Dict[str, torch.Tensor]] = None,
                                device: str = 'cpu') -> torch.Tensor:
        """
        获取长度概率分布
        
        Args:
            conditions: 条件信息
            device: 设备
            
        Returns:
            长度概率分布 [batch_size, num_lengths]
        """
        if conditions is None:
            # 使用基础分布
            lengths = torch.arange(
                self.config.min_length,
                self.config.max_length + 1,
                device=device
            ).float()
            
            distribution = self.base_distributions[self.config.distribution_type]
            log_probs = distribution.log_prob(lengths)
            return F.softmax(log_probs, dim=-1).unsqueeze(0)
        
        # 使用自适应预测
        condition_embeddings = self._get_condition_embeddings(conditions, device)
        length_logits = self._predict_length_distribution(condition_embeddings)
        return F.softmax(length_logits, dim=-1)
    
    def condition_on_peptide_type(self, peptide_type: str) -> 'LengthSamplerConfig':
        """
        根据多肽类型调整配置
        
        Args:
            peptide_type: 多肽类型
            
        Returns:
            调整后的配置
        """
        new_config = LengthSamplerConfig(**self.config.__dict__)
        
        if peptide_type in self.config.peptide_type_length_prefs:
            mean, std = self.config.peptide_type_length_prefs[peptide_type]
            new_config.normal_mean = mean
            new_config.normal_std = std
            new_config.default_length = int(mean)
        
        return new_config


class LengthConstrainedSampler:
    """
    长度约束采样器
    在生成过程中维持目标长度
    """
    
    def __init__(self, length_sampler: AdaptiveLengthSampler):
        self.length_sampler = length_sampler
    
    def create_length_mask(self, 
                          target_lengths: torch.Tensor,
                          max_seq_length: int,
                          device: str = 'cpu') -> torch.Tensor:
        """
        创建长度掩码
        
        Args:
            target_lengths: 目标长度 [batch_size]
            max_seq_length: 最大序列长度
            device: 设备
            
        Returns:
            注意力掩码 [batch_size, max_seq_length]
        """
        batch_size = target_lengths.shape[0]
        
        # 创建位置索引
        positions = torch.arange(max_seq_length, device=device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        
        # 创建掩码
        mask = positions < target_lengths.unsqueeze(1)
        
        return mask.float()
    
    def enforce_length_constraint(self,
                                 sequences: torch.Tensor,
                                 target_lengths: torch.Tensor,
                                 pad_token_id: int = 0) -> torch.Tensor:
        """
        强制执行长度约束
        
        Args:
            sequences: 生成的序列 [batch_size, seq_length]
            target_lengths: 目标长度 [batch_size]
            pad_token_id: 填充标记ID
            
        Returns:
            长度调整后的序列
        """
        batch_size, seq_length = sequences.shape
        device = sequences.device
        
        # 创建新的序列张量
        adjusted_sequences = torch.full_like(sequences, pad_token_id)
        
        for i, target_len in enumerate(target_lengths):
            target_len = int(target_len.item())
            
            if target_len <= seq_length:
                # 截断
                adjusted_sequences[i, :target_len] = sequences[i, :target_len]
            else:
                # 填充（保持原序列，其余位置已经是pad_token_id）
                adjusted_sequences[i, :seq_length] = sequences[i]
        
        return adjusted_sequences


def create_length_sampler(config: LengthSamplerConfig) -> AdaptiveLengthSampler:
    """创建长度采样器的工厂函数"""
    return AdaptiveLengthSampler(config)


def get_default_length_config() -> LengthSamplerConfig:
    """获取默认长度采样配置"""
    return LengthSamplerConfig(
        min_length=5,
        max_length=50,
        distribution_type="normal",
        normal_mean=25.0,
        normal_std=8.0,
        use_adaptive_sampling=True,
        condition_dependent=True
    )