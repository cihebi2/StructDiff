#!/usr/bin/env python3
"""
CFG + 长度采样器集成测试
验证Classifier-Free Guidance和长度分布采样器的功能正确性
"""

import torch
import torch.nn as nn
import numpy as np
import pytest
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from structdiff.models.classifier_free_guidance import (
    CFGConfig, ClassifierFreeGuidance, CFGTrainingMixin, create_cfg_model
)
from structdiff.sampling.length_sampler import (
    LengthSamplerConfig, AdaptiveLengthSampler, LengthConstrainedSampler,
    NormalLengthDistribution, UniformLengthDistribution,
    get_default_length_config
)


class TestCFGFunctionality:
    """测试CFG功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 4
        self.seq_length = 20
        self.hidden_dim = 128
        
        # 创建CFG配置
        self.cfg_config = CFGConfig(
            dropout_prob=0.1,
            guidance_scale=2.0,
            adaptive_guidance=True
        )
        
        # 创建CFG实例
        self.cfg = ClassifierFreeGuidance(self.cfg_config)
    
    def test_cfg_initialization(self):
        """测试CFG初始化"""
        assert self.cfg.config.dropout_prob == 0.1
        assert self.cfg.config.guidance_scale == 2.0
        assert self.cfg.config.adaptive_guidance == True
    
    def test_condition_preparation(self):
        """测试条件准备功能"""
        # 有条件输入
        conditions = {
            'peptide_type': torch.tensor([0, 1, 2, 0], device=self.device)
        }
        
        # 训练模式（应用丢弃）
        processed = self.cfg.prepare_conditions(conditions, self.batch_size, training=True)
        assert 'peptide_type' in processed
        assert 'is_unconditional' in processed
        
        # 推理模式（不丢弃）
        processed = self.cfg.prepare_conditions(conditions, self.batch_size, training=False)
        assert torch.equal(processed['peptide_type'], conditions['peptide_type'])
    
    def test_unconditional_batch_creation(self):
        """测试无条件批次创建"""
        uncond_batch = self.cfg._create_unconditional_batch(self.batch_size)
        
        assert uncond_batch['peptide_type'].shape == (self.batch_size,)
        assert torch.all(uncond_batch['peptide_type'] == -1)
        assert torch.all(uncond_batch['is_unconditional'] == True)
    
    def test_condition_dropout(self):
        """测试条件丢弃"""
        conditions = {
            'peptide_type': torch.tensor([0, 1, 2, 0], device=self.device)
        }
        
        # 多次测试以验证随机性
        dropout_counts = 0
        num_tests = 100
        
        for _ in range(num_tests):
            processed = self.cfg._apply_condition_dropout(conditions, self.batch_size)
            dropout_mask = processed['is_unconditional']
            dropout_counts += torch.sum(dropout_mask).item()
        
        # 验证丢弃率接近设定值
        dropout_rate = dropout_counts / (num_tests * self.batch_size)
        assert 0.05 <= dropout_rate <= 0.2  # 允许一定范围的随机波动
    
    def test_adaptive_guidance_scale(self):
        """测试自适应引导强度"""
        total_timesteps = 100
        base_scale = 2.0
        
        # 测试不同时间步的引导强度
        for timestep in [0, 25, 50, 75, 99]:
            scale = self.cfg.adaptive_guidance_scale(timestep, total_timesteps, base_scale)
            assert isinstance(scale, float)
            assert scale > 0
    
    def test_guided_denoising(self):
        """测试引导去噪"""
        # 创建简单的模型函数
        def simple_model(x_t, t, conditions):
            # 简单返回输入（仅用于测试）
            return x_t * 0.9
        
        x_t = torch.randn(self.batch_size, self.seq_length, self.hidden_dim, device=self.device)
        t = torch.randint(0, 100, (self.batch_size,), device=self.device)
        conditions = {
            'peptide_type': torch.tensor([0, 1, 2, 0], device=self.device)
        }
        
        # 测试有引导和无引导的差异
        guided_output = self.cfg.guided_denoising(simple_model, x_t, t, conditions, 2.0)
        unguided_output = self.cfg.guided_denoising(simple_model, x_t, t, conditions, 1.0)
        
        assert guided_output.shape == x_t.shape
        assert unguided_output.shape == x_t.shape
        # 引导输出应该与无引导输出不同
        assert not torch.allclose(guided_output, unguided_output)


class TestLengthSampler:
    """测试长度采样器功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 8
        
        # 创建长度采样配置
        self.length_config = LengthSamplerConfig(
            min_length=5,
            max_length=50,
            distribution_type="normal",
            normal_mean=25.0,
            normal_std=8.0,
            use_adaptive_sampling=True
        )
        
        # 创建长度采样器
        self.length_sampler = AdaptiveLengthSampler(self.length_config)
        self.length_sampler = self.length_sampler.to(self.device)
    
    def test_length_sampler_initialization(self):
        """测试长度采样器初始化"""
        assert self.length_sampler.config.min_length == 5
        assert self.length_sampler.config.max_length == 50
        assert 'normal' in self.length_sampler.base_distributions
        assert 'uniform' in self.length_sampler.base_distributions
    
    def test_basic_length_sampling(self):
        """测试基础长度采样"""
        # 测试不同分布类型
        for dist_type in ['normal', 'uniform', 'gamma', 'beta']:
            lengths = self.length_sampler.sample_lengths(
                batch_size=self.batch_size,
                distribution_type=dist_type,
                device=self.device
            )
            
            assert lengths.shape == (self.batch_size,)
            assert torch.all(lengths >= self.length_config.min_length)
            assert torch.all(lengths <= self.length_config.max_length)
            assert lengths.dtype == torch.long
    
    def test_conditional_length_sampling(self):
        """测试条件长度采样"""
        conditions = {
            'peptide_type': torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], device=self.device)
        }
        
        lengths = self.length_sampler.sample_lengths(
            batch_size=self.batch_size,
            conditions=conditions,
            device=self.device
        )
        
        assert lengths.shape == (self.batch_size,)
        assert torch.all(lengths >= self.length_config.min_length)
        assert torch.all(lengths <= self.length_config.max_length)
    
    def test_length_distribution_properties(self):
        """测试长度分布特性"""
        num_samples = 1000
        
        # 正态分布测试
        normal_dist = self.length_sampler.base_distributions['normal']
        lengths = normal_dist.sample(num_samples, self.device)
        
        # 验证均值和标准差接近设定值
        mean_val = torch.mean(lengths.float()).item()
        std_val = torch.std(lengths.float()).item()
        
        assert abs(mean_val - self.length_config.normal_mean) < 2.0
        assert abs(std_val - self.length_config.normal_std) < 2.0
    
    def test_length_probability_calculation(self):
        """测试长度概率计算"""
        probs = self.length_sampler.get_length_probabilities(device=self.device)
        
        # 验证概率分布的性质
        assert probs.shape[1] == self.length_config.max_length - self.length_config.min_length + 1
        assert torch.allclose(torch.sum(probs, dim=-1), torch.ones(probs.shape[0]))
        assert torch.all(probs >= 0)
    
    def test_temperature_effects(self):
        """测试温度对采样的影响"""
        # 低温度应该产生更集中的分布
        low_temp_lengths = self.length_sampler.sample_lengths(
            batch_size=100,
            temperature=0.5,
            device=self.device
        )
        
        # 高温度应该产生更分散的分布
        high_temp_lengths = self.length_sampler.sample_lengths(
            batch_size=100,
            temperature=1.5,
            device=self.device
        )
        
        # 计算方差
        low_temp_var = torch.var(low_temp_lengths.float()).item()
        high_temp_var = torch.var(high_temp_lengths.float()).item()
        
        # 高温度的方差应该更大（允许一定的随机性）
        assert high_temp_var >= low_temp_var * 0.8


class TestLengthConstrainedSampler:
    """测试长度约束采样器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.length_config = get_default_length_config()
        self.length_sampler = AdaptiveLengthSampler(self.length_config)
        self.constrainer = LengthConstrainedSampler(self.length_sampler)
    
    def test_length_mask_creation(self):
        """测试长度掩码创建"""
        target_lengths = torch.tensor([10, 15, 20, 8], device=self.device)
        max_seq_length = 25
        
        mask = self.constrainer.create_length_mask(target_lengths, max_seq_length, self.device)
        
        assert mask.shape == (4, max_seq_length)
        
        # 验证掩码正确性
        for i, length in enumerate(target_lengths):
            expected_mask = torch.zeros(max_seq_length, device=self.device)
            expected_mask[:length] = 1.0
            assert torch.equal(mask[i], expected_mask)
    
    def test_length_constraint_enforcement(self):
        """测试长度约束强制执行"""
        batch_size = 4
        seq_length = 30
        target_lengths = torch.tensor([10, 15, 20, 25], device=self.device)
        
        # 创建随机序列
        sequences = torch.randint(0, 20, (batch_size, seq_length), device=self.device)
        
        # 应用长度约束
        constrained = self.constrainer.enforce_length_constraint(sequences, target_lengths, pad_token_id=0)
        
        assert constrained.shape == sequences.shape
        
        # 验证每个序列的长度约束
        for i, target_len in enumerate(target_lengths):
            # 检查目标长度内的部分保持不变
            assert torch.equal(constrained[i, :target_len], sequences[i, :target_len])
            # 检查超出部分被填充为0
            if target_len < seq_length:
                assert torch.all(constrained[i, target_len:] == 0)


class TestCFGLengthIntegration:
    """测试CFG和长度采样器的集成"""
    
    def setup_method(self):
        """设置测试环境"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = 4
        
        # 创建配置
        self.cfg_config = CFGConfig(dropout_prob=0.1, guidance_scale=2.0)
        self.length_config = get_default_length_config()
        
        # 创建组件
        self.cfg = ClassifierFreeGuidance(self.cfg_config)
        self.length_sampler = AdaptiveLengthSampler(self.length_config)
        self.length_sampler = self.length_sampler.to(self.device)
    
    def test_integrated_condition_processing(self):
        """测试集成的条件处理"""
        # 创建条件
        conditions = {
            'peptide_type': torch.tensor([0, 1, 2, 0], device=self.device)
        }
        
        # CFG条件处理
        cfg_conditions = self.cfg.prepare_conditions(conditions, self.batch_size, training=True)
        
        # 长度采样（使用CFG处理后的条件）
        lengths = self.length_sampler.sample_lengths(
            batch_size=self.batch_size,
            conditions=cfg_conditions,
            device=self.device
        )
        
        assert lengths.shape == (self.batch_size,)
        assert torch.all(lengths >= self.length_config.min_length)
        assert torch.all(lengths <= self.length_config.max_length)
    
    def test_condition_dropout_with_length_sampling(self):
        """测试条件丢弃对长度采样的影响"""
        conditions = {
            'peptide_type': torch.tensor([0, 1, 2, 0], device=self.device)
        }
        
        # 多次测试以观察条件丢弃的影响
        lengths_with_dropout = []
        lengths_without_dropout = []
        
        for _ in range(10):
            # 带条件丢弃
            cfg_conditions = self.cfg.prepare_conditions(conditions, self.batch_size, training=True)
            lengths_dropout = self.length_sampler.sample_lengths(
                batch_size=self.batch_size,
                conditions=cfg_conditions,
                device=self.device
            )
            lengths_with_dropout.append(lengths_dropout)
            
            # 不带条件丢弃
            lengths_no_dropout = self.length_sampler.sample_lengths(
                batch_size=self.batch_size,
                conditions=conditions,
                device=self.device
            )
            lengths_without_dropout.append(lengths_no_dropout)
        
        # 验证有差异（由于条件丢弃的随机性）
        all_dropout_lengths = torch.stack(lengths_with_dropout)
        all_no_dropout_lengths = torch.stack(lengths_without_dropout)
        
        # 计算方差，有条件丢弃的应该方差更大
        dropout_var = torch.var(all_dropout_lengths.float(), dim=0).mean().item()
        no_dropout_var = torch.var(all_no_dropout_lengths.float(), dim=0).mean().item()
        
        # 允许一定的随机性
        assert dropout_var >= no_dropout_var * 0.8


class TestPerformance:
    """性能测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def test_cfg_performance(self):
        """测试CFG性能"""
        cfg_config = CFGConfig()
        cfg = ClassifierFreeGuidance(cfg_config)
        
        batch_size = 32
        seq_length = 50
        hidden_dim = 768
        
        def mock_model(x, t, cond):
            return torch.randn_like(x)
        
        x_t = torch.randn(batch_size, seq_length, hidden_dim, device=self.device)
        t = torch.randint(0, 1000, (batch_size,), device=self.device)
        conditions = {'peptide_type': torch.randint(0, 3, (batch_size,), device=self.device)}
        
        # 测量时间
        import time
        start_time = time.time()
        
        for _ in range(10):
            output = cfg.guided_denoising(mock_model, x_t, t, conditions, 2.0)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        print(f"CFG平均推理时间: {avg_time:.4f}s")
        assert avg_time < 1.0  # 应该在1秒内完成
    
    def test_length_sampler_performance(self):
        """测试长度采样器性能"""
        length_config = get_default_length_config()
        sampler = AdaptiveLengthSampler(length_config).to(self.device)
        
        batch_size = 128
        
        # 测量时间
        import time
        start_time = time.time()
        
        for _ in range(100):
            lengths = sampler.sample_lengths(batch_size=batch_size, device=self.device)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 100
        
        print(f"长度采样器平均采样时间: {avg_time:.4f}s")
        assert avg_time < 0.1  # 应该在0.1秒内完成


def run_all_tests():
    """运行所有测试"""
    print("🧪 开始CFG + 长度采样器集成测试")
    print("=" * 50)
    
    # CFG功能测试
    print("🔸 测试CFG功能...")
    cfg_test = TestCFGFunctionality()
    cfg_test.setup_method()
    cfg_test.test_cfg_initialization()
    cfg_test.test_condition_preparation()
    cfg_test.test_unconditional_batch_creation()
    cfg_test.test_condition_dropout()
    cfg_test.test_adaptive_guidance_scale()
    cfg_test.test_guided_denoising()
    print("✅ CFG功能测试通过")
    
    # 长度采样器测试
    print("🔸 测试长度采样器...")
    length_test = TestLengthSampler()
    length_test.setup_method()
    length_test.test_length_sampler_initialization()
    length_test.test_basic_length_sampling()
    length_test.test_conditional_length_sampling()
    length_test.test_length_distribution_properties()
    length_test.test_length_probability_calculation()
    length_test.test_temperature_effects()
    print("✅ 长度采样器测试通过")
    
    # 长度约束测试
    print("🔸 测试长度约束...")
    constraint_test = TestLengthConstrainedSampler()
    constraint_test.setup_method()
    constraint_test.test_length_mask_creation()
    constraint_test.test_length_constraint_enforcement()
    print("✅ 长度约束测试通过")
    
    # 集成测试
    print("🔸 测试CFG+长度采样器集成...")
    integration_test = TestCFGLengthIntegration()
    integration_test.setup_method()
    integration_test.test_integrated_condition_processing()
    integration_test.test_condition_dropout_with_length_sampling()
    print("✅ 集成测试通过")
    
    # 性能测试
    print("🔸 测试性能...")
    perf_test = TestPerformance()
    perf_test.setup_method()
    perf_test.test_cfg_performance()
    perf_test.test_length_sampler_performance()
    print("✅ 性能测试通过")
    
    print("\n🎉 所有测试通过！CFG和长度采样器功能正常")


if __name__ == "__main__":
    run_all_tests()