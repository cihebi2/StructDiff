#!/usr/bin/env python3
"""
Classifier-Free Guidance (CFG) implementation for StructDiff
基于CPL-Diff论文的分类器自由引导机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
from dataclasses import dataclass


@dataclass
class CFGConfig:
    """CFG配置类"""
    # 训练时配置
    dropout_prob: float = 0.1  # 条件丢弃概率
    unconditional_token: str = "<UNCOND>"  # 无条件标记
    
    # 推理时配置
    guidance_scale: float = 2.0  # 默认引导强度
    guidance_scale_range: Tuple[float, float] = (1.0, 5.0)  # 引导强度范围
    
    # 高级配置
    adaptive_guidance: bool = True  # 自适应引导强度
    multi_level_guidance: bool = True  # 多级引导
    guidance_schedule: str = "constant"  # constant, linear, cosine


class ClassifierFreeGuidance(nn.Module):
    """
    分类器自由引导模块
    实现训练时的条件丢弃和推理时的引导采样
    """
    
    def __init__(self, config: CFGConfig):
        super().__init__()
        self.config = config
        
        # 条件处理
        self.unconditional_embedding = None
        self.condition_embeddings = nn.ModuleDict()
        
        # 引导强度调度器
        self.guidance_scheduler = self._create_guidance_scheduler()
    
    def _create_guidance_scheduler(self):
        """创建引导强度调度器"""
        if self.config.guidance_schedule == "constant":
            return lambda t: self.config.guidance_scale
        elif self.config.guidance_schedule == "linear":
            return lambda t: self.config.guidance_scale * (1 - t)
        elif self.config.guidance_schedule == "cosine":
            return lambda t: self.config.guidance_scale * (0.5 * (1 + np.cos(np.pi * t)))
        else:
            return lambda t: self.config.guidance_scale
    
    def prepare_conditions(self, 
                          conditions: Optional[Dict[str, torch.Tensor]], 
                          batch_size: int,
                          training: bool = True) -> Dict[str, torch.Tensor]:
        """
        准备条件信息，支持训练时的随机丢弃
        
        Args:
            conditions: 输入条件字典
            batch_size: 批次大小
            training: 是否为训练模式
            
        Returns:
            处理后的条件字典
        """
        if conditions is None:
            # 返回无条件标记
            return self._create_unconditional_batch(batch_size)
        
        if training and self.config.dropout_prob > 0:
            # 训练时随机丢弃条件
            return self._apply_condition_dropout(conditions, batch_size)
        
        return conditions
    
    def _create_unconditional_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """创建无条件批次"""
        return {
            'peptide_type': torch.full((batch_size,), -1, dtype=torch.long),  # -1表示无条件
            'is_unconditional': torch.ones(batch_size, dtype=torch.bool)
        }
    
    def _apply_condition_dropout(self, 
                                conditions: Dict[str, torch.Tensor], 
                                batch_size: int) -> Dict[str, torch.Tensor]:
        """
        应用条件丢弃
        
        Args:
            conditions: 原始条件
            batch_size: 批次大小
            
        Returns:
            应用丢弃后的条件
        """
        device = next(iter(conditions.values())).device
        
        # 创建丢弃掩码
        dropout_mask = torch.rand(batch_size, device=device) < self.config.dropout_prob
        
        processed_conditions = {}
        for key, value in conditions.items():
            if isinstance(value, torch.Tensor):
                # 对标量条件应用丢弃
                if value.dim() == 1:
                    processed_value = value.clone()
                    processed_value[dropout_mask] = -1  # 用-1表示丢弃的条件
                    processed_conditions[key] = processed_value
                else:
                    processed_conditions[key] = value
            else:
                processed_conditions[key] = value
        
        # 添加丢弃标记
        processed_conditions['is_unconditional'] = dropout_mask
        
        return processed_conditions
    
    def guided_denoising(self,
                        model: nn.Module,
                        x_t: torch.Tensor,
                        t: torch.Tensor,
                        conditions: Optional[Dict[str, torch.Tensor]] = None,
                        guidance_scale: Optional[float] = None) -> torch.Tensor:
        """
        执行引导去噪
        
        Args:
            model: 去噪模型
            x_t: 噪声输入
            t: 时间步
            conditions: 条件信息
            guidance_scale: 引导强度（可覆盖配置）
            
        Returns:
            引导后的模型输出
        """
        if guidance_scale is None:
            guidance_scale = self.config.guidance_scale
        
        if conditions is None or guidance_scale <= 1.0:
            # 无条件生成或不使用引导
            return model(x_t, t, conditions)
        
        # CFG双路预测
        batch_size = x_t.shape[0]
        
        # 1. 条件预测
        cond_output = model(x_t, t, conditions)
        
        # 2. 无条件预测
        uncond_conditions = self._create_unconditional_batch(batch_size)
        # 确保设备一致
        for key, value in uncond_conditions.items():
            if isinstance(value, torch.Tensor):
                uncond_conditions[key] = value.to(x_t.device)
        
        uncond_output = model(x_t, t, uncond_conditions)
        
        # 3. CFG组合：ε_uncond + w * (ε_cond - ε_uncond)
        guided_output = uncond_output + guidance_scale * (cond_output - uncond_output)
        
        return guided_output
    
    def adaptive_guidance_scale(self, 
                               timestep: int, 
                               total_timesteps: int,
                               base_scale: Optional[float] = None) -> float:
        """
        自适应引导强度调整
        
        Args:
            timestep: 当前时间步
            total_timesteps: 总时间步数
            base_scale: 基础引导强度
            
        Returns:
            调整后的引导强度
        """
        if base_scale is None:
            base_scale = self.config.guidance_scale
        
        if not self.config.adaptive_guidance:
            return base_scale
        
        # 归一化时间步 (0->1)
        t_norm = timestep / total_timesteps
        
        # 应用调度函数
        scale_multiplier = self.guidance_scheduler(t_norm)
        
        return base_scale * scale_multiplier
    
    def multi_level_guidance(self,
                            model: nn.Module,
                            x_t: torch.Tensor,
                            t: torch.Tensor,
                            conditions: Dict[str, torch.Tensor],
                            guidance_scales: Dict[str, float]) -> torch.Tensor:
        """
        多级引导：对不同条件类型应用不同的引导强度
        
        Args:
            model: 去噪模型
            x_t: 噪声输入  
            t: 时间步
            conditions: 条件信息
            guidance_scales: 各条件的引导强度
            
        Returns:
            多级引导后的输出
        """
        if not self.config.multi_level_guidance:
            # 使用标准CFG
            return self.guided_denoising(model, x_t, t, conditions)
        
        batch_size = x_t.shape[0]
        device = x_t.device
        
        # 无条件预测
        uncond_conditions = self._create_unconditional_batch(batch_size)
        for key, value in uncond_conditions.items():
            if isinstance(value, torch.Tensor):
                uncond_conditions[key] = value.to(device)
        
        uncond_output = model(x_t, t, uncond_conditions)
        
        # 累积引导效果
        total_guidance = torch.zeros_like(uncond_output)
        
        for condition_type, guidance_scale in guidance_scales.items():
            if condition_type in conditions and guidance_scale > 0:
                # 创建单一条件
                single_condition = self._create_single_condition(
                    conditions, condition_type, batch_size, device
                )
                
                # 单条件预测
                single_cond_output = model(x_t, t, single_condition)
                
                # 累积引导
                guidance_effect = guidance_scale * (single_cond_output - uncond_output)
                total_guidance += guidance_effect
        
        return uncond_output + total_guidance
    
    def _create_single_condition(self,
                               conditions: Dict[str, torch.Tensor],
                               target_condition: str,
                               batch_size: int,
                               device: torch.device) -> Dict[str, torch.Tensor]:
        """创建单一条件字典"""
        single_condition = self._create_unconditional_batch(batch_size)
        for key, value in single_condition.items():
            if isinstance(value, torch.Tensor):
                single_condition[key] = value.to(device)
        
        # 只保留目标条件
        if target_condition in conditions:
            single_condition[target_condition] = conditions[target_condition]
            single_condition['is_unconditional'] = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        return single_condition


class CFGTrainingMixin:
    """
    CFG训练混入类
    为现有模型添加CFG训练支持
    """
    
    def __init__(self, cfg_config: CFGConfig):
        self.cfg = ClassifierFreeGuidance(cfg_config)
        self.cfg_config = cfg_config
    
    def forward_with_cfg(self,
                        model_forward_fn,
                        x_t: torch.Tensor,
                        t: torch.Tensor,
                        conditions: Optional[Dict[str, torch.Tensor]] = None,
                        training: bool = True) -> torch.Tensor:
        """
        带CFG的前向传播
        
        Args:
            model_forward_fn: 原始模型的前向函数
            x_t: 噪声输入
            t: 时间步
            conditions: 条件信息
            training: 是否训练模式
            
        Returns:
            模型输出
        """
        batch_size = x_t.shape[0]
        
        # 准备条件（训练时可能丢弃）
        processed_conditions = self.cfg.prepare_conditions(
            conditions, batch_size, training
        )
        
        # 执行前向传播
        return model_forward_fn(x_t, t, processed_conditions)
    
    def sample_with_cfg(self,
                       model_forward_fn,
                       x_t: torch.Tensor,
                       t: torch.Tensor,
                       conditions: Optional[Dict[str, torch.Tensor]] = None,
                       guidance_scale: Optional[float] = None,
                       timestep: Optional[int] = None,
                       total_timesteps: Optional[int] = None) -> torch.Tensor:
        """
        带CFG的采样
        
        Args:
            model_forward_fn: 原始模型的前向函数
            x_t: 噪声输入
            t: 时间步
            conditions: 条件信息
            guidance_scale: 引导强度
            timestep: 当前时间步索引
            total_timesteps: 总时间步数
            
        Returns:
            引导后的模型输出
        """
        # 自适应引导强度
        if (self.cfg_config.adaptive_guidance and 
            timestep is not None and total_timesteps is not None):
            guidance_scale = self.cfg.adaptive_guidance_scale(
                timestep, total_timesteps, guidance_scale
            )
        
        # 执行引导去噪
        return self.cfg.guided_denoising(
            lambda x, time_step, cond: model_forward_fn(x, time_step, cond),
            x_t, t, conditions, guidance_scale
        )


def create_cfg_model(base_model: nn.Module, cfg_config: CFGConfig) -> nn.Module:
    """
    为现有模型添加CFG支持
    
    Args:
        base_model: 基础模型
        cfg_config: CFG配置
        
    Returns:
        支持CFG的模型
    """
    class CFGWrappedModel(base_model.__class__, CFGTrainingMixin):
        def __init__(self, original_model, cfg_config):
            # 复制原始模型的所有属性
            for attr_name in dir(original_model):
                if not attr_name.startswith('_') and hasattr(original_model, attr_name):
                    attr_value = getattr(original_model, attr_name)
                    if not callable(attr_value):
                        setattr(self, attr_name, attr_value)
            
            # 复制模型参数
            self.load_state_dict(original_model.state_dict())
            
            # 初始化CFG
            CFGTrainingMixin.__init__(self, cfg_config)
        
        def forward(self, x_t, t, conditions=None, use_cfg=False, guidance_scale=None, **kwargs):
            # 获取原始前向函数
            original_forward = super(CFGWrappedModel, self).forward
            
            if use_cfg and not self.training:
                # 推理时使用CFG
                return self.sample_with_cfg(
                    original_forward, x_t, t, conditions, guidance_scale, **kwargs
                )
            elif self.training:
                # 训练时使用CFG（条件丢弃）
                return self.forward_with_cfg(
                    original_forward, x_t, t, conditions, training=True
                )
            else:
                # 标准前向传播
                return original_forward(x_t, t, conditions, **kwargs)
    
    return CFGWrappedModel(base_model, cfg_config)