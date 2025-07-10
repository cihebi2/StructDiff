#!/usr/bin/env python3
"""
GPU优化训练脚本 - 使用预计算的结构特征
优化配置：
- 批次大小：16 (从原来的2大幅提升)
- 数据加载工作进程：4 (启用多进程)
- 混合精度训练：启用
- 梯度累积：2 (减少以增加更新频率)
- 预计算结构特征：启用缓存系统
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import logging
from pathlib import Path
from datetime import datetime
import gc
import psutil
import GPUtil
import warnings
warnings.filterwarnings('ignore')

# 设置环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 导入自定义模块
sys.path.append('/home/qlyu/sequence/StructDiff-7.0.0')
from precompute_structure_features import StructureFeatureCache
from transformers import EsmModel, EsmTokenizer
from sklearn.model_selection import train_test_split

# 配置日志
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"gpu_optimized_training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class OptimizedPeptideDataset(Dataset):
    """优化的肽段数据集，使用预计算的结构特征"""
    
    def __init__(self, sequences, labels, structure_cache, max_length=512, use_structure=True):
        self.sequences = sequences
        self.labels = labels
        self.structure_cache = structure_cache
        self.max_length = max_length
        self.use_structure = use_structure
        
        # ESM tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        
        logger.info(f"数据集初始化完成: {len(sequences)} 个序列")
        logger.info(f"使用结构特征: {use_structure}")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        label = self.labels[idx]
        
        # ESM编码
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # 获取结构特征
        structure_features = None
        if self.use_structure and self.structure_cache is not None:
            try:
                structure_features = self.structure_cache.get_cached_structure(sequence)
                if structure_features is not None:
                    # 调整维度以匹配序列长度
                    seq_len = attention_mask.sum().item()
                    if structure_features.shape[0] > seq_len:
                        structure_features = structure_features[:seq_len]
                    elif structure_features.shape[0] < seq_len:
                        # 填充到匹配长度
                        padding = torch.zeros(seq_len - structure_features.shape[0], structure_features.shape[1])
                        structure_features = torch.cat([structure_features, padding], dim=0)
                else:
                    # 如果没有结构特征，创建零填充
                    seq_len = attention_mask.sum().item()
                    structure_features = torch.zeros(seq_len, 1024)  # ESMFold特征维度
            except Exception as e:
                logger.warning(f"获取结构特征失败 (序列 {idx}): {e}")
                seq_len = attention_mask.sum().item()
                structure_features = torch.zeros(seq_len, 1024)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.float32)
        }
        
        if structure_features is not None:
            # 填充结构特征到max_length
            padded_features = torch.zeros(self.max_length, structure_features.shape[1])
            actual_len = min(structure_features.shape[0], self.max_length)
            padded_features[:actual_len] = structure_features[:actual_len]
            result['structure_features'] = padded_features
        
        return result

class StructureFusionTransformer(nn.Module):
    """结构融合Transformer模型"""
    
    def __init__(self, esm_model_name="facebook/esm2_t33_650M_UR50D", num_labels=1, 
                 structure_dim=1024, fusion_dim=512, dropout=0.1):
        super().__init__()
        
        # ESM2模型
        self.esm_model = EsmModel.from_pretrained(esm_model_name)
        self.esm_dim = self.esm_model.config.hidden_size
        
        # 结构特征投影
        self.structure_projection = nn.Linear(structure_dim, fusion_dim)
        
        # 序列特征投影
        self.sequence_projection = nn.Linear(self.esm_dim, fusion_dim)
        
        # 融合层
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 4, num_labels)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask, structure_features=None):
        # ESM2编码
        with autocast():
            esm_outputs = self.esm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            sequence_features = esm_outputs.last_hidden_state
        
        # 投影到融合维度
        seq_projected = self.sequence_projection(sequence_features)
        
        if structure_features is not None:
            # 结构特征投影
            struct_projected = self.structure_projection(structure_features)
            
            # 融合注意力
            fused_features, _ = self.fusion_attention(
                query=seq_projected,
                key=struct_projected,
                value=struct_projected,
                key_padding_mask=~attention_mask.bool()
            )
            
            # 残差连接
            features = seq_projected + fused_features
        else:
            features = seq_projected
        
        # 池化：使用注意力掩码进行平均池化
        mask_expanded = attention_mask.unsqueeze(-1).expand(features.size()).float()
        sum_embeddings = torch.sum(features * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_features = sum_embeddings / sum_mask
        
        # 分类
        pooled_features = self.dropout(pooled_features)
        logits = self.classifier(pooled_features)
        
        return logits

def calculate_metrics(predictions, labels):
    """计算评估指标"""
    predictions = torch.sigmoid(predictions).cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 处理多维标签：如果labels是多维的，展平为1维
    if labels.ndim > 1:
        labels = labels.flatten()
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    
    # 确保是二分类问题
    unique_labels = np.unique(labels)
    logger.info(f"标签唯一值: {unique_labels}")
    
    # 二分类指标
    pred_binary = (predictions > 0.5).astype(int)
    accuracy = (pred_binary == labels).mean()
    
    # 计算其他指标
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    try:
        # 如果是多类问题，使用macro平均
        if len(unique_labels) > 2:
            precision = precision_score(labels, pred_binary, average='macro', zero_division=0)
            recall = recall_score(labels, pred_binary, average='macro', zero_division=0)
            f1 = f1_score(labels, pred_binary, average='macro', zero_division=0)
        else:
            precision = precision_score(labels, pred_binary, zero_division=0)
            recall = recall_score(labels, pred_binary, zero_division=0)
            f1 = f1_score(labels, pred_binary, zero_division=0)
        
        auc = roc_auc_score(labels, predictions) if len(unique_labels) == 2 else 0.0
    except Exception as e:
        logger.warning(f"指标计算失败: {e}")
        precision = recall = f1 = auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def monitor_resources():
    """监控系统资源"""
    # CPU和内存
    cpu_percent = psutil.cpu_percent()
    memory_info = psutil.virtual_memory()
    
    # GPU
    gpu_info = []
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info.append({
                'id': gpu.id,
                'utilization': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                'temperature': gpu.temperature
            })
    except:
        pass
    
    return {
        'cpu_percent': cpu_percent,
        'memory_percent': memory_info.percent,
        'memory_used_gb': memory_info.used / (1024**3),
        'gpus': gpu_info
    }

def main():
    # 训练配置（GPU 1 平衡版本 - 修复多进程CUDA冲突）
    config = {
        'batch_size': 4,  # 适中批次大小，适应剩余内存
        'num_workers': 0,  # 禁用多进程避免CUDA冲突
        'learning_rate': 2e-5,
        'num_epochs': 200,
        'accumulation_steps': 8,  # 增加梯度累积保持等效批次大小32
        'max_length': 512,
        'device': 'cuda:1',  # 使用GPU 1
        'use_amp': True,  # 启用混合精度节省内存
        'pin_memory': False,  # 禁用内存固定（单进程时不需要）
        'prefetch_factor': None,  # 单进程时不使用预取
        'structure_cache_dir': './structure_cache'
    }
    
    logger.info("🚀 开始GPU优化训练")
    logger.info(f"训练配置: {config}")
    
    # 设置设备
    device = torch.device(config['device'])
    logger.info(f"使用设备: {device}")
    
    # 初始化结构特征缓存
    logger.info("📁 初始化结构特征缓存...")
    structure_cache = StructureFeatureCache(cache_dir=config['structure_cache_dir'])
    
    # 加载数据
    logger.info("📊 加载训练数据...")
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/val.csv')
    
    logger.info(f"训练集大小: {len(train_df)}")
    logger.info(f"验证集大小: {len(val_df)}")
    
    # 创建数据集
    train_dataset = OptimizedPeptideDataset(
        sequences=train_df['sequence'].tolist(),
        labels=train_df['label'].tolist(),
        structure_cache=structure_cache,
        max_length=config['max_length'],
        use_structure=True
    )
    
    val_dataset = OptimizedPeptideDataset(
        sequences=val_df['sequence'].tolist(),
        labels=val_df['label'].tolist(),
        structure_cache=structure_cache,
        max_length=config['max_length'],
        use_structure=True
    )
    
    # 创建数据加载器（单进程配置，避免CUDA冲突）
    train_loader_kwargs = {
        'batch_size': config['batch_size'],
        'shuffle': True,
        'num_workers': config['num_workers'],
        'pin_memory': config['pin_memory']
    }
    
    val_loader_kwargs = {
        'batch_size': config['batch_size'],
        'shuffle': False,
        'num_workers': config['num_workers'],
        'pin_memory': config['pin_memory']
    }
    
    # 只有在使用多进程时才添加prefetch_factor和persistent_workers
    if config['num_workers'] > 0 and config['prefetch_factor'] is not None:
        train_loader_kwargs.update({
            'prefetch_factor': config['prefetch_factor'],
            'persistent_workers': True
        })
        val_loader_kwargs.update({
            'prefetch_factor': config['prefetch_factor'],
            'persistent_workers': True
        })
    
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)
    
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")
    
    # 创建模型
    logger.info("🧠 初始化模型...")
    model = StructureFusionTransformer(
        num_labels=1,
        structure_dim=1024,
        fusion_dim=512,
        dropout=0.1
    ).to(device)
    
    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=1e-6
    )
    
    # 混合精度训练
    scaler = GradScaler() if config['use_amp'] else None
    
    # 损失函数
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练循环
    logger.info("🏃 开始训练...")
    best_val_f1 = 0.0
    train_losses = []
    val_losses = []
    
    for epoch in range(config['num_epochs']):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_labels = []
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            structure_features = batch.get('structure_features')
            if structure_features is not None:
                structure_features = structure_features.to(device, non_blocking=True)
            
            # 前向传播
            if config['use_amp']:
                with autocast():
                    logits = model(input_ids, attention_mask, structure_features)
                    loss = criterion(logits.squeeze(), labels)
                    loss = loss / config['accumulation_steps']
                
                # 反向传播
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config['accumulation_steps'] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model(input_ids, attention_mask, structure_features)
                loss = criterion(logits.squeeze(), labels)
                loss = loss / config['accumulation_steps']
                
                loss.backward()
                
                if (batch_idx + 1) % config['accumulation_steps'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * config['accumulation_steps']
            train_predictions.append(logits.squeeze().detach())
            train_labels.append(labels.detach())
            
            # 实时监控
            if batch_idx % 10 == 0:
                resources = monitor_resources()
                gpu_util = resources['gpus'][1]['utilization'] if len(resources['gpus']) > 1 else 0
                gpu_memory = resources['gpus'][1]['memory_percent'] if len(resources['gpus']) > 1 else 0
                
                logger.info(f"Epoch {epoch+1}/{config['num_epochs']}, "
                          f"Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, "
                          f"GPU利用率: {gpu_util:.1f}%, "
                          f"GPU内存: {gpu_memory:.1f}%")
        
        # 计算训练指标
        train_predictions = torch.cat(train_predictions)
        train_labels = torch.cat(train_labels)
        train_metrics = calculate_metrics(train_predictions, train_labels)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                structure_features = batch.get('structure_features')
                if structure_features is not None:
                    structure_features = structure_features.to(device, non_blocking=True)
                
                if config['use_amp']:
                    with autocast():
                        logits = model(input_ids, attention_mask, structure_features)
                        loss = criterion(logits.squeeze(), labels)
                else:
                    logits = model(input_ids, attention_mask, structure_features)
                    loss = criterion(logits.squeeze(), labels)
                
                val_loss += loss.item()
                val_predictions.append(logits.squeeze())
                val_labels.append(labels)
        
        # 计算验证指标
        val_predictions = torch.cat(val_predictions)
        val_labels = torch.cat(val_labels)
        val_metrics = calculate_metrics(val_predictions, val_labels)
        
        # 学习率调度
        scheduler.step()
        
        # 记录损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 计算epoch时间
        epoch_time = time.time() - epoch_start_time
        
        # 资源监控
        resources = monitor_resources()
        
        # 日志记录
        logger.info(f"\n=== Epoch {epoch+1}/{config['num_epochs']} 完成 ===")
        logger.info(f"时间: {epoch_time:.2f}s")
        logger.info(f"训练损失: {avg_train_loss:.4f}")
        logger.info(f"验证损失: {avg_val_loss:.4f}")
        logger.info(f"训练指标: {train_metrics}")
        logger.info(f"验证指标: {val_metrics}")
        logger.info(f"学习率: {scheduler.get_last_lr()[0]:.2e}")
        
        if len(resources['gpus']) > 1:
            gpu_info = resources['gpus'][1]
            logger.info(f"GPU利用率: {gpu_info['utilization']:.1f}%")
            logger.info(f"GPU内存: {gpu_info['memory_percent']:.1f}%")
            logger.info(f"GPU温度: {gpu_info['temperature']}°C")
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['f1'],
                'config': config
            }, 'best_model_optimized.pth')
            logger.info(f"💾 保存最佳模型 (F1: {val_metrics['f1']:.4f})")
        
        # 内存清理
        if epoch % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    logger.info("✅ 训练完成!")
    logger.info(f"最佳验证F1分数: {best_val_f1:.4f}")

if __name__ == "__main__":
    main() 