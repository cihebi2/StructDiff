# train_classification.py - 抗菌肽分类训练脚本
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from tqdm import tqdm
import os
import sys
import gc
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体，避免中文显示警告
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import seaborn as sns

# 设置路径，确保能导入项目模块
sys.path.append('/home/qlyu/StructDiff')

class AMPClassifier(nn.Module):
    """抗菌肽分类器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 序列编码器 - 使用ESM2
        from transformers import EsmModel, EsmTokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(config.model.sequence_encoder.pretrained_model)
        self.sequence_encoder = EsmModel.from_pretrained(config.model.sequence_encoder.pretrained_model)
        
        # 冻结encoder参数（可选）
        if config.model.sequence_encoder.get('freeze_encoder', False):
            for param in self.sequence_encoder.parameters():
                param.requires_grad = False
        
        # 获取编码器的隐藏维度
        encoder_dim = self.sequence_encoder.config.hidden_size
        
        # 分类头
        classifier_layers = []
        hidden_dim = config.model.classifier.hidden_dim
        num_layers = config.model.classifier.get('num_layers', 2)
        dropout = config.model.classifier.get('dropout', 0.1)
        
        # 输入层
        classifier_layers.append(nn.Linear(encoder_dim, hidden_dim))
        classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for _ in range(num_layers - 1):
            classifier_layers.append(nn.Linear(hidden_dim, hidden_dim))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(dropout))
        
        # 输出层
        num_classes = config.task.num_classes
        classifier_layers.append(nn.Linear(hidden_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        print(f"✓ 初始化分类器: {num_classes} 类")
        print(f"  编码器维度: {encoder_dim}")
        print(f"  分类器隐藏维度: {hidden_dim}")
        print(f"  分类器层数: {num_layers}")
    
    def forward(self, sequences, attention_mask=None, labels=None, weights=None):
        """前向传播"""
        # 编码序列
        encoder_outputs = self.sequence_encoder(
            input_ids=sequences,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 使用CLS token的表示或者平均池化
        if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None:
            sequence_representation = encoder_outputs.pooler_output
        else:
            # 使用平均池化
            hidden_states = encoder_outputs.last_hidden_state
            if attention_mask is not None:
                # 加权平均（忽略padding部分）
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
                sequence_representation = sum_embeddings / sum_mask
            else:
                sequence_representation = hidden_states.mean(dim=1)
        
        # 分类
        logits = self.classifier(sequence_representation)
        
        outputs = {'logits': logits}
        
        # 计算损失
        if labels is not None:
            if weights is not None:
                # 加权交叉熵损失
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                losses = loss_fct(logits, labels)
                loss = (losses * weights).mean()
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            
            outputs['loss'] = loss
            outputs['classification_loss'] = loss
        
        return outputs
    
    def count_parameters(self):
        """计算模型参数数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }

class AMPDataset(torch.utils.data.Dataset):
    """抗菌肽数据集"""
    
    def __init__(self, csv_path, tokenizer, config, is_training=True):
        self.config = config
        self.tokenizer = tokenizer
        self.is_training = is_training
        
        # 读取数据
        self.data = pd.read_csv(csv_path)
        print(f"加载数据: {len(self.data)} 样本")
        
        # 检查必要列
        required_cols = ['sequence', 'label']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"数据缺少必要列: {col}")
        
        # 序列长度过滤
        max_length = config.data.get('max_length', 100)
        min_length = config.data.get('min_length', 5)
        
        original_len = len(self.data)
        seq_lengths = self.data['sequence'].str.len()
        self.data = self.data[(seq_lengths >= min_length) & (seq_lengths <= max_length)]
        filtered_len = len(self.data)
        
        if filtered_len < original_len:
            print(f"过滤序列长度: {original_len} -> {filtered_len}")
        
        # 标签和权重
        self.labels = self.data['label'].values
        if 'weight' in self.data.columns:
            self.weights = self.data['weight'].values
        else:
            self.weights = np.ones(len(self.data))
        
        print(f"标签分布: {dict(zip(*np.unique(self.labels, return_counts=True)))}")
        print(f"权重分布: {dict(zip(*np.unique(self.weights, return_counts=True)))}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['sequence']
        label = row['label']
        weight = self.weights[idx]
        
        # 数据增强（仅训练时）
        if self.is_training and self.config.data.augmentation.get('enable', False):
            sequence = self._augment_sequence(sequence)
        
        return {
            'sequence': sequence,
            'label': label,
            'weight': weight
        }
    
    def _augment_sequence(self, sequence):
        """简单的序列数据增强"""
        # 随机掩盖部分氨基酸
        if np.random.random() < self.config.data.augmentation.get('mask_prob', 0):
            seq_list = list(sequence)
            mask_positions = np.random.choice(
                len(seq_list),
                size=max(1, int(len(seq_list) * 0.1)),
                replace=False
            )
            for pos in mask_positions:
                seq_list[pos] = '<mask>'
            sequence = ''.join(seq_list)
        
        return sequence

def collate_fn(batch, tokenizer, config):
    """数据批处理函数"""
    sequences = [item['sequence'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    weights = torch.tensor([item['weight'] for item in batch], dtype=torch.float)
    
    # 编码序列
    max_length = config.data.get('max_length', 100)
    encoded = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    return {
        'sequences': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'labels': labels,
        'weights': weights
    }

def calculate_metrics(predictions, labels, weights=None, num_classes=3):
    """计算评估指标"""
    # 基本指标
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    # 混淆矩阵
    cm = confusion_matrix(labels, predictions)
    
    # ROC AUC (多分类)
    try:
        # 需要预测概率来计算AUC
        auc_roc = None  # 这里需要logits来计算
    except:
        auc_roc = None
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'auc_roc': auc_roc
    }
    
    # 每类指标
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    for i in range(num_classes):
        metrics[f'precision_class_{i}'] = per_class_precision[i] if i < len(per_class_precision) else 0
        metrics[f'recall_class_{i}'] = per_class_recall[i] if i < len(per_class_recall) else 0
        metrics[f'f1_class_{i}'] = per_class_f1[i] if i < len(per_class_f1) else 0
        metrics[f'support_class_{i}'] = support[i] if i < len(support) else 0
    
    return metrics

def train_epoch(model, train_loader, optimizer, device, config, epoch, writer=None, logger=None):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_weights = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} 训练")
    
    for batch_idx, batch in enumerate(pbar):
        # 移动到设备
        sequences = batch['sequences'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        weights = batch['weights'].to(device) if config.task.get('use_weights', False) else None
        
        # 前向传播
        outputs = model(
            sequences=sequences,
            attention_mask=attention_mask,
            labels=labels,
            weights=weights
        )
        
        loss = outputs['loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if config.training.get('max_grad_norm', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        
        # 预测
        with torch.no_grad():
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if weights is not None:
                all_weights.extend(weights.cpu().numpy())
        
        # 更新进度条
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # 记录到TensorBoard
        if writer:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
    
    # 计算epoch指标
    avg_loss = total_loss / len(train_loader)
    metrics = calculate_metrics(all_predictions, all_labels, all_weights, config.task.num_classes)
    metrics['loss'] = avg_loss
    
    return metrics

def validate_model(model, val_loader, device, config, epoch, writer=None, logger=None):
    """验证模型"""
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_weights = []
    all_logits = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="验证")
        
        for batch in pbar:
            # 移动到设备
            sequences = batch['sequences'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            weights = batch['weights'].to(device) if config.task.get('use_weights', False) else None
            
            # 前向传播
            outputs = model(
                sequences=sequences,
                attention_mask=attention_mask,
                labels=labels,
                weights=weights
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            total_loss += loss.item()
            
            # 预测
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(F.softmax(logits, dim=-1).cpu().numpy())
            if weights is not None:
                all_weights.extend(weights.cpu().numpy())
            
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
    
    # 计算指标
    avg_loss = total_loss / len(val_loader)
    metrics = calculate_metrics(all_predictions, all_labels, all_weights, config.task.num_classes)
    metrics['loss'] = avg_loss
    
    # 计算AUC（多分类）
    try:
        all_logits = np.array(all_logits)
        all_labels_np = np.array(all_labels)
        if len(np.unique(all_labels_np)) > 2:
            # 多分类AUC
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(all_labels_np, classes=list(range(config.task.num_classes)))
            auc_scores = []
            for i in range(config.task.num_classes):
                if np.sum(y_test_bin[:, i]) > 0:  # 确保类别存在
                    auc = roc_auc_score(y_test_bin[:, i], all_logits[:, i])
                    auc_scores.append(auc)
            if auc_scores:
                metrics['auc_roc'] = np.mean(auc_scores)
    except Exception as e:
        if logger:
            logger.warning(f"计算AUC失败: {e}")
    
    # 记录到TensorBoard
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/Accuracy', metrics['accuracy'], epoch)
        writer.add_scalar('Val/F1', metrics['f1'], epoch)
        
        # 混淆矩阵
        if 'confusion_matrix' in metrics:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', ax=ax, 
                       xticklabels=config.task.class_names,
                       yticklabels=config.task.class_names)
            ax.set_xlabel('预测')
            ax.set_ylabel('真实')
            ax.set_title(f'Epoch {epoch+1} 混淆矩阵')
            writer.add_figure('Val/Confusion_Matrix', fig, epoch)
            plt.close(fig)
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='抗菌肽分类训练')
    parser.add_argument('--config', type=str, default='configs/classification_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='outputs/classification',
                       help='输出目录')
    parser.add_argument('--data_file', type=str, 
                       default='/home/qlyu/StructDiff/data/train/mulit_peptide_val.csv',
                       help='数据文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 开始抗菌肽分类训练")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"数据文件: {args.data_file}")
    
    # 准备数据
    logger.info("📊 准备数据...")
    
    # 如果processed数据不存在，先处理原始数据
    if not (Path("data/processed/train.csv").exists() and Path("data/processed/val.csv").exists()):
        logger.info("处理原始数据...")
        from prepare_classification_data import prepare_classification_data
        train_path, val_path, test_path = prepare_classification_data(
            input_file=args.data_file,
            output_dir="data/processed"
        )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建模型
    logger.info("🔧 创建模型...")
    model = AMPClassifier(config).to(device)
    
    param_info = model.count_parameters()
    logger.info(f"模型参数: 总计 {param_info['total']:,}, 可训练 {param_info['trainable']:,}")
    
    # 创建数据集和数据加载器
    logger.info("📚 创建数据集...")
    
    # 获取tokenizer
    tokenizer = model.tokenizer
    
    # 创建数据集
    train_dataset = AMPDataset("data/processed/train.csv", tokenizer, config, is_training=True)
    val_dataset = AMPDataset("data/processed/val.csv", tokenizer, config, is_training=False)
    
    # 创建数据加载器
    from functools import partial
    collate_fn_with_config = partial(collate_fn, tokenizer=tokenizer, config=config)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_config,
        num_workers=config.data.get('num_workers', 4),
        pin_memory=config.data.get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_config,
        num_workers=config.data.get('num_workers', 4),
        pin_memory=config.data.get('pin_memory', True)
    )
    
    logger.info(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    logger.info(f"训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay,
        betas=config.training.optimizer.betas
    )
    
    # 创建学习率调度器
    scheduler = None
    if config.training.get('scheduler'):
        if config.training.scheduler.name == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config.training.num_epochs,
                eta_min=config.training.scheduler.min_lr
            )
    
    # TensorBoard
    writer = SummaryWriter(output_dir / "tensorboard")
    
    # 训练循环
    logger.info(f"🏋️ 开始训练 {config.training.num_epochs} epochs...")
    
    best_val_f1 = 0.0
    patience_counter = 0
    patience = config.training.get('early_stopping', {}).get('patience', 10)
    
    for epoch in range(config.training.num_epochs):
        logger.info(f"\n📅 Epoch {epoch + 1}/{config.training.num_epochs}")
        
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, device, config, epoch, writer, logger)
        
        logger.info(f"训练 - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}, "
                   f"F1: {train_metrics['f1']:.4f}")
        
        # 验证
        if (epoch + 1) % config.training.get('validate_every', 1) == 0:
            val_metrics = validate_model(model, val_loader, device, config, epoch, writer, logger)
            
            logger.info(f"验证 - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}")
            
            # 保存最佳模型
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                
                # 保存模型
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'val_metrics': val_metrics,
                    'train_metrics': train_metrics
                }
                
                torch.save(checkpoint, output_dir / "best_model.pth")
                logger.info(f"💾 保存最佳模型 (F1: {best_val_f1:.4f})")
                
                # 保存详细结果
                results = {
                    'epoch': epoch,
                    'train_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                                    for k, v in train_metrics.items() if k != 'confusion_matrix'},
                    'val_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                                  for k, v in val_metrics.items() if k != 'confusion_matrix'}
                }
                
                with open(output_dir / "best_results.json", 'w') as f:
                    json.dump(results, f, indent=2)
                    
            else:
                patience_counter += 1
                
            # 早停
            if patience_counter >= patience:
                logger.info(f"🛑 早停 (patience: {patience})")
                break
        
        # 学习率调度
        if scheduler:
            scheduler.step()
        
        # 清理内存
        if epoch % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 关闭资源
    writer.close()
    
    logger.info("🎉 训练完成！")
    logger.info(f"📊 最佳F1分数: {best_val_f1:.4f}")
    logger.info(f"📁 结果保存在: {output_dir}")

if __name__ == "__main__":
    main() 