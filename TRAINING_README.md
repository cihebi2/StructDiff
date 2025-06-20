# StructDiff 多肽生成训练指南

## 概述

本项目使用StructDiff模型进行多肽生成，集成ESMFold进行结构预测。

## 数据集信息

- **训练集**: 2,815 个样本
- **验证集**: 939 个样本  
- **测试集**: 939 个样本
- **多肽类型**: 抗菌肽(0), 抗真菌肽(1), 抗病毒肽(2)
- **序列长度**: 5-50 个氨基酸

## 快速开始

### 1. 测试运行 (推荐首次使用)
```bash
./run_training.sh test
```
- 仅训练3个epochs
- 快速验证环境和数据

### 2. Debug模式
```bash
./run_training.sh debug
```
- 使用小数据集 (100训练+50验证)
- 适合调试代码

### 3. 完整训练
```bash
./run_training.sh
```
- 训练50个epochs
- 使用完整数据集

## 配置说明

配置文件: `configs/peptide_esmfold_config.yaml`

### 关键配置项：

```yaml
# ESMFold配置
model:
  structure_encoder:
    use_esmfold: true  # 启用ESMFold
    
data:
  use_predicted_structures: true  # 使用预测结构
  batch_size: 16  # 批量大小
  
training:
  num_epochs: 50
  lr: 5e-5
  gradient_accumulation_steps: 2
```

## 内存优化

- **批量大小**: 16 (可根据GPU内存调整)
- **梯度累积**: 2步 (有效批量大小32)
- **混合精度**: 启用FP16
- **结构缓存**: 自动缓存ESMFold预测结果

## 监控训练

### TensorBoard
```bash
tensorboard --logdir outputs/peptide_esmfold_generation/tensorboard
```

### Weights & Biases
- 项目名: `StructDiff-Peptide-ESMFold`
- 自动记录损失、学习率等指标

## 输出结构

```
outputs/peptide_esmfold_generation/
├── checkpoints/        # 模型检查点
├── logs/              # 训练日志
└── tensorboard/       # TensorBoard日志
```

## 常见问题

### 1. 内存不足
- 减小batch_size (如8)
- 增加gradient_accumulation_steps
- 减少num_workers

### 2. ESMFold加载失败
- 检查网络连接
- 确保有足够磁盘空间
- 查看日志中的错误信息

### 3. 训练中断恢复
```bash
python scripts/train_peptide_esmfold.py \
    --config configs/peptide_esmfold_config.yaml \
    --resume outputs/peptide_esmfold_generation/checkpoints/best_model.pt
```

## 性能预期

### GPU内存使用
- ESMFold: ~6GB
- 训练模型: ~4GB
- 建议总显存: ≥12GB

### 训练时间 (V100/A100)
- 测试运行: ~10分钟
- 完整训练: ~6-8小时

## 结果评估

训练完成后，使用evaluate.py评估生成的多肽:

```bash
python scripts/evaluate.py \
    --peptides generated_peptides.fasta \
    --metrics all \
    --predict_structure
```

## 生成新多肽

使用训练好的模型生成新多肽:

```bash
python scripts/generate.py \
    --checkpoint outputs/peptide_esmfold_generation/checkpoints/best_model.pt \
    --num_samples 100 \
    --peptide_type antimicrobial \
    --output generated_antimicrobial.fasta
```

## 技术支持

如遇问题，请检查:
1. 日志文件 (`outputs/*/logs/`)
2. GPU显存使用情况
3. 数据文件完整性

---

**祝你训练顺利！** 🚀 