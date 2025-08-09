# StructDiff数据流图文档

## 概述
本文档通过详细的数据流图展示StructDiff系统中数据的流动过程，帮助理解各组件之间的交互关系。

## 1. 整体系统数据流

```
┌─────────────────┐
│   输入数据      │
│  (CSV文件)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ PeptideDataset  │────▶│ ESMFold(可选)    │
│   数据加载      │     │  结构预测        │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│  DataCollator   │◀────│ Structure Cache  │
│   批处理        │     │  结构缓存        │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐
│   DataLoader    │
│   数据加载器    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Training Loop   │
│   训练循环      │
└─────────────────┘
```

## 2. 训练阶段数据流

### 2.1 阶段1：去噪器训练

```
输入批次 (sequences, structures, conditions)
    │
    ▼
┌─────────────────────┐
│  ESM-2 Encoder      │
│  序列编码(冻结)     │
└──────────┬──────────┘
           │ embeddings
           ▼
┌─────────────────────┐     ┌─────────────────────┐
│  Add Noise          │     │ Structure Encoder   │
│  添加噪声 t∈[0,T]   │     │ 结构编码(可训练)    │
└──────────┬──────────┘     └──────────┬──────────┘
           │ noisy_emb                  │ struct_features
           ▼                            ▼
        ┌──────────────────────────────────┐
        │      Structure-Aware Denoiser    │
        │         结构感知去噪器            │
        │  (Cross-Attention + Self-Attn)   │
        └──────────────┬───────────────────┘
                       │ predicted_noise
                       ▼
        ┌──────────────────────────────────┐
        │         MSE Loss                 │
        │      均方误差损失计算             │
        └──────────────────────────────────┘
```

### 2.2 阶段2：解码器训练

```
输入批次 (sequences, attention_mask)
    │
    ▼
┌─────────────────────┐
│  ESM-2 Encoder      │
│  序列编码(冻结)     │
└──────────┬──────────┘
           │ clean_embeddings
           ▼
┌─────────────────────┐
│  Decoder Layers     │
│  解码器层(可训练)    │
└──────────┬──────────┘
           │ decoded_embeddings
           ▼
┌─────────────────────┐
│ Decode Projection   │
│  投影到词汇表       │
└──────────┬──────────┘
           │ logits
           ▼
┌─────────────────────┐
│  Cross Entropy Loss │
│   交叉熵损失        │
└─────────────────────┘
```

## 3. 生成阶段数据流

```
随机噪声 x_T ~ N(0, I)
    │
    ▼
┌─────────────────────────────────┐
│  Iterative Denoising Process    │
│       迭代去噪过程               │
│                                 │
│  for t in T-1 to 0:            │
│    ┌────────────────────┐      │
│    │ Denoise at step t  │      │
│    └─────────┬──────────┘      │
│              ▼                  │
│    ┌────────────────────┐      │
│    │ Apply conditions   │      │
│    │ (CFG, length, etc) │      │
│    └─────────┬──────────┘      │
│              ▼                  │
│    x_{t-1} = sample            │
└──────────────┬──────────────────┘
               │ x_0 (clean embeddings)
               ▼
┌─────────────────────┐
│   Decode to Tokens  │
│   解码为序列        │
└──────────┬──────────┘
           │ token_ids
           ▼
┌─────────────────────┐
│  Tokenizer Decode   │
│   转换为氨基酸序列   │
└──────────┬──────────┘
           │
           ▼
    生成的肽序列
```

## 4. 条件控制数据流

```
条件输入 (peptide_type, target_length)
    │
    ├─────────────────┐
    │                 ▼
    │    ┌────────────────────────┐
    │    │ Condition Embedding    │
    │    │   条件嵌入              │
    │    └──────────┬─────────────┘
    │               │
    ▼               ▼
┌───────────────────────────────┐
│  Classifier-Free Guidance     │
│      分类器自由引导            │
│                               │
│  - Conditional prediction     │
│  - Unconditional prediction   │
│  - Guided combination         │
└───────────────┬───────────────┘
                │
                ▼
         引导后的预测
```

## 5. 评估流程数据流

```
生成的序列集合
    │
    ├────────────────┬────────────────┬─────────────────┐
    ▼                ▼                ▼                 ▼
┌─────────┐    ┌─────────┐    ┌──────────┐    ┌──────────┐
│ESM-2    │    │ESMFold  │    │modlAMP   │    │BioPython │
│伪困惑度  │    │pLDDT    │    │不稳定性   │    │相似性     │
└────┬────┘    └────┬────┘    └────┬─────┘    └────┬─────┘
     │              │               │                │
     └──────────────┴───────────────┴────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ 综合评估报告  │
                    └───────────────┘
```

## 6. 结构特征处理流程

```
输入序列
    │
    ▼
┌─────────────────┐
│   ESMFold       │
│  结构预测       │
└────────┬────────┘
         │ 3D coordinates
         ▼
┌─────────────────────────────┐
│   Feature Extraction        │
│      特征提取               │
│ - Distance matrix           │
│ - Contact map               │
│ - Dihedral angles          │
│ - Secondary structure       │
│ - pLDDT scores             │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Multi-Scale Encoder        │
│    多尺度编码器             │
│ ┌─────────┐  ┌──────────┐  │
│ │ Local   │  │ Global   │  │
│ │ Encoder │  │ Encoder  │  │
│ └────┬────┘  └────┬─────┘  │
│      └──────┬─────┘        │
│             ▼              │
│      Feature Fusion        │
└──────────┬──────────────────┘
           │
           ▼
    结构特征向量
```

## 7. 检查点和恢复流程

```
训练状态
    │
    ▼
┌─────────────────┐
│ CheckpointManager│
│   检查点管理器   │
└────────┬────────┘
         │ 保存
         ▼
┌─────────────────────────┐
│   Checkpoint File       │
│ - Model state dict      │
│ - Optimizer state       │
│ - Scheduler state       │
│ - Training statistics   │
│ - EMA state (if used)   │
└────────┬────────────────┘
         │ 加载
         ▼
┌─────────────────┐
│  Resume Training │
│   恢复训练       │
└─────────────────┘
```

## 8. 内存优化数据流

```
大批次数据
    │
    ▼
┌─────────────────────────┐
│  Gradient Accumulation  │
│     梯度累积            │
│                         │
│  for mini_batch:       │
│    forward()           │
│    loss.backward()     │
│    accumulate grads    │
│                         │
│  optimizer.step()      │
└─────────────────────────┘
    │
    ▼
有效大批次训练
```

## 9. 并行处理数据流（多GPU）

```
     主GPU (rank 0)
         │
         ▼
┌─────────────────────┐
│  DistributedSampler │
│   分布式采样器      │
└────────┬────────────┘
         │ 数据分片
    ┌────┴────┬────┬────┐
    ▼         ▼    ▼    ▼
  GPU 0    GPU 1 GPU 2 GPU 3
    │         │    │    │
    ▼         ▼    ▼    ▼
  前向      前向  前向  前向
    │         │    │    │
    ▼         ▼    ▼    ▼
  反向      反向  反向  反向
    │         │    │    │
    └────┬────┴────┴────┘
         ▼
   AllReduce梯度
         │
         ▼
   参数更新（所有GPU）
```

## 10. 长度控制数据流

```
目标长度 L
    │
    ▼
┌─────────────────┐
│ Length Sampler  │
│  长度采样器     │
└────────┬────────┘
         │ length_embedding
         ▼
┌─────────────────────────┐
│  Length Controller      │
│    长度控制器           │
│                         │
│ - Encode length info    │
│ - Guide generation      │
│ - Enforce constraints   │
└────────┬────────────────┘
         │
         ▼
  长度受控的生成
```

## 关键数据结构

### 1. 批次数据结构
```python
batch = {
    'sequences': Tensor[B, L],           # token IDs
    'attention_mask': Tensor[B, L],      # attention mask
    'structures': {                      # 可选
        'positions': Tensor[B, L, 37, 3],
        'plddt': Tensor[B, L],
        'angles': Tensor[B, L, 10],
        ...
    },
    'conditions': {                      # 可选
        'peptide_type': Tensor[B],
        'target_length': Tensor[B],
        ...
    }
}
```

### 2. 模型输出结构
```python
output = {
    'loss': Tensor[1],                   # 总损失
    'losses': {                          # 分项损失
        'diffusion_loss': Tensor[1],
        'structure_loss': Tensor[1],
        ...
    },
    'predictions': Tensor[B, L, D],      # 预测结果
    'attention_weights': Tensor[...],    # 注意力权重
}
```

## 性能考虑

### 1. 数据加载优化
- 使用多进程数据加载 (num_workers > 0)
- 预取数据 (prefetch_factor)
- Pin memory for GPU transfer

### 2. 内存管理
- 结构特征缓存
- 梯度检查点（gradient checkpointing）
- 混合精度训练（AMP）

### 3. 计算优化
- 批次动态填充
- 序列长度分组
- 高效的注意力实现

## 总结

StructDiff的数据流设计体现了以下特点：
1. **模块化**: 各组件职责明确，数据流清晰
2. **灵活性**: 支持多种条件和控制方式
3. **效率**: 通过缓存和优化减少重复计算
4. **可扩展**: 易于添加新的特征和条件

理解这些数据流对于：
- 调试问题
- 优化性能
- 添加新功能
- 理解系统行为

都具有重要意义。

---

最后更新：2025-08-04
作者：StructDiff架构改进项目