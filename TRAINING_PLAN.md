# StructDiff-7.0.0 下一阶段训练计划

## 📊 当前状态总结

### ✅ 已修复的关键问题
1. **CUDA内存不足** - 禁用动态ESMFold加载，使用预计算缓存
2. **缓存文件路径不一致** - 统一MD5哈希命名和目录结构
3. **EMA属性错误** - 修复`AttributeError: 'EMA' object has no attribute 'ema_model'`
4. **结构特征形状不匹配** - 添加动态填充逻辑
5. **DataLoader CUDA张量问题** - 强制CPU张量pin_memory
6. **masked_fill类型错误** - 转换Long类型为Bool类型
7. **检查点保存TypeError** - 修复参数传递问题

### 🔍 当前配置分析
- **配置文件**: `configs/separated_training_production.yaml`
- **结构特征**: 当前 `use_esmfold: false`（已禁用）
- **预计算缓存**: 已启用 `use_predicted_structures: true`
- **训练阶段**: 已完成v1和v2_no_esmfold版本

## 🎯 推荐行动方案

### 方案A：启用结构特征的完整训练（强烈推荐）

**目标**: 启用ESMFold结构特征，验证结构感知训练的效果

**配置修改**:
```yaml
# 需要修改的关键配置
model:
  structure_encoder:
    use_esmfold: true      # 从false改为true
    use_cache: true        # 确保使用预计算缓存
    
data:
  use_predicted_structures: true   # 确保启用
  structure_cache_dir: "./structure_cache"  # 使用正确的缓存目录
```

### 方案B：优化现有无结构特征训练

**目标**: 在当前配置基础上进一步优化性能

**优化参数**:
- 增加批次大小（从2→4或8）
- 调整学习率
- 启用更多GPU并行训练

## 🚀 具体执行步骤

### 步骤1：验证预计算缓存
```bash
# 检查结构缓存是否存在
ls -la sequence/StructDiff-7.0.0/structure_cache/
ls -la sequence/StructDiff-7.0.0/cache/
```

### 步骤2：创建新的配置文件
创建 `configs/separated_training_structure_enabled.yaml`:
- 启用 `use_esmfold: true`
- 保持批次大小为2（避免内存问题）
- 使用预计算缓存

### 步骤3：启动新的训练
```bash
# 阶段1训练（启用结构特征）
python scripts/train_separated.py \
  --config configs/separated_training_structure_enabled.yaml \
  --output-dir ./outputs/separated_structure_v1 \
  --stage 1 \
  --batch-size 2 \
  --stage1-lr 1e-4 \
  --use-amp --use-ema --use-cfg --use-length-control

# 阶段2训练
python scripts/train_separated.py \
  --config configs/separated_training_structure_enabled.yaml \
  --output-dir ./outputs/separated_structure_v1 \
  --stage 2 \
  --stage1-checkpoint ./outputs/separated_structure_v1/checkpoints/latest.pth \
  --batch-size 8 \
  --stage2-lr 5e-5 \
  --use-amp --use-ema --use-cfg --use-length-control
```

## 📈 监控指标

### 训练监控
- **阶段1损失**: 去噪器MSE损失
- **阶段2损失**: 序列重建交叉熵损失
- **验证损失**: 每个epoch的验证集表现
- **GPU利用率**: 实时监控显存使用

### 评估指标
- **伪困惑度**: ESM-2评估的序列质量
- **结构合理性**: 预测结构的plDDT分数
- **新颖性比例**: 与训练集的差异度
- **活性预测**: 抗菌肽活性预测分数

## ⚠️ 风险提示

### 内存管理
- **批次大小**: 保持较小批次（2-4）避免OOM
- **梯度累积**: 使用梯度累积达到有效批次大小
- **混合精度**: 启用AMP减少内存使用

### 缓存验证
- **缓存完整性**: 确保所有训练样本都有预计算结构
- **缓存路径**: 验证缓存目录权限和可用空间

## 🔄 备选方案

如果结构特征训练遇到问题：

1. **回退到无结构特征**: 使用现有v2_no_esmfold配置
2. **混合训练**: 先无结构特征，后微调添加结构特征
3. **分阶段启用**: 逐步增加结构特征的权重

## 📋 下一步行动清单

- [ ] 确认预计算缓存完整性
- [ ] 创建结构感知配置文件
- [ ] 启动阶段1训练（结构感知）
- [ ] 监控训练过程
- [ ] 评估结构特征效果
- [ ] 对比有无结构特征的性能差异

## 🎯 决策建议

**推荐立即执行**: 启用结构特征的完整训练
**理由**: 
1. 所有关键技术问题已修复
2. 预计算缓存机制已验证
3. 结构特征可能显著提升模型性能
4. 当前硬件配置支持结构感知训练

**预期收益**: 
- 更好的结构-序列一致性
- 更高的生物活性预测准确性
- 更合理的肽段结构设计