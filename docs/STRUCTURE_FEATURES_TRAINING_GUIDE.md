# 结构特征训练指南

## 🎯 概述

基于已完成的序列特征训练，现在可以启用结构特征来进一步提升模型性能。这个阶段将集成 ESMFold 结构预测，让模型同时学习序列和结构信息。

## 📊 当前状态

### ✅ 已完成的序列特征训练
- **训练轮数**: 200 epochs
- **最终训练损失**: 0.159720
- **最佳验证损失**: 0.160608
- **模型参数**: 211,932,007 (约2.12亿参数)
- **ESMFold状态**: 可用但未使用
- **结构特征**: 未启用

### 🎯 下一步目标
- 启用 ESMFold 结构预测
- 集成结构特征到训练过程
- 进一步降低损失并提升生成质量

## 🔧 结构特征训练配置

### 1. 主要变化

#### 1.1 配置修改
```python
# 启用结构特征
config.data.use_predicted_structures = True
config.model.structure_encoder.use_esmfold = True
```

#### 1.2 训练参数调整
```python
# 为适应ESMFold的内存需求
batch_size = 2  # 从8降低到2
gradient_accumulation_steps = 8  # 从2增加到8
effective_batch_size = 16  # 保持相同的有效批次大小
learning_rate = 5e-5  # 从1e-4降低到5e-5
num_epochs = 100  # 结构特征训练较少的epoch
```

### 2. 结构特征提取

#### 2.1 提取的结构信息
```python
# 从ESMFold输出中提取的特征
features = [
    'plddt',              # 置信度分数
    'distance_matrix',    # 距离矩阵统计
    'contact_map',        # 接触图
    'secondary_structure' # 二级结构
]
```

#### 2.2 特征处理流程
```python
def extract_structure_tensor(structure_dict, device, max_length=52):
    """提取并处理结构特征"""
    features = []
    
    # 1. pLDDT分数处理
    if 'plddt' in structure_dict:
        plddt = structure_dict['plddt'].to(device)
        # 长度标准化
        plddt = normalize_length(plddt, max_length)
        features.append(plddt.unsqueeze(-1))
    
    # 2. 距离矩阵统计
    if 'distance_matrix' in structure_dict:
        dist_matrix = structure_dict['distance_matrix'].to(device)
        mean_distances = dist_matrix.mean(dim=-1)
        features.append(mean_distances.unsqueeze(-1))
    
    # 3. 接触图统计
    if 'contact_map' in structure_dict:
        contact_map = structure_dict['contact_map'].to(device)
        contact_counts = contact_map.sum(dim=-1)
        features.append(contact_counts.unsqueeze(-1))
    
    # 4. 二级结构one-hot编码
    if 'secondary_structure' in structure_dict:
        ss = structure_dict['secondary_structure'].to(device)
        ss_onehot = torch.nn.functional.one_hot(ss, num_classes=3).float()
        features.append(ss_onehot)
    
    return torch.cat(features, dim=-1)
```

## 🚀 启动结构特征训练

### 1. 检查前提条件

```bash
# 1. 确认预训练模型存在
ls -la outputs/full_training_200_esmfold_fixed/best_model.pt

# 2. 检查GPU状态
nvidia-smi

# 3. 检查可用内存
free -h
```

### 2. 启动训练

#### 方法1: 使用启动脚本
```bash
# 给脚本执行权限
chmod +x start_structure_training.sh

# 运行启动脚本
./start_structure_training.sh
```

#### 方法2: 直接运行
```bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/home/qlyu/sequence/StructDiff-7.0.0:$PYTHONPATH"

# 运行训练脚本
python full_train_with_structure_features_enabled.py
```

#### 方法3: 后台运行
```bash
# 后台运行并记录日志
nohup python full_train_with_structure_features_enabled.py > structure_training.log 2>&1 &

# 查看进程
ps aux | grep full_train_with_structure_features_enabled
```

### 3. 监控训练

```bash
# 查看实时日志
tail -f outputs/structure_feature_training/training.log

# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看训练指标
cat outputs/structure_feature_training/training_metrics.json | jq '.'
```

## 📈 预期训练表现

### 1. 性能指标

#### 1.1 系统资源使用
```
GPU内存使用: ~18-20GB (包含ESMFold)
训练速度: ~8-12秒/批次 (比之前慢4-5倍)
每epoch时间: ~2-3小时
预计总训练时间: ~200-300小时 (100 epochs)
```

#### 1.2 损失收敛预期
```
初始损失: ~0.16 (从预训练模型开始)
目标损失: ~0.12-0.14 (预期改善)
收敛时间: 50-80 epochs
```

### 2. 训练阶段

#### 阶段1: 结构特征适应 (Epochs 1-20)
- 模型学习结构特征表示
- 损失可能先上升后下降
- 预期损失: 0.16 → 0.15

#### 阶段2: 序列-结构协同 (Epochs 21-60)
- 序列和结构特征开始协同工作
- 损失稳定下降
- 预期损失: 0.15 → 0.13

#### 阶段3: 精细调优 (Epochs 61-100)
- 模型性能进一步优化
- 损失缓慢收敛
- 预期损失: 0.13 → 0.12

## 🔍 关键监控指标

### 1. 训练指标
```python
# 关键指标监控
metrics_to_watch = {
    'train_loss': '训练损失',
    'val_loss': '验证损失',
    'gpu_memory': 'GPU内存使用',
    'batch_time': '批次处理时间',
    'structure_success_rate': '结构预测成功率'
}
```

### 2. 异常检测
```python
# 异常情况检测
def check_training_health():
    # 1. GPU内存超限
    if gpu_memory > 22_000:  # 超过22GB
        alert("GPU内存使用过高")
    
    # 2. 损失异常
    if current_loss > previous_loss * 1.5:
        alert("损失突然增加")
    
    # 3. 结构预测失败率过高
    if structure_failure_rate > 0.3:
        alert("结构预测失败率过高")
    
    # 4. 训练时间异常
    if batch_time > 30:  # 超过30秒/批次
        alert("训练速度异常慢")
```

## 🛠️ 故障排除

### 1. 常见问题

#### 1.1 GPU内存不足
```bash
# 解决方案
# 1. 进一步降低批次大小
batch_size = 1
gradient_accumulation_steps = 16

# 2. 清理GPU内存
torch.cuda.empty_cache()
gc.collect()

# 3. 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

#### 1.2 ESMFold初始化失败
```python
# 检查ESMFold状态
if not esmfold_wrapper.available:
    logger.error("ESMFold不可用，检查以下问题：")
    logger.error("1. 是否有足够的GPU内存")
    logger.error("2. 是否正确安装了ESMFold依赖")
    logger.error("3. 是否有网络连接下载模型")
```

#### 1.3 结构特征维度不匹配
```python
# 调试结构特征维度
def debug_structure_features(structure_dict):
    print("结构特征维度:")
    for key, value in structure_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
```

### 2. 性能优化

#### 2.1 内存优化
```python
# 优化策略
optimizations = [
    "减少批次大小",
    "增加梯度累积步数",
    "定期清理GPU内存",
    "使用混合精度训练",
    "优化数据加载器"
]
```

#### 2.2 速度优化
```python
# 加速技巧
speed_improvements = [
    "缓存结构预测结果",
    "使用多进程数据加载",
    "优化结构特征提取",
    "减少日志输出频率"
]
```

## 📋 检查清单

### 启动前检查
- [ ] 预训练模型文件存在
- [ ] GPU内存充足 (>20GB)
- [ ] ESMFold依赖已安装
- [ ] 训练数据可访问
- [ ] 输出目录已创建

### 训练中监控
- [ ] GPU内存使用正常
- [ ] 训练损失下降
- [ ] 结构预测成功率 >70%
- [ ] 训练速度在预期范围内
- [ ] 日志文件正常写入

### 完成后验证
- [ ] 最终模型文件生成
- [ ] 训练指标文件完整
- [ ] 验证损失优于基线
- [ ] 检查点文件正常保存

## 🎯 成功标准

### 1. 训练收敛标准
- 最终训练损失 < 0.14
- 验证损失稳定且无过拟合
- 结构特征成功集成

### 2. 性能提升标准
- 相比序列训练模型损失降低 >15%
- 生成质量评估指标改善
- 结构合理性提升

### 3. 稳定性标准
- 训练过程无异常中断
- 内存使用稳定
- 检查点保存完整

## 📝 下一步计划

训练完成后，可以进行：

1. **生成测试**: 使用新模型生成肽段序列
2. **质量评估**: 评估生成序列的生物学合理性
3. **性能对比**: 与序列训练模型对比
4. **进一步优化**: 根据结果调整训练策略

通过结构特征的集成，模型将能够生成更加生物学合理和功能性更强的肽段序列。 