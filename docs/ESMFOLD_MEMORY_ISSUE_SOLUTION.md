# ESMFold 内存问题解决方案

## 🚨 问题描述

在运行结构特征训练时遇到的ESMFold内存分配失败问题：

```
CUDA out of memory. Tried to allocate 100.00 MiB. GPU 0 has a total capacity of 23.64 GiB of which 64.12 MiB is free. Process 29424 has 496.00 MiB memory in use. Including non-PyTorch memory, this process has 23.09 GiB memory in use.
```

## 🔍 问题分析

### 1. 根本原因
- **双重ESMFold加载**：模型内部尝试再次加载ESMFold，导致内存重复占用
- **内存碎片化**：PyTorch内存分配器的碎片化问题
- **缺乏内存管理策略**：没有采用成功脚本中的内存优化技巧

### 2. 影响评估
- ❌ **严重影响训练效果**：ESMFold加载失败导致回退到"虚拟结构预测"
- ❌ **结构特征缺失**：实际上没有使用真正的结构特征
- ❌ **训练质量下降**：变成了伪结构特征训练

## ✅ 解决方案

### 1. 基于成功脚本的优化策略

#### 1.1 内存分配策略
```bash
# 关键环境变量设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

#### 1.2 共享ESMFold实例
```python
def setup_shared_esmfold(device):
    """创建共享的ESMFold实例以节省内存"""
    # 更激进的内存清理
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    
    # 设置内存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    
    # 再次清理
    torch.cuda.empty_cache()
    
    # 创建ESMFold实例
    shared_esmfold = ESMFoldWrapper(device=device)
    return shared_esmfold
```

#### 1.3 避免双重加载
```python
def setup_model_and_training(config, device, shared_esmfold):
    # 备份原始配置
    original_use_esmfold = config.model.structure_encoder.get('use_esmfold', False)
    
    # 临时禁用模型内部的ESMFold加载
    if shared_esmfold and shared_esmfold.available:
        config.model.structure_encoder.use_esmfold = False
    
    # 创建模型
    model = StructDiff(config.model).to(device)
    
    # 恢复配置并设置共享实例
    config.model.structure_encoder.use_esmfold = original_use_esmfold
    
    # 手动设置共享ESMFold实例
    if shared_esmfold and shared_esmfold.available:
        model.structure_encoder.esmfold = shared_esmfold
        model.structure_encoder.use_esmfold = True
```

### 2. 数据加载优化

#### 2.1 保守的数据加载器设置
```python
train_loader = DataLoader(
    train_dataset,
    batch_size=2,  # 小批次大小
    shuffle=True,
    num_workers=0,  # 关键：避免多进程缓存竞争
    pin_memory=False,  # 禁用pin_memory节省内存
    collate_fn=collator,
    drop_last=True
)
```

#### 2.2 共享ESMFold实例传递
```python
train_dataset = PeptideStructureDataset(
    data_path="...",
    config=config,
    is_training=True,
    shared_esmfold=shared_esmfold  # 传递共享实例
)
```

### 3. 训练过程优化

#### 3.1 定期内存清理
```python
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# 在训练循环中定期调用
if batch_idx % 10 == 0:
    clear_memory()
```

#### 3.2 梯度累积策略
```python
# 使用小批次 + 大梯度累积
batch_size = 2
gradient_accumulation_steps = 8
effective_batch_size = 16  # 保持相同的有效批次大小
```

## 🛠️ 修复版本实现

### 1. 新的训练脚本
创建了 `full_train_with_structure_features_fixed_v2.py`，包含：

- ✅ **共享ESMFold实例管理**
- ✅ **内存分配策略优化**
- ✅ **避免双重加载机制**
- ✅ **保守的数据加载设置**
- ✅ **定期内存清理**

### 2. 启动脚本优化
创建了 `start_structure_training_fixed.sh`，包含：

```bash
# 关键环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# 预清理GPU内存
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f'GPU内存已清理')
"
```

## 🚀 使用方法

### 1. 停止当前训练
```bash
# 找到并停止当前训练进程
ps aux | grep full_train_with_structure_features_enabled
kill <PID>
```

### 2. 启动修复版本
```bash
cd /home/qlyu/sequence/StructDiff-7.0.0
./start_structure_training_fixed.sh
```

### 3. 监控训练
```bash
# 查看日志
tail -f outputs/structure_feature_training_fixed/training.log

# 监控GPU内存
watch -n 1 nvidia-smi
```

## 📊 预期改进效果

### 1. 内存使用优化
- **ESMFold内存使用**: ~14GB（共享实例）
- **模型内存使用**: ~4GB
- **总内存使用**: ~18GB（在24GB范围内）

### 2. 训练稳定性
- ✅ **ESMFold成功加载**：不再回退到虚拟结构预测
- ✅ **真正的结构特征**：使用实际的pLDDT、距离矩阵等
- ✅ **训练收敛性**：结构特征真正参与训练

### 3. 性能指标
- **批次处理时间**: 8-12秒/批次
- **内存稳定性**: 无内存泄漏
- **训练质量**: 结构感知的序列生成

## 🔍 验证方法

### 1. 检查ESMFold状态
在训练日志中查找：
```
✅ 共享ESMFold GPU实例创建成功
✅ 共享 ESMFold 实例已设置到模型中
```

### 2. 监控内存使用
```bash
# 应该看到稳定的内存使用，无突然增长
nvidia-smi -l 1
```

### 3. 验证结构特征
在训练日志中应该看到结构特征相关的处理信息，而不是"虚拟结构预测"。

## 📝 关键经验总结

1. **共享实例是关键**：避免重复加载大型模型
2. **内存分配策略很重要**：`expandable_segments:True`解决碎片化
3. **数据加载器设置影响稳定性**：`num_workers=0`避免竞争
4. **定期清理必不可少**：防止内存泄漏累积
5. **小批次+梯度累积**：在内存限制下保持训练效果

通过这些优化，可以成功启用真正的结构特征训练，而不是之前的伪训练模式。 