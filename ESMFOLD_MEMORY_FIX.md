# ESMFold 内存问题解决方案

## 问题描述
训练过程中ESMFold出现CUDA内存不足错误：
```
CUDA out of memory. Tried to allocate 21.67 GiB (GPU 0; 24.00 GiB total capacity; 22.36 GiB already allocated)
```

## 解决方案

### 1. 自动内存优化
训练脚本已更新，包含以下优化：

**激进内存清理**：
- 在ESMFold初始化前清理所有GPU缓存
- 强制Python垃圾回收
- 设置PyTorch内存分配策略

**CPU备用方案**：
- GPU内存不足时自动切换到CPU
- 保持训练流程不中断

### 2. 配置优化
在 `configs/peptide_esmfold_config.yaml` 中添加了：
```yaml
memory_optimization:
  esmfold_memory_optimization:
    enabled: true
    use_cpu_fallback: true
    aggressive_cleanup: true
    chunked_processing: true
```

### 3. 使用工具

**内存监控**：
```bash
python3 memory_monitor.py
```

**优化启动**：
```bash
./start_training_optimized.sh
```

**CPU模式**：
```bash
python3 scripts/train_peptide_esmfold.py \
    --config configs/esmfold_cpu_config.yaml
```

### 4. 环境变量
在训练前设置：
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

### 5. 手动解决方案
如果仍有问题：

1. **减小批次大小**：
   - 将 `data.batch_size` 从 8 改为 4 或 2

2. **限制序列长度**：
   - 将 `data.max_length` 从 50 改为 30

3. **强制CPU模式**：
   - 设置 `memory_optimization.evaluation_only_cpu: true`

## 预期效果
- ✅ ESMFold可以成功初始化
- ✅ 训练流程不中断  
- ✅ 自动处理内存不足情况
- ✅ 保持模型性能

现在您可以重新运行训练，ESMFold应该能够正常工作！