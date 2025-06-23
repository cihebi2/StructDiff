# StructDiff 大规模训练指南

这个指南介绍如何使用新的大规模训练系统来训练 StructDiff 模型。

## 🆕 新训练系统特性

相比于 `simple_train.py`，新的训练系统具有以下优势：

### 🚀 核心功能
- **分布式训练**: 支持单GPU和多GPU训练
- **完整的日志系统**: 详细的训练日志和TensorBoard可视化
- **智能检查点管理**: 自动保存最新和最佳模型
- **内存优化**: 梯度累积、共享ESMFold实例等
- **训练监控**: 实时监控训练进度和GPU状态
- **错误恢复**: 从检查点恢复训练，错误处理机制

### 📊 监控和可视化
- **TensorBoard集成**: 损失、学习率、内存使用等指标
- **训练监控脚本**: 实时监控训练状态
- **训练曲线绘制**: 自动生成训练进度图表

## 📁 文件结构

```
StructDiff/
├── train_full.py           # 主要训练脚本
├── launch_train.sh         # 训练启动脚本
├── monitor_training.py     # 训练监控脚本
├── simple_train.py         # 简单训练脚本（用于测试）
├── configs/
│   └── test_train.yaml     # 训练配置文件
└── outputs/                # 训练输出目录（自动创建）
    ├── logs/              # 训练日志
    ├── tensorboard/       # TensorBoard日志
    ├── checkpoints/       # 模型检查点
    └── config.yaml        # 保存的配置文件
```

## 🚀 快速开始

### 1. 单GPU训练

```bash
# 使用默认配置
./launch_train.sh

# 使用自定义配置
./launch_train.sh --config configs/test_train.yaml --output_dir my_training_run
```

### 2. 多GPU分布式训练

```bash
# 使用2个GPU
./launch_train.sh --config configs/test_train.yaml --num_gpus 2

# 使用4个GPU，自定义输出目录
./launch_train.sh --config configs/test_train.yaml --num_gpus 4 --output_dir outputs/multi_gpu_run
```

### 3. 恢复训练

```bash
# 从最新检查点恢复
./launch_train.sh --config configs/test_train.yaml --resume outputs/checkpoints/checkpoint_latest.pth

# 从最佳模型恢复
./launch_train.sh --config configs/test_train.yaml --resume outputs/checkpoints/checkpoint_best.pth
```

## 🔧 训练配置

### 主要配置选项

在 `configs/test_train.yaml` 中可以配置以下参数：

```yaml
# 模型配置
model:
  structure_encoder:
    use_esmfold: true    # 是否使用ESMFold

# 数据配置
data:
  train_path: "data/train.csv"
  val_path: "data/val.csv"
  use_predicted_structures: true
  batch_size: 8
  num_workers: 4

# 训练配置
training:
  num_epochs: 100
  batch_size: 8
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  validate_every: 5
  
  # 优化器配置
  optimizer:
    name: "adamw"
    lr: 1e-4
    weight_decay: 0.01
    
  # 学习率调度器配置
  scheduler:
    name: "cosine"
    min_lr: 1e-6

# 扩散模型配置
diffusion:
  num_timesteps: 1000
```

### 批量大小设置

有效批量大小 = `batch_size` × `gradient_accumulation_steps` × `num_gpus`

例如：
- `batch_size: 4`
- `gradient_accumulation_steps: 8`  
- `num_gpus: 2`
- 有效批量大小 = 4 × 8 × 2 = 64

## 📊 训练监控

### 1. 查看训练状态

```bash
# 显示当前训练状态
python monitor_training.py --output_dir outputs/your_training_run --mode status
```

### 2. 持续监控

```bash
# 每30秒更新一次状态
python monitor_training.py --output_dir outputs/your_training_run --mode monitor --interval 30
```

### 3. 绘制训练曲线

```bash
# 绘制并显示训练曲线
python monitor_training.py --output_dir outputs/your_training_run --mode plot

# 保存训练曲线图
python monitor_training.py --output_dir outputs/your_training_run --mode plot --save_plot training_curves.png
```

### 4. TensorBoard可视化

```bash
# 启动TensorBoard
tensorboard --logdir outputs/your_training_run/tensorboard

# 然后在浏览器中打开 http://localhost:6006
```

## 📋 训练日志

训练日志保存在 `outputs/logs/training_TIMESTAMP.log` 中，包含：

- 训练进度和损失
- 验证结果
- GPU内存使用情况
- 错误信息和警告
- 检查点保存信息

示例日志输出：
```
2024-01-15 10:30:00 - INFO - 开始 StructDiff 大规模训练
2024-01-15 10:30:01 - INFO - 模型参数数量: 125,432,123
2024-01-15 10:30:05 - INFO - 🚀 Epoch 1/100
2024-01-15 10:35:20 - INFO - 训练损失 - 总计: 2.3456, 扩散: 1.8932, 结构: 0.4524
2024-01-15 10:40:10 - INFO - 验证损失 - 总计: 2.1234, 扩散: 1.7123, 结构: 0.4111
2024-01-15 10:40:11 - INFO - 💾 保存最佳模型 (验证损失: 2.1234)
```

## 💾 检查点管理

系统自动管理三种类型的检查点：

1. **最新检查点** (`checkpoint_latest.pth`): 每个epoch更新
2. **最佳模型** (`checkpoint_best.pth`): 验证损失最低的模型
3. **定期检查点** (`checkpoint_epoch_N.pth`): 每10个epoch保存

检查点包含：
- 模型状态字典
- 优化器状态
- 学习率调度器状态
- 训练损失历史
- 配置信息
- 时间戳

## 🔍 故障排除

### 常见问题

1. **内存不足 (CUDA OOM)**
   - 减少 `batch_size`
   - 增加 `gradient_accumulation_steps`
   - 禁用 ESMFold (`use_esmfold: false`)

2. **训练速度慢**
   - 增加 `num_workers`
   - 使用多GPU训练
   - 检查数据加载瓶颈

3. **分布式训练失败**
   - 检查GPU数量和可用性
   - 确保端口没有被占用
   - 查看错误日志

4. **ESMFold加载失败**
   - 检查网络连接
   - 临时禁用ESMFold进行测试
   - 查看详细错误信息

### 调试技巧

1. **启用调试模式**
   ```bash
   export CUDA_LAUNCH_BLOCKING=1
   python train_full.py --config configs/test_train.yaml
   ```

2. **减少数据量进行测试**
   - 在配置中限制数据集大小
   - 使用 `simple_train.py` 进行快速验证

3. **监控GPU使用情况**
   ```bash
   watch -n 1 nvidia-smi
   ```

## 🎯 最佳实践

### 训练策略
1. **从小规模开始**: 先用小数据集和小模型验证流程
2. **逐步扩大**: 逐渐增加批量大小和数据量
3. **定期验证**: 设置合理的验证频率
4. **保存检查点**: 确保定期保存，避免训练中断损失

### 资源管理
1. **内存优化**: 使用梯度累积代替大批量
2. **计算优化**: 合理设置num_workers避免CPU瓶颈
3. **存储管理**: 定期清理旧的检查点文件

### 监控建议
1. **持续监控**: 使用监控脚本跟踪训练进度
2. **可视化分析**: 定期查看TensorBoard中的指标
3. **日志分析**: 关注训练日志中的警告和错误

## 📞 支持

如果遇到问题，请：
1. 查看训练日志中的错误信息
2. 使用监控脚本检查训练状态
3. 尝试从较小的配置开始
4. 检查系统资源（GPU内存、磁盘空间等）

祝您训练顺利！🚀 