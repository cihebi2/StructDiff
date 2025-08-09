# StructDiff配置文件结构指南

## 概述
本文档详细说明StructDiff项目中各配置文件的作用、关系和使用方法。项目采用YAML格式的分层配置系统，支持配置继承和覆盖。

## 配置文件层次结构

```
configs/
├── default.yaml                    # 基础默认配置
├── model/
│   ├── baseline.yaml              # 基线模型配置
│   └── structdiff.yaml            # StructDiff模型配置
├── separated_training*.yaml        # 各种训练配置变体
├── generate.yaml                   # 生成配置
├── peptide_*.yaml                  # 肽相关特殊配置
└── 其他专用配置
```

## 核心配置文件详解

### 1. default.yaml - 基础配置
**作用**: 定义所有其他配置的基础默认值

**主要部分**:
```yaml
experiment:         # 实验设置
  name: 实验名称
  seed: 随机种子
  output_dir: 输出目录

model:             # 模型架构
  sequence_encoder: ESM-2配置
  structure_encoder: 结构编码器配置
  denoiser: 去噪器配置

data:              # 数据配置
  paths: 数据路径
  preprocessing: 预处理参数
  dataloader: 加载器设置

training:          # 训练参数
  optimizer: 优化器设置
  scheduler: 学习率调度
  checkpointing: 检查点策略

diffusion:         # 扩散过程
  timesteps: 时间步数
  noise_schedule: 噪声调度
```

### 2. separated_training_production.yaml - 生产训练配置
**作用**: 生产环境的两阶段分离训练配置

**关键特性**:
- 继承并覆盖default.yaml
- 定义两阶段训练参数
- 生产级别的超参数设置

**独特配置**:
```yaml
training:
  strategy: "separated"  # 分离式训练
  stage1:
    epochs: 200
    lr: 1e-4
    freeze_decoder: true
  stage2:
    epochs: 100
    lr: 5e-5
    freeze_encoder: true
```

### 3. model/structdiff.yaml - 模型架构配置
**作用**: 定义StructDiff模型的详细架构

**内容**:
- 各组件的维度设置
- 层数和注意力头数
- 特殊模块配置（CFG、长度控制等）

### 4. generate.yaml - 生成配置
**作用**: 序列生成时的参数配置

**关键参数**:
- 采样方法
- 引导尺度
- 温度控制
- 批次大小

## 配置文件关系图

```
                    default.yaml
                         |
        +----------------+----------------+
        |                |                |
    model配置        training配置      data配置
        |                |                |
   structdiff.yaml  separated_*.yaml  peptide_*.yaml
        |                |                |
        +----------------+----------------+
                         |
                   具体实验配置
```

## 配置继承规则

### 1. 基础继承
- 所有配置都隐式继承default.yaml
- 子配置覆盖父配置的相应字段

### 2. 配置合并策略
```python
# 伪代码
final_config = deep_merge(default_config, specific_config)
```

### 3. 覆盖优先级
1. 命令行参数（最高）
2. 指定的配置文件
3. default.yaml（最低）

## 配置文件分类

### 训练策略配置
1. **separated_training.yaml** - 基础分离训练
2. **separated_training_production.yaml** - 生产版本（推荐）
3. **separated_training_optimized.yaml** - 优化版本
4. **separated_training_fixed_v2.yaml** - 修复版本
5. **separated_training_structure_enabled.yaml** - 启用结构

### 模型变体配置
1. **small_model.yaml** - 小模型（快速实验）
2. **peptide_large_model.yaml** - 大模型（高质量）
3. **baseline.yaml** - 基线模型（对比实验）

### 特殊功能配置
1. **cfg_length_config.yaml** - CFG和长度控制
2. **peptide_adaptive_conditioning.yaml** - 自适应条件
3. **esmfold_cpu_config.yaml** - CPU上的ESMFold
4. **structure_enabled_training.yaml** - 结构特征训练

### 测试和实验配置
1. **minimal_test.yaml** - 最小测试配置
2. **test_train.yaml** - 测试训练配置
3. **classification_config.yaml** - 分类实验

## 配置使用指南

### 1. 选择合适的配置
```bash
# 生产训练
python scripts/train_separated.py --config configs/separated_training_production.yaml

# 快速测试
python scripts/train.py --config configs/minimal_test.yaml

# 自定义配置
python scripts/train.py --config configs/default.yaml --override batch_size=64
```

### 2. 配置覆盖
```bash
# 命令行覆盖
python scripts/train.py \
  --config configs/default.yaml \
  --override training.num_epochs=50 \
  --override data.batch_size=16
```

### 3. 创建新配置
```yaml
# my_config.yaml
# 只需要指定与default.yaml不同的部分
training:
  num_epochs: 200
  optimizer:
    lr: 5e-5

model:
  denoiser:
    num_layers: 16
```

## 配置验证

### 必需字段
- `model.type`: 模型类型
- `data.train_path`: 训练数据路径
- `training.num_epochs`: 训练轮数

### 条件必需
- 如果`use_esmfold=true`，需要足够的GPU内存
- 如果`use_cfg=true`，需要配置CFG参数

## 配置最佳实践

### 1. 生产环境
使用`separated_training_production.yaml`作为基础：
- 经过验证的超参数
- 合理的资源使用
- 稳定的训练流程

### 2. 实验开发
从`minimal_test.yaml`开始：
- 快速迭代
- 小批次大小
- 较少的epochs

### 3. 性能优化
参考`separated_training_optimized.yaml`：
- 梯度累积
- 混合精度训练
- 优化的数据加载

## 配置清理建议

### 保留的配置（核心）
1. `default.yaml` - 基础配置
2. `separated_training_production.yaml` - 主要训练配置
3. `model/structdiff.yaml` - 模型定义
4. `generate.yaml` - 生成配置

### 可以删除的配置
1. 各种`*_fixed_v*.yaml` - 临时修复版本
2. `*.yaml.backup` - 备份文件
3. 过时的实验配置

### 需要评估的配置
1. 特殊功能配置 - 根据是否使用决定
2. 测试配置 - 可能合并到单一测试配置

## 配置迁移指南

### 从旧配置迁移
1. 识别使用的特殊参数
2. 在新配置中找到对应位置
3. 验证参数兼容性
4. 逐步测试迁移

### 配置升级路径
```
旧配置 → default.yaml + 自定义覆盖 → 验证 → 新配置文件
```

## 常见问题

### Q: 如何知道使用哪个配置？
A: 
- 生产训练：`separated_training_production.yaml`
- 快速测试：`minimal_test.yaml`
- 生成序列：`generate.yaml`

### Q: 配置冲突怎么办？
A: 优先级：命令行 > 指定配置 > default.yaml

### Q: 如何调试配置问题？
A: 
1. 使用`--print-config`查看最终配置
2. 检查日志中的配置加载信息
3. 验证路径和参数类型

## 未来改进建议

1. **简化配置数量**: 合并相似配置
2. **配置验证器**: 自动检查配置有效性
3. **配置生成器**: 基于需求自动生成配置
4. **版本管理**: 配置版本控制和兼容性检查

---

最后更新：2025-08-04
作者：StructDiff架构改进项目