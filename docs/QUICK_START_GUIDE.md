# StructDiff 快速启动指南

## 🚀 即刻开始

### 1. 环境确认
```bash
# 检查GPU内存
python3 memory_monitor.py

# 设置环境变量 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
```

### 2. 开始第一阶段训练
```bash
# 自动配置并启动第一阶段
python3 scripts/stage_controller.py --start stage1_validation

# 或者直接运行优化启动脚本
./start_training_optimized.sh
```

### 3. 监控训练进度
```bash
# 查看阶段状态
python3 scripts/stage_controller.py --status

# 监控内存使用
python3 memory_monitor.py
```

## 📂 核心文件说明

### 配置文件
- `configs/peptide_esmfold_config.yaml` - 主配置文件
- `configs/esmfold_cpu_config.yaml` - CPU备用配置  
- `configs/stage_*.yaml` - 自动生成的阶段配置

### 训练脚本
- `scripts/train_peptide_esmfold.py` - 主训练脚本
- `scripts/stage_controller.py` - 阶段控制器
- `start_training_optimized.sh` - 优化启动脚本

### 评估工具
- `scripts/evaluation_suite.py` - 模型评估套件
- `memory_monitor.py` - 内存监控工具

### 文档
- `PROJECT_DEVELOPMENT_PLAN.md` - 完整开发规划
- `ESMFOLD_MEMORY_FIX.md` - 内存问题解决方案

## 🎯 六阶段开发流程

### 阶段1: 基础验证 (1-2周)
```bash
python3 scripts/stage_controller.py --start stage1_validation
# 目标: 验证模型能正常工作，ESMFold集成成功
```

### 阶段2: 中等规模优化 (2-3周)  
```bash
python3 scripts/stage_controller.py --start stage2_optimization
# 目标: 扩大数据规模，优化超参数
```

### 阶段3: 大规模训练 (3-4周)
```bash
python3 scripts/stage_controller.py --start stage3_scaling  
# 目标: 全量数据训练，性能提升
```

### 阶段4: 模型精调 (2-3周)
- 领域特定微调
- 强化学习集成
- 专业化模型开发

### 阶段5: 评估基准测试 (1-2周)
```bash
python3 scripts/evaluation_suite.py
# 综合评估模型性能
```

### 阶段6: 部署应用 (1-2周)
- Web界面开发
- API服务部署
- 用户工具开发

## 🔧 常用命令

### 训练相关
```bash
# 标准训练
python3 scripts/train_peptide_esmfold.py --config configs/peptide_esmfold_config.yaml

# 调试模式 (小数据集)
python3 scripts/train_peptide_esmfold.py --config configs/peptide_esmfold_config.yaml --debug

# CPU模式 (内存不足时)
python3 scripts/train_peptide_esmfold.py --config configs/esmfold_cpu_config.yaml
```

### 评估相关
```bash
# 生成评估报告
python3 scripts/evaluation_suite.py

# 监控系统资源
python3 memory_monitor.py
```

### 阶段管理
```bash
# 开始新阶段
python3 scripts/stage_controller.py --start stage1_validation

# 查看当前状态  
python3 scripts/stage_controller.py --status

# 完成当前阶段
python3 scripts/stage_controller.py --complete stage1_validation
```

## 💡 关键技术特性

### AlphaFold3 自适应条件控制
- ✅ 多方面条件控制 (电荷、疏水性、结构、功能)
- ✅ 生物学启发的初始化模式
- ✅ 自适应强度学习

### ESMFold结构预测集成
- ✅ GPU/CPU自动切换
- ✅ 激进内存管理 
- ✅ 批处理优化

### 三种功能条件
- 🦠 抗菌肽生成
- 🍄 抗真菌肽生成  
- 🦠 抗病毒肽生成

## 📊 预期结果

### 短期目标 (1个月)
- [x] 稳定训练流程
- [x] 基础多肽生成
- [x] 条件控制验证
- [x] ESMFold集成

### 中期目标 (2-3个月)
- 🎯 >80% 有效序列生成率
- 🎯 >90% 条件一致性
- 🎯 >0.7 ESMFold置信度
- 🎯 >75% 生物活性预测准确率

### 长期目标 (6个月)
- 🚀 超越现有方法性能
- 🚀 新型多肽设计能力
- 🚀 实际应用验证
- 🚀 开源社区建设

## ⚠️ 常见问题解决

### GPU内存不足
```bash
# 使用CPU模式
python3 scripts/train_peptide_esmfold.py --config configs/esmfold_cpu_config.yaml

# 或减小批次大小
# 编辑配置文件中的 data.batch_size: 4 -> 2
```

### 训练不收敛
```bash
# 降低学习率
# 编辑配置文件中的 training.optimizer.lr: 5e-5 -> 1e-5

# 增加梯度累积
# 编辑配置文件中的 training.gradient_accumulation_steps: 4 -> 8
```

### ESMFold初始化失败
```bash
# 运行内存清理
python3 memory_monitor.py

# 使用CPU backup
# 配置文件中设置 memory_optimization.esmfold_cpu_fallback: true
```

## 📞 支持

遇到问题时：
1. 📖 查看 `PROJECT_DEVELOPMENT_PLAN.md` 详细规划
2. 🔧 运行 `python3 memory_monitor.py` 检查资源
3. 📊 使用 `scripts/evaluation_suite.py` 评估模型
4. 🎮 通过 `scripts/stage_controller.py` 管理训练阶段

现在您已经拥有了完整的工具链，可以开始循序渐进地开发和优化StructDiff多肽生成模型！