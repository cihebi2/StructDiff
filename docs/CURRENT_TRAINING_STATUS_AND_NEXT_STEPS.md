# 当前训练状态与下一步计划

## 📊 当前训练状态

### 1. 正在运行的训练任务

**训练脚本**: `full_train_200_epochs_with_esmfold_fixed.py`

**当前配置**:
```python
# 训练参数
batch_size = 8
gradient_accumulation_steps = 2
effective_batch_size = 16
num_epochs = 200
learning_rate = 1e-4
scheduler = CosineAnnealingLR(T_max=200, eta_min=1e-6)

# 模型状态
structure_features_enabled = False  # 当前禁用
esmfold_status = "备用状态"  # 已初始化但未使用
```

### 2. 训练进度监控

#### 2.1 关键文件位置
```bash
# 训练日志
/home/qlyu/sequence/StructDiff-7.0.0/outputs/full_training_200_esmfold_fixed/training.log

# 检查点目录
/home/qlyu/sequence/StructDiff-7.0.0/outputs/full_training_200_esmfold_fixed/checkpoints/

# 训练指标
/home/qlyu/sequence/StructDiff-7.0.0/outputs/full_training_200_esmfold_fixed/training_metrics.json
```

#### 2.2 监控命令
```bash
# 实时查看训练日志
tail -f /home/qlyu/sequence/StructDiff-7.0.0/outputs/full_training_200_esmfold_fixed/training.log

# 查看GPU使用情况
watch -n 1 nvidia-smi

# 查看训练进程
ps aux | grep python | grep full_train

# 查看当前训练指标
cat /home/qlyu/sequence/StructDiff-7.0.0/outputs/full_training_200_esmfold_fixed/training_metrics.json | jq '.'
```

### 3. 预期训练表现

#### 3.1 损失收敛模式
```
Epoch 1-20:   训练损失 1.0 → 0.6    (快速下降阶段)
Epoch 21-50:  训练损失 0.6 → 0.3    (稳定下降阶段)
Epoch 51-100: 训练损失 0.3 → 0.2    (缓慢收敛阶段)
Epoch 101-200: 训练损失 0.2 → 0.15  (精细调优阶段)
```

#### 3.2 系统资源使用
```
GPU内存使用: ~4GB (不含ESMFold)
训练速度: ~2-3秒/批次
每epoch时间: ~10-15分钟
预计总训练时间: ~40-50小时
```

## 🎯 下一步计划

### 阶段1: 当前训练完成 (预计1-2天)

#### 1.1 等待当前训练完成
```bash
# 监控训练是否完成
ls -la /home/qlyu/sequence/StructDiff-7.0.0/outputs/full_training_200_esmfold_fixed/final_model_epoch_200.pt

# 检查最终指标
cat /home/qlyu/sequence/StructDiff-7.0.0/outputs/full_training_200_esmfold_fixed/final_metrics.json
```

#### 1.2 评估训练结果
```python
# 评估脚本
def evaluate_current_training():
    """评估当前训练结果"""
    
    # 加载最终指标
    with open('final_metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # 检查收敛情况
    final_train_loss = metrics['final_train_loss']
    best_val_loss = metrics['best_val_loss']
    
    print(f"最终训练损失: {final_train_loss:.6f}")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    # 判断是否需要继续训练
    if final_train_loss > 0.25:
        print("建议继续训练以获得更好的收敛")
    elif final_train_loss < 0.15:
        print("训练收敛良好，可以进入下一阶段")
    else:
        print("训练收敛正常，可以考虑进入下一阶段")
```

### 阶段2: 结构特征集成 (预计2-3天)

#### 2.1 创建结构特征训练脚本
```python
# 文件名: full_train_with_structure_features.py

def create_structure_training_script():
    """创建包含结构特征的训练脚本"""
    
    # 基于当前脚本修改
    script_content = """
#!/usr/bin/env python3

# 在原有脚本基础上的修改
def full_train_with_structure_features():
    # 启用结构特征
    config.data.use_predicted_structures = True
    config.model.structure_encoder.use_esmfold = True
    
    # 调整训练参数以适应ESMFold
    batch_size = 2  # 降低批次大小
    gradient_accumulation_steps = 8  # 增加梯度累积
    effective_batch_size = 16  # 保持相同的有效批次大小
    
    # 从之前的检查点开始
    checkpoint_path = "outputs/full_training_200_esmfold_fixed/best_model.pt"
    
    # 其余逻辑保持不变...
"""
    
    return script_content
```

#### 2.2 渐进式结构特征集成
```python
# 阶段2A: 结构特征预训练 (50 epochs)
def stage_2a_structure_pretraining():
    """结构特征预训练阶段"""
    
    # 配置
    config = {
        'epochs': 50,
        'batch_size': 2,
        'gradient_accumulation_steps': 8,
        'learning_rate': 5e-5,  # 降低学习率
        'structure_weight': 0.1,  # 较小的结构损失权重
        'freeze_sequence_encoder': True  # 冻结序列编码器
    }
    
    # 从最佳检查点开始
    start_from_checkpoint = "best_model.pt"

# 阶段2B: 端到端微调 (50 epochs)
def stage_2b_end_to_end_finetuning():
    """端到端微调阶段"""
    
    # 配置
    config = {
        'epochs': 50,
        'batch_size': 2,
        'gradient_accumulation_steps': 8,
        'learning_rate': 1e-5,  # 更低的学习率
        'structure_weight': 0.2,  # 增加结构损失权重
        'freeze_sequence_encoder': False  # 解冻序列编码器
    }
```

### 阶段3: 生成和评估 (预计1天)

#### 3.1 生成测试脚本
```python
# 文件名: test_generation.py

def test_peptide_generation():
    """测试肽段生成功能"""
    
    # 加载训练好的模型
    model = StructDiff.from_pretrained("outputs/final_model/")
    diffusion = GaussianDiffusion(...)
    
    # 生成不同类型的肽段
    test_conditions = [
        {'peptide_type': 0, 'length': 20},  # 抗菌肽
        {'peptide_type': 1, 'length': 25},  # 抗真菌肽
        {'peptide_type': 2, 'length': 30},  # 抗病毒肽
    ]
    
    for condition in test_conditions:
        sequences = generate_peptides(
            model=model,
            diffusion=diffusion,
            num_samples=100,
            condition=condition
        )
        
        # 评估生成质量
        evaluate_generated_sequences(sequences, condition)
```

#### 3.2 评估指标
```python
def evaluate_generated_sequences(sequences, condition):
    """评估生成的序列质量"""
    
    metrics = {}
    
    # 1. 基本统计
    metrics['num_sequences'] = len(sequences)
    metrics['avg_length'] = np.mean([len(seq) for seq in sequences])
    metrics['length_std'] = np.std([len(seq) for seq in sequences])
    
    # 2. 序列多样性
    metrics['unique_sequences'] = len(set(sequences))
    metrics['diversity_ratio'] = metrics['unique_sequences'] / metrics['num_sequences']
    
    # 3. 氨基酸组成
    all_aas = ''.join(sequences)
    aa_counts = Counter(all_aas)
    metrics['aa_distribution'] = dict(aa_counts)
    
    # 4. 生物学评估 (如果有工具)
    # metrics['antimicrobial_score'] = predict_antimicrobial_activity(sequences)
    # metrics['toxicity_score'] = predict_toxicity(sequences)
    
    return metrics
```

## 🛠️ 具体操作步骤

### 步骤1: 监控当前训练

```bash
# 1. 检查训练是否还在运行
ps aux | grep full_train_200_epochs_with_esmfold_fixed

# 2. 查看最新日志
tail -n 50 /home/qlyu/sequence/StructDiff-7.0.0/outputs/full_training_200_esmfold_fixed/training.log

# 3. 检查GPU使用情况
nvidia-smi

# 4. 查看当前epoch进度
grep "Epoch.*完成" /home/qlyu/sequence/StructDiff-7.0.0/outputs/full_training_200_esmfold_fixed/training.log | tail -5
```

### 步骤2: 准备下一阶段训练

```bash
# 1. 创建新的训练脚本
cp full_train_200_epochs_with_esmfold_fixed.py full_train_with_structure_features.py

# 2. 修改配置以启用结构特征
# (需要手动编辑脚本)

# 3. 创建新的输出目录
mkdir -p outputs/structure_feature_training

# 4. 准备从检查点恢复的脚本
```

### 步骤3: 创建监控脚本

```python
# 文件名: monitor_training.py

import json
import time
import os
from datetime import datetime

def monitor_training_progress():
    """监控训练进度"""
    
    log_file = "outputs/full_training_200_esmfold_fixed/training.log"
    metrics_file = "outputs/full_training_200_esmfold_fixed/training_metrics.json"
    
    while True:
        try:
            # 检查日志文件
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1]
                        print(f"[{datetime.now()}] 最新日志: {last_line.strip()}")
            
            # 检查指标文件
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    current_epoch = metrics.get('epoch', 0)
                    current_loss = metrics.get('train_losses', [])
                    if current_loss:
                        print(f"[{datetime.now()}] 当前epoch: {current_epoch}, 最新损失: {current_loss[-1]:.6f}")
            
            time.sleep(300)  # 每5分钟检查一次
            
        except Exception as e:
            print(f"监控错误: {e}")
            time.sleep(60)

if __name__ == "__main__":
    monitor_training_progress()
```

### 步骤4: 创建自动化脚本

```bash
# 文件名: auto_training_pipeline.sh

#!/bin/bash

# 自动化训练流水线
echo "开始自动化训练流水线..."

# 1. 等待当前训练完成
echo "等待当前训练完成..."
while [ ! -f "outputs/full_training_200_esmfold_fixed/final_model_epoch_200.pt" ]; do
    echo "训练仍在进行中，等待中..."
    sleep 1800  # 等待30分钟
done

echo "当前训练已完成！"

# 2. 评估训练结果
echo "评估训练结果..."
python evaluate_training_results.py

# 3. 开始结构特征训练
echo "开始结构特征训练..."
python full_train_with_structure_features.py

# 4. 完成后进行生成测试
echo "进行生成测试..."
python test_generation.py

echo "自动化流水线完成！"
```

## 📋 检查清单

### 当前阶段完成标准
- [ ] 训练损失收敛到 < 0.20
- [ ] 验证损失稳定且无明显过拟合
- [ ] 生成 `final_model_epoch_200.pt` 文件
- [ ] 生成完整的训练指标文件

### 下一阶段准备清单
- [ ] 创建结构特征训练脚本
- [ ] 设置正确的检查点恢复路径
- [ ] 调整批次大小和梯度累积参数
- [ ] 准备结构特征提取函数
- [ ] 设置内存优化策略

### 最终评估清单
- [ ] 生成质量评估
- [ ] 序列多样性分析
- [ ] 条件控制效果验证
- [ ] 与基线模型比较
- [ ] 生物学活性预测（如果可能）

## 🚨 注意事项

### 1. 资源管理
- 当前训练预计需要40-50小时
- 结构特征训练需要更多GPU内存
- 建议在训练间隙进行系统维护

### 2. 备份策略
```bash
# 定期备份重要文件
rsync -av outputs/ backup/outputs_$(date +%Y%m%d_%H%M%S)/
```

### 3. 故障恢复
```bash
# 如果训练意外中断
python full_train_200_epochs_with_esmfold_fixed.py --resume_from_checkpoint outputs/full_training_200_esmfold_fixed/checkpoint_epoch_XXX.pt
```

### 4. 监控告警
```python
# 简单的告警脚本
def check_training_health():
    """检查训练健康状态"""
    
    # 检查GPU内存
    gpu_mem = torch.cuda.memory_allocated() // 1024**3
    if gpu_mem > 20:  # 超过20GB
        send_alert("GPU内存使用过高")
    
    # 检查损失是否异常
    if os.path.exists("training_metrics.json"):
        with open("training_metrics.json", "r") as f:
            metrics = json.load(f)
            recent_losses = metrics.get("train_losses", [])[-10:]
            if recent_losses and np.mean(recent_losses) > 1.0:
                send_alert("训练损失异常高")
```

通过这个详细的计划，您可以系统地管理当前的训练过程，并为下一阶段的结构特征集成做好准备。 

graph TB
    A[开始训练] --> B[环境初始化]
    B --> C[加载配置文件]
    C --> D[初始化ESMFold<br/>备用状态]
    D --> E[创建数据集]
    E --> F[创建数据加载器]
    F --> G[创建StructDiff模型]
    G --> H[创建扩散过程]
    H --> I[设置优化器和调度器]
    I --> J[开始训练循环]
    
    J --> K[遍历训练批次]
    K --> L[获取序列嵌入]
    L --> M[采样时间步]
    M --> N[添加噪声]
    N --> O[去噪预测]
    O --> P[计算MSE损失]
    P --> Q[反向传播]
    Q --> R[梯度累积]
    R --> S{是否达到累积步数?}
    S -->|是| T[更新参数]
    S -->|否| K
    T --> U[更新学习率]
    U --> V{是否完成epoch?}
    V -->|否| K
    V -->|是| W[计算epoch损失]
    W --> X{是否需要验证?}
    X -->|是| Y[验证步骤]
    X -->|否| Z
    Y --> Z{是否需要保存?}
    Z -->|是| AA[保存检查点]
    Z -->|否| BB
    AA --> BB{是否完成训练?}
    BB -->|否| J
    BB -->|是| CC[保存最终模型]
    CC --> DD[训练完成]
    
    style A fill:#e1f5fe
    style DD fill:#c8e6c9
    style G fill:#fff3e0
    style H fill:#fff3e0
    style P fill:#ffebee