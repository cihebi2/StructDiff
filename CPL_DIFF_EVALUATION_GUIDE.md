# CPL-Diff标准评估套件使用指南

## 📋 概述

本文档介绍了新集成的CPL-Diff标准评估套件，该套件实现了CPL-Diff论文中的5个核心评估指标，确保与原论文的评估方法完全一致。

## 🎯 CPL-Diff核心评估指标

### 1. **Perplexity ↓** (伪困惑度)
- **目标**: 越低越好
- **原理**: 使用ESM-2蛋白质语言模型计算序列的伪困惑度
- **公式**: 对序列的负伪对数概率取指数
- **实现**: 需要L次正向传播（L为序列长度）
- **Fallback**: 基于自然氨基酸频率的简化计算

### 2. **pLDDT ↑** (置信度分数)
- **目标**: 越高越好
- **原理**: 使用ESMFold预测蛋白质结构的置信度
- **计算**: 所有氨基酸置信度分数的平均值
- **范围**: 0-100，>70表示高置信度
- **Fallback**: 基于序列特征的结构可信度估计

### 3. **Instability ↓** (不稳定性指数)
- **目标**: 越低越好
- **原理**: 使用modlAMP计算基于氨基酸组成的肽稳定性
- **标准**: <40为稳定，>40为不稳定
- **Fallback**: 使用Kyte-Doolittle疏水性指数

### 4. **Similarity ↓** (相似性分数)
- **目标**: 越低越好（表示更新颖）
- **原理**: 使用BLOSUM62矩阵与参考序列进行比对
- **参数**: 开放间隙罚分-10，延伸间隙罚分-0.5
- **Fallback**: 基于编辑距离的相似性计算

### 5. **Activity ↑** (活性预测)
- **目标**: 越高越好
- **原理**: 使用外部分类器预测序列活性
- **分类器**: 
  - AMP: CAMPR4 Random Forest
  - AFP: Antifungipept分类器
  - AVP: Stack-AVP分类器
- **Fallback**: 基于序列特征的经验规则预测

## 🛠️ 使用方法

### 基本使用

```python
from scripts.cpldiff_standard_evaluation import CPLDiffStandardEvaluator

# 创建评估器
evaluator = CPLDiffStandardEvaluator(output_dir="evaluation_results")

# 准备数据
generated_sequences = ["KRWWKWIRWKK", "FRLKWFKRLLK", ...]
reference_sequences = ["MAGAININ", "CECROPIN", ...]

# 运行评估
results = evaluator.comprehensive_cpldiff_evaluation(
    generated_sequences=generated_sequences,
    reference_sequences=reference_sequences,
    peptide_type='antimicrobial'  # 'antifungal', 'antiviral'
)

# 生成报告
evaluator.generate_cpldiff_report(results, "my_evaluation")
```

### 快速演示

```bash
# 运行演示脚本
python3 demo_cpldiff_evaluation.py
```

### 单独使用各指标

```python
# 1. ESM-2伪困惑度
perplexity_results = evaluator.evaluate_esm2_pseudo_perplexity(sequences)

# 2. pLDDT分数
plddt_results = evaluator.evaluate_plddt_scores(sequences)

# 3. 不稳定性指数
instability_results = evaluator.evaluate_instability_index(sequences)

# 4. BLOSUM62相似性
similarity_results = evaluator.evaluate_blosum62_similarity(generated_seqs, reference_seqs)

# 5. 活性预测
activity_results = evaluator.evaluate_activity_prediction(sequences, 'antimicrobial')
```

## 📊 评估结果解释

### 指标期望值参考

对于高质量的抗菌肽生成:

| 指标 | 期望范围 | 说明 |
|------|----------|------|
| Perplexity | 1-50 | 越低表示越符合自然蛋白质模式 |
| pLDDT | 50-100 | 越高表示结构预测置信度越高 |
| Instability | 0-40 | 越低表示肽越稳定 |
| Similarity | 取决于新颖性需求 | 越低表示与已知序列差异越大 |
| Activity | 0.5-1.0 | 越高表示具有目标活性的比例越高 |

### 结果文件说明

- `{name}.json`: 完整评估结果的JSON格式
- `{name}_summary.txt`: 人类可读的摘要报告

## 🔧 依赖管理

### 完整依赖（推荐）

```bash
pip install transformers torch biopython scipy modlamp numpy pandas
```

### 必要依赖（最小化）

```bash
# 仅需Python标准库
# 评估器会自动使用fallback方法
```

### 依赖状态检查

```python
evaluator = CPLDiffStandardEvaluator()
print(evaluator.available_libs)
# {'esm2': False, 'esmfold': False, 'modlamp': False, 'biopython': False}
```

## 📈 性能优化建议

### 1. 批量评估

```python
# 对于大量序列，建议分批处理
batch_size = 100
for i in range(0, len(sequences), batch_size):
    batch = sequences[i:i+batch_size]
    results = evaluator.comprehensive_cpldiff_evaluation(batch, references)
```

### 2. 内存管理

```python
# 对于ESM-2模型，使用CPU避免GPU内存不足
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 强制使用CPU
```

### 3. 并行处理

```python
# 可以并行运行不同的评估指标
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(evaluator.evaluate_esm2_pseudo_perplexity, sequences),
        executor.submit(evaluator.evaluate_instability_index, sequences),
        # ... 其他指标
    ]
```

## 🔍 故障排除

### 常见问题

1. **ModuleNotFoundError**: 
   - 解决：安装缺失依赖或使用fallback方法

2. **CUDA out of memory**:
   - 解决：设置`CUDA_VISIBLE_DEVICES=''`使用CPU

3. **评估速度慢**:
   - 解决：减少序列数量或使用批处理

4. **pLDDT预测失败**:
   - 解决：检查序列长度（建议5-50氨基酸）

### 调试模式

```python
# 启用详细日志
evaluator = CPLDiffStandardEvaluator()
evaluator.debug = True  # 输出详细信息
```

## 🎯 与StructDiff集成

### 训练时评估

```python
# 在训练循环中使用
def evaluate_generation(model, test_data):
    generated_seqs = model.generate(test_data)
    results = evaluator.comprehensive_cpldiff_evaluation(
        generated_seqs, 
        test_data.reference_sequences,
        peptide_type='antimicrobial'
    )
    return results['cpldiff_core_metrics']
```

### 模型选择

```python
# 基于CPL-Diff指标选择最佳模型
best_model = None
best_score = float('inf')

for model_path in model_candidates:
    results = evaluate_model(model_path)
    # 综合评分（可根据需要调整权重）
    score = (results['pseudo_perplexity']['mean_pseudo_perplexity'] - 
             results['plddt']['mean_plddt'] + 
             results['instability']['mean_instability'] -
             results['activity']['activity_ratio'] * 100)
    
    if score < best_score:
        best_score = score
        best_model = model_path
```

## 📚 参考资料

1. **CPL-Diff论文**: 原始评估指标定义
2. **ESM-2模型**: facebook/esm2_t6_8M_UR50D
3. **BLOSUM62矩阵**: BioPython默认实现
4. **modlAMP工具包**: 肽描述符计算

---

## 💡 最佳实践

1. **标准化评估**: 始终使用相同的参考数据集
2. **多次运行**: 对随机性结果进行多次评估取平均
3. **结果验证**: 将结果与已发表的基准进行对比
4. **文档记录**: 保存评估参数和环境信息

这套CPL-Diff标准评估套件确保了StructDiff与最新研究的评估标准保持一致，为模型性能提供了可靠的量化指标。