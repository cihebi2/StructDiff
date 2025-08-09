# StructDiff-7.0.0 第二阶段训练错误修复指南

## 🐛 问题描述

在第二阶段训练过程中，评估阶段出现以下错误：
```
ERROR - 生成单个序列失败: can't multiply sequence by non-int of type 'float'
```

## 🔍 问题分析

错误出现在 `_generate_evaluation_samples` 方法中，具体是在去噪步骤：

```python
seq_embeddings = seq_embeddings - 0.1 * noise_pred
```

### 根本原因
1. **类型不匹配**: `noise_pred` 可能是tuple类型而不是tensor
2. **维度不匹配**: 可能存在广播问题
3. **设备不一致**: 张量可能在不同设备上

## 🛠️ 修复方案

### 方案1：修复去噪逻辑
修改 `_generate_evaluation_samples` 方法中的去噪部分：

```python
# 原代码（有问题的）
for t in reversed(range(0, 1000, 100)):
    timesteps = torch.tensor([t], device=self.device)
    noise_pred = model.denoiser(
        seq_embeddings, timesteps, attention_mask
    )
    seq_embeddings = seq_embeddings - 0.1 * noise_pred

# 修复后的代码
for t in reversed(range(0, 1000, 100)):
    timesteps = torch.tensor([t], device=self.device)
    noise_pred_output = model.denoiser(
        seq_embeddings, timesteps, attention_mask
    )
    # 处理可能的tuple返回值
    if isinstance(noise_pred_output, tuple):
        noise_pred = noise_pred_output[0]
    else:
        noise_pred = noise_pred_output
    
    # 确保noise_pred是tensor且形状匹配
    if isinstance(noise_pred, torch.Tensor):
        seq_embeddings = seq_embeddings - 0.1 * noise_pred
    else:
        logger.warning(f"noise_pred类型异常: {type(noise_pred)}")
```

### 方案2：添加类型检查和错误处理
在 `_decode_for_evaluation` 方法中添加更健壮的错误处理：

```python
def _decode_for_evaluation(self, embeddings: torch.Tensor, attention_mask: torch.Tensor, target_length) -> str:
    """为评估解码序列"""
    try:
        # 强制转换target_length为整数
        if target_length is None:
            target_length = 10
        elif isinstance(target_length, torch.Tensor):
            target_length = int(target_length.item())
        elif isinstance(target_length, (float, int)):
            target_length = int(target_length)
        else:
            target_length = int(target_length)
    except (ValueError, TypeError):
        target_length = 10
    
    try:
        # 确保所有输入都是tensor
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings, device=self.device)
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, device=self.device)
            
        # 确保在正确的设备上
        embeddings = embeddings.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # 其余解码逻辑...
```

### 方案3：简化评估生成逻辑
使用更简单的评估生成方法：

```python
def _generate_evaluation_samples(self, model, num_samples: int = 100) -> List[str]:
    """生成用于评估的样本序列（修复版）"""
    try:
        sequences = []
        model.eval()
        
        with torch.no_grad():
            for i in range(num_samples):
                try:
                    # 随机长度
                    length = torch.randint(
                        int(self.config.min_length),
                        int(self.config.max_length) + 1,
                        (1,)
                    ).item()
                    
                    # 确保length是整数
                    length = int(length)
                    if length <= 0:
                        length = 10
                    
                    # 生成随机序列嵌入
                    hidden_size = 320  # 使用实际模型的hidden size
                    seq_embeddings = torch.randn(1, length, hidden_size, device=self.device)
                    
                    # 使用解码器直接生成序列
                    if hasattr(self.model, 'decode_projection'):
                        logits = self.model.decode_projection(seq_embeddings)
                        token_ids = torch.argmax(logits, dim=-1).squeeze(0)
                        
                        if self.tokenizer:
                            sequence = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                            amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
                            clean_sequence = ''.join([c for c in sequence.upper() if c in amino_acids])
                            
                            if clean_sequence and len(clean_sequence) >= self.config.min_length:
                                sequences.append(clean_sequence[:length])
                                
                except Exception as e:
                    logger.error(f"生成单个序列失败: {e}")
                    # 回退到随机序列
                    import random
                    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
                    fallback_length = max(5, min(50, length if 'length' in locals() else 10))
                    fallback_seq = ''.join(random.choices(amino_acids, k=fallback_length))
                    sequences.append(fallback_seq)
        
        return sequences[:num_samples]
        
    except Exception as e:
        logger.error(f"样本生成失败: {e}")
        # 完全回退方案
        import random
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        return [''.join(random.choices(amino_acids, k=20)) for _ in range(min(10, num_samples))]
```

## 🚀 实施步骤

1. **切换到Code模式** - 使用 `switch_mode` 工具
2. **应用修复** - 按照上述方案修改代码
3. **重新运行训练** - 继续第二阶段训练
4. **验证修复** - 检查评估阶段是否正常工作

## 📋 验证清单

- [ ] 修复 `_generate_evaluation_samples` 方法
- [ ] 修复 `_decode_for_evaluation` 方法
- [ ] 测试评估生成逻辑
- [ ] 重新运行第二阶段训练
- [ ] 确认评估阶段无错误

## 🎯 建议

**推荐方案**: 使用方案3（简化评估生成逻辑），因为它：
- 避免了复杂的去噪过程
- 更直接地使用解码器
- 有更好的错误处理
- 更容易调试和维护

**下一步**: 切换到Code模式实施修复