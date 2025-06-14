#!/usr/bin/env python3
"""
Fix for ESMFold one_hot LongTensor error
"""

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

# 主要修复点：
# 1. 确保使用 add_special_tokens=False
# 2. 确保 input_ids 是 LongTensor 类型
# 3. 正确设置模型精度和设备

def fix_esmfold_prediction(sequence):
    """
    修复后的 ESMFold 预测函数
    """
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage=True  # 减少内存使用
    )
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 重要：将 ESM 部分转换为 float32
    model.esm = model.esm.float()
    model = model.to(device)
    
    # 设置 chunk size 以减少内存使用
    model.trunk.set_chunk_size(64)
    
    # 设置为评估模式
    model.eval()
    
    # 分词 - 关键修复点
    # 必须使用 add_special_tokens=False 避免特殊标记导致的问题
    inputs = tokenizer(
        [sequence],  # 注意是列表格式
        return_tensors="pt",
        add_special_tokens=False  # 这是关键！
    )
    
    # 移动到正确的设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 确保 input_ids 是 LongTensor（修复 one_hot 错误的关键）
    if 'input_ids' in inputs:
        inputs['input_ids'] = inputs['input_ids'].long()
    
    # 如果还有 attention_mask，也确保是正确类型
    if 'attention_mask' in inputs:
        inputs['attention_mask'] = inputs['attention_mask'].long()
    
    # 运行预测
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs


# 如果您的原始代码中有自定义的处理函数，这里是修复版本
def fix_custom_processing(sequence, model, tokenizer, device):
    """
    修复自定义处理中的数据类型问题
    """
    # 1. 正确的分词方式
    inputs = tokenizer(
        [sequence],
        return_tensors="pt",
        add_special_tokens=False,
        padding=False,  # 不要自动填充
        truncation=False  # 不要自动截断
    )
    
    # 2. 确保所有张量都在正确的设备上且类型正确
    processed_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            # 移动到设备
            value = value.to(device)
            
            # 确保整数类型的张量是 LongTensor
            if value.dtype in [torch.int32, torch.int16, torch.int8]:
                value = value.long()
            
            processed_inputs[key] = value
    
    # 3. 如果需要手动创建 one-hot 编码，确保输入是 LongTensor
    if 'input_ids' in processed_inputs:
        input_ids = processed_inputs['input_ids'].long()  # 确保是 LongTensor
        
        # 如果需要 one-hot 编码（通常 ESMFold 不需要）
        # vocab_size = tokenizer.vocab_size
        # one_hot = torch.nn.functional.one_hot(input_ids, num_classes=vocab_size)
    
    return processed_inputs


# 测试修复后的代码
if __name__ == "__main__":
    # 测试序列
    test_sequence = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETARDLLISEQNVVNGITKGEMLPVSDTTGFPYT"
    
    print("测试修复后的 ESMFold...")
    
    try:
        # 使用修复后的函数
        outputs = fix_esmfold_prediction(test_sequence)
        
        print("✓ 预测成功！")
        print(f"  - Positions shape: {outputs.positions.shape}")
        print(f"  - pLDDT shape: {outputs.plddt.shape}")
        
        # 如果需要额外的处理
        # 加载模型用于自定义处理
        tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 测试自定义处理
        processed_inputs = fix_custom_processing(test_sequence, model, tokenizer, device)
        print("\n✓ 自定义处理成功！")
        print(f"  - Input keys: {list(processed_inputs.keys())}")
        print(f"  - Input IDs dtype: {processed_inputs['input_ids'].dtype}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()


# 额外的调试函数
def debug_tensor_types(inputs):
    """
    调试函数：检查所有张量的类型
    """
    print("\n=== Tensor Types Debug ===")
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}:")
            print(f"  - Shape: {value.shape}")
            print(f"  - Dtype: {value.dtype}")
            print(f"  - Device: {value.device}")
            print(f"  - Is LongTensor: {value.dtype == torch.long}")