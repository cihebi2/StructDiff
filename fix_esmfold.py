"""
ESMFold Quick Fix - 立即可用的修复方案
直接运行此脚本来修复 one_hot LongTensor 错误
"""

import torch
import torch.nn.functional as F

# 全局修复：重写 one_hot 函数
_original_one_hot = F.one_hot

def safe_one_hot(input, num_classes=-1):
    """
    安全的 one_hot 函数，自动转换为 LongTensor
    """
    if hasattr(input, 'dtype') and input.dtype != torch.long:
        input = input.long()
    return _original_one_hot(input, num_classes)

# 应用全局补丁
F.one_hot = safe_one_hot
torch.nn.functional.one_hot = safe_one_hot

print("✓ ESMFold one_hot 补丁已应用")

# 现在可以正常导入和使用 ESMFold
from transformers import AutoTokenizer, EsmForProteinFolding

def predict_structure(sequence, device=None):
    """
    使用修复后的 ESMFold 预测蛋白质结构
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载 ESMFold 模型...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage=True
    )
    
    # 配置模型
    model.esm = model.esm.float()
    model = model.to(device)
    model.trunk.set_chunk_size(64)
    model.eval()
    
    print("✓ 模型加载完成")
    
    # 预测
    print(f"预测长度为 {len(sequence)} 的序列...")
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 额外的类型检查
    for key in inputs:
        if hasattr(inputs[key], 'dtype') and inputs[key].dtype in [torch.int32, torch.int16, torch.int8]:
            inputs[key] = inputs[key].long()
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs

# 主程序
if __name__ == "__main__":
    # 测试序列
    test_sequence = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETARDLLISEQNVVNGITKGEMLPVSDTTGFPYT"
    
    print("=== ESMFold 快速修复测试 ===")
    
    try:
        outputs = predict_structure(test_sequence)
        
        print("\n✓ 预测成功！")
        print(f"  - Positions shape: {outputs.positions.shape}")
        print(f"  - pLDDT shape: {outputs.plddt.shape}")
        print(f"  - 平均 pLDDT: {outputs.plddt.mean().item():.2f}")
        print(f"  - 最小 pLDDT: {outputs.plddt.min().item():.2f}")
        print(f"  - 最大 pLDDT: {outputs.plddt.max().item():.2f}")
        
        # 保存结果（可选）
        # save_pdb(outputs, "output.pdb")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n建议尝试以下方法：")
        print("1. 降级 transformers: pip install transformers==4.30.0")
        print("2. 使用 Linux/WSL2 环境")
        print("3. 使用 Google Colab: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb")
