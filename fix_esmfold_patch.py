#!/usr/bin/env python3
"""
ESMFold Bug Fix - Patch for one_hot LongTensor error
This patches the transformers library's internal ESMFold implementation
"""

import torch
import transformers
from transformers.models.esm import openfold_utils

# 保存原始函数
original_frames_func = openfold_utils.feats.frames_and_literature_positions_to_atom14_pos

def patched_frames_and_literature_positions_to_atom14_pos(
    r: openfold_utils.rigid_utils.Rigid,
    aatype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    """
    修补后的函数，确保所有整数张量都是 LongTensor
    """
    # 确保 aatype 是 LongTensor
    if aatype.dtype != torch.long:
        aatype = aatype.long()
    
    # 确保 group_idx 是 LongTensor（如果存在）
    if hasattr(group_idx, 'dtype') and group_idx.dtype != torch.long:
        group_idx = group_idx.long()
    
    # 调用原始函数
    return original_frames_func(r, aatype, default_frames, group_idx, atom_mask, lit_positions)

# 应用补丁
def apply_esmfold_patch():
    """应用 ESMFold one_hot bug 修复补丁"""
    print("应用 ESMFold bug 修复补丁...")
    
    # 替换函数
    openfold_utils.feats.frames_and_literature_positions_to_atom14_pos = patched_frames_and_literature_positions_to_atom14_pos
    
    # 如果需要，也可以修补其他可能出问题的函数
    # 修补 nn.functional.one_hot 的调用
    original_one_hot = torch.nn.functional.one_hot
    
    def safe_one_hot(tensor, num_classes=-1):
        """确保输入是 LongTensor 的 one_hot 包装器"""
        if tensor.dtype != torch.long:
            tensor = tensor.long()
        return original_one_hot(tensor, num_classes)
    
    torch.nn.functional.one_hot = safe_one_hot
    
    print("✓ 补丁应用成功！")

# 使用修复后的 ESMFold
def run_esmfold_with_patch(sequence):
    """
    运行带补丁的 ESMFold
    """
    # 应用补丁
    apply_esmfold_patch()
    
    # 正常加载和使用 ESMFold
    from transformers import AutoTokenizer, EsmForProteinFolding
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        low_cpu_mem_usage=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.esm = model.esm.float()
    model = model.to(device)
    model.trunk.set_chunk_size(64)
    model.eval()
    
    # 分词
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 确保所有整数张量都是 LongTensor
    for key in inputs:
        if inputs[key].dtype in [torch.int32, torch.int16, torch.int8]:
            inputs[key] = inputs[key].long()
    
    # 运行预测
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs

# 替代方案：使用 monkey patch 修复内部错误
def monkey_patch_esmfold():
    """
    Monkey patch transformers 库的内部函数
    """
    import transformers.models.esm.openfold_utils.feats as feats
    
    # 保存原始函数
    _original_frames = feats.frames_and_literature_positions_to_atom14_pos
    
    def _patched_frames(r, aatype, default_frames, group_idx, atom_mask, lit_positions):
        # 转换数据类型
        if hasattr(aatype, 'dtype') and aatype.dtype != torch.long:
            aatype = aatype.long()
        
        # 在函数内部再次检查和转换
        import torch.nn.functional as F
        _orig_one_hot = F.one_hot
        
        def _safe_one_hot(tensor, num_classes=-1):
            if tensor.dtype != torch.long:
                tensor = tensor.long()
            return _orig_one_hot(tensor, num_classes)
        
        # 临时替换 one_hot
        F.one_hot = _safe_one_hot
        
        try:
            result = _original_frames(r, aatype, default_frames, group_idx, atom_mask, lit_positions)
        finally:
            # 恢复原始 one_hot
            F.one_hot = _orig_one_hot
        
        return result
    
    # 应用补丁
    feats.frames_and_literature_positions_to_atom14_pos = _patched_frames
    print("✓ Monkey patch 应用成功！")

# 主测试函数
if __name__ == "__main__":
    # 测试序列
    test_sequence = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETARDLLISEQNVVNGITKGEMLPVSDTTGFPYT"
    
    print("=== 测试 ESMFold Bug 修复 ===")
    
    try:
        # 方法1：使用补丁函数
        print("\n方法1：使用补丁函数")
        outputs = run_esmfold_with_patch(test_sequence)
        print("✓ 成功！pLDDT shape:", outputs.plddt.shape)
        
    except Exception as e:
        print(f"方法1失败: {e}")
        
        # 方法2：使用 monkey patch
        print("\n方法2：使用 monkey patch")
        try:
            monkey_patch_esmfold()
            
            # 重新运行
            from transformers import AutoTokenizer, EsmForProteinFolding
            
            tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
            model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            
            inputs = tokenizer([test_sequence], return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(device).long() if v.dtype in [torch.int32, torch.int16] else v.to(device) 
                     for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            print("✓ 成功！pLDDT shape:", outputs.plddt.shape)
            
        except Exception as e2:
            print(f"方法2也失败: {e2}")
            print("\n建议：考虑使用其他方案如 ColabFold 或降级 transformers 版本")