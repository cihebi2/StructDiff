# load_esmfold.py
"""
专门从 Huggingface 加载 ESMFold 模型的脚本
使用fix_esmfold_patch.py中的补丁修复CUDA错误
"""

import os
import sys
import subprocess
import torch
import warnings
from typing import Optional, Tuple

# 导入补丁
from fix_esmfold_patch import apply_esmfold_patch, monkey_patch_esmfold

def install_dependencies():
    """安装必需的依赖包"""
    print("正在检查和安装依赖包...")
    
    # 检查和安装 accelerate
    try:
        import accelerate
        print(f"✓ accelerate 已安装，版本: {accelerate.__version__}")
    except ImportError:
        print("❌ accelerate 未安装，正在安装...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "accelerate>=0.26.0", "-q"
        ])
        print("✓ accelerate 安装完成")
    
    # 检查其他依赖
    dependencies = {
        'transformers': '4.30.0',
        'torch': '2.0.0',
    }
    
    for package, min_version in dependencies.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {package} 已安装，版本: {version}")
        except ImportError:
            print(f"⚠️ {package} 未安装，请手动安装")

def load_esmfold_huggingface(
    model_name: str = "facebook/esmfold_v1",
    device: Optional[str] = None,
    use_low_cpu_mem: bool = True,
    cache_dir: Optional[str] = None
) -> Tuple[Optional[object], Optional[object], bool]:
    """
    专门从 Huggingface 加载 ESMFold 模型
    
    Args:
        model_name: 模型名称，默认为 facebook/esmfold_v1
        device: 设备，默认自动选择
        use_low_cpu_mem: 是否使用低CPU内存模式
        cache_dir: 缓存目录
        
    Returns:
        (model, tokenizer, success): 模型对象、分词器和是否成功加载的标志
    """
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"正在从 Huggingface 加载 ESMFold 模型: {model_name}")
    print(f"目标设备: {device}")
    
    try:
        # 首先应用补丁修复已知问题
        print("应用ESMFold补丁...")
        apply_esmfold_patch()
        monkey_patch_esmfold()
        
        from transformers import EsmForProteinFolding, AutoTokenizer
        
        print("正在加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("正在加载模型...")
        # 设置加载参数
        load_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "trust_remote_code": True
        }
        
        if use_low_cpu_mem:
            load_kwargs["low_cpu_mem_usage"] = True
            # 如果在CUDA设备上，使用device_map
            if device == "cuda":
                load_kwargs["device_map"] = "auto"
        
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        
        # 加载模型
        model = EsmForProteinFolding.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # 如果没有使用device_map，手动移动到设备
        if "device_map" not in load_kwargs:
            model = model.to(device)
        
        # 设置模型为评估模式
        model.eval()
        
        # 优化设置
        if hasattr(model, 'trunk') and hasattr(model.trunk, 'set_chunk_size'):
            model.trunk.set_chunk_size(64)  # 减少内存使用
        
        print("✓ ESMFold 模型加载成功 (Huggingface)")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, tokenizer, True
        
    except Exception as e:
        print(f"❌ Huggingface方法失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None, None, False

def test_esmfold_inference_safe(model, tokenizer, sequence: str = None, device: str = "cuda"):
    """安全地测试ESMFold推理功能"""
    if sequence is None:
        # 使用较短的测试序列避免索引错误
        sequence = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    
    print(f"\n正在测试 ESMFold 推理...")
    print(f"测试序列: {sequence}")
    print(f"序列长度: {len(sequence)}")
    
    try:
        with torch.no_grad():
            # 处理输入序列
            print("正在处理输入序列...")
            
            # 使用add_special_tokens=False避免索引问题
            tokenized = tokenizer(
                sequence, 
                return_tensors="pt",
                add_special_tokens=False,
                padding=False,
                truncation=False
            )
            
            print(f"Tokenized keys: {list(tokenized.keys())}")
            print(f"Input shape: {tokenized['input_ids'].shape}")
            
            # 移动到设备并确保数据类型正确
            if device == "cuda":
                for key in tokenized:
                    tokenized[key] = tokenized[key].to(device)
                    # 确保整数张量是LongTensor
                    if tokenized[key].dtype in [torch.int32, torch.int16, torch.int8]:
                        tokenized[key] = tokenized[key].long()
            
            print("正在执行模型推理...")
            
            # 尝试推理
            outputs = model(tokenized['input_ids'])
            
            print("✓ 推理测试成功")
            if hasattr(outputs, 'positions'):
                print(f"结构坐标形状: {outputs.positions.shape}")
            if hasattr(outputs, 'plddt'):
                print(f"pLDDT分数形状: {outputs.plddt.shape}")
                avg_plddt = outputs.plddt.mean().item()
                print(f"平均pLDDT分数: {avg_plddt:.3f}")
            
            return outputs
            
    except Exception as e:
        print(f"❌ 推理测试失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        
        # 尝试更短的序列
        if len(sequence) > 50:
            print("尝试使用更短的序列...")
            short_sequence = sequence[:30]  # 截短到30个氨基酸
            return test_esmfold_inference_safe(model, tokenizer, short_sequence, device)
        
        return None

def main():
    """主函数"""
    print("=== ESMFold Huggingface 加载脚本 ===\n")
    
    # 1. 安装依赖
    install_dependencies()
    print()
    
    # 2. 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cuda":
        print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        print(f"当前 GPU 内存使用: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    print()
    
    # 3. 加载模型
    model, tokenizer, success = load_esmfold_huggingface(device=device)
    
    if success and model is not None:
        print(f"\n🎉 ESMFold 模型加载成功！")
        
        # 4. 测试推理
        test_result = test_esmfold_inference_safe(model, tokenizer, device=device)
        
        if test_result is not None:
            print("✓ 模型可以正常进行结构预测")
        else:
            print("⚠️ 模型加载成功但推理测试失败")
            
        # 5. 显示内存使用
        if device == "cuda":
            print(f"\n模型加载后 GPU 内存使用: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
            
        return model, tokenizer
            
    else:
        print("\n❌ ESMFold 模型加载失败")
        print("请检查以下问题：")
        print("1. 是否安装了所有必需的依赖包")
        print("2. 网络连接是否正常")
        print("3. 是否有足够的内存/显存")
        print("\n建议手动安装依赖：")
        print("pip install accelerate>=0.26.0 transformers torch")
        
        return None, None

if __name__ == "__main__":
    main() 