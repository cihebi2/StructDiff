import torch
import transformers
import esm
from structdiff import StructDiff
from structdiff.models.esmfold_wrapper import ESMFoldWrapper

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# 测试ESM模型加载
try:
    from transformers import AutoModel, AutoTokenizer
    model_name = "facebook/esm2_t6_8M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    print("✓ ESM-2 loaded successfully")
except Exception as e:
    print(f"✗ ESM-2 loading failed: {e}")

# 测试ESMFold（可选）
try:
    esmfold = ESMFoldWrapper()
    print("✓ ESMFold loaded successfully")
except Exception as e:
    print(f"✗ ESMFold loading failed: {e}")
    print("  (This is optional, you can continue without it)")

print("\nInstallation check complete!")