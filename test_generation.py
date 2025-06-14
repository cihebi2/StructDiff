# test_generation.py
import torch
from omegaconf import OmegaConf, DictConfig
from structdiff.models.structdiff import StructDiff

# 添加 DictConfig 到安全全局变量列表
torch.serialization.add_safe_globals([DictConfig])

# 加载模型
config = OmegaConf.load("configs/minimal_test.yaml")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = StructDiff(config).to(device)

# 修复加载检查点的问题
try:
    # 使用 weights_only=False 来加载包含配置的检查点
    checkpoint = torch.load("checkpoints/model_epoch_10.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ 成功加载训练好的模型")
except FileNotFoundError:
    print("⚠️  检查点文件不存在，使用随机初始化的模型进行生成")
except Exception as e:
    print(f"⚠️  加载检查点失败: {e}")
    print("使用随机初始化的模型进行生成")

model.eval()

# 生成序列
print("开始生成序列...")
with torch.no_grad():
    samples = model.sample(
        batch_size=5,  # 减少批量大小
        seq_length=15,  # 减少序列长度
        sampling_method='ddpm',
        temperature=1.0,
        progress_bar=True
    )

print("\n生成的序列:")
for i, seq in enumerate(samples['sequences']):
    print(f"{i+1}: {seq}")
    
print(f"\n生成质量分数: {samples['scores'].mean().item():.3f}")