# test_model_init.py
import torch
from omegaconf import OmegaConf
from structdiff.models.structdiff import StructDiff

# 加载配置
config = OmegaConf.load("configs/small_model.yaml")

# 创建模型
try:
    model = StructDiff(config)
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 测试前向传播
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 创建假数据
    batch_size = 2
    seq_len = 20
    sequences = torch.randint(0, 20, (batch_size, seq_len)).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    timesteps = torch.randint(0, config.diffusion.num_timesteps, (batch_size,)).to(device)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(
            sequences=sequences,
            attention_mask=attention_mask,
            timesteps=timesteps,
            return_loss=True
        )
    
    print("✓ Forward pass successful")
    print(f"  Loss: {outputs['total_loss'].item():.4f}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()