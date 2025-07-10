import torch
import numpy as np
from structdiff.models.structdiff import StructDiff
from omegaconf import OmegaConf

# 简单的生成测试
config = OmegaConf.create({
    'sequence_encoder': {
        'pretrained_model': 'facebook/esm2_t6_8M_UR50D',
        'freeze_encoder': False,
        'use_lora': True,
        'lora_rank': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1
    },
    'structure_encoder': {
        'hidden_dim': 256,
        'num_layers': 3,
        'use_esmfold': False,
        'use_coordinates': False,
        'use_distances': False,
        'use_angles': False,
        'use_secondary_structure': True
    },
    'denoiser': {
        'hidden_dim': 320,
        'num_layers': 6,
        'num_heads': 8,
        'dropout': 0.1,
        'use_cross_attention': True,
        'use_cfg': True,
        'cfg_dropout': 0.1
    },
    'data': {'max_length': 512}
})

print("创建模型...")
model = StructDiff(config)

print("加载权重...")
checkpoint = torch.load("./outputs/structdiff_fixed/best_model.pt", weights_only=False)
model.load_state_dict(checkpoint, strict=False)
model.eval()

print("生成随机嵌入...")
device = torch.device('cuda')
model.to(device)

# 生成一个简单的序列
batch_size = 1
seq_len = 32  # 包括CLS和SEP
hidden_dim = 320

x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
attention_mask = torch.ones(batch_size, seq_len, device=device)

print("解码为序列...")
sequences = model._decode_embeddings(x, attention_mask)
print(f"生成的序列: {sequences}")
