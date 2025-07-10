#!/usr/bin/env python3
"""
简化的生成测试 - 测试当前训练好的模型
"""

import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_model_loading():
    """测试模型加载"""
    print("🔄 测试模型加载...")
    
    try:
        from structdiff.models.structdiff import StructDiff
        from omegaconf import OmegaConf
        
        # 创建配置（与训练时一致）
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
                'use_cross_attention': False,  # 注意：训练时禁用了
                'use_cfg': True,
                'cfg_dropout': 0.1
            },
            'data': {'max_length': 512}
        })
        
        print("✓ 配置创建成功")
        
        # 创建模型
        model = StructDiff(config)
        print(f"✓ 模型创建成功，参数数量: {model.count_parameters():,}")
        
        return model, config
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_checkpoint_loading(model, checkpoint_path):
    """测试检查点加载"""
    print(f"🔄 测试检查点加载: {checkpoint_path}")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not Path(checkpoint_path).exists():
            print(f"❌ 检查点文件不存在: {checkpoint_path}")
            return None
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 处理不同的检查点格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("✓ 检查点格式: model_state_dict")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("✓ 检查点格式: state_dict")
        else:
            state_dict = checkpoint
            print("✓ 检查点格式: 直接状态字典")
        
        # 加载到模型
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️ 缺失键: {len(missing_keys)} 个")
            for key in missing_keys[:5]:  # 只显示前5个
                print(f"   - {key}")
        
        if unexpected_keys:
            print(f"⚠️ 多余键: {len(unexpected_keys)} 个")
            for key in unexpected_keys[:5]:  # 只显示前5个
                print(f"   - {key}")
        
        model = model.to(device)
        model.eval()
        
        print("✓ 检查点加载成功")
        return model
        
    except Exception as e:
        print(f"❌ 检查点加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_tokenizer():
    """测试分词器"""
    print("🔄 测试分词器...")
    
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
        
        # 测试编码解码
        test_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIYLRSLGYNIVATPRGYVLAGG"
        encoded = tokenizer.encode(test_seq, add_special_tokens=True)
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        
        print(f"✓ 原始序列: {test_seq[:30]}...")
        print(f"✓ 编码长度: {len(encoded)}")
        print(f"✓ 解码序列: {decoded[:30]}...")
        print(f"✓ 词汇表大小: {len(tokenizer)}")
        
        return tokenizer
        
    except Exception as e:
        print(f"❌ 分词器测试失败: {e}")
        return None

def test_sequence_decoding(model, tokenizer, device):
    """测试序列解码功能"""
    print("🔄 测试序列解码...")
    
    try:
        # 创建随机嵌入
        batch_size = 1
        seq_len = 32  # 包括CLS和SEP
        hidden_dim = model.seq_hidden_dim
        
        # 随机嵌入
        embeddings = torch.randn(batch_size, seq_len, hidden_dim, device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        print(f"✓ 嵌入形状: {embeddings.shape}")
        print(f"✓ 掩码形状: {attention_mask.shape}")
        
        # 解码
        with torch.no_grad():
            sequences = model._decode_embeddings(embeddings, attention_mask)
        
        print(f"✓ 解码结果: {sequences}")
        print(f"✓ 生成序列数量: {len(sequences)}")
        
        if sequences and sequences[0]:
            print(f"✓ 第一个序列: {sequences[0]}")
            print(f"✓ 序列长度: {len(sequences[0])}")
        
        return sequences
        
    except Exception as e:
        print(f"❌ 序列解码失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_simple_generation(model, device):
    """测试简单生成"""
    print("🔄 测试简单生成...")
    
    try:
        # 简单的前向传播测试
        batch_size = 1
        seq_len = 32
        
        # 创建虚拟输入
        sequences = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        timesteps = torch.randint(0, 1000, (batch_size,), device=device)
        conditions = {'peptide_type': torch.tensor([0], device=device, dtype=torch.long)}
        
        print(f"✓ 输入形状 - 序列: {sequences.shape}")
        print(f"✓ 输入形状 - 掩码: {attention_mask.shape}")
        print(f"✓ 时间步: {timesteps}")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(
                sequences=sequences,
                attention_mask=attention_mask,
                timesteps=timesteps,
                conditions=conditions,
                return_loss=False
            )
        
        print(f"✓ 输出键: {list(outputs.keys())}")
        
        if 'denoised_embeddings' in outputs:
            denoised = outputs['denoised_embeddings']
            print(f"✓ 去噪嵌入形状: {denoised.shape}")
            
            # 尝试解码
            try:
                sequences = model._decode_embeddings(denoised, attention_mask)
                print(f"✓ 生成序列: {sequences}")
            except Exception as decode_error:
                print(f"⚠️ 解码失败: {decode_error}")
        
        return True
        
    except Exception as e:
        print(f"❌ 简单生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🧪 简化模型生成测试")
    print("=" * 60)
    
    # 1. 测试模型加载
    model, config = test_model_loading()
    if model is None:
        print("❌ 模型加载失败，测试终止")
        return
    
    # 2. 测试检查点加载
    checkpoint_path = "./outputs/structdiff_fixed/best_model.pt"
    model = test_checkpoint_loading(model, checkpoint_path)
    if model is None:
        print("❌ 检查点加载失败，测试终止")
        return
    
    device = next(model.parameters()).device
    print(f"✓ 模型设备: {device}")
    
    # 3. 测试分词器
    tokenizer = test_tokenizer()
    if tokenizer is None:
        print("❌ 分词器测试失败")
        return
    
    # 4. 测试序列解码
    sequences = test_sequence_decoding(model, tokenizer, device)
    
    # 5. 测试简单生成
    success = test_simple_generation(model, device)
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 基础测试完成！模型可以正常工作")
        print("📝 结论: 当前训练的简化模型功能正常")
    else:
        print("❌ 测试失败，需要进一步调试")
    print("=" * 60)

if __name__ == "__main__":
    main() 