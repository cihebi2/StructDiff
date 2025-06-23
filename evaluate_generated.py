# evaluate_generated.py
import torch
from omegaconf import OmegaConf, DictConfig
from structdiff.models.structdiff import StructDiff
from structdiff.metrics import compute_sequence_metrics
import pandas as pd
import numpy as np

# 添加 DictConfig 到安全全局变量列表
torch.serialization.add_safe_globals([DictConfig])

def main():
    print("正在评估生成的序列...")
    
    # 加载模型配置
    config = OmegaConf.load("configs/small_model.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = StructDiff(config).to(device)
    
    # 尝试加载训练好的模型
    try:
        checkpoint = torch.load("checkpoints/model_epoch_10.pth", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ 成功加载训练好的模型")
    except FileNotFoundError:
        print("⚠️  检查点文件不存在，使用随机初始化的模型")
    except Exception as e:
        print(f"⚠️  加载检查点失败: {e}")
        print("使用随机初始化的模型")
    
    model.eval()
    
    # 生成序列用于评估
    print("正在生成序列...")
    with torch.no_grad():
        samples = model.sample(
            batch_size=20,  # 生成更多序列用于评估
            seq_length=20,
            sampling_method='ddpm',
            temperature=1.0,
            progress_bar=True
        )
    
    generated_sequences = samples['sequences']
    print(f"生成了 {len(generated_sequences)} 个序列")
    
    # 加载参考序列
    try:
        ref_df = pd.read_csv("data/processed/train.csv")  # 使用训练数据作为参考
        reference_sequences = ref_df['sequence'].tolist()[:100]  # 取前100个作为参考
        print(f"加载了 {len(reference_sequences)} 个参考序列")
    except FileNotFoundError:
        print("⚠️  参考数据文件不存在，使用默认参考序列")
        # 使用一些典型的抗菌肽序列作为参考
        reference_sequences = [
            "GLRKRLRKFRNKIKYLRPRRN",
            "KWKLFKKIEKVGQNIRDGIVGGAAYAAGKYA", 
            "GIGKFLHSAKKFGKAFVGEIMNS",
            "GIGDPVTCLKSGAICHPVFCG",
            "FLPIIAKLLGLL",
            "FMNDNEDFFVA",
            "QITDVQGWGEDAPDQYAYQPKFNAFING",
            "DHMVYVYKVMYQMQNHIEC",
            "KMRYTPVRHY",
            "RHICQEEVKSDN"
        ]
    
    # 计算评估指标
    print("\n正在计算评估指标...")
    metrics = compute_sequence_metrics(generated_sequences, reference_sequences)
    
    print("\n📊 评估结果:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 显示一些生成的序列示例
    print("\n📋 生成序列示例:")
    print("=" * 50)
    for i, seq in enumerate(generated_sequences[:10]):
        quality_score = samples['scores'][i].item()
        print(f"{i+1:2d}: {seq:30s} (质量分数: {quality_score:.3f})")
    
    # 保存结果
    results = {
        'generated_sequences': generated_sequences,
        'metrics': {k: float(v) for k, v in metrics.items()},  # 转换为 Python float
        'quality_scores': [float(score) for score in samples['scores'].tolist()]  # 转换为 Python float
    }
    
    import json
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 评估结果已保存到 evaluation_results.json")

if __name__ == "__main__":
    main()