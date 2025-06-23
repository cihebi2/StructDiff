# prepare_classification_data.py - 准备抗菌肽分类数据
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def prepare_classification_data(input_file, output_dir="data/processed", test_size=0.2, val_size=0.2, random_state=42):
    """
    准备抗菌肽分类数据
    
    Args:
        input_file: 输入的CSV文件路径
        output_dir: 输出目录
        test_size: 测试集比例
        val_size: 验证集比例（从训练集中分出）
        random_state: 随机种子
    """
    
    print(f"🔄 正在处理数据文件: {input_file}")
    
    # 读取数据
    df = pd.read_csv(input_file)
    print(f"📊 原始数据形状: {df.shape}")
    
    # 检查数据格式
    required_columns = ['sequence', 'label', 'weight']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")
    
    # 数据统计
    print("\n📈 标签分布:")
    label_counts = df['label'].value_counts().sort_index()
    label_names = {0: '抗菌肽', 1: '抗真菌肽', 2: '抗病毒肽'}
    for label, count in label_counts.items():
        print(f"  {label} ({label_names.get(label, '未知')}): {count} 样本")
    
    print("\n⚖️ 权重分布:")
    weight_counts = df['weight'].value_counts().sort_index()
    for weight, count in weight_counts.items():
        print(f"  权重 {weight}: {count} 样本")
    
    # 序列长度统计
    df['seq_length'] = df['sequence'].str.len()
    print(f"\n📏 序列长度统计:")
    print(f"  最短: {df['seq_length'].min()}")
    print(f"  最长: {df['seq_length'].max()}")
    print(f"  平均: {df['seq_length'].mean():.1f}")
    print(f"  中位数: {df['seq_length'].median():.1f}")
    
    # 数据清洗
    print("\n🧹 数据清洗...")
    original_size = len(df)
    
    # 移除过短或过长的序列
    min_length = 5
    max_length = 100
    df = df[(df['seq_length'] >= min_length) & (df['seq_length'] <= max_length)]
    print(f"  移除长度不合适的序列: {original_size - len(df)} 条")
    
    # 移除包含异常字符的序列
    valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    def is_valid_sequence(seq):
        return all(aa in valid_amino_acids for aa in seq.upper())
    
    valid_mask = df['sequence'].apply(is_valid_sequence)
    df = df[valid_mask]
    print(f"  移除包含异常字符的序列: {original_size - len(df)} 条")
    
    print(f"  清洗后数据形状: {df.shape}")
    
    # 准备最终数据格式，保留所有原有字段
    df_clean = df.copy()
    df_clean['sequence'] = df_clean['sequence'].str.upper()  # 统一大写
    
    # 分层采样，确保各类别在训练、验证、测试集中都有代表
    # 首先分出测试集
    X = df_clean.drop(['label'], axis=1)  # 保留除label外的所有字段
    y = df_clean['label']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    # 再从剩余数据中分出验证集
    val_size_adjusted = val_size / (1 - test_size)  # 调整验证集比例
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=y_temp
    )
    
    # 重新组合数据
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    print(f"\n📊 数据集分割结果:")
    print(f"  训练集: {len(train_df)} 样本")
    print(f"  验证集: {len(val_df)} 样本")
    print(f"  测试集: {len(test_df)} 样本")
    
    # 检查各集合的标签分布
    for name, data in [('训练集', train_df), ('验证集', val_df), ('测试集', test_df)]:
        print(f"\n{name}标签分布:")
        label_dist = data['label'].value_counts().sort_index()
        for label, count in label_dist.items():
            percentage = count / len(data) * 100
            print(f"  {label} ({label_names.get(label)}): {count} ({percentage:.1f}%)")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存数据集
    train_path = output_path / "train.csv"
    val_path = output_path / "val.csv"
    test_path = output_path / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n💾 数据已保存:")
    print(f"  训练集: {train_path}")
    print(f"  验证集: {val_path}")
    print(f"  测试集: {test_path}")
    
    # 保存统计信息
    stats = {
        'total_samples': len(df_clean),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'label_distribution': {
            name: {
                'train': int(train_df[train_df['label'] == label].shape[0]),
                'val': int(val_df[val_df['label'] == label].shape[0]),
                'test': int(test_df[test_df['label'] == label].shape[0])
            }
            for label, name in label_names.items()
            if label in df_clean['label'].values
        },
        'sequence_length': {
            'min': int(df_clean['sequence'].str.len().min()),
            'max': int(df_clean['sequence'].str.len().max()),
            'mean': float(df_clean['sequence'].str.len().mean()),
            'median': float(df_clean['sequence'].str.len().median())
        },
        'weight_distribution': {float(k): int(v) for k, v in df_clean['weight'].value_counts().items()}
    }
    
    import json
    stats_path = output_path / "dataset_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"  统计信息: {stats_path}")
    
    return train_path, val_path, test_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='准备抗菌肽分类数据')
    parser.add_argument('--input', type=str, required=True,
                       help='输入的CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                       help='输出目录')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.2,
                       help='验证集比例')
    parser.add_argument('--random_state', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    try:
        train_path, val_path, test_path = prepare_classification_data(
            input_file=args.input,
            output_dir=args.output_dir,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state
        )
        
        print("\n✅ 数据准备完成！")
        print("\n🚀 接下来可以使用以下命令开始训练:")
        print(f"cd /home/qlyu/StructDiff")
        print(f"python train_full.py --config configs/classification_config.yaml --output_dir outputs/classification_run")
        
    except Exception as e:
        print(f"\n❌ 数据准备失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 