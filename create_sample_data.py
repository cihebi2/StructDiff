# create_sample_data.py
import pandas as pd
import numpy as np
from pathlib import Path

# 创建数据目录
Path("data/raw").mkdir(parents=True, exist_ok=True)

# 生成示例肽段数据
def generate_peptide(length, peptide_type):
    """生成随机肽段序列"""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    # 根据类型调整氨基酸组成
    if peptide_type == 'antimicrobial':
        # 抗菌肽通常富含阳离子氨基酸
        amino_acids = 'ACDEFGHIKLMNPQRSTVWYKR' * 2  # 增加K和R的比例
    elif peptide_type == 'antifungal':
        amino_acids = 'ACDEFGHIKLMNPQRSTVWYHR' * 2  # 增加H和R
    
    return ''.join(np.random.choice(list(amino_acids), length))

# 生成数据集
data = []
peptide_types = ['antimicrobial', 'antifungal', 'antiviral']

for i in range(1000):  # 生成1000个样本
    length = np.random.randint(10, 30)
    ptype = np.random.choice(peptide_types)
    sequence = generate_peptide(length, ptype)
    
    data.append({
        'id': f'PEP_{i:04d}',
        'sequence': sequence,
        'peptide_type': ptype,
        'length': length
    })

# 保存数据
df = pd.DataFrame(data)
df.to_csv('data/raw/peptides.csv', index=False)
print(f"Created {len(df)} peptide sequences")
print(df.head())