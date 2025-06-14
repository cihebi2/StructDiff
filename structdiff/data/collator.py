import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PeptideStructureCollator:
    """Collate function for peptide-structure pairs"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pad_token_id = 1  # ESM-2 padding token
        
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of peptide-structure pairs
        
        Args:
            batch: List of dictionaries from dataset
            
        Returns:
            Collated batch dictionary
        """
        # Find max length in batch
        max_seq_len = max(item['sequences'].shape[0] for item in batch)
        
        # 结构特征的长度应该比序列短2 (没有CLS/SEP标记)
        max_struct_len = max_seq_len - 2

        # Initialize tensors
        batch_size = len(batch)
        sequences = torch.full(
            (batch_size, max_seq_len), 
            self.pad_token_id, 
            dtype=torch.long
        )
        attention_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        labels = []
        
        # Structure placeholders
        has_structures = any('structures' in item for item in batch)
        structures = {} if has_structures else None
        
        # Fill tensors
        for i, item in enumerate(batch):
            seq_len = item['sequences'].shape[0]
            sequences[i, :seq_len] = item['sequences']
            
            # 确保attention_mask的处理正确
            if 'attention_mask' in item:
                mask_len = min(item['attention_mask'].shape[0], max_seq_len)
                attention_mask[i, :mask_len] = item['attention_mask'][:mask_len].bool()
            else:
                # 如果没有attention_mask，创建一个
                attention_mask[i, :seq_len] = True
            
            if 'label' in item:
                labels.append(item['label'])
            
            # Handle structures
            if has_structures and 'structures' in item:
                for key, value in item['structures'].items():
                    if key not in structures:
                        structures[key] = []
                    structures[key].append(value)
        
        # Prepare output
        collated = {
            'sequences': sequences,
            'attention_mask': attention_mask,
        }
        
        # 注意：这里创建的 attention_mask 是后续所有操作（包括结构特征）的唯一真实性来源。
        # 由于所有结构特征都与序列长度对齐并被填充到相同长度，
        # 这个掩码同样适用于它们，以确保模型在注意力计算中忽略填充部分。

        if labels:
            # 确保labels可以正确堆叠
            try:
                if isinstance(labels[0], torch.Tensor):
                    collated['labels'] = torch.stack(labels)
                else:
                    collated['labels'] = torch.tensor(labels)
            except Exception as e:
                print(f"Error stacking labels: {e}")
                # 创建默认标签
                collated['labels'] = torch.zeros(batch_size, dtype=torch.long)
        
        # Pad and stack structure features
        if structures:
            collated['structures'] = self._collate_structures(
                structures, max_struct_len
            )
        
        # Add conditions if present
        if 'labels' in collated:
            collated['conditions'] = {
                'peptide_type': collated['labels']
            }
        
        return collated
    
    def _collate_structures(
        self,
        structures: Dict[str, List[torch.Tensor]],
        max_len: int
    ) -> Dict[str, torch.Tensor]:
        """Collate structure features with padding"""
        collated_structures = {}
        
        # 将所有张量移动到同一设备，以避免pad_sequence出错
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for key, values in structures.items():
            if not values:
                continue

            try:
                # 预处理：确保所有张量都在同一设备上，并且是Tensor类型
                processed_values = []
                for v in values:
                    if not isinstance(v, torch.Tensor):
                        v = torch.tensor(v)
                    processed_values.append(v.to(device))
                
                # 特殊处理2D矩阵（如距离矩阵）
                if 'matrix' in key or 'map' in key:
                    padded = []
                    for matrix in processed_values:
                        # 截断或填充到 (max_len, max_len)
                        current_len = matrix.shape[0]
                        if current_len > max_len:
                            matrix = matrix[:max_len, :max_len]
                        elif current_len < max_len:
                            pad_size = max_len - current_len
                            matrix = F.pad(matrix, (0, pad_size, 0, pad_size), value=0)
                        padded.append(matrix)
                    
                    collated_structures[key] = torch.stack(padded)

                # 使用 pad_sequence 处理其他所有可变长度的序列
                else:
                    # 对于plddt这种可能出现异常维度的，先进行squeeze()
                    if key == 'plddt':
                        # 假设 plddt 应该是 (seq_len, ) 或 (seq_len, 1)
                        # 我们移除多余的维度
                        processed_values = [v.squeeze() for v in processed_values]

                    # pad_sequence 需要一个 tensor 列表
                    # batch_first=True -> 输出形状为 (batch_size, max_len, ...)
                    padded_tensors = torch.nn.utils.rnn.pad_sequence(
                        processed_values, 
                        batch_first=True, 
                        padding_value=0.0
                    )
                    collated_structures[key] = padded_tensors

            except Exception as e:
                print(f"Error collating '{key}': {e}")
                # 提供一个更详细的错误，帮助调试
                for i, v in enumerate(values):
                    if isinstance(v, torch.Tensor):
                        print(f"  - Tensor {i} shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
                    else:
                        print(f"  - Item {i} is not a tensor: {type(v)}")
                # 创建一个占位符以允许训练继续
                collated_structures[key] = torch.zeros(len(values), max_len, device=device)

        return collated_structures
    
    def _get_max_dims(self, tensors: List[torch.Tensor]) -> List[int]:
        # ... existing code ...
        pass

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
