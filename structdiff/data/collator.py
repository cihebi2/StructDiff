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
        """Collate structure features with padding - 修复版本"""
        collated_structures = {}
        
        # 将所有张量移动到同一设备，以避免pad_sequence出错
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size = len(next(iter(structures.values())))

        for key, values in structures.items():
            if not values:
                continue

            try:
                # 预处理：确保所有张量都在同一设备上，并且是Tensor类型
                processed_values = []
                max_shapes = []
                
                # 先分析所有张量的形状，找出维度不一致的问题
                for i, v in enumerate(values):
                    if not isinstance(v, torch.Tensor):
                        v = torch.tensor(v)
                    v = v.to(device)
                    
                    # 检查是否有异常的维度
                    if len(v.shape) > 3:
                        # 只在第一次遇到时警告，避免日志过多
                        if not hasattr(self, '_shape_warnings'):
                            self._shape_warnings = set()
                        warning_key = f"{key}_{len(v.shape)}"
                        if warning_key not in self._shape_warnings:
                            print(f"Info: Detected unusual {key} shape: {v.shape}, auto-correcting...")
                            self._shape_warnings.add(warning_key)
                        # 尝试reshape到合理的形状
                        if key == 'positions' and len(v.shape) == 5:
                            # 对于positions，期望形状是 (seq_len, atom_num, 3)
                            # 如果是 (batch, seq_len, atom_num, seq_len, 3)，取第一个batch和atom
                            v = v[0, :, 0, :, :]  # 变成 (seq_len, seq_len, 3)
                            v = v[:, :3]  # 只取前3个atom坐标，变成 (seq_len, 3)
                        elif key == 'distance_matrix' and len(v.shape) == 4:
                            # 对于距离矩阵，期望形状是 (seq_len, seq_len)
                            v = v[0, 0, :, :]  # 取第一个batch和第一个head
                        elif key == 'contact_map' and len(v.shape) == 4:
                            # 对于接触图，期望形状是 (seq_len, seq_len)
                            v = v[0, 0, :, :]  # 取第一个batch和第一个head
                    
                    processed_values.append(v)
                    
                    # 记录每个维度的最大值
                    if not max_shapes:
                        max_shapes = list(v.shape)
                    else:
                        for j in range(min(len(max_shapes), len(v.shape))):
                            max_shapes[j] = max(max_shapes[j], v.shape[j])
                
                # 特殊处理2D矩阵（如距离矩阵、接触图）
                if 'matrix' in key or 'map' in key:
                    padded = []
                    for matrix in processed_values:
                        # 确保是2D矩阵
                        while len(matrix.shape) > 2:
                            matrix = matrix[0]
                        
                        # 截断或填充到 (max_len, max_len)
                        current_len = min(matrix.shape[0], matrix.shape[1])
                        if current_len > max_len:
                            matrix = matrix[:max_len, :max_len]
                        elif current_len < max_len:
                            pad_h = max_len - matrix.shape[0]
                            pad_w = max_len - matrix.shape[1]
                            matrix = F.pad(matrix, (0, pad_w, 0, pad_h), value=0)
                        padded.append(matrix)
                    
                    collated_structures[key] = torch.stack(padded)

                # 处理positions（3D坐标）
                elif key == 'positions':
                    padded = []
                    for pos in processed_values:
                        # 确保是2D矩阵 (seq_len, 3) 或 (seq_len, atom_num, 3)
                        if len(pos.shape) > 3:
                            pos = pos.reshape(-1, 3)  # 展平到 (seq_len*atom_num, 3)
                        elif len(pos.shape) == 3:
                            # (seq_len, atom_num, 3) -> (seq_len, 3)，只取第一个原子
                            pos = pos[:, 0, :]
                        
                        # 截断或填充到 (max_len, 3)
                        if pos.shape[0] > max_len:
                            pos = pos[:max_len, :]
                        elif pos.shape[0] < max_len:
                            pad_size = max_len - pos.shape[0]
                            pos = F.pad(pos, (0, 0, 0, pad_size), value=0)
                        
                        # 确保最后一个维度是3（xyz坐标）
                        if pos.shape[-1] != 3:
                            if pos.shape[-1] > 3:
                                pos = pos[:, :3]
                            else:
                                pad_feat = 3 - pos.shape[-1]
                                pos = F.pad(pos, (0, pad_feat), value=0)
                        
                        padded.append(pos)
                    
                    collated_structures[key] = torch.stack(padded)

                # 处理1D特征（如plddt、angles等）
                else:
                    padded = []
                    for feat in processed_values:
                        # 移除多余的维度
                        feat = feat.squeeze()
                        
                        # 如果还是多维的，展平
                        if len(feat.shape) > 1:
                            feat = feat.flatten()
                        
                        # 截断或填充到max_len
                        if feat.shape[0] > max_len:
                            feat = feat[:max_len]
                        elif feat.shape[0] < max_len:
                            pad_size = max_len - feat.shape[0]
                            feat = F.pad(feat, (0, pad_size), value=0)
                        
                        padded.append(feat)
                    
                    collated_structures[key] = torch.stack(padded)

            except Exception as e:
                print(f"Error collating '{key}': {e}")
                # 提供更详细的错误信息
                for i, v in enumerate(values):
                    if isinstance(v, torch.Tensor):
                        print(f"  - Tensor {i} shape: {v.shape}, dtype: {v.dtype}, device: {v.device}")
                    else:
                        print(f"  - Item {i} is not a tensor: {type(v)}")
                
                # 根据key的类型创建合适的占位符
                if 'matrix' in key or 'map' in key:
                    # 2D矩阵
                    collated_structures[key] = torch.zeros(batch_size, max_len, max_len, device=device)
                elif key == 'positions':
                    # 3D坐标
                    collated_structures[key] = torch.zeros(batch_size, max_len, 3, device=device)
                else:
                    # 1D特征
                    collated_structures[key] = torch.zeros(batch_size, max_len, device=device)

        return collated_structures
    
    def _get_max_dims(self, tensors: List[torch.Tensor]) -> List[int]:
        # ... existing code ...
        pass

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
