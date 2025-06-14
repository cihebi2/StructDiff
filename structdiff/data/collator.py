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
        max_len = max(item['sequences'].shape[0] for item in batch)
        
        # Initialize tensors
        batch_size = len(batch)
        sequences = torch.full(
            (batch_size, max_len), 
            self.pad_token_id, 
            dtype=torch.long
        )
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
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
                mask_len = min(item['attention_mask'].shape[0], max_len)
                attention_mask[i, :mask_len] = item['attention_mask'][:mask_len]
            else:
                # 如果没有attention_mask，创建一个
                attention_mask[i, :seq_len] = 1
            
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
                structures, max_len
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
        
        for key, values in structures.items():
            if not values:  # 跳过空列表
                continue
                
            # 确保所有张量在同一设备上
            device = values[0].device if hasattr(values[0], 'device') else torch.device('cpu')
            
            if key == 'distance_matrix' or key == 'contact_map':
                # Handle 2D matrices - need to pad both dimensions
                padded = []
                for matrix in values:
                    # 确保在正确设备上
                    if hasattr(matrix, 'to'):
                        matrix = matrix.to(device)
                    
                    current_len = matrix.shape[0]
                    if current_len > max_len:
                        # 如果太大，截断
                        matrix = matrix[:max_len, :max_len]
                    elif current_len < max_len:
                        # 如果太小，填充
                        pad_size = max_len - current_len
                        matrix = F.pad(
                            matrix, 
                            (0, pad_size, 0, pad_size), 
                            value=0
                        )
                    
                    # 确保最终尺寸正确
                    if matrix.shape[0] != max_len or matrix.shape[1] != max_len:
                        new_matrix = torch.zeros(max_len, max_len, device=device, dtype=matrix.dtype)
                        min_size = min(matrix.shape[0], max_len)
                        new_matrix[:min_size, :min_size] = matrix[:min_size, :min_size]
                        matrix = new_matrix
                        
                    padded.append(matrix)
                
                try:
                    collated_structures[key] = torch.stack(padded)
                except Exception as e:
                    print(f"Error stacking {key}: {e}")
                    # 创建零张量作为后备
                    dtype = values[0].dtype if len(values) > 0 else torch.float32
                    collated_structures[key] = torch.zeros(len(values), max_len, max_len, device=device, dtype=dtype)
                    
            elif key == 'positions':
                # Handle 3D position data (seq_len, n_atoms, 3)
                padded = []
                for positions in values:
                    # 确保在正确设备上
                    if hasattr(positions, 'to'):
                        positions = positions.to(device)
                    
                    current_len = positions.shape[0]
                    if current_len > max_len:
                        # 截断到max_len
                        positions = positions[:max_len]
                    elif current_len < max_len:
                        # 填充序列长度维度
                        pad_size = max_len - current_len
                        positions = F.pad(
                            positions, 
                            (0, 0, 0, 0, 0, pad_size), 
                            value=0
                        )
                    
                    # 确保最终形状正确
                    if positions.shape[0] != max_len:
                        n_atoms, n_dims = positions.shape[1], positions.shape[2]
                        new_positions = torch.zeros(max_len, n_atoms, n_dims, device=device, dtype=positions.dtype)
                        copy_len = min(positions.shape[0], max_len)
                        new_positions[:copy_len] = positions[:copy_len]
                        positions = new_positions
                        
                    padded.append(positions)
                
                try:
                    collated_structures[key] = torch.stack(padded)
                except Exception as e:
                    print(f"Error stacking {key}: {e}")
                    # 创建零张量作为后备
                    n_atoms = values[0].shape[1] if len(values) > 0 and len(values[0].shape) > 1 else 37
                    dtype = values[0].dtype if len(values) > 0 else torch.float32
                    collated_structures[key] = torch.zeros(len(values), max_len, n_atoms, 3, device=device, dtype=dtype)
                    
            else:
                # Handle 1D and 2D features (plddt, angles, secondary_structure, etc.)
                padded = []
                for feat in values:
                    # 确保在正确设备上
                    if hasattr(feat, 'to'):
                        feat = feat.to(device)
                    
                    current_len = feat.shape[0]
                    if current_len > max_len:
                        # 截断
                        feat = feat[:max_len]
                    elif current_len < max_len:
                        # 填充
                        pad_size = max_len - current_len
                        if feat.dim() == 1:
                            # 1D features like plddt, secondary_structure
                            feat = F.pad(feat, (0, pad_size), value=0)
                        elif feat.dim() == 2:
                            # 2D features like angles (seq_len, n_angles)
                            feat = F.pad(feat, (0, 0, 0, pad_size), value=0)
                        else:
                            # Higher dimensional features
                            padding = [0, 0] * (feat.dim() - 1) + [0, pad_size]
                            feat = F.pad(feat, padding, value=0)
                    
                    # 确保最终长度正确
                    if feat.shape[0] != max_len:
                        if feat.dim() == 1:
                            new_feat = torch.zeros(max_len, device=device, dtype=feat.dtype)
                            copy_len = min(feat.shape[0], max_len)
                            new_feat[:copy_len] = feat[:copy_len]
                            feat = new_feat
                        elif feat.dim() == 2:
                            new_feat = torch.zeros(max_len, feat.shape[1], device=device, dtype=feat.dtype)
                            copy_len = min(feat.shape[0], max_len)
                            new_feat[:copy_len] = feat[:copy_len]
                            feat = new_feat
                    
                    padded.append(feat)
                
                try:
                    collated_structures[key] = torch.stack(padded)
                except Exception as e:
                    print(f"Error stacking {key}: {e}")
                    # 创建零张量作为后备
                    if values and len(values) > 0:
                        sample_feat = values[0]
                        dtype = sample_feat.dtype if hasattr(sample_feat, 'dtype') else torch.float32
                        if sample_feat.dim() == 1:
                            collated_structures[key] = torch.zeros(len(values), max_len, device=device, dtype=dtype)
                        elif sample_feat.dim() == 2:
                            collated_structures[key] = torch.zeros(len(values), max_len, sample_feat.shape[1], device=device, dtype=dtype)
        
        return collated_structures
# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
