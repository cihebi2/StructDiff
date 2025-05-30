import torch
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
        attention_mask = torch.zeros(batch_size, max_len)
        labels = []
        
        # Structure placeholders
        has_structures = any('structures' in item for item in batch)
        structures = {} if has_structures else None
        
        # Fill tensors
        for i, item in enumerate(batch):
            seq_len = item['sequences'].shape[0]
            sequences[i, :seq_len] = item['sequences']
            attention_mask[i, :seq_len] = item['attention_mask']
            
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
            collated['labels'] = torch.stack(labels)
        
        # Pad and stack structure features
        if structures:
            collated['structures'] = self._collate_structures(
                structures, max_len
            )
        
        # Add conditions if present
        if 'label' in batch[0]:
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
            if key == 'distance_matrix':
                # Handle 2D matrices
                padded = []
                for matrix in values:
                    pad_size = max_len - matrix.shape[0]
                    if pad_size > 0:
                        matrix = F.pad(
                            matrix, 
                            (0, pad_size, 0, pad_size), 
                            value=0
                        )
                    padded.append(matrix)
                collated_structures[key] = torch.stack(padded)
            else:
                # Handle 1D features
                padded = []
                for feat in values:
                    pad_size = max_len - feat.shape[0]
                    if pad_size > 0:
                        feat = F.pad(feat, (0, 0, 0, pad_size), value=0)
                    padded.append(feat)
                collated_structures[key] = torch.stack(padded)
        
        return collated_structures