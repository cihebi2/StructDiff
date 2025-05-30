import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import random


class SequenceAugmentation:
    """Augmentation strategies for peptide sequences"""
    
    def __init__(
        self,
        mask_prob: float = 0.15,
        random_prob: float = 0.1,
        unchanged_prob: float = 0.1,
        mask_token_id: int = 32,  # ESM-2 mask token
        vocab_size: int = 33
    ):
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.unchanged_prob = unchanged_prob
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        
    def mask_sequence(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply masked language modeling augmentation
        
        Returns:
            - Augmented sequences
            - Mask indicating which positions were modified
        """
        augmented = sequences.clone()
        batch_size, seq_len = sequences.shape
        
        # Create probability matrix for masking
        probability_matrix = torch.full(sequences.shape, self.mask_prob)
        
        # Don't mask special tokens (first and last)
        probability_matrix[:, 0] = 0.0
        probability_matrix[attention_mask == 0] = 0.0
        
        # Create mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # 80% of time, replace with mask token
        indices_replaced = torch.bernoulli(
            torch.full(sequences.shape, 0.8)
        ).bool() & masked_indices
        augmented[indices_replaced] = self.mask_token_id
        
        # 10% of time, replace with random token
        indices_random = torch.bernoulli(
            torch.full(sequences.shape, 0.5)
        ).bool() & masked_indices & ~indices_replaced
        random_tokens = torch.randint(
            5, self.vocab_size - 1, sequences.shape, dtype=torch.long
        )
        augmented[indices_random] = random_tokens[indices_random]
        
        # 10% of time, keep original (already done)
        
        return augmented, masked_indices
    
    def shuffle_sequence(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        shuffle_prob: float = 0.1
    ) -> torch.Tensor:
        """Randomly shuffle subsequences"""
        batch_size, seq_len = sequences.shape
        augmented = sequences.clone()
        
        for i in range(batch_size):
            if random.random() < shuffle_prob:
                # Get valid sequence (excluding special tokens)
                valid_len = attention_mask[i].sum().item() - 2
                if valid_len > 2:
                    # Shuffle middle portion
                    start = 1
                    end = start + valid_len
                    
                    perm = torch.randperm(valid_len) + start
                    augmented[i, start:end] = sequences[i, perm]
        
        return augmented
    
    def reverse_sequence(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        reverse_prob: float = 0.05
    ) -> torch.Tensor:
        """Reverse sequences"""
        batch_size = sequences.shape[0]
        augmented = sequences.clone()
        
        for i in range(batch_size):
            if random.random() < reverse_prob:
                valid_len = attention_mask[i].sum().item() - 2
                if valid_len > 0:
                    start = 1
                    end = start + valid_len
                    augmented[i, start:end] = sequences[i, start:end].flip(0)
        
        return augmented


class StructureAugmentation:
    """Augmentation strategies for structure features"""
    
    def __init__(
        self,
        noise_level: float = 0.1,
        dropout_prob: float = 0.1,
        jitter_scale: float = 0.05
    ):
        self.noise_level = noise_level
        self.dropout_prob = dropout_prob
        self.jitter_scale = jitter_scale
    
    def add_noise(
        self,
        features: Dict[str, torch.Tensor],
        feature_types: Dict[str, str]
    ) -> Dict[str, torch.Tensor]:
        """Add Gaussian noise to continuous features"""
        augmented = {}
        
        for key, value in features.items():
            if key in feature_types and feature_types[key] == 'continuous':
                noise = torch.randn_like(value) * self.noise_level
                augmented[key] = value + noise
            else:
                augmented[key] = value
        
        return augmented
    
    def dropout_features(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Randomly dropout structure features"""
        augmented = {}
        
        for key, value in features.items():
            if random.random() < self.dropout_prob:
                # Replace with zeros
                augmented[key] = torch.zeros_like(value)
            else:
                augmented[key] = value
        
        return augmented
    
    def jitter_coordinates(
        self,
        coords: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Add small perturbations to 3D coordinates"""
        jitter = torch.randn_like(coords) * self.jitter_scale
        
        # Only jitter valid positions
        mask = attention_mask.unsqueeze(-1).expand_as(coords)
        jittered = coords + jitter * mask
        
        return jittered
    
    def augment_angles(
        self,
        angles: torch.Tensor,
        angle_noise: float = 0.1
    ) -> torch.Tensor:
        """Add noise to torsion angles"""
        noise = torch.randn_like(angles) * angle_noise
        
        # Add noise and wrap to [-pi, pi]
        augmented = angles + noise
        augmented = torch.atan2(torch.sin(augmented), torch.cos(augmented))
        
        return augmented
    
    def mask_distance_matrix(
        self,
        dist_matrix: torch.Tensor,
        mask_prob: float = 0.1
    ) -> torch.Tensor:
        """Randomly mask entries in distance matrix"""
        mask = torch.bernoulli(
            torch.full_like(dist_matrix, 1 - mask_prob)
        )
        
        # Ensure symmetry
        mask = mask * mask.T
        
        # Apply mask (set masked entries to large value)
        augmented = dist_matrix * mask + (1 - mask) * 999.9
        
        return augmented


def augment_batch(
    batch: Dict[str, torch.Tensor],
    seq_aug: Optional[SequenceAugmentation] = None,
    struct_aug: Optional[StructureAugmentation] = None,
    augment_prob: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    Apply augmentation to a batch of data
    
    Args:
        batch: Batch dictionary from dataloader
        seq_aug: Sequence augmentation instance
        struct_aug: Structure augmentation instance
        augment_prob: Probability of applying augmentation
        
    Returns:
        Augmented batch
    """
    if random.random() > augment_prob:
        return batch
    
    augmented = batch.copy()
    
    # Augment sequences
    if seq_aug and 'sequences' in batch:
        # Choose augmentation type
        aug_type = random.choice(['mask', 'shuffle', 'reverse'])
        
        if aug_type == 'mask':
            augmented['sequences'], mask = seq_aug.mask_sequence(
                batch['sequences'],
                batch['attention_mask']
            )
            augmented['masked_positions'] = mask
        elif aug_type == 'shuffle':
            augmented['sequences'] = seq_aug.shuffle_sequence(
                batch['sequences'],
                batch['attention_mask']
            )
        elif aug_type == 'reverse':
            augmented['sequences'] = seq_aug.reverse_sequence(
                batch['sequences'],
                batch['attention_mask']
            )
    
    # Augment structures
    if struct_aug and 'structures' in batch:
        structures = batch['structures']
        
        # Define feature types
        feature_types = {
            'angles': 'continuous',
            'distance_matrix': 'continuous',
            'sasa': 'continuous',
            'secondary_structure': 'discrete',
            'contact_map': 'discrete'
        }
        
        # Apply augmentation
        aug_type = random.choice(['noise', 'dropout', 'specific'])
        
        if aug_type == 'noise':
            augmented['structures'] = struct_aug.add_noise(
                structures, feature_types
            )
        elif aug_type == 'dropout':
            augmented['structures'] = struct_aug.dropout_features(structures)
        elif aug_type == 'specific':
            # Apply specific augmentations
            aug_structures = structures.copy()
            
            if 'angles' in structures:
                aug_structures['angles'] = struct_aug.augment_angles(
                    structures['angles']
                )
            
            if 'distance_matrix' in structures:
                aug_structures['distance_matrix'] = struct_aug.mask_distance_matrix(
                    structures['distance_matrix']
                )
            
            augmented['structures'] = aug_structures
    
    return augmented
# Updated: 05/30/2025 22:59:09
