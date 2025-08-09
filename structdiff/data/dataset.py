import os
import pickle
import hashlib
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from ..utils.logger import get_logger
from .structure_utils import extract_structure_features, predict_structure_with_esmfold

logger = get_logger(__name__)


class PeptideStructureDataset(Dataset):
    """
    Dataset for peptide sequences with optional structure information
    """
    
    def __init__(
        self,
        data_path: str,
        config: Dict,
        is_training: bool = True,
        cache_dir: Optional[str] = None,  # No longer used, kept for compatibility
        shared_esmfold: Optional[object] = None
    ):
        self.config = config
        self.is_training = is_training
        self.max_length = config.data.max_length
        self.min_length = config.data.get('min_length', 8)
        self.use_predicted_structures = config.data.get('use_predicted_structures', False)
        
        # 外部ESMFold实例优先级最高
        self.structure_predictor = shared_esmfold
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.sequence_encoder.pretrained_model
        )
        
        # Load data
        logger.info(f"Loading dataset from {data_path}")
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            self.data = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Structure predictor is no longer initialized here.
        # It is assumed that features are pre-computed.
        if self.use_predicted_structures:
            logger.info("预计算的结构特征已启用。将从缓存加载。")
            self.structure_predictor = None # Ensure no dynamic prediction
        else:
            logger.info("结构特征未启用。")
        
        # The cache directory is now determined by the _get_structure_features method
        # to match the pre-computation script's logic. No need to create a directory here.
        
        # Filter sequences by length
        self.data = self.data[
            (self.data['sequence'].str.len() >= self.min_length) &
            (self.data['sequence'].str.len() <= self.max_length)
        ].reset_index(drop=True)
        
        logger.info(f"Dataset loaded: {len(self.data)} sequences")
        if self.use_predicted_structures and self.structure_predictor:
            logger.info("Structure prediction enabled")
        else:
            logger.info("Structure prediction disabled")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset"""
        row = self.data.iloc[idx]
        sequence = row['sequence']
        label = row.get('label', 0)  # 0: antimicrobial, 1: antifungal, 2: antiviral
        
        # Tokenize sequence
        encoding = self.tokenizer(
            sequence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length + 2,  # +2 for CLS and SEP tokens
            return_tensors='pt'
        )
        
        item = {
            'sequence': sequence,
            'sequences': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        # Get structure features
        structure_features = self._get_structure_features(sequence, idx)
        if structure_features is not None:
            item['structures'] = structure_features
        
        # Apply data augmentation if training
        augmentation_config = getattr(self.config.data, 'augmentation', None)
        if (self.is_training and augmentation_config and 
            getattr(augmentation_config, 'enable', False)):
            item = self._apply_augmentation(item)
        
        return item
    
    def _get_structure_features(
        self,
        sequence: str,
        idx: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Get structure features from pre-computed cache."""
        if not self.use_predicted_structures:
            return None

        # Use the same hashing logic as the pre-computation script
        seq_hash = hashlib.md5(sequence.encode()).hexdigest()
        cache_base_dir = "./structure_cache"
        subdir = seq_hash[:2]
        cache_path = os.path.join(cache_base_dir, subdir, f"{seq_hash}.pkl")

        if not os.path.exists(cache_path):
            logger.error(f"错误：找不到预计算的结构特征文件: {cache_path}")
            logger.error(f"序列 (索引 {idx}, 前10个字符: {sequence[:10]}): {sequence}")
            logger.error("请确保已成功运行 precompute_structure_features.py 脚本。")
            raise FileNotFoundError(f"缓存文件不存在: {cache_path}")

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"加载缓存文件 {cache_path} 失败: {e}")
            raise
    
    def _apply_augmentation(self, item: Dict) -> Dict:
        """Apply data augmentation"""
        # Mask random positions
        augmentation_config = getattr(self.config.data, 'augmentation', None)
        mask_prob = getattr(augmentation_config, 'mask_prob', 0.15) if augmentation_config else 0.15
        
        if np.random.random() < mask_prob:
            mask_positions = np.random.choice(
                range(1, len(item['sequence']) + 1),  # Skip CLS token
                size=int(len(item['sequence']) * 0.15),
                replace=False
            )
            item['sequences'][mask_positions] = self.tokenizer.mask_token_id
        
        return item


class PeptideStructureDatasetInference(Dataset):
    """
    Dataset for inference with pre-computed structure features
    """
    
    def __init__(
        self,
        sequences: List[str],
        structures: Optional[List[Dict]] = None,
        config: Optional[Dict] = None
    ):
        self.sequences = sequences
        self.structures = structures or [None] * len(sequences)
        self.config = config
        
        # Initialize tokenizer
        model_path = config.model.sequence_encoder.path if config else "facebook/esm2_t6_8M_UR50D"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.max_length = config.data.max_length if config else 50
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            sequence,
            padding='max_length',
            truncation=True,
            max_length=self.max_length + 2,
            return_tensors='pt'
        )
        
        item = {
            'sequence': sequence,
            'sequences': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }
        
        # Add structure if available
        if self.structures[idx] is not None:
            item['structures'] = self.structures[idx]
        
        return item
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
