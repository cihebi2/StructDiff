import os
import pickle
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import esm

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
        cache_dir: Optional[str] = None
    ):
        self.config = config
        self.is_training = is_training
        self.max_length = config.data.max_length
        self.min_length = config.data.min_length
        
        # Load data
        self.data = self._load_data(data_path)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.sequence_encoder.path
        )
        
        # Structure cache
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(data_path), "structure_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize ESMFold if needed
        self.use_predicted_structures = config.data.get('use_predicted_structures', True)
        if self.use_predicted_structures:
            self._init_structure_predictor()
        
        logger.info(f"Loaded {len(self.data)} peptide sequences")
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load peptide data from CSV file"""
        df = pd.read_csv(data_path)
        
        # Filter by length
        df = df[
            (df['sequence'].str.len() >= self.min_length) &
            (df['sequence'].str.len() <= self.max_length)
        ]
        
        return df.reset_index(drop=True)
    
    def _init_structure_predictor(self):
        """Initialize ESMFold for structure prediction"""
        try:
            self.structure_predictor = esm.pretrained.esmfold_v1()
            self.structure_predictor.eval()
            if torch.cuda.is_available():
                self.structure_predictor = self.structure_predictor.cuda()
            logger.info("Initialized ESMFold for structure prediction")
        except Exception as e:
            logger.warning(f"Could not initialize ESMFold: {e}")
            self.structure_predictor = None
            self.use_predicted_structures = False
    
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
        if self.is_training and self.config.data.augmentation.enable:
            item = self._apply_augmentation(item)
        
        return item
    
    def _get_structure_features(
        self,
        sequence: str,
        idx: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Get or compute structure features for a sequence"""
        # Check cache first
        cache_path = os.path.join(
            self.cache_dir,
            f"{idx}_{sequence[:10]}.pkl"
        )
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load cached structure: {e}")
        
        # Predict structure if needed
        if self.use_predicted_structures and self.structure_predictor is not None:
            try:
                structure_features = predict_structure_with_esmfold(
                    sequence,
                    self.structure_predictor
                )
                
                # Cache the results
                with open(cache_path, 'wb') as f:
                    pickle.dump(structure_features, f)
                
                return structure_features
            except Exception as e:
                logger.warning(f"Could not predict structure for sequence {idx}: {e}")
        
        return None
    
    def _apply_augmentation(self, item: Dict) -> Dict:
        """Apply data augmentation"""
        # Mask random positions
        if np.random.random() < self.config.data.augmentation.mask_prob:
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