#!/usr/bin/env python3
"""
Preprocess peptide data for training
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import pandas as pd
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from structdiff.data.structure_utils import (
    extract_structure_features,
    predict_structure_with_esmfold
)
from structdiff.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess peptide data")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input data file (FASTA, CSV, or directory of PDB files)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--structure_dir",
        type=str,
        default=None,
        help="Directory containing PDB structure files"
    )
    parser.add_argument(
        "--predict_structures",
        action="store_true",
        help="Predict structures using ESMFold"
    )
    parser.add_argument(
        "--split_ratios",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help="Train/val/test split ratios"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=5,
        help="Minimum sequence length"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    return parser.parse_args()


def load_sequences(file_path: str) -> List[Dict]:
    """Load sequences from various formats"""
    sequences = []
    
    if file_path.endswith('.fasta') or file_path.endswith('.fa'):
        # Load from FASTA
        for record in SeqIO.parse(file_path, "fasta"):
            seq_data = {
                'id': record.id,
                'sequence': str(record.seq),
                'description': record.description
            }
            
            # Parse peptide type from description if available
            if 'antimicrobial' in record.description.lower():
                seq_data['peptide_type'] = 'antimicrobial'
            elif 'antifungal' in record.description.lower():
                seq_data['peptide_type'] = 'antifungal'
            elif 'antiviral' in record.description.lower():
                seq_data['peptide_type'] = 'antiviral'
            
            sequences.append(seq_data)
    
    elif file_path.endswith('.csv'):
        # Load from CSV
        df = pd.read_csv(file_path)
        
        # Expected columns: sequence, peptide_type (optional)
        for _, row in df.iterrows():
            seq_data = {
                'id': row.get('id', f"seq_{_}"),
                'sequence': row['sequence']
            }
            
            if 'peptide_type' in row:
                seq_data['peptide_type'] = row['peptide_type']
            
            sequences.append(seq_data)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    logger.info(f"Loaded {len(sequences)} sequences")
    return sequences


def filter_sequences(
    sequences: List[Dict],
    min_length: int,
    max_length: int
) -> List[Dict]:
    """Filter sequences by length and validity"""
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    filtered = []
    
    for seq_data in sequences:
        seq = seq_data['sequence']
        
        # Check length
        if len(seq) < min_length or len(seq) > max_length:
            continue
        
        # Check valid amino acids
        if not all(aa in valid_aa for aa in seq):
            continue
        
        filtered.append(seq_data)
    
    logger.info(f"Filtered to {len(filtered)} valid sequences")
    return filtered


def load_structures(
    sequences: List[Dict],
    structure_dir: Optional[str]
) -> Dict[str, Dict]:
    """Load pre-computed structures"""
    structures = {}
    
    if structure_dir is None:
        return structures
    
    structure_dir = Path(structure_dir)
    
    for seq_data in tqdm(sequences, desc="Loading structures"):
        seq_id = seq_data['id']
        
        # Look for PDB file
        pdb_path = structure_dir / f"{seq_id}.pdb"
        if pdb_path.exists():
            try:
                features = extract_structure_features(
                    str(pdb_path),
                    len(seq_data['sequence'])
                )
                structures[seq_id] = features
            except Exception as e:
                logger.warning(f"Failed to load structure for {seq_id}: {e}")
    
    logger.info(f"Loaded {len(structures)} structures")
    return structures


def predict_missing_structures(
    sequences: List[Dict],
    structures: Dict[str, Dict],
    batch_size: int = 8
) -> Dict[str, Dict]:
    """Predict structures for sequences without PDB files"""
    # Import ESMFold
    try:
        import esm
        model = esm.pretrained.esmfold_v1().eval()
        if torch.cuda.is_available():
            model = model.cuda()
    except ImportError:
        logger.error("ESMFold not available. Install with: pip install fair-esm")
        return structures
    
    # Find sequences without structures
    missing = [s for s in sequences if s['id'] not in structures]
    
    if not missing:
        return structures
    
    logger.info(f"Predicting {len(missing)} missing structures...")
    
    # Predict in batches
    for i in tqdm(range(0, len(missing), batch_size)):
        batch = missing[i:i+batch_size]
        
        for seq_data in batch:
            try:
                features = predict_structure_with_esmfold(
                    seq_data['sequence'],
                    model
                )
                structures[seq_data['id']] = features
            except Exception as e:
                logger.warning(f"Failed to predict structure for {seq_data['id']}: {e}")
    
    return structures


def create_splits(
    sequences: List[Dict],
    split_ratios: List[float],
    seed: int
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create train/val/test splits"""
    # Set random seed
    np.random.seed(seed)
    
    # Get indices
    indices = np.arange(len(sequences))
    np.random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(split_ratios[0] * len(sequences))
    val_size = int(split_ratios[1] * len(sequences))
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create splits
    train_data = [sequences[i] for i in train_indices]
    val_data = [sequences[i] for i in val_indices]
    test_data = [sequences[i] for i in test_indices]
    
    logger.info(f"Split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def save_processed_data(
    data: List[Dict],
    structures: Dict[str, Dict],
    output_path: str
):
    """Save processed data to CSV"""
    # Create dataframe
    df_data = []
    
    for seq_data in data:
        row = {
            'id': seq_data['id'],
            'sequence': seq_data['sequence'],
            'length': len(seq_data['sequence'])
        }
        
        # Add peptide type if available
        if 'peptide_type' in seq_data:
            row['peptide_type'] = seq_data['peptide_type']
        
        # Add structure availability
        row['has_structure'] = seq_data['id'] in structures
        
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} sequences to {output_path}")
    
    # Save structures separately
    if structures:
        structure_path = Path(output_path).parent / "structures.pt"
        torch.save(structures, structure_path)
        logger.info(f"Saved structures to {structure_path}")


def compute_statistics(sequences: List[Dict]) -> Dict:
    """Compute dataset statistics"""
    lengths = [len(s['sequence']) for s in sequences]
    
    # Peptide type distribution
    type_counts = {}
    for s in sequences:
        ptype = s.get('peptide_type', 'unknown')
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
    
    # Amino acid frequencies
    aa_counts = {}
    total_aa = 0
    for s in sequences:
        for aa in s['sequence']:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
            total_aa += 1
    
    aa_freq = {aa: count/total_aa for aa, count in aa_counts.items()}
    
    stats = {
        'num_sequences': len(sequences),
        'length_stats': {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': min(lengths),
            'max': max(lengths),
            'percentiles': {
                '25': np.percentile(lengths, 25),
                '50': np.percentile(lengths, 50),
                '75': np.percentile(lengths, 75)
            }
        },
        'peptide_types': type_counts,
        'amino_acid_frequencies': aa_freq
    }
    
    return stats


def main():
    args = parse_args()
    
    # Setup logging
    setup_logger()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load sequences
    logger.info(f"Loading sequences from {args.input}")
    sequences = load_sequences(args.input)
    
    # Filter sequences
    sequences = filter_sequences(
        sequences,
        args.min_length,
        args.max_length
    )
    
    # Load or predict structures
    structures = {}
    if args.structure_dir:
        structures = load_structures(sequences, args.structure_dir)
    
    if args.predict_structures:
        structures = predict_missing_structures(
            sequences,
            structures
        )
    
    # Create splits
    train_data, val_data, test_data = create_splits(
        sequences,
        args.split_ratios,
        args.seed
    )
    
    # Save processed data
    save_processed_data(train_data, structures, output_dir / "train.csv")
    save_processed_data(val_data, structures, output_dir / "val.csv")
    save_processed_data(test_data, structures, output_dir / "test.csv")
    
    # Compute and save statistics
    stats = {
        'train': compute_statistics(train_data),
        'val': compute_statistics(val_data),
        'test': compute_statistics(test_data),
        'total': compute_statistics(sequences)
    }
    
    with open(output_dir / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Data preprocessing completed!")
    
    # Print summary
    logger.info("\nDataset Summary:")
    logger.info(f"Total sequences: {stats['total']['num_sequences']}")
    logger.info(f"Average length: {stats['total']['length_stats']['mean']:.1f} ± {stats['total']['length_stats']['std']:.1f}")
    logger.info(f"Peptide types: {stats['total']['peptide_types']}")
    logger.info(f"Structures available: {len(structures)}")


if __name__ == "__main__":
    main()
# Updated: 05/30/2025 22:59:09
