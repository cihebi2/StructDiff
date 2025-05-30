#!/usr/bin/env python3
"""
Evaluation script for generated peptides
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

import torch
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from structdiff.metrics import (
    compute_sequence_metrics,
    compute_structure_metrics,
    compute_functional_metrics,
    compute_diversity_metrics
)
from structdiff.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated peptides")
    parser.add_argument(
        "--peptides",
        type=str,
        required=True,
        help="Path to generated peptides (FASTA/CSV/JSON)"
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Path to reference peptides for comparison"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        default=["all"],
        choices=["all", "sequence", "structure", "function", "diversity"],
        help="Metrics to compute"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--predict_structure",
        action="store_true",
        help="Predict structures for evaluation (slow)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for structure prediction"
    )
    return parser.parse_args()


def load_peptides(file_path: str) -> List[Dict]:
    """Load peptides from file"""
    peptides = []
    
    if file_path.endswith('.fasta'):
        for record in SeqIO.parse(file_path, "fasta"):
            peptide = {
                'id': record.id,
                'sequence': str(record.seq),
                'description': record.description
            }
            
            # Parse metadata from description
            if '|' in record.description:
                for field in record.description.split('|')[1:]:
                    if ':' in field:
                        key, value = field.split(':', 1)
                        peptide[key] = value
            
            peptides.append(peptide)
    
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        peptides = df.to_dict('records')
    
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            peptides = json.load(f)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    logger.info(f"Loaded {len(peptides)} peptides from {file_path}")
    return peptides


def evaluate_sequences(
    generated: List[Dict],
    reference: Optional[List[Dict]] = None
) -> Dict[str, float]:
    """Evaluate sequence-level metrics"""
    logger.info("Computing sequence metrics...")
    
    gen_sequences = [p['sequence'] for p in generated]
    ref_sequences = [p['sequence'] for p in reference] if reference else None
    
    metrics = compute_sequence_metrics(gen_sequences, ref_sequences)
    
    return metrics


def evaluate_structures(
    generated: List[Dict],
    reference: Optional[List[Dict]] = None,
    predict: bool = False,
    batch_size: int = 32
) -> Dict[str, float]:
    """Evaluate structure-level metrics"""
    logger.info("Computing structure metrics...")
    
    if predict:
        logger.info("Predicting structures with ESMFold...")
        # Import structure prediction
        from structdiff.data.structure_utils import predict_structure_with_esmfold
        
        # Initialize ESMFold
        import esm
        esmfold = esm.pretrained.esmfold_v1().eval()
        if torch.cuda.is_available():
            esmfold = esmfold.cuda()
        
        # Predict structures in batches
        gen_structures = []
        for i in tqdm(range(0, len(generated), batch_size)):
            batch = generated[i:i+batch_size]
            for peptide in batch:
                try:
                    structure = predict_structure_with_esmfold(
                        peptide['sequence'], esmfold
                    )
                    gen_structures.append(structure)
                except Exception as e:
                    logger.warning(f"Failed to predict structure: {e}")
                    gen_structures.append(None)
    else:
        # Use pre-computed structures if available
        gen_structures = [p.get('structure') for p in generated]
    
    # Reference structures
    ref_structures = None
    if reference and predict:
        ref_structures = []
        for peptide in reference[:100]:  # Limit reference predictions
            try:
                structure = predict_structure_with_esmfold(
                    peptide['sequence'], esmfold
                )
                ref_structures.append(structure)
            except:
                ref_structures.append(None)
    
    metrics = compute_structure_metrics(gen_structures, ref_structures)
    
    return metrics


def evaluate_function(
    generated: List[Dict],
    reference: Optional[List[Dict]] = None
) -> Dict[str, float]:
    """Evaluate functional metrics"""
    logger.info("Computing functional metrics...")
    
    gen_sequences = [p['sequence'] for p in generated]
    ref_sequences = [p['sequence'] for p in reference] if reference else None
    
    metrics = compute_functional_metrics(gen_sequences, ref_sequences)
    
    return metrics


def evaluate_diversity(
    generated: List[Dict],
    reference: Optional[List[Dict]] = None
) -> Dict[str, float]:
    """Evaluate diversity metrics"""
    logger.info("Computing diversity metrics...")
    
    gen_sequences = [p['sequence'] for p in generated]
    ref_sequences = [p['sequence'] for p in reference] if reference else None
    
    metrics = compute_diversity_metrics(gen_sequences, ref_sequences)
    
    return metrics


def main():
    args = parse_args()
    
    # Setup logging
    setup_logger()
    
    # Load peptides
    generated = load_peptides(args.peptides)
    reference = load_peptides(args.reference) if args.reference else None
    
    # Compute metrics
    all_metrics = {}
    
    if "all" in args.metrics or "sequence" in args.metrics:
        sequence_metrics = evaluate_sequences(generated, reference)
        all_metrics['sequence'] = sequence_metrics
    
    if "all" in args.metrics or "structure" in args.metrics:
        structure_metrics = evaluate_structures(
            generated, reference, 
            predict=args.predict_structure,
            batch_size=args.batch_size
        )
        all_metrics['structure'] = structure_metrics
    
    if "all" in args.metrics or "function" in args.metrics:
        function_metrics = evaluate_function(generated, reference)
        all_metrics['function'] = function_metrics
    
    if "all" in args.metrics or "diversity" in args.metrics:
        diversity_metrics = evaluate_diversity(generated, reference)
        all_metrics['diversity'] = diversity_metrics
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Evaluation Results Summary:")
    logger.info("="*50)
    
    for category, metrics in all_metrics.items():
        logger.info(f"\n{category.upper()} METRICS:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")
    
    logger.info(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
# Updated: 05/30/2025 22:59:09
