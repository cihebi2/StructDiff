#!/usr/bin/env python3
"""
Generation script for StructDiff
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json

import torch
import torch.nn.functional as F
from tqdm import tqdm
import yaml
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from structdiff.models.structdiff import StructDiff
from structdiff.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate peptides with StructDiff")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of peptides to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--peptide_type",
        type=str,
        choices=["antimicrobial", "antifungal", "antiviral", "random"],
        default="random",
        help="Type of peptide to generate"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="Length of peptides to generate (random if not specified)"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="Minimum peptide length"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=30,
        help="Maximum peptide length"
    )
    parser.add_argument(
        "--target_structure",
        type=str,
        default=None,
        help="Target secondary structure string (H: helix, E: sheet, C: coil)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_peptides.fasta",
        help="Output file path"
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["fasta", "csv", "json"],
        default="fasta",
        help="Output format"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> StructDiff:
    """Load model from checkpoint"""
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = StructDiff(config).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, config


def generate_peptides(
    model: StructDiff,
    config: Dict,
    args: argparse.Namespace
) -> List[Dict]:
    """Generate peptides using the model"""
    device = torch.device(args.device)
    
    # Prepare conditions
    conditions = None
    if args.peptide_type != "random":
        peptide_type_map = {
            "antimicrobial": 0,
            "antifungal": 1,
            "antiviral": 2
        }
        type_id = peptide_type_map[args.peptide_type]
        conditions = {
            'peptide_type': torch.tensor([type_id] * args.batch_size, device=device)
        }
    
    # Generate in batches
    all_peptides = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        # Determine batch size (handle last batch)
        current_batch_size = min(
            args.batch_size,
            args.num_samples - batch_idx * args.batch_size
        )
        
        # Sample lengths
        if args.length is not None:
            lengths = [args.length] * current_batch_size
        else:
            lengths = torch.randint(
                args.min_length,
                args.max_length + 1,
                (current_batch_size,)
            ).tolist()
        
        # Generate for each length
        batch_peptides = []
        for length in lengths:
            # Generate single peptide
            with torch.no_grad():
                output = model.sample(
                    batch_size=1,
                    seq_length=length,
                    conditions=conditions,
                    target_structure=args.target_structure,
                    guidance_scale=args.guidance_scale
                )
            
            # Decode sequence
            sequence = decode_sequence(
                output['embeddings'][0],
                output['attention_mask'][0],
                config
            )
            
            peptide_info = {
                'sequence': sequence,
                'length': length,
                'type': args.peptide_type,
                'guidance_scale': args.guidance_scale
            }
            
            # Add structure if available
            if args.target_structure:
                peptide_info['target_structure'] = args.target_structure[:length]
            
            batch_peptides.append(peptide_info)
        
        all_peptides.extend(batch_peptides)
    
    return all_peptides[:args.num_samples]


def decode_sequence(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    config: Dict
) -> str:
    """Decode embeddings to amino acid sequence"""
    # This is a placeholder - actual implementation would use
    # the ESM tokenizer and a learned decoder
    
    # For now, return a random sequence
    import random
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    length = int(attention_mask.sum().item()) - 2  # Remove CLS/SEP tokens
    
    return ''.join(random.choices(amino_acids, k=length))


def save_results(
    peptides: List[Dict],
    output_path: str,
    output_format: str
):
    """Save generated peptides to file"""
    logger.info(f"Saving {len(peptides)} peptides to {output_path}")
    
    if output_format == "fasta":
        with open(output_path, 'w') as f:
            for i, peptide in enumerate(peptides):
                header = f">Generated_{i}"
                if peptide['type'] != 'random':
                    header += f"|type:{peptide['type']}"
                if 'target_structure' in peptide:
                    header += f"|structure:{peptide['target_structure']}"
                
                f.write(f"{header}\n")
                f.write(f"{peptide['sequence']}\n")
    
    elif output_format == "csv":
        import pandas as pd
        df = pd.DataFrame(peptides)
        df.to_csv(output_path, index=False)
    
    elif output_format == "json":
        with open(output_path, 'w') as f:
            json.dump(peptides, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


def main():
    args = parse_args()
    
    # Setup logging
    setup_logger()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Generate peptides
    peptides = generate_peptides(model, config, args)
    
    # Save results
    save_results(peptides, args.output, args.output_format)
    
    # Print summary
    logger.info("Generation completed!")
    logger.info(f"Generated {len(peptides)} peptides")
    if args.output_format == "fasta":
        logger.info(f"Example: {peptides[0]['sequence']}")


if __name__ == "__main__":
    main()
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
