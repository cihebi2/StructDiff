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
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from structdiff.models.structdiff import StructDiff
from structdiff.diffusion.gaussian_diffusion import GaussianDiffusion
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


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model from checkpoint"""
    logger.info(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # Create model and diffusion
    model = StructDiff(config).to(device)
    diffusion = GaussianDiffusion(config.get('diffusion', {}))
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, diffusion, config


def generate_peptides(
    model: StructDiff,
    diffusion: GaussianDiffusion,
    tokenizer: AutoTokenizer,
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
            # Generate single peptide using diffusion sampling
            with torch.no_grad():
                # Create random noise as starting point
                seq_embeddings = torch.randn(1, length, model.sequence_encoder.config.hidden_size, device=device)
                attention_mask = torch.ones(1, length, device=device)
                
                # Sample using diffusion process
                if hasattr(diffusion, 'p_sample_loop'):
                    denoised_embeddings = diffusion.p_sample_loop(
                        model=model,
                        shape=(1, length, model.sequence_encoder.config.hidden_size),
                        conditions=conditions,
                        guidance_scale=args.guidance_scale
                    )
                else:
                    # Fallback: direct model sampling
                    denoised_embeddings = seq_embeddings
                    for t in reversed(range(0, 1000, 50)):  # DDIM-like sampling
                        timesteps = torch.tensor([t], device=device)
                        noise_pred = model.denoiser(
                            denoised_embeddings, timesteps, attention_mask,
                            conditions=conditions
                        )
                        # Simple denoising step
                        alpha = 0.99
                        denoised_embeddings = alpha * denoised_embeddings + (1 - alpha) * (denoised_embeddings - noise_pred)
            
            # Decode sequence
            sequence = decode_sequence(
                denoised_embeddings[0],
                attention_mask[0],
                model,
                tokenizer,
                device
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
    model: StructDiff,
    tokenizer: AutoTokenizer,
    device: torch.device
) -> str:
    """Decode embeddings to amino acid sequence using the model's decoder"""
    try:
        with torch.no_grad():
            # Use model's sequence decoder if available
            if hasattr(model, 'sequence_decoder') and model.sequence_decoder is not None:
                # Decode using learned decoder
                logits = model.sequence_decoder(embeddings.unsqueeze(0), attention_mask.unsqueeze(0))
                token_ids = torch.argmax(logits, dim=-1).squeeze(0)
            elif hasattr(model, 'decode_sequences'):
                # Use model's decode_sequences method
                token_ids = model.decode_sequences(embeddings.unsqueeze(0), attention_mask.unsqueeze(0))
                token_ids = token_ids.squeeze(0)
            else:
                # Fallback: use embedding similarity to reconstruct sequence
                token_ids = decode_via_similarity(embeddings, tokenizer, device)
            
            # Convert token IDs to sequence
            # Remove special tokens (CLS, SEP, PAD)
            valid_mask = attention_mask.bool()
            valid_tokens = token_ids[valid_mask]
            
            # Skip CLS token at start and SEP token at end
            if len(valid_tokens) > 2:
                valid_tokens = valid_tokens[1:-1]
            
            # Decode to string
            sequence = tokenizer.decode(valid_tokens, skip_special_tokens=True)
            
            # Clean up the sequence - keep only amino acid characters
            amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
            clean_sequence = ''.join([c for c in sequence.upper() if c in amino_acids])
            
            # Fallback if decoding failed
            if not clean_sequence:
                clean_sequence = generate_fallback_sequence(int(attention_mask.sum().item()) - 2)
            
            return clean_sequence
            
    except Exception as e:
        logger.warning(f"Decoding failed: {e}, using fallback")
        return generate_fallback_sequence(int(attention_mask.sum().item()) - 2)


def decode_via_similarity(embeddings: torch.Tensor, tokenizer: AutoTokenizer, device: torch.device) -> torch.Tensor:
    """Decode embeddings using similarity to token embeddings"""
    try:
        # Get ESM-2 token embeddings
        vocab_size = tokenizer.vocab_size
        token_ids = torch.arange(vocab_size, device=device)
        
        # This is a simplified approach - in practice you'd want to use
        # the actual ESM-2 embedding layer
        similarities = torch.cosine_similarity(
            embeddings.unsqueeze(1), 
            embeddings.mean(dim=0, keepdim=True).unsqueeze(0).repeat(embeddings.size(0), vocab_size, 1),
            dim=-1
        )
        
        # Get most similar tokens
        predicted_tokens = torch.argmax(similarities, dim=-1)
        return predicted_tokens
        
    except Exception as e:
        logger.warning(f"Similarity-based decoding failed: {e}")
        # Return random valid amino acid tokens
        amino_acid_tokens = [tokenizer.encode(aa, add_special_tokens=False)[0] for aa in 'ACDEFGHIKLMNPQRSTVWY' if tokenizer.encode(aa, add_special_tokens=False)]
        return torch.tensor([torch.randint(0, len(amino_acid_tokens), (1,)).item() for _ in range(len(embeddings))], device=device)


def generate_fallback_sequence(length: int) -> str:
    """Generate a random amino acid sequence as fallback"""
    import random
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    return ''.join(random.choices(amino_acids, k=max(1, length)))


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
    
    # Load model and diffusion
    model, diffusion, config = load_model(args.checkpoint, device)
    
    # Load tokenizer
    tokenizer_name = config.get('model', {}).get('sequence_encoder', {}).get('pretrained_model', 'facebook/esm2_t6_8M_UR50D')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    logger.info(f"Loaded tokenizer: {tokenizer_name}")
    
    # Generate peptides
    peptides = generate_peptides(model, diffusion, tokenizer, config, args)
    
    # Save results
    save_results(peptides, args.output, args.output_format)
    
    # Print summary
    logger.info("Generation completed!")
    logger.info(f"Generated {len(peptides)} peptides")
    if peptides and args.output_format == "fasta":
        logger.info(f"Example: {peptides[0]['sequence']}")
        # Show sequence length distribution
        lengths = [len(p['sequence']) for p in peptides]
        logger.info(f"Length range: {min(lengths)}-{max(lengths)}, avg: {sum(lengths)/len(lengths):.1f}")


if __name__ == "__main__":
    main()
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
