import numpy as np
from typing import List, Optional, Dict
from collections import Counter
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis


def compute_sequence_metrics(
    generated: List[str],
    reference: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute sequence-level metrics"""
    metrics = {}
    
    # Basic statistics
    lengths = [len(seq) for seq in generated]
    metrics['avg_length'] = np.mean(lengths)
    metrics['std_length'] = np.std(lengths)
    metrics['min_length'] = min(lengths)
    metrics['max_length'] = max(lengths)
    
    # Amino acid composition
    aa_counts = Counter()
    for seq in generated:
        aa_counts.update(seq)
    
    total_aa = sum(aa_counts.values())
    aa_freq = {aa: count/total_aa for aa, count in aa_counts.items()}
    
    # Natural amino acid distribution
    natural_aa_freq = {
        'A': 0.074, 'C': 0.025, 'D': 0.054, 'E': 0.054,
        'F': 0.047, 'G': 0.074, 'H': 0.026, 'I': 0.068,
        'K': 0.058, 'L': 0.099, 'M': 0.025, 'N': 0.045,
        'P': 0.039, 'Q': 0.034, 'R': 0.052, 'S': 0.057,
        'T': 0.051, 'V': 0.073, 'W': 0.013, 'Y': 0.032
    }
    
    # KL divergence from natural distribution
    kl_div = 0.0
    for aa in natural_aa_freq:
        p = natural_aa_freq[aa]
        q = aa_freq.get(aa, 1e-10)
        kl_div += p * np.log(p / q)
    
    metrics['aa_kl_divergence'] = kl_div
    
    # Physicochemical properties
    properties = {
        'molecular_weight': [],
        'aromaticity': [],
        'instability_index': [],
        'isoelectric_point': [],
        'gravy': []  # Grand average of hydropathy
    }
    
    for seq in generated[:100]:  # Analyze first 100 for efficiency
        try:
            analyzed = ProteinAnalysis(seq)
            properties['molecular_weight'].append(analyzed.molecular_weight())
            properties['aromaticity'].append(analyzed.aromaticity())
            properties['instability_index'].append(analyzed.instability_index())
            properties['isoelectric_point'].append(analyzed.isoelectric_point())
            properties['gravy'].append(analyzed.gravy())
        except Exception:
            continue
    
    for prop, values in properties.items():
        if values:
            metrics[f'avg_{prop}'] = np.mean(values)
            metrics[f'std_{prop}'] = np.std(values)
    
    # Compare with reference if provided
    if reference:
        # Sequence similarity
        from difflib import SequenceMatcher
        
        similarities = []
        for gen_seq in generated[:100]:
            max_sim = max(
                SequenceMatcher(None, gen_seq, ref_seq).ratio()
                for ref_seq in reference[:100]
            )
            similarities.append(max_sim)
        
        metrics['avg_similarity_to_reference'] = np.mean(similarities)
    
    return metrics
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04
