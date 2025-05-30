import numpy as np
from typing import List, Optional, Dict
from collections import defaultdict
import re


def compute_functional_metrics(
    generated: List[str],
    reference: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute functional property metrics"""
    metrics = {}
    
    # Antimicrobial peptide features
    amp_features = compute_amp_features(generated)
    metrics.update(amp_features)
    
    # Charge and hydrophobicity
    charge_features = compute_charge_features(generated)
    metrics.update(charge_features)
    
    # Motif analysis
    motif_features = compute_motif_features(generated)
    metrics.update(motif_features)
    
    return metrics


def compute_amp_features(sequences: List[str]) -> Dict[str, float]:
    """Compute features relevant to antimicrobial peptides"""
    features = {}
    
    # Typical AMP characteristics
    positive_charges = []
    hydrophobic_ratios = []
    amphipathic_scores = []
    
    positive_aa = set('RKH')
    hydrophobic_aa = set('AILMFVWY')
    
    for seq in sequences:
        # Net positive charge
        pos_charge = sum(1 for aa in seq if aa in positive_aa)
        positive_charges.append(pos_charge)
        
        # Hydrophobic ratio
        hydro_ratio = sum(1 for aa in seq if aa in hydrophobic_aa) / len(seq)
        hydrophobic_ratios.append(hydro_ratio)
        
        # Simple amphipathicity (alternating pattern)
        amphipathic_score = compute_amphipathicity(seq)
        amphipathic_scores.append(amphipathic_score)
    
    features['avg_positive_charge'] = np.mean(positive_charges)
    features['avg_hydrophobic_ratio'] = np.mean(hydrophobic_ratios)
    features['avg_amphipathicity'] = np.mean(amphipathic_scores)
    
    # Percentage likely to be AMPs (simple heuristic)
    amp_like = [
        (pc >= 2 and 0.3 <= hr <= 0.7)
        for pc, hr in zip(positive_charges, hydrophobic_ratios)
    ]
    features['amp_likelihood'] = np.mean(amp_like)
    
    return features


def compute_charge_features(sequences: List[str]) -> Dict[str, float]:
    """Compute charge-related features"""
    features = {}
    
    charge_map = {
        'R': 1, 'K': 1, 'H': 0.5,  # Positive
        'D': -1, 'E': -1,           # Negative
    }
    
    net_charges = []
    charge_densities = []
    
    for seq in sequences:
        charge = sum(charge_map.get(aa, 0) for aa in seq)
        net_charges.append(charge)
        charge_densities.append(abs(charge) / len(seq))
    
    features['avg_net_charge'] = np.mean(net_charges)
    features['avg_charge_density'] = np.mean(charge_densities)
    features['fraction_cationic'] = np.mean([c > 0 for c in net_charges])
    
    return features


def compute_motif_features(sequences: List[str]) -> Dict[str, float]:
    """Analyze sequence motifs"""
    features = {}
    
    # Common AMP motifs
    motifs = {
        'KK_motif': r'KK',
        'RR_motif': r'RR',
        'KXXK_motif': r'K..K',
        'proline_hinge': r'P{2,}',
        'glycine_bracket': r'G.{1,10}G'
    }
    
    motif_counts = defaultdict(int)
    
    for seq in sequences:
        for motif_name, pattern in motifs.items():
            if re.search(pattern, seq):
                motif_counts[motif_name] += 1
    
    for motif_name in motifs:
        features[f'{motif_name}_frequency'] = motif_counts[motif_name] / len(sequences)
    
    return features


def compute_amphipathicity(sequence: str) -> float:
    """Simple amphipathicity score based on hydrophobic moment"""
    hydrophobic_aa = set('AILMFVWY')
    
    # Simplified: check alternating pattern
    hydro_positions = [i for i, aa in enumerate(sequence) if aa in hydrophobic_aa]
    
    if len(hydro_positions) < 2:
        return 0.0
    
    # Check for regular spacing (indicative of amphipathic helix)
    spacings = np.diff(hydro_positions)
    
    # Ideal spacing for amphipathic helix is ~3-4 residues
    ideal_spacings = np.sum((spacings >= 2) & (spacings <= 5))
    
    return ideal_spacings / max(len(spacings), 1)
# Updated: 05/30/2025 22:59:09
