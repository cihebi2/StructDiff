# structdiff/metrics/advanced_metrics.py
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter, defaultdict
import re
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_comprehensive_metrics(
    generated: Union[List[str], torch.Tensor],
    reference: Optional[Union[List[str], torch.Tensor]] = None,
    structures: Optional[List[Dict[str, torch.Tensor]]] = None,
    peptide_type: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for generated peptides
    
    Args:
        generated: Generated sequences or embeddings
        reference: Reference sequences for comparison
        structures: Predicted structures for each sequence
        peptide_type: Type of peptides for functional evaluation
        
    Returns:
        Dictionary of metrics organized by category
    """
    metrics = {}
    
    # Convert embeddings to sequences if needed
    if isinstance(generated, torch.Tensor):
        # Placeholder - would need actual decoder
        generated = [f"SEQ{i}" for i in range(len(generated))]
    
    # Sequence metrics
    sequence_metrics = compute_sequence_quality_metrics(generated)
    metrics.update({f"sequence_{k}": v for k, v in sequence_metrics.items()})
    
    # Structure metrics
    if structures:
        structure_metrics = compute_structure_quality_metrics(structures)
        metrics.update({f"structure_{k}": v for k, v in structure_metrics.items()})
    
    # Functional metrics
    if peptide_type:
        functional_metrics = compute_functional_likelihood(generated, peptide_type)
        metrics.update({f"function_{k}": v for k, v in functional_metrics.items()})
    
    # Diversity metrics
    diversity_metrics = compute_advanced_diversity_metrics(generated)
    metrics.update({f"diversity_{k}": v for k, v in diversity_metrics.items()})
    
    # Comparison with reference
    if reference:
        comparison_metrics = compute_reference_comparison_metrics(generated, reference)
        metrics.update({f"comparison_{k}": v for k, v in comparison_metrics.items()})
    
    return metrics


def compute_sequence_quality_metrics(sequences: List[str]) -> Dict[str, float]:
    """Compute sequence quality metrics"""
    metrics = {}
    
    if not sequences:
        return metrics
    
    # Basic statistics
    lengths = [len(seq) for seq in sequences]
    metrics['length_mean'] = np.mean(lengths)
    metrics['length_std'] = np.std(lengths)
    
    # Amino acid composition entropy
    aa_entropies = []
    for seq in sequences:
        if len(seq) > 0:
            aa_counts = Counter(seq)
            total = sum(aa_counts.values())
            probs = [count/total for count in aa_counts.values()]
            aa_entropies.append(entropy(probs))
    
    metrics['aa_entropy_mean'] = np.mean(aa_entropies) if aa_entropies else 0
    
    # Complexity score (sliding window entropy)
    complexities = []
    window_size = 3
    for seq in sequences:
        if len(seq) >= window_size:
            windows = [seq[i:i+window_size] for i in range(len(seq)-window_size+1)]
            unique_windows = len(set(windows))
            max_windows = min(len(windows), 20**window_size)
            complexity = unique_windows / max_windows
            complexities.append(complexity)
    
    metrics['complexity_mean'] = np.mean(complexities) if complexities else 0
    
    # Repetitiveness score
    repetitiveness_scores = []
    for seq in sequences:
        if len(seq) > 1:
            # Check for repeated patterns
            repeated = 0
            for length in range(2, min(len(seq)//2 + 1, 6)):
                for i in range(len(seq) - length):
                    pattern = seq[i:i+length]
                    if seq.count(pattern) > 1:
                        repeated += 1
            repetitiveness_scores.append(repeated / len(seq))
    
    metrics['repetitiveness_mean'] = np.mean(repetitiveness_scores) if repetitiveness_scores else 0
    
    return metrics


def compute_structure_quality_metrics(structures: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """Compute structure quality metrics"""
    metrics = {}
    
    if not structures:
        return metrics
    
    # pLDDT statistics
    plddt_scores = []
    for struct in structures:
        if 'plddt' in struct and struct['plddt'] is not None:
            plddt_mean = struct['plddt'].mean().item()
            plddt_scores.append(plddt_mean)
    
    if plddt_scores:
        metrics['plddt_mean'] = np.mean(plddt_scores)
        metrics['plddt_std'] = np.std(plddt_scores)
        metrics['high_confidence_ratio'] = np.mean([s > 70 for s in plddt_scores])
        metrics['very_high_confidence_ratio'] = np.mean([s > 90 for s in plddt_scores])
    
    # Secondary structure diversity
    ss_entropies = []
    for struct in structures:
        if 'secondary_structure' in struct and struct['secondary_structure'] is not None:
            ss = struct['secondary_structure']
            if isinstance(ss, torch.Tensor):
                ss = ss.cpu().numpy()
            
            # Calculate secondary structure content
            unique, counts = np.unique(ss, return_counts=True)
            if len(counts) > 0:
                probs = counts / counts.sum()
                ss_entropies.append(entropy(probs))
    
    if ss_entropies:
        metrics['ss_entropy_mean'] = np.mean(ss_entropies)
    
    # Contact density
    contact_densities = []
    for struct in structures:
        if 'contact_map' in struct and struct['contact_map'] is not None:
            contacts = struct['contact_map']
            if isinstance(contacts, torch.Tensor):
                contacts = contacts.cpu().numpy()
            
            # Calculate long-range contacts (|i-j| > 5)
            n = contacts.shape[0]
            long_range_contacts = 0
            total_possible = 0
            
            for i in range(n):
                for j in range(i+6, n):  # Long-range
                    if contacts[i, j] > 0.5:
                        long_range_contacts += 1
                    total_possible += 1
            
            if total_possible > 0:
                density = long_range_contacts / total_possible
                contact_densities.append(density)
    
    if contact_densities:
        metrics['long_range_contact_density'] = np.mean(contact_densities)
    
    return metrics


def compute_functional_likelihood(sequences: List[str], peptide_type: str) -> Dict[str, float]:
    """Compute functional likelihood scores"""
    metrics = {}
    
    # Define characteristic features for each peptide type
    features = {
        'antimicrobial': {
            'positive_residues': set('RKH'),
            'hydrophobic_residues': set('AILMFVWY'),
            'preferred_length': (12, 50),
            'min_positive_charge': 2,
            'preferred_hydrophobicity': (0.3, 0.6)
        },
        'antifungal': {
            'positive_residues': set('RK'),
            'hydrophobic_residues': set('AILMFVW'),
            'preferred_length': (10, 40),
            'min_positive_charge': 2,
            'preferred_hydrophobicity': (0.35, 0.65)
        },
        'antiviral': {
            'positive_residues': set('RKH'),
            'hydrophobic_residues': set('AILMFV'),
            'preferred_length': (8, 30),
            'min_positive_charge': 1,
            'preferred_hydrophobicity': (0.25, 0.55)
        }
    }
    
    if peptide_type not in features:
        return metrics
    
    feat = features[peptide_type]
    likelihood_scores = []
    
    for seq in sequences:
        score = 0.0
        factors = 0
        
        # Length score
        length = len(seq)
        if feat['preferred_length'][0] <= length <= feat['preferred_length'][1]:
            score += 1.0
        factors += 1
        
        # Charge score
        positive_charge = sum(1 for aa in seq if aa in feat['positive_residues'])
        if positive_charge >= feat['min_positive_charge']:
            score += 1.0
        factors += 1
        
        # Hydrophobicity score
        hydrophobic_ratio = sum(1 for aa in seq if aa in feat['hydrophobic_residues']) / len(seq)
        if feat['preferred_hydrophobicity'][0] <= hydrophobic_ratio <= feat['preferred_hydrophobicity'][1]:
            score += 1.0
        factors += 1
        
        # Amphipathicity score (simple approximation)
        if length >= 7:
            # Check for alternating hydrophobic/hydrophilic pattern
            pattern_score = 0
            for i in range(len(seq) - 3):
                window = seq[i:i+4]
                hydrophobic_positions = [j for j, aa in enumerate(window) if aa in feat['hydrophobic_residues']]
                if len(hydrophobic_positions) == 2 and hydrophobic_positions[1] - hydrophobic_positions[0] == 2:
                    pattern_score += 1
            
            if pattern_score > length / 8:  # Threshold for amphipathic pattern
                score += 0.5
            factors += 0.5
        
        likelihood_scores.append(score / factors)
    
    metrics[f'{peptide_type}_likelihood_mean'] = np.mean(likelihood_scores)
    metrics[f'{peptide_type}_likelihood_std'] = np.std(likelihood_scores)
    metrics[f'{peptide_type}_high_likelihood_ratio'] = np.mean([s > 0.7 for s in likelihood_scores])
    
    return metrics


def compute_advanced_diversity_metrics(sequences: List[str]) -> Dict[str, float]:
    """Compute advanced diversity metrics"""
    metrics = {}
    
    if len(sequences) < 2:
        return metrics
    
    # Sequence uniqueness
    unique_sequences = set(sequences)
    metrics['uniqueness_ratio'] = len(unique_sequences) / len(sequences)
    
    # Pairwise sequence similarity distribution
    similarities = []
    for i in range(min(len(sequences), 100)):  # Limit for efficiency
        for j in range(i + 1, min(len(sequences), 100)):
            sim = sequence_similarity(sequences[i], sequences[j])
            similarities.append(sim)
    
    if similarities:
        metrics['pairwise_similarity_mean'] = np.mean(similarities)
        metrics['pairwise_similarity_std'] = np.std(similarities)
        metrics['high_similarity_pairs_ratio'] = np.mean([s > 0.8 for s in similarities])
    
    # Motif diversity
    motif_sets = []
    for seq in sequences:
        motifs = set()
        for k in [2, 3, 4]:
            for i in range(len(seq) - k + 1):
                motifs.add(seq[i:i+k])
        motif_sets.append(motifs)
    
    # Jaccard diversity between motif sets
    if len(motif_sets) > 1:
        jaccard_distances = []
        for i in range(min(len(motif_sets), 50)):
            for j in range(i + 1, min(len(motif_sets), 50)):
                intersection = len(motif_sets[i] & motif_sets[j])
                union = len(motif_sets[i] | motif_sets[j])
                if union > 0:
                    jaccard_distances.append(1 - intersection / union)
        
        if jaccard_distances:
            metrics['motif_diversity'] = np.mean(jaccard_distances)
    
    # Compositional diversity (amino acid usage)
    aa_distributions = []
    for seq in sequences:
        if len(seq) > 0:
            aa_count = Counter(seq)
            aa_freq = np.zeros(20)
            aa_map = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
            for aa, count in aa_count.items():
                if aa in aa_map:
                    aa_freq[aa_map[aa]] = count / len(seq)
            aa_distributions.append(aa_freq)
    
    if len(aa_distributions) > 1:
        aa_distributions = np.array(aa_distributions)
        # Average pairwise cosine distance
        cos_distances = []
        for i in range(min(len(aa_distributions), 50)):
            for j in range(i + 1, min(len(aa_distributions), 50)):
                sim = cosine_similarity([aa_distributions[i]], [aa_distributions[j]])[0, 0]
                cos_distances.append(1 - sim)
        
        if cos_distances:
            metrics['compositional_diversity'] = np.mean(cos_distances)
    
    return metrics


def compute_reference_comparison_metrics(
    generated: List[str],
    reference: List[str]
) -> Dict[str, float]:
    """Compare generated sequences with reference set"""
    metrics = {}
    
    # Novelty (sequences not in reference)
    reference_set = set(reference)
    novel_sequences = [seq for seq in generated if seq not in reference_set]
    metrics['exact_novelty_ratio'] = len(novel_sequences) / len(generated)
    
    # Nearest neighbor analysis
    nn_similarities = []
    for gen_seq in generated[:100]:  # Limit for efficiency
        max_sim = 0
        for ref_seq in reference[:100]:
            sim = sequence_similarity(gen_seq, ref_seq)
            max_sim = max(max_sim, sim)
        nn_similarities.append(max_sim)
    
    if nn_similarities:
        metrics['nn_similarity_mean'] = np.mean(nn_similarities)
        metrics['nn_similarity_std'] = np.std(nn_similarities)
    
    # Distribution comparison (JS divergence of amino acid frequencies)
    def get_aa_distribution(sequences):
        aa_counts = Counter()
        for seq in sequences:
            aa_counts.update(seq)
        total = sum(aa_counts.values())
        
        # Create probability vector
        probs = np.zeros(20)
        aa_map = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
        for aa, count in aa_counts.items():
            if aa in aa_map:
                probs[aa_map[aa]] = count / total
        return probs
    
    gen_dist = get_aa_distribution(generated)
    ref_dist = get_aa_distribution(reference)
    
    # JS divergence
    m = (gen_dist + ref_dist) / 2
    js_div = (entropy(gen_dist, m) + entropy(ref_dist, m)) / 2
    metrics['aa_distribution_js_divergence'] = js_div
    
    return metrics


def sequence_similarity(seq1: str, seq2: str) -> float:
    """Compute normalized sequence similarity"""
    if len(seq1) == 0 or len(seq2) == 0:
        return 0.0
    
    # Use Levenshtein distance
    from difflib import SequenceMatcher
    return SequenceMatcher(None, seq1, seq2).ratio()


class AdvancedMetrics:
    """Advanced metrics calculator with caching and batch processing"""
    
    def __init__(self):
        self.cache = {}
    
    def compute_motif_enrichment(self, sequences: List[str]) -> Dict[str, float]:
        """Compute enrichment of functional motifs"""
        motif_patterns = {
            'cationic_cluster': r'[RKH]{2,}',
            'hydrophobic_cluster': r'[AILMFVWY]{3,}',
            'proline_kink': r'P{2,}',
            'cysteine_pair': r'C.{2,4}C',
            'glycine_flexibility': r'G{2,}',
            'aromatic_cluster': r'[FWY]{2,}'
        }
        
        results = {}
        for motif_name, pattern in motif_patterns.items():
            count = sum(1 for seq in sequences if re.search(pattern, seq))
            results[f'{motif_name}_frequency'] = count / len(sequences)
        
        return results
    
    def compute_biophysical_properties(self, sequences: List[str]) -> Dict[str, float]:
        """Compute aggregated biophysical properties"""
        properties = defaultdict(list)
        
        for seq in sequences:
            try:
                analyzed = ProteinAnalysis(seq)
                properties['hydrophobicity'].append(analyzed.gravy())
                properties['charge'].append(analyzed.charge_at_pH(7.0))
                properties['instability'].append(analyzed.instability_index())
                properties['aromaticity'].append(analyzed.aromaticity())
                properties['molecular_weight'].append(analyzed.molecular_weight())
                
                # Secondary structure propensity
                ss = analyzed.secondary_structure_fraction()
                properties['helix_propensity'].append(ss[0])
                properties['turn_propensity'].append(ss[1])
                properties['sheet_propensity'].append(ss[2])
            except:
                continue
        
        results = {}
        for prop, values in properties.items():
            if values:
                results[f'{prop}_mean'] = np.mean(values)
                results[f'{prop}_std'] = np.std(values)
        
        return results
# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
