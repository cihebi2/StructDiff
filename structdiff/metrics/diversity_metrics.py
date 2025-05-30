import numpy as np
from typing import List, Optional, Dict
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import itertools


def compute_diversity_metrics(
    generated: List[str],
    reference: Optional[List[str]] = None
) -> Dict[str, float]:
    """Compute diversity metrics for generated sequences"""
    metrics = {}
    
    # Sequence diversity
    seq_diversity = compute_sequence_diversity(generated)
    metrics.update(seq_diversity)
    
    # Nearest neighbor diversity
    nn_diversity = compute_nearest_neighbor_diversity(generated)
    metrics.update(nn_diversity)
    
    # Motif diversity
    motif_diversity = compute_motif_diversity(generated)
    metrics.update(motif_diversity)
    
    # Compare with reference if provided
    if reference:
        novelty = compute_novelty(generated, reference)
        metrics.update(novelty)
    
    return metrics


def compute_sequence_diversity(sequences: List[str]) -> Dict[str, float]:
    """Basic sequence diversity metrics"""
    metrics = {}
    
    # Unique sequences
    unique_seqs = set(sequences)
    metrics['unique_ratio'] = len(unique_seqs) / len(sequences)
    
    # Edit distance diversity
    if len(sequences) <= 1000:  # Compute for smaller sets
        edit_distances = []
        for seq1, seq2 in itertools.combinations(sequences[:100], 2):
            dist = edit_distance(seq1, seq2)
            normalized_dist = dist / max(len(seq1), len(seq2))
            edit_distances.append(normalized_dist)
        
        if edit_distances:
            metrics['avg_edit_distance'] = np.mean(edit_distances)
            metrics['edit_distance_std'] = np.std(edit_distances)
    
    # Length diversity
    lengths = [len(seq) for seq in sequences]
    metrics['length_diversity'] = np.std(lengths) / np.mean(lengths)
    
    return metrics


def compute_nearest_neighbor_diversity(sequences: List[str]) -> Dict[str, float]:
    """Compute nearest neighbor diversity in embedding space"""
    metrics = {}
    
    # Convert to simple one-hot embeddings
    embeddings = sequences_to_embeddings(sequences[:500])  # Limit for efficiency
    
    # Compute pairwise similarities
    similarities = cosine_similarity(embeddings)
    
    # For each sequence, find its nearest neighbor
    np.fill_diagonal(similarities, -1)  # Exclude self
    nn_similarities = similarities.max(axis=1)
    
    metrics['avg_nn_similarity'] = np.mean(nn_similarities)
    metrics['nn_similarity_std'] = np.std(nn_similarities)
    metrics['diversity_score'] = 1 - np.mean(nn_similarities)
    
    return metrics


def compute_motif_diversity(sequences: List[str]) -> Dict[str, float]:
    """Compute diversity of sequence motifs"""
    metrics = {}
    
    # Extract k-mers
    kmer_counts = defaultdict(Counter)
    
    for k in [2, 3, 4]:
        for seq in sequences:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                kmer_counts[k][kmer] += 1
    
    # Compute entropy for each k
    for k, counts in kmer_counts.items():
        total = sum(counts.values())
        probs = np.array([count/total for count in counts.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(min(20**k, len(sequences) * (50-k+1)))
        metrics[f'{k}mer_entropy'] = entropy / max_entropy
    
    return metrics


def compute_novelty(generated: List[str], reference: List[str]) -> Dict[str, float]:
    """Compute novelty with respect to reference set"""
    metrics = {}
    
    reference_set = set(reference)
    
    # Exact novelty
    novel_sequences = [seq for seq in generated if seq not in reference_set]
    metrics['exact_novelty'] = len(novel_sequences) / len(generated)
    
    # Approximate novelty (no close matches)
    if len(generated) <= 100 and len(reference) <= 1000:
        approx_novel = 0
        for gen_seq in generated:
            similarities = [
                sequence_similarity(gen_seq, ref_seq)
                for ref_seq in reference[:100]
            ]
            if max(similarities) < 0.8:  # 80% similarity threshold
                approx_novel += 1
        
        metrics['approximate_novelty'] = approx_novel / len(generated)
    
    return metrics


def edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def sequence_similarity(s1: str, s2: str) -> float:
    """Compute normalized similarity between sequences"""
    dist = edit_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1 - (dist / max_len)


def sequences_to_embeddings(sequences: List[str]) -> np.ndarray:
    """Convert sequences to simple embeddings"""
    # Simple one-hot encoding
    aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    max_len = max(len(seq) for seq in sequences)
    
    embeddings = []
    for seq in sequences:
        # Create one-hot matrix
        matrix = np.zeros((max_len, 20))
        for i, aa in enumerate(seq):
            if aa in aa_to_idx:
                matrix[i, aa_to_idx[aa]] = 1
        
        # Flatten to vector
        embeddings.append(matrix.flatten())
    
    return np.array(embeddings)


from collections import defaultdict