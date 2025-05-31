import pytest
import numpy as np
import torch

from structdiff.metrics import (
    compute_sequence_metrics,
    compute_structure_metrics,
    compute_diversity_metrics
)


class TestSequenceMetrics:
    """Test sequence metrics"""
    
    def test_basic_metrics(self):
        """Test basic sequence metrics"""
        sequences = [
            'ACDEFGHIKLM',
            'NPRSTVWY',
            'AAAAGGGG'
        ]
        
        metrics = compute_sequence_metrics(sequences)
        
        assert 'avg_length' in metrics
        assert metrics['avg_length'] == pytest.approx(9.0, 0.1)
        assert 'aa_kl_divergence' in metrics
    
    def test_with_reference(self):
        """Test metrics with reference sequences"""
        generated = ['ACDEF', 'GHIKL']
        reference = ['ACDEF', 'MNPQR']
        
        metrics = compute_sequence_metrics(generated, reference)
        
        assert 'avg_similarity_to_reference' in metrics


class TestStructureMetrics:
    """Test structure metrics"""
    
    def test_structure_metrics(self):
        """Test structure metric computation"""
        # Mock structure data
        structures = [
            {
                'plddt': torch.tensor([80.0, 75.0, 70.0]),
                'secondary_structure': torch.tensor([0, 1, 2]),
                'contact_map': torch.ones(3, 3)
            }
        ]
        
        metrics = compute_structure_metrics(structures)
        
        assert 'avg_plddt' in metrics
        assert metrics['avg_plddt'] == pytest.approx(75.0, 0.1)
        assert 'high_confidence_ratio' in metrics


class TestDiversityMetrics:
    """Test diversity metrics"""
    
    def test_diversity_metrics(self):
        """Test diversity computation"""
        sequences = [
            'ACDEFGHIKL',
            'ACDEFGHIKL',  # Duplicate
            'MNPQRSTVWY',
            'AAAABBBBCC'
        ]
        
        metrics = compute_diversity_metrics(sequences)
        
        assert 'unique_ratio' in metrics
        assert metrics['unique_ratio'] == 0.75
        assert 'diversity_score' in metrics
        
    def test_empty_sequences(self):
        """Test with empty sequence list"""
        sequences = []
        
        metrics = compute_diversity_metrics(sequences)
        
        # Should handle empty input gracefully
        assert isinstance(metrics, dict)
# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
