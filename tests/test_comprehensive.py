# tests/test_comprehensive.py
import pytest
import torch
import numpy as np
from typing import Dict, List, Optional
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from structdiff.models import StructDiff
from structdiff.data import PeptideStructureDataset, PeptideStructureCollator
from structdiff.diffusion import GaussianDiffusion
from structdiff.metrics import compute_sequence_metrics
from structdiff.utils import load_config, set_seed


# Fixtures
@pytest.fixture(scope="session")
def device():
    """Get test device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def test_config():
    """Create test configuration"""
    return {
        'model': {
            'sequence_encoder': {
                'pretrained_model': 'facebook/esm2_t6_8M_UR50D',
                'freeze_encoder': True,
                'use_lora': False
            },
            'structure_encoder': {
                'type': 'multi_scale',
                'hidden_dim': 128,
                'use_esmfold': False
            },
            'denoiser': {
                'hidden_dim': 128,
                'num_layers': 2,
                'num_heads': 4,
                'dropout': 0.1,
                'use_cross_attention': True
            }
        },
        'diffusion': {
            'num_timesteps': 100,
            'noise_schedule': 'linear',
            'beta_start': 0.0001,
            'beta_end': 0.02
        },
        'training_config': {
            'loss_weights': {
                'diffusion_loss': 1.0,
                'structure_consistency_loss': 0.1,
                'auxiliary_loss': 0.01
            }
        }
    }


@pytest.fixture
def sample_batch(device):
    """Create sample batch for testing"""
    batch_size = 4
    seq_len = 20
    
    return {
        'sequences': torch.randint(0, 20, (batch_size, seq_len), device=device),
        'attention_mask': torch.ones(batch_size, seq_len, device=device),
        'structures': {
            'angles': torch.randn(batch_size, seq_len, 3, device=device),
            'secondary_structure': torch.randint(0, 3, (batch_size, seq_len), device=device),
            'distance_matrix': torch.rand(batch_size, seq_len, seq_len, device=device) * 20,
            'contact_map': torch.rand(batch_size, seq_len, seq_len, device=device) > 0.5
        },
        'labels': torch.randint(0, 3, (batch_size,), device=device)
    }


# Model Tests
class TestStructDiff:
    """Comprehensive tests for StructDiff model"""
    
    def test_model_initialization(self, test_config, device):
        """Test model initialization with various configurations"""
        model = StructDiff(test_config).to(device)
        
        # Check components exist
        assert hasattr(model, 'sequence_encoder')
        assert hasattr(model, 'structure_encoder')
        assert hasattr(model, 'denoiser')
        assert hasattr(model, 'diffusion')
        
        # Check parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        
        # With frozen encoder
        assert trainable_params < total_params  # Some params should be frozen
    
    def test_forward_pass(self, test_config, sample_batch, device):
        """Test forward pass with different configurations"""
        model = StructDiff(test_config).to(device)
        model.eval()
        
        # Test without structures
        timesteps = torch.randint(0, 100, (sample_batch['sequences'].shape[0],), device=device)
        
        with torch.no_grad():
            outputs = model(
                sequences=sample_batch['sequences'],
                attention_mask=sample_batch['attention_mask'],
                timesteps=timesteps,
                return_loss=False
            )
        
        assert 'denoised_embeddings' in outputs
        assert outputs['denoised_embeddings'].shape[0] == sample_batch['sequences'].shape[0]
        
        # Test with structures
        with torch.no_grad():
            outputs = model(
                sequences=sample_batch['sequences'],
                attention_mask=sample_batch['attention_mask'],
                timesteps=timesteps,
                structures=sample_batch['structures'],
                return_loss=False
            )
        
        assert 'cross_attention_weights' in outputs
    
    def test_loss_computation(self, test_config, sample_batch, device):
        """Test loss computation"""
        model = StructDiff(test_config).to(device)
        model.train()
        
        timesteps = torch.randint(0, 100, (sample_batch['sequences'].shape[0],), device=device)
        
        outputs = model(
            sequences=sample_batch['sequences'],
            attention_mask=sample_batch['attention_mask'],
            timesteps=timesteps,
            structures=sample_batch['structures'],
            return_loss=True
        )
        
        # Check all loss components
        assert 'total_loss' in outputs
        assert 'diffusion_loss' in outputs
        assert outputs['total_loss'].requires_grad
        
        # Test backward pass
        outputs['total_loss'].backward()
        
        # Check gradients exist
        has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_gradients
    
    def test_sampling(self, test_config, device):
        """Test sequence generation"""
        model = StructDiff(test_config).to(device)
        model.eval()
        
        # Basic sampling
        with torch.no_grad():
            samples = model.sample(
                batch_size=2,
                seq_length=15,
                guidance_scale=1.0
            )
        
        assert 'sequences' in samples
        assert len(samples['sequences']) == 2
        assert all(isinstance(seq, str) for seq in samples['sequences'])
        
        # Conditional sampling
        conditions = {'peptide_type': torch.tensor([0, 1], device=device)}
        
        with torch.no_grad():
            samples = model.sample(
                batch_size=2,
                seq_length=15,
                conditions=conditions,
                guidance_scale=2.0
            )
        
        assert len(samples['sequences']) == 2
    
    @pytest.mark.parametrize("sampling_method", ["ddpm", "ddim", "pndm"])
    def test_sampling_methods(self, test_config, device, sampling_method):
        """Test different sampling methods"""
        # Mock the sampling to be faster for tests
        test_config['diffusion']['num_timesteps'] = 10
        
        model = StructDiff(test_config).to(device)
        model.eval()
        
        # Test sampling doesn't crash
        with torch.no_grad():
            samples = model.sample(
                batch_size=2,
                seq_length=10,
                sampling_method=sampling_method,
                num_inference_steps=5
            )
        
        assert 'sequences' in samples
        assert len(samples['sequences']) == 2


# Data Pipeline Tests
class TestDataPipeline:
    """Test data loading and processing"""
    
    def test_dataset_creation(self, tmp_path):
        """Test dataset creation and loading"""
        # Create dummy data
        data = {
            'id': ['pep1', 'pep2', 'pep3'],
            'sequence': ['ACDEFG', 'KLMNPQ', 'RSTVWY'],
            'label': [0, 1, 2],
            'length': [6, 6, 6]
        }
        
        data_path = tmp_path / 'test_peptides.csv'
        pd.DataFrame(data).to_csv(data_path, index=False)
        
        # Create dataset
        config = {'data': {'max_length': 50, 'min_length': 3}}
        dataset = PeptideStructureDataset(str(data_path), config, is_training=True)
        
        assert len(dataset) == 3
        
        # Test __getitem__
        item = dataset[0]
        assert 'sequences' in item
        assert 'attention_mask' in item
        assert 'label' in item
    
    def test_collator(self, test_config):
        """Test batch collation"""
        collator = PeptideStructureCollator(test_config)
        
        # Create sample batch
        batch = [
            {
                'sequences': torch.tensor([1, 2, 3, 4]),
                'attention_mask': torch.ones(4),
                'label': torch.tensor(0)
            },
            {
                'sequences': torch.tensor([5, 6, 7, 8, 9, 10]),
                'attention_mask': torch.ones(6),
                'label': torch.tensor(1)
            }
        ]
        
        collated = collator(batch)
        
        # Check padding
        assert collated['sequences'].shape == (2, 6)  # Padded to max length
        assert collated['attention_mask'].shape == (2, 6)
        assert collated['labels'].shape == (2,)
        
        # Check padding values
        assert collated['sequences'][0, 4:].sum() == 2  # Padding token is 1
        assert collated['attention_mask'][0, 4:].sum() == 0
    
    def test_data_augmentation(self):
        """Test data augmentation"""
        from structdiff.data.augmentation import SequenceAugmentation
        
        aug = SequenceAugmentation(mask_prob=0.15)
        
        sequences = torch.randint(5, 30, (4, 20))
        attention_mask = torch.ones(4, 20)
        
        augmented, mask = aug.mask_sequence(sequences, attention_mask)
        
        # Check some positions are masked
        assert (augmented != sequences).any()
        assert mask.any()
        
        # Check mask probability is roughly correct
        mask_ratio = mask.float().mean().item()
        assert 0.1 < mask_ratio < 0.2  # Should be around 0.15


# Diffusion Process Tests
class TestDiffusionProcess:
    """Test diffusion process components"""
    
    def test_noise_schedules(self):
        """Test different noise schedules"""
        timesteps = 1000
        
        for schedule in ['linear', 'cosine', 'sqrt']:
            diffusion = GaussianDiffusion(
                num_timesteps=timesteps,
                noise_schedule=schedule
            )
            
            # Check beta values
            assert len(diffusion.betas) == timesteps
            assert diffusion.betas[0] < diffusion.betas[-1]
            assert (diffusion.betas >= 0).all()
            assert (diffusion.betas <= 1).all()
            
            # Check cumulative products
            assert (diffusion.alphas_cumprod >= 0).all()
            assert (diffusion.alphas_cumprod <= 1).all()
            assert diffusion.alphas_cumprod[0] > diffusion.alphas_cumprod[-1]
    
    def test_forward_diffusion(self):
        """Test forward diffusion process"""
        diffusion = GaussianDiffusion(num_timesteps=1000)
        
        x_start = torch.randn(2, 10, 128)
        
        # Test different timesteps
        for t in [0, 100, 500, 999]:
            t_tensor = torch.tensor([t, t])
            x_noisy = diffusion.q_sample(x_start, t_tensor)
            
            assert x_noisy.shape == x_start.shape
            
            # Check noise level increases with time
            noise_level = (x_noisy - x_start).abs().mean()
            if t > 0:
                assert noise_level > 0
    
    def test_reverse_diffusion(self):
        """Test reverse diffusion step"""
        diffusion = GaussianDiffusion(num_timesteps=1000)
        
        x_t = torch.randn(2, 10, 128)
        model_output = torch.randn_like(x_t)
        t = torch.tensor([500, 500])
        
        x_prev = diffusion.p_sample(model_output, x_t, t)
        
        assert x_prev.shape == x_t.shape
        assert not torch.allclose(x_prev, x_t)  # Should be different


# Metrics Tests
class TestMetrics:
    """Test evaluation metrics"""
    
    def test_sequence_metrics(self):
        """Test sequence-based metrics"""
        generated = ['ACDEFGHIKL', 'MNPQRSTVWY', 'AAAAAAAAAA']
        reference = ['ACDEFGHIKL', 'KLMNPQRST', 'GGGGGGGGGG']
        
        metrics = compute_sequence_metrics(generated, reference)
        
        # Check expected metrics exist
        assert 'avg_length' in metrics
        assert 'aa_kl_divergence' in metrics
        assert 'avg_similarity_to_reference' in metrics
        
        # Check values are reasonable
        assert metrics['avg_length'] == 10.0
        assert 0 <= metrics['aa_kl_divergence'] <= 10
        assert 0 <= metrics['avg_similarity_to_reference'] <= 1
    
    def test_empty_inputs(self):
        """Test metrics with empty inputs"""
        # Should handle gracefully
        metrics = compute_sequence_metrics([])
        assert isinstance(metrics, dict)
    
    def test_advanced_metrics(self):
        """Test advanced metrics computation"""
        from structdiff.metrics.advanced_metrics import AdvancedMetrics
        
        advanced = AdvancedMetrics()
        sequences = ['KRRWKWFKK', 'ACDEFGHIK', 'LLLLAAAAA']
        
        # Test motif enrichment
        motifs = advanced.compute_motif_enrichment(sequences)
        assert 'cationic_cluster_frequency' in motifs
        
        # Test biophysical properties
        props = advanced.compute_biophysical_properties(sequences)
        assert 'hydrophobicity_mean' in props
        assert 'charge_mean' in props
        
        # Test functional likelihood
        func = advanced.compute_functional_likelihood(sequences, 'antimicrobial')
        assert 'antimicrobial_likelihood_mean' in func
        assert 0 <= func['antimicrobial_likelihood_mean'] <= 1


# Integration Tests
class TestIntegration:
    """End-to-end integration tests"""
    
    def test_training_step(self, test_config, sample_batch, device):
        """Test complete training step"""
        model = StructDiff(test_config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        model.train()
        
        # Forward pass
        timesteps = torch.randint(0, 100, (sample_batch['sequences'].shape[0],), device=device)
        outputs = model(
            sequences=sample_batch['sequences'],
            attention_mask=sample_batch['attention_mask'],
            timesteps=timesteps,
            structures=sample_batch['structures'],
            return_loss=True
        )
        
        # Backward pass
        loss = outputs['total_loss']
        loss.backward()
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert grad_norm > 0
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
    
    def test_generation_pipeline(self, test_config, device):
        """Test full generation pipeline"""
        model = StructDiff(test_config).to(device)
        model.eval()
        
        # Generate sequences
        with torch.no_grad():
            samples = model.sample(
                batch_size=4,
                seq_length=20,
                conditions={'peptide_type': torch.tensor([0, 0, 1, 1], device=device)},
                guidance_scale=2.0,
                temperature=0.8
            )
        
        sequences = samples['sequences']
        
        # Validate sequences
        assert len(sequences) == 4
        for seq in sequences:
            assert isinstance(seq, str)
            assert 10 <= len(seq) <= 30  # Reasonable length
            assert all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in seq)
        
        # Compute metrics
        metrics = compute_sequence_metrics(sequences)
        assert 'avg_length' in metrics
        assert 'aa_kl_divergence' in metrics


# Memory and Performance Tests
class TestPerformance:
    """Test memory usage and performance"""
    
    def test_memory_efficiency(self, test_config, device):
        """Test memory-efficient features"""
        if device.type != 'cuda':
            pytest.skip("GPU required for memory tests")
        
        from structdiff.utils.memory_optimization import MemoryMonitor
        
        monitor = MemoryMonitor(device)
        
        # Create model with memory optimizations
        test_config['training_config']['gradient_checkpointing'] = True
        model = StructDiff(test_config).to(device)
        
        # Track memory during forward pass
        with monitor.track_memory("forward_pass"):
            batch_size = 8
            seq_len = 50
            
            sequences = torch.randint(0, 20, (batch_size, seq_len), device=device)
            attention_mask = torch.ones(batch_size, seq_len, device=device)
            timesteps = torch.randint(0, 100, (batch_size,), device=device)
            
            outputs = model(
                sequences=sequences,
                attention_mask=attention_mask,
                timesteps=timesteps,
                return_loss=True
            )
        
        # Memory should not grow excessively
        memory_stats = monitor.memory_stats
        if len(memory_stats) >= 2:
            start_memory = memory_stats[-2]['pytorch_allocated_mb']
            end_memory = memory_stats[-1]['pytorch_allocated_mb']
            memory_increase = end_memory - start_memory
            
            # Should be reasonable (depends on model size)
            assert memory_increase < 1000  # Less than 1GB increase
    
    def test_data_loading_performance(self, tmp_path):
        """Test data loading efficiency"""
        from structdiff.data.efficient_loader import CachedPeptideDataset
        
        # Create test data
        num_samples = 100
        data = {
            'id': [f'pep_{i}' for i in range(num_samples)],
            'sequence': ['ACDEFGHIKLMNPQRSTVWY'] * num_samples,
            'label': np.random.randint(0, 3, num_samples).tolist()
        }
        
        data_path = tmp_path / 'test_data.csv'
        pd.DataFrame(data).to_csv(data_path, index=False)
        
        # Create cached dataset
        cache_dir = tmp_path / 'cache'
        dataset = CachedPeptideDataset(
            str(data_path),
            cache_dir=str(cache_dir),
            precompute_features=True,
            num_workers=2
        )
        
        # Time data loading
        import time
        start = time.time()
        
        for i in range(min(10, len(dataset))):
            _ = dataset[i]
        
        load_time = time.time() - start
        
        # Should be fast (cached)
        assert load_time < 1.0  # Less than 1 second for 10 samples


# Error Handling Tests
class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_inputs(self, test_config, device):
        """Test handling of invalid inputs"""
        model = StructDiff(test_config).to(device)
        
        # Test with mismatched dimensions
        with pytest.raises((ValueError, RuntimeError)):
            model(
                sequences=torch.randint(0, 20, (2, 10), device=device),
                attention_mask=torch.ones(3, 10, device=device),  # Wrong batch size
                timesteps=torch.tensor([0, 0], device=device)
            )
    
    def test_empty_batch(self, test_config, device):
        """Test with empty batch"""
        model = StructDiff(test_config).to(device)
        
        # Should handle gracefully or raise clear error
        with pytest.raises((ValueError, RuntimeError)):
            model.sample(batch_size=0, seq_length=10)
    
    def test_out_of_memory_handling(self, test_config, device):
        """Test OOM handling"""
        if device.type != 'cuda':
            pytest.skip("GPU required for OOM test")
        
        model = StructDiff(test_config).to(device)
        
        # Try to trigger OOM with huge batch
        try:
            model.sample(batch_size=10000, seq_length=1000)
            pytest.fail("Should have raised OOM error")
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            assert "out of memory" in str(e).lower()
            # Clean up
            torch.cuda.empty_cache()


# Utility function tests
def test_reproducibility():
    """Test reproducibility with seed setting"""
    from structdiff.utils import set_seed
    
    # Set seed and generate random numbers
    set_seed(42)
    random1 = torch.randn(10)
    
    # Reset seed and generate again
    set_seed(42)
    random2 = torch.randn(10)
    
    # Should be identical
    assert torch.allclose(random1, random2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
