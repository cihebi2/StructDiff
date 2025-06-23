import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from structdiff.models import StructDiff, Denoiser, StructureEncoder
from structdiff.models.cross_attention import CrossModalAttention


class TestStructDiff:
    """Test StructDiff model"""
    
    @pytest.fixture
    def config(self):
        """Model configuration"""
        return OmegaConf.create({
            'model': {
                'hidden_dim': 128,
                'num_layers': 2,
                'num_attention_heads': 4,
                'sequence_encoder': {
                    'pretrained_model': None,  # Use mock
                    'hidden_dim': 128
                },
                'structure_encoder': {
                    'type': 'multi_scale',
                    'local': {
                        'hidden_dim': 64,
                        'num_layers': 2
                    },
                    'global': {
                        'hidden_dim': 64,
                        'num_layers': 2
                    }
                },
                'cross_attention': {
                    'num_layers': 2,
                    'hidden_dim': 128,
                    'num_heads': 4
                },
                'denoiser': {
                    'hidden_dim': 128,
                    'num_layers': 2,
                    'num_attention_heads': 4
                }
            },
            'diffusion': {
                'num_timesteps': 100
            }
        })
    
    def test_model_creation(self, config):
        """Test model instantiation"""
        model = StructDiff(config)
        assert isinstance(model, nn.Module)
    
    def test_forward_pass(self, config):
        """Test forward pass"""
        model = StructDiff(config)
        model.eval()
        
        batch_size = 2
        seq_len = 10
        
        # Mock inputs
        sequences = torch.randint(0, 20, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        timesteps = torch.randint(0, 100, (batch_size,))
        
        with torch.no_grad():
            output = model(sequences, attention_mask, timesteps)
        
        assert output.shape == (batch_size, seq_len, config.model.hidden_dim)
    
    def test_conditional_generation(self, config):
        """Test conditional generation"""
        model = StructDiff(config)
        model.eval()
        
        conditions = {
            'peptide_type': torch.tensor([0, 1])
        }
        
        with torch.no_grad():
            samples = model.sample(
                batch_size=2,
                seq_length=15,
                conditions=conditions
            )
        
        assert 'embeddings' in samples
        assert samples['embeddings'].shape == (2, 15, config.model.hidden_dim)


class TestDenoiser:
    """Test Denoiser module"""
    
    def test_denoiser_forward(self):
        """Test denoiser forward pass"""
        config = OmegaConf.create({
            'hidden_dim': 128,
            'num_layers': 2,
            'num_attention_heads': 4,
            'dropout': 0.1
        })
        
        denoiser = Denoiser(config)
        
        batch_size = 2
        seq_len = 10
        
        embeddings = torch.randn(batch_size, seq_len, 128)
        timesteps = torch.randint(0, 100, (batch_size,))
        attention_mask = torch.ones(batch_size, seq_len)
        
        output = denoiser(embeddings, timesteps, attention_mask)
        
        assert output.shape == embeddings.shape


class TestCrossAttention:
    """Test cross-attention modules"""
    
    def test_cross_modal_attention(self):
        """Test cross-modal attention"""
        hidden_dim = 128
        num_heads = 4
        
        attn = CrossModalAttention(hidden_dim, num_heads)
        
        batch_size = 2
        seq_len = 10
        
        query = torch.randn(batch_size, seq_len, hidden_dim)
        key_value = torch.randn(batch_size, seq_len, hidden_dim)
        
        output, weights = attn(query, key_value)
        
        assert output.shape == query.shape
        assert weights.shape == (batch_size, seq_len, seq_len)
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
