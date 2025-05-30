import pytest
import torch

from structdiff.diffusion import GaussianDiffusion, get_sampler
from structdiff.diffusion.noise_schedule import get_noise_schedule


class TestGaussianDiffusion:
    """Test Gaussian diffusion process"""
    
    @pytest.fixture
    def diffusion(self):
        """Create diffusion instance"""
        return GaussianDiffusion(
            num_timesteps=100,
            noise_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02
        )
    
    def test_noise_schedule(self, diffusion):
        """Test noise schedule"""
        assert len(diffusion.betas) == 100
        assert diffusion.betas[0] < diffusion.betas[-1]
        assert torch.all(diffusion.alphas_cumprod >= 0)
        assert torch.all(diffusion.alphas_cumprod <= 1)
    
    def test_forward_diffusion(self, diffusion):
        """Test forward diffusion process"""
        batch_size = 2
        seq_len = 10
        hidden_dim = 128
        
        x_start = torch.randn(batch_size, seq_len, hidden_dim)
        t = torch.tensor([10, 50])
        
        x_noisy = diffusion.q_sample(x_start, t)
        
        assert x_noisy.shape == x_start.shape
        assert not torch.allclose(x_noisy, x_start)
    
    def test_reverse_step(self, diffusion):
        """Test reverse diffusion step"""
        batch_size = 2
        seq_len = 10
        hidden_dim = 128
        
        x_t = torch.randn(batch_size, seq_len, hidden_dim)
        t = torch.tensor([50, 50])
        model_output = torch.randn_like(x_t)
        
        x_prev = diffusion.p_sample(model_output, x_t, t)
        
        assert x_prev.shape == x_t.shape


class TestSamplers:
    """Test different sampling methods"""
    
    @pytest.fixture
    def diffusion(self):
        return GaussianDiffusion(num_timesteps=100)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        class MockModel(torch.nn.Module):
            def __init__(self, output_dim):
                super().__init__()
                self.output_dim = output_dim
            
            def forward(self, x, t, conditions=None):
                return torch.randn_like(x)
        
        return MockModel(128)
    
    def test_ddpm_sampler(self, diffusion, mock_model):
        """Test DDPM sampler"""
        sampler = get_sampler('ddpm', diffusion)
        
        shape = (2, 10, 128)
        samples = sampler.sample(
            mock_model,
            shape,
            device='cpu',
            verbose=False
        )
        
        assert samples.shape == shape
    
    def test_ddim_sampler(self, diffusion, mock_model):
        """Test DDIM sampler"""
        sampler = get_sampler('ddim', diffusion, num_inference_steps=10)
        
        shape = (2, 10, 128)
        samples = sampler.sample(
            mock_model,
            shape,
            device='cpu',
            verbose=False
        )
        
        assert samples.shape == shape
        assert len(sampler.timesteps) == 10