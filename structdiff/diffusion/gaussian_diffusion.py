import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from functools import partial

from .noise_schedule import get_noise_schedule


class GaussianDiffusion:
    """
    Gaussian diffusion process for continuous embeddings
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        noise_schedule: str = "sqrt",
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        self.num_timesteps = num_timesteps
        
        # Get noise schedule
        betas = get_noise_schedule(
            noise_schedule,
            num_timesteps,
            beta_start,
            beta_end
        )
        
        # Pre-compute diffusion parameters
        self.betas = torch.tensor(betas, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        
        # Pre-compute parameters for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Pre-compute parameters for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / 
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) /
            (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) /
            (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        
        Args:
            x_start: Clean data x_0
            t: Timestep indices
            noise: Optional pre-sampled noise
            
        Returns:
            Noisy data x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Move parameters to same device
        device = x_start.device
        
        # Move t to CPU for indexing, then move result to target device
        t_cpu = t.cpu()
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t_cpu].to(device)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_cpu].to(device)
        
        # Expand dimensions for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return (
            sqrt_alphas_cumprod_t * x_start +
            sqrt_one_minus_alphas_cumprod_t * noise
        )
    
    def add_noise(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to data (wrapper for q_sample)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        noisy_data = self.q_sample(x_start, timesteps, noise)
        
        # Fix device mismatch for indexing
        timesteps_cpu = timesteps.cpu()
        noise_level = self.sqrt_one_minus_alphas_cumprod[timesteps_cpu].to(x_start.device)
        
        return noisy_data, noise_level
    
    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior distribution q(x_{t-1} | x_t, x_0)
        """
        device = x_start.device
        
        # Fix device mismatch for indexing
        t_cpu = t.cpu()
        
        posterior_mean = (
            self.posterior_mean_coef1[t_cpu].to(device).unsqueeze(-1).unsqueeze(-1) * x_start +
            self.posterior_mean_coef2[t_cpu].to(device).unsqueeze(-1).unsqueeze(-1) * x_t
        )
        posterior_variance = self.posterior_variance[t_cpu].to(device).unsqueeze(-1).unsqueeze(-1)
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t_cpu].to(device).unsqueeze(-1).unsqueeze(-1)
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for p(x_{t-1} | x_t)
        """
        device = x_t.device
        
        # Predict x_0 from model output
        if model_output.shape != x_t.shape:
            raise ValueError(f"Model output shape {model_output.shape} != x_t shape {x_t.shape}")
        
        # Use model output as prediction of x_0
        pred_x_start = model_output
        
        if clip_denoised:
            # Optionally clip predictions
            pred_x_start = torch.clamp(pred_x_start, -1.0, 1.0)
        
        # Compute posterior mean and variance
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            x_start=pred_x_start, x_t=x_t, t=t
        )
        
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_sample(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t)
        """
        model_mean, _, model_log_variance = self.p_mean_variance(
            model_output, x_t, t, clip_denoised
        )
        
        noise = torch.randn_like(x_t)
        
        # No noise when t == 0
        nonzero_mask = (t != 0).float().unsqueeze(-1).unsqueeze(-1)
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    def denoise_step(
        self,
        x_t: torch.Tensor,
        t: int,
        model_output: torch.Tensor,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        Single denoising step with optional guidance
        """
        batch_size = x_t.shape[0]
        device = x_t.device
        
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Apply guidance if scale > 1
        if guidance_scale > 1.0:
            # Classifier-free guidance
            # Assume model_output contains both conditional and unconditional predictions
            # Split in half
            cond_output, uncond_output = model_output.chunk(2)
            model_output = uncond_output + guidance_scale * (cond_output - uncond_output)
        
        return self.p_sample(model_output, x_t, t_batch)
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
