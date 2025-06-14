import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, List
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm


class BaseSampler(ABC):
    """Abstract base class for diffusion samplers"""
    
    def __init__(self, diffusion_model):
        self.diffusion = diffusion_model
        
    @abstractmethod
    def sample(
        self,
        model: nn.Module,
        shape: torch.Size,
        conditions: Optional[Dict] = None,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
        device: str = 'cuda',
        verbose: bool = True
    ) -> torch.Tensor:
        """Generate samples from noise"""
        pass


class DDPMSampler(BaseSampler):
    """
    Standard DDPM sampler (slowest but highest quality)
    Uses full denoising chain
    """
    
    def sample(
        self,
        model: nn.Module,
        shape: torch.Size,
        conditions: Optional[Dict] = None,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
        device: str = 'cuda',
        verbose: bool = True
    ) -> torch.Tensor:
        """Sample using DDPM (full denoising chain)"""
        # 只对模型调用 eval()，对函数则跳过
        if hasattr(model, 'eval'):
            model.eval()
        
        # Start from pure noise
        x_t = torch.randn(shape, device=device) * temperature
        
        timesteps = reversed(range(0, self.diffusion.num_timesteps))
        if verbose:
            timesteps = tqdm(list(timesteps), desc="DDPM Sampling")
        
        with torch.no_grad():
            for t in timesteps:
                t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
                
                # Get model prediction
                if guidance_scale > 1.0 and conditions is not None:
                    # Classifier-free guidance
                    model_output_cond = model(x_t, t_batch, conditions)
                    model_output_uncond = model(x_t, t_batch, None)
                    
                    model_output = model_output_uncond + guidance_scale * (
                        model_output_cond - model_output_uncond
                    )
                else:
                    model_output = model(x_t, t_batch, conditions)
                
                # DDPM step using the diffusion model's p_sample method
                x_t = self.diffusion.p_sample(
                    model_output=model_output,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=True
                )
        
        return x_t


class DDIMSampler(BaseSampler):
    """
    Denoising Diffusion Implicit Models sampler
    Allows for faster sampling with fewer steps
    """
    
    def __init__(self, diffusion_model, num_inference_steps: int = 50):
        super().__init__(diffusion_model)
        self.num_inference_steps = num_inference_steps
        
        # Create sub-sampled timesteps
        self.timesteps = self._create_timesteps()
        
    def _create_timesteps(self) -> List[int]:
        """Create sub-sampled timestep schedule"""
        # Linearly spaced timesteps
        step_ratio = self.diffusion.num_timesteps // self.num_inference_steps
        timesteps = list(range(0, self.diffusion.num_timesteps, step_ratio))[::-1]
        
        return timesteps
    
    def sample(
        self,
        model: nn.Module,
        shape: torch.Size,
        conditions: Optional[Dict] = None,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
        eta: float = 0.0,  # 0 for deterministic, 1 for DDPM
        device: str = 'cuda',
        verbose: bool = True
    ) -> torch.Tensor:
        """Sample using DDIM"""
        # 只对模型调用 eval()，对函数则跳过
        if hasattr(model, 'eval'):
            model.eval()
        
        # Start from pure noise
        x_t = torch.randn(shape, device=device) * temperature
        
        timesteps = self.timesteps
        if verbose:
            timesteps = tqdm(timesteps, desc="DDIM Sampling")
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
                
                # Get model prediction
                if guidance_scale > 1.0 and conditions is not None:
                    model_output_cond = model(x_t, t_batch, conditions)
                    model_output_uncond = model(x_t, t_batch, None)
                    
                    model_output = model_output_uncond + guidance_scale * (
                        model_output_cond - model_output_uncond
                    )
                else:
                    model_output = model(x_t, t_batch, conditions)
                
                # DDIM step
                t_prev = self.timesteps[i + 1] if i < len(self.timesteps) - 1 else 0
                x_t = self._ddim_step(
                    model_output,
                    x_t,
                    t,
                    t_prev,
                    eta,
                    device
                )
        
        return x_t
    
    def _ddim_step(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float,
        device: str
    ) -> torch.Tensor:
        """Single DDIM denoising step"""
        # Get alphas
        alpha_t = self.diffusion.alphas_cumprod[t].to(device)
        alpha_t_prev = self.diffusion.alphas_cumprod[t_prev].to(device) if t_prev >= 0 else torch.tensor(1.0).to(device)
        
        # Compute x_0 prediction
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
        
        # Clip prediction
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Compute variance
        sigma_t = eta * torch.sqrt(
            (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)
        )
        
        # Compute mean
        mean_pred = (
            torch.sqrt(alpha_t_prev) * x_0_pred +
            torch.sqrt(1 - alpha_t_prev - sigma_t ** 2) * model_output
        )
        
        # Add noise
        noise = torch.randn_like(x_t) if t_prev > 0 else torch.zeros_like(x_t)
        x_t_prev = mean_pred + sigma_t * noise
        
        return x_t_prev


class PNDMSampler(BaseSampler):
    """
    Pseudo Numerical methods for Diffusion Models
    Even faster sampling than DDIM
    """
    
    def __init__(self, diffusion_model, num_inference_steps: int = 50):
        super().__init__(diffusion_model)
        self.num_inference_steps = num_inference_steps
        
        # Create timesteps
        self.timesteps = self._create_timesteps()
        
        # History for multi-step method
        self.ets = []
        
    def _create_timesteps(self) -> List[int]:
        """Create timestep schedule for PNDM"""
        # Use quadratic spacing for better results
        steps = np.linspace(0, self.diffusion.num_timesteps ** 0.5, self.num_inference_steps) ** 2
        timesteps = steps.round().astype(int)[::-1]
        
        return timesteps.tolist()
    
    def sample(
        self,
        model: nn.Module,
        shape: torch.Size,
        conditions: Optional[Dict] = None,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
        device: str = 'cuda',
        verbose: bool = True
    ) -> torch.Tensor:
        """Sample using PNDM"""
        # 只对模型调用 eval()，对函数则跳过
        if hasattr(model, 'eval'):
            model.eval()
        
        # Start from pure noise
        x_t = torch.randn(shape, device=device) * temperature
        
        # Clear history
        self.ets = []
        
        timesteps = self.timesteps
        if verbose:
            timesteps = tqdm(timesteps, desc="PNDM Sampling")
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
                
                # Get model prediction
                if guidance_scale > 1.0 and conditions is not None:
                    model_output_cond = model(x_t, t_batch, conditions)
                    model_output_uncond = model(x_t, t_batch, None)
                    
                    model_output = model_output_uncond + guidance_scale * (
                        model_output_cond - model_output_uncond
                    )
                else:
                    model_output = model(x_t, t_batch, conditions)
                
                # PNDM step
                x_t = self._pndm_step(
                    model_output,
                    x_t,
                    t,
                    self.timesteps[i + 1] if i < len(self.timesteps) - 1 else 0,
                    device
                )
        
        return x_t
    
    def _pndm_step(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        device: str
    ) -> torch.Tensor:
        """Single PNDM step using linear multi-step method"""
        # Add to history
        self.ets.append(model_output)
        
        # Only keep last 4 predictions
        if len(self.ets) > 4:
            self.ets.pop(0)
        
        # Get alphas
        alpha_t = self.diffusion.alphas_cumprod[t].to(device)
        alpha_t_prev = self.diffusion.alphas_cumprod[t_prev].to(device) if t_prev >= 0 else torch.tensor(1.0).to(device)
        
        # Linear multi-step coefficients
        if len(self.ets) == 1:
            # First step: use simple DDIM
            et = self.ets[-1]
        elif len(self.ets) == 2:
            # Second step: linear extrapolation
            et = 2 * self.ets[-1] - self.ets[-2]
        elif len(self.ets) == 3:
            # Third step: quadratic extrapolation
            et = 3 * self.ets[-1] - 3 * self.ets[-2] + self.ets[-3]
        else:
            # Fourth step and beyond: cubic extrapolation
            et = (55 * self.ets[-1] - 59 * self.ets[-2] + 
                  37 * self.ets[-3] - 9 * self.ets[-4]) / 24
        
        # Compute x_0 prediction
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * et) / torch.sqrt(alpha_t)
        
        # Clip prediction
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Compute x_{t-1}
        x_t_prev = (
            torch.sqrt(alpha_t_prev) * x_0_pred +
            torch.sqrt(1 - alpha_t_prev) * et
        )
        
        return x_t_prev


def get_sampler(
    name: str,
    diffusion_model,
    **kwargs
) -> BaseSampler:
    """
    Get sampler by name
    
    Args:
        name: Sampler name (ddpm, ddim, pndm)
        diffusion_model: Diffusion model instance
        **kwargs: Additional arguments for sampler
        
    Returns:
        Sampler instance
    """
    samplers = {
        'ddpm': DDPMSampler,
        'ddim': DDIMSampler,
        'pndm': PNDMSampler
    }
    
    if name not in samplers:
        raise ValueError(f"Unknown sampler: {name}")
    
    # 只为支持的采样器传递特定参数
    if name == 'ddpm':
        # DDPMSampler 不需要额外参数
        return samplers[name](diffusion_model)
    else:
        # DDIM 和 PNDM 支持 num_inference_steps
        return samplers[name](diffusion_model, **kwargs)
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
