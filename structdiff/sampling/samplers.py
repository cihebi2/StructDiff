# structdiff/sampling/advanced_samplers.py
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Callable
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod

from ..diffusion.sampling import BaseSampler
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DPMSolver(BaseSampler):
    """
    DPM-Solver: Fast ODE solver for diffusion models
    Reference: Lu et al., "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps" (2022)
    """
    
    def __init__(
        self,
        diffusion_model,
        algorithm: str = "dpmsolver++",
        order: int = 2,
        correcting_x0_fn: Optional[Callable] = None
    ):
        super().__init__(diffusion_model)
        self.algorithm = algorithm
        self.order = order
        self.correcting_x0_fn = correcting_x0_fn
        
    def sample(
        self,
        model: nn.Module,
        shape: torch.Size,
        conditions: Optional[Dict] = None,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
        num_inference_steps: int = 20,
        device: str = 'cuda',
        verbose: bool = True
    ) -> torch.Tensor:
        """Sample using DPM-Solver"""
        model.eval()
        
        # Start from noise
        x_t = torch.randn(shape, device=device) * temperature
        
        # Setup timesteps
        timesteps = self._get_timesteps(num_inference_steps)
        
        if verbose:
            timesteps = tqdm(timesteps, desc="DPM-Solver Sampling")
        
        # Initialize solver
        if self.order == 1:
            return self._first_order_sample(
                model, x_t, timesteps, conditions, guidance_scale, device
            )
        elif self.order == 2:
            return self._second_order_sample(
                model, x_t, timesteps, conditions, guidance_scale, device
            )
        else:
            return self._multistep_sample(
                model, x_t, timesteps, conditions, guidance_scale, device
            )
    
    def _first_order_sample(
        self, model, x_t, timesteps, conditions, guidance_scale, device
    ):
        """First-order DPM-Solver (equivalent to DDIM)"""
        with torch.no_grad():
            for i, t in enumerate(timesteps[:-1]):
                t_prev = timesteps[i + 1]
                
                # Model prediction
                model_output = self._get_model_output(
                    model, x_t, t, conditions, guidance_scale, device
                )
                
                # Update x_t
                x_t = self._dpm_solver_first_order_update(
                    x_t, model_output, t, t_prev, device
                )
        
        return x_t
    
    def _second_order_sample(
        self, model, x_t, timesteps, conditions, guidance_scale, device
    ):
        """Second-order DPM-Solver"""
        model_outputs = []
        
        with torch.no_grad():
            for i, t in enumerate(timesteps[:-1]):
                t_prev = timesteps[i + 1]
                
                # Model prediction
                model_output = self._get_model_output(
                    model, x_t, t, conditions, guidance_scale, device
                )
                model_outputs.append(model_output)
                
                if i == 0:
                    # First step: use first-order
                    x_t = self._dpm_solver_first_order_update(
                        x_t, model_output, t, t_prev, device
                    )
                else:
                    # Second-order update
                    x_t = self._dpm_solver_second_order_update(
                        x_t, model_outputs[-2:], timesteps[i-1:i+2], device
                    )
        
        return x_t
    
    def _get_timesteps(self, num_steps: int) -> List[int]:
        """Get timestep schedule for DPM-Solver"""
        # Use exponential schedule for better results
        t_start = self.diffusion.num_timesteps - 1
        t_end = 0
        
        if self.algorithm == "dpmsolver++":
            # DPM-Solver++ schedule
            lambda_t = lambda t: np.log(self.diffusion.alphas_cumprod[t])
            lambdas = np.linspace(lambda_t(t_start), lambda_t(t_end), num_steps + 1)
            
            timesteps = []
            for lambda_val in lambdas:
                # Find t such that lambda_t(t) ≈ lambda_val
                dists = [abs(lambda_t(t) - lambda_val) for t in range(self.diffusion.num_timesteps)]
                timesteps.append(np.argmin(dists))
            
            return timesteps
        else:
            # Linear schedule
            return np.linspace(t_start, t_end, num_steps + 1).round().astype(int).tolist()


class AnalyticDPM(BaseSampler):
    """
    Analytic-DPM: Training-free fast sampling
    Reference: Bao et al., "Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models" (2022)
    """
    
    def __init__(self, diffusion_model):
        super().__init__(diffusion_model)
        self._precompute_coefficients()
    
    def _precompute_coefficients(self):
        """Precompute analytic coefficients"""
        # Compute optimal variance schedule analytically
        alphas = self.diffusion.alphas_cumprod
        betas = 1 - alphas
        
        # Analytic variance
        self.analytic_variance = []
        for t in range(len(alphas)):
            if t == 0:
                var = 0
            else:
                var = betas[t] * (1 - alphas[t-1]) / (1 - alphas[t])
            self.analytic_variance.append(var)
        
        self.analytic_variance = torch.tensor(self.analytic_variance)
    
    def sample(
        self,
        model: nn.Module,
        shape: torch.Size,
        conditions: Optional[Dict] = None,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
        num_inference_steps: int = 20,
        device: str = 'cuda',
        verbose: bool = True
    ) -> torch.Tensor:
        """Sample using Analytic-DPM"""
        model.eval()
        
        # Start from noise
        x_t = torch.randn(shape, device=device) * temperature
        
        # Get timestep schedule
        skip = self.diffusion.num_timesteps // num_inference_steps
        timesteps = list(range(self.diffusion.num_timesteps - 1, 0, -skip))
        
        if verbose:
            timesteps = tqdm(timesteps, desc="Analytic-DPM Sampling")
        
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
                
                # Analytic update
                x_t = self._analytic_update(x_t, model_output, t, device)
        
        return x_t
    
    def _analytic_update(self, x_t, model_output, t, device):
        """Update using analytic variance"""
        # Get coefficients
        alpha_t = self.diffusion.alphas_cumprod[t].to(device)
        alpha_prev = self.diffusion.alphas_cumprod[t-1].to(device) if t > 0 else torch.tensor(1.0).to(device)
        beta_t = 1 - alpha_t
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(beta_t) * model_output) / torch.sqrt(alpha_t)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Get analytic variance
        variance = self.analytic_variance[t].to(device)
        
        # Compute mean
        mean = (
            torch.sqrt(alpha_prev) * beta_t / (1 - alpha_t) * x_0_pred +
            torch.sqrt(alpha_t) * (1 - alpha_prev) / (1 - alpha_t) * x_t
        )
        
        # Add noise
        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(variance) * noise
        else:
            x_t = mean
        
        return x_t


class EDMSampler(BaseSampler):
    """
    EDM (Elucidating the Design Space of Diffusion-Based Generative Models) Sampler
    Reference: Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models" (2022)
    """
    
    def __init__(
        self,
        diffusion_model,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: float = 7
    ):
        super().__init__(diffusion_model)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
    
    def sample(
        self,
        model: nn.Module,
        shape: torch.Size,
        conditions: Optional[Dict] = None,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
        num_inference_steps: int = 50,
        device: str = 'cuda',
        verbose: bool = True,
        s_churn: float = 0,
        s_tmin: float = 0,
        s_tmax: float = float('inf'),
        s_noise: float = 1
    ) -> torch.Tensor:
        """Sample using EDM formulation"""
        model.eval()
        
        # Generate noise schedule
        sigmas = self._get_sigmas(num_inference_steps, device)
        
        # Start from noise
        x = torch.randn(shape, device=device) * sigmas[0] * temperature
        
        if verbose:
            sigma_steps = tqdm(enumerate(sigmas[:-1]), total=len(sigmas)-1, desc="EDM Sampling")
        else:
            sigma_steps = enumerate(sigmas[:-1])
        
        with torch.no_grad():
            for i, sigma in sigma_steps:
                sigma_next = sigmas[i + 1]
                
                # Add noise (stochastic sampling)
                if s_churn > 0 and s_tmin < sigma < s_tmax:
                    gamma = min(s_churn / num_inference_steps, 2 ** 0.5 - 1)
                    sigma_hat = sigma * (1 + gamma)
                    x = x + (sigma_hat ** 2 - sigma ** 2) ** 0.5 * s_noise * torch.randn_like(x)
                else:
                    sigma_hat = sigma
                
                # Denoise
                denoised = self._denoise(model, x, sigma_hat, conditions, guidance_scale)
                
                # Euler step
                d = (x - denoised) / sigma_hat
                dt = sigma_next - sigma_hat
                x = x + d * dt
                
                # Second-order correction
                if sigma_next > 0 and i < num_inference_steps - 1:
                    denoised_2 = self._denoise(model, x, sigma_next, conditions, guidance_scale)
                    d_2 = (x - denoised_2) / sigma_next
                    x = x + (d + d_2) * dt / 2
        
        return x
    
    def _get_sigmas(self, n: int, device: torch.device) -> torch.Tensor:
        """Generate sigma schedule"""
        ramp = torch.linspace(0, 1, n + 1, device=device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas
    
    def _denoise(self, model, x, sigma, conditions, guidance_scale):
        """Denoise at given noise level"""
        # Convert sigma to timestep
        t = self._sigma_to_t(sigma)
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        
        # Model prediction
        if guidance_scale > 1.0 and conditions is not None:
            # Classifier-free guidance
            model_output_cond = model(x, t_batch, conditions)
            model_output_uncond = model(x, t_batch, None)
            model_output = model_output_uncond + guidance_scale * (
                model_output_cond - model_output_uncond
            )
        else:
            model_output = model(x, t_batch, conditions)
        
        # Denoise
        return x - sigma * model_output
    
    def _sigma_to_t(self, sigma: float) -> int:
        """Convert sigma to timestep"""
        # Approximate mapping (would need proper calibration)
        log_sigma = np.log(sigma)
        log_sigma_min = np.log(self.sigma_min)
        log_sigma_max = np.log(self.sigma_max)
        
        # Linear interpolation in log space
        frac = (log_sigma - log_sigma_min) / (log_sigma_max - log_sigma_min)
        t = int(frac * (self.diffusion.num_timesteps - 1))
        
        return np.clip(t, 0, self.diffusion.num_timesteps - 1)


class ConsistencyModel(nn.Module):
    """
    Consistency Models for single-step generation
    Reference: Song et al., "Consistency Models" (2023)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        target_model: Optional[nn.Module] = None,
        consistency_training: bool = True
    ):
        super().__init__()
        self.base_model = base_model
        self.target_model = target_model or base_model
        self.consistency_training = consistency_training
        
        # Consistency function parameters
        self.sigma_min = 0.002
        self.sigma_max = 80
        
    def forward(self, x: torch.Tensor, sigma: torch.Tensor, conditions: Optional[Dict] = None):
        """Forward pass of consistency model"""
        # Skip connection
        skip_scaling = self.sigma_min / (sigma ** 2 + self.sigma_min ** 2) ** 0.5
        out_scaling = sigma * self.sigma_min / (sigma ** 2 + self.sigma_min ** 2) ** 0.5
        
        # Model prediction
        model_output = self.base_model(x, self._sigma_to_t(sigma), conditions)
        
        # Consistency parameterization
        return skip_scaling * x + out_scaling * model_output
    
    @torch.no_grad()
    def sample(
        self,
        shape: torch.Size,
        conditions: Optional[Dict] = None,
        num_steps: int = 1,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """Generate samples using consistency model"""
        # Start from noise
        x = torch.randn(shape, device=device) * self.sigma_max
        
        if num_steps == 1:
            # Single-step generation
            sigma = torch.full((shape[0],), self.sigma_max, device=device)
            return self.forward(x, sigma, conditions)
        else:
            # Multi-step refinement
            sigmas = torch.linspace(self.sigma_max, self.sigma_min, num_steps + 1, device=device)
            
            for i in range(num_steps):
                sigma = sigmas[i].expand(shape[0])
                x = self.forward(x, sigma, conditions)
                
                # Add noise for next step (except last)
                if i < num_steps - 1:
                    noise_level = (sigmas[i+1] ** 2 - self.sigma_min ** 2) ** 0.5
                    x = x + noise_level * torch.randn_like(x)
            
            return x
    
    def _sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to timestep"""
        # Implement proper mapping based on noise schedule
        # This is a placeholder
        return (sigma / self.sigma_max * 999).long()


# Latent Consistency Models (LCM)
class LatentConsistencyModel(ConsistencyModel):
    """
    Latent Consistency Models for ultra-fast generation
    Reference: Luo et al., "Latent Consistency Models" (2023)
    """
    
    def __init__(self, base_model: nn.Module, num_inference_steps: int = 4):
        super().__init__(base_model)
        self.num_inference_steps = num_inference_steps
        
        # LCM-specific parameters
        self.w = 3.0  # Guidance embedding scale
        
    @torch.no_grad()
    def sample_lcm(
        self,
        shape: torch.Size,
        conditions: Optional[Dict] = None,
        guidance_scale: float = 8.0,
        num_steps: Optional[int] = None,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """Ultra-fast sampling with LCM"""
        num_steps = num_steps or self.num_inference_steps
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        # LCM sampling schedule
        c_skip = 0.25
        c_out = 0.7
        
        for i in range(num_steps):
            # Compute sigma for current step
            t = (i + 0.5) / num_steps
            sigma = self._t_to_sigma(t)
            
            # Consistency model forward with guidance
            if conditions is not None:
                # Embed guidance scale
                w_embedding = torch.full((shape[0], 1), self.w * guidance_scale, device=device)
                conditions['w'] = w_embedding
            
            # Apply consistency function
            out = self.forward(x, sigma.expand(shape[0]), conditions)
            
            # Skip connection parameterization
            x = c_skip * x + c_out * out
        
        return x
    
    def _t_to_sigma(self, t: float) -> torch.Tensor:
        """Convert normalized timestep to sigma"""
        # LCM schedule
        return torch.tensor(self.sigma_max ** (1 - t) * self.sigma_min ** t)


# Rectified Flow
class RectifiedFlowSampler(BaseSampler):
    """
    Rectified Flow for straight generation trajectories
    Reference: Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (2022)
    """
    
    def __init__(self, diffusion_model, reflow_steps: int = 1):
        super().__init__(diffusion_model)
        self.reflow_steps = reflow_steps
    
    def sample(
        self,
        model: nn.Module,
        shape: torch.Size,
        conditions: Optional[Dict] = None,
        guidance_scale: float = 1.0,
        temperature: float = 1.0,
        num_inference_steps: int = 50,
        device: str = 'cuda',
        verbose: bool = True
    ) -> torch.Tensor:
        """Sample using rectified flow"""
        model.eval()
        
        # Start from noise
        z1 = torch.randn(shape, device=device) * temperature
        
        # Time steps
        times = torch.linspace(1, 0, num_inference_steps + 1, device=device)
        
        if verbose:
            time_pairs = tqdm(
                zip(times[:-1], times[1:]),
                total=num_inference_steps,
                desc="Rectified Flow Sampling"
            )
        else:
            time_pairs = zip(times[:-1], times[1:])
        
        x = z1
        with torch.no_grad():
            for t_curr, t_next in time_pairs:
                # Predict velocity
                t_batch = t_curr.expand(shape[0])
                
                if guidance_scale > 1.0 and conditions is not None:
                    # Classifier-free guidance on velocity
                    v_cond = self._predict_velocity(model, x, t_batch, conditions)
                    v_uncond = self._predict_velocity(model, x, t_batch, None)
                    v = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    v = self._predict_velocity(model, x, t_batch, conditions)
                
                # Euler step
                dt = t_next - t_curr
                x = x + v * dt
        
        return x
    
    def _predict_velocity(self, model, x, t, conditions):
        """Predict velocity field"""
        # Convert to diffusion model timestep
        diffusion_t = (t * self.diffusion.num_timesteps).long()
        
        # Get model prediction (noise)
        noise_pred = model(x, diffusion_t, conditions)
        
        # Convert to velocity
        alpha_t = self.diffusion.alphas_cumprod[diffusion_t].view(-1, 1, 1)
        sigma_t = torch.sqrt(1 - alpha_t)
        
        # Velocity = (data - noise * sigma) / alpha - x / (1 - t)
        velocity = -noise_pred * sigma_t / alpha_t
        
        return velocity
# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
