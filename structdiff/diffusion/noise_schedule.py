import numpy as np
import torch
from typing import Union


def get_noise_schedule(
    schedule_type: str,
    num_timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02
) -> np.ndarray:
    """
    Get noise schedule for diffusion process
    
    Args:
        schedule_type: Type of schedule ('linear', 'cosine', 'sqrt', 'alphafold3')
        num_timesteps: Number of diffusion timesteps
        beta_start: Starting value of beta
        beta_end: Ending value of beta
        
    Returns:
        Beta schedule array
    """
    if schedule_type == "linear":
        return np.linspace(beta_start, beta_end, num_timesteps)
    
    elif schedule_type == "cosine":
        # Cosine schedule from Nichol & Dhariwal
        def alpha_bar_fn(t):
            return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
        
        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            alpha_bar_t1 = alpha_bar_fn(t1)
            alpha_bar_t2 = alpha_bar_fn(t2)
            beta = 1 - alpha_bar_t2 / alpha_bar_t1
            betas.append(np.clip(beta, 0, 0.999))
        
        return np.array(betas)
    
    elif schedule_type == "sqrt":
        # Square root schedule
        return np.sqrt(np.linspace(beta_start**2, beta_end**2, num_timesteps))
    
    elif schedule_type == "alphafold3":
        # AlphaFold3 noise schedule - parameterized approach
        # More stable for protein sequences
        SIGMA_DATA = 0.5  # Data standard deviation
        smin, smax, p = 0.0004, 160.0, 7
        
        # Generate timesteps 
        timesteps = np.linspace(0, 1, num_timesteps)
        
        # AF3 noise schedule function
        sigmas = SIGMA_DATA * (smax**(1/p) + timesteps * (smin**(1/p) - smax**(1/p)))**p
        
        # Convert sigmas to betas (approximation for compatibility)
        # sigma^2 = beta / (1 - beta) approximately for small beta
        betas = sigmas**2 / (1 + sigmas**2)
        betas = np.clip(betas, beta_start, beta_end)
        
        return betas
    
    else:
        raise ValueError(f"Unknown noise schedule: {schedule_type}")
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
