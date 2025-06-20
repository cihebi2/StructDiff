import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class AF3FourierEmbedding(nn.Module):
    """
    AlphaFold3-style Fourier time embedding with pre-defined weights
    This provides more stable time embeddings for diffusion models
    """
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Pre-defined Fourier weights (similar to AF3)
        # These are learned once and fixed for stability
        frequencies = torch.randn(embedding_dim // 2) * 0.02
        phases = torch.randn(embedding_dim // 2) * 2 * np.pi
        
        self.register_buffer('frequencies', frequencies)
        self.register_buffer('phases', phases)
        
        # Linear projection to desired dimension
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create Fourier embeddings for timesteps
        
        Args:
            timesteps: Tensor of shape (batch_size,) with timestep values
            
        Returns:
            Fourier embeddings of shape (batch_size, embedding_dim)
        """
        # Ensure timesteps are float and on correct device
        timesteps = timesteps.float().to(self.frequencies.device)
        
        # Expand timesteps for broadcasting: (batch_size, 1)
        t_expanded = timesteps.unsqueeze(-1)
        
        # Compute Fourier features
        # (batch_size, embedding_dim//2)
        cos_features = torch.cos(t_expanded * self.frequencies + self.phases)
        sin_features = torch.sin(t_expanded * self.frequencies + self.phases)
        
        # Concatenate cos and sin features
        fourier_features = torch.cat([cos_features, sin_features], dim=-1)
        
        # Apply linear projection
        embeddings = self.linear(fourier_features)
        
        return embeddings


class AF3AdaptiveLayerNorm(nn.Module):
    """
    AlphaFold3-style adaptive layer normalization
    Modulates normalization based on conditions (time, peptide type, etc.)
    """
    
    def __init__(self, hidden_dim: int, condition_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Standard layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Condition-dependent scale and shift
        self.scale_net = nn.Linear(condition_dim, hidden_dim)
        self.shift_net = nn.Linear(condition_dim, hidden_dim)
        
        # Initialize with small values for stability
        nn.init.zeros_(self.scale_net.weight)
        nn.init.zeros_(self.scale_net.bias)
        nn.init.zeros_(self.shift_net.weight)
        nn.init.zeros_(self.shift_net.bias)
        
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply adaptive layer normalization
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)
            condition: Condition tensor (batch_size, condition_dim)
            
        Returns:
            Normalized tensor with same shape as input
        """
        # Standard layer normalization
        normalized = self.layer_norm(x)
        
        if condition is None:
            return normalized
        
        # Compute condition-dependent scale and shift
        scale = torch.sigmoid(self.scale_net(condition))  # Ensure positive scale
        shift = self.shift_net(condition)
        
        # Expand for broadcasting with sequence dimension
        if x.dim() == 3:  # (batch, seq_len, hidden_dim)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        # Apply adaptive modulation
        return scale * normalized + shift


class AF3AdaptiveZeroInit(nn.Module):
    """
    AlphaFold3-style adaptive output layer with zero initialization
    Ensures stable training by starting with small outputs
    """
    
    def __init__(self, input_dim: int, output_dim: int, condition_dim: int):
        super().__init__()
        
        # Main output projection
        self.output_proj = nn.Linear(input_dim, output_dim)
        
        # Condition-dependent gating
        self.gate_net = nn.Linear(condition_dim, output_dim)
        
        # Initialize output projection normally
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # Initialize gate with small negative bias for near-zero initial output
        nn.init.zeros_(self.gate_net.weight)
        nn.init.constant_(self.gate_net.bias, -2.0)  # sigmoid(-2) ≈ 0.12
        
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply adaptive zero-initialized output
        
        Args:
            x: Input tensor
            condition: Condition tensor
            
        Returns:
            Gated output tensor
        """
        output = self.output_proj(x)
        
        if condition is None:
            return output
        
        # Compute condition-dependent gate
        gate = torch.sigmoid(self.gate_net(condition))
        
        # Expand gate for broadcasting if needed
        if x.dim() == 3 and condition.dim() == 2:  # (batch, seq_len, dim)
            gate = gate.unsqueeze(1)
        
        return gate * output


class AF3TimestepEmbedding(nn.Module):
    """
    Complete AlphaFold3-style timestep embedding system
    Combines Fourier embedding with adaptive conditioning
    """
    
    def __init__(self, hidden_dim: int, fourier_dim: int = 256):
        super().__init__()
        
        # Fourier time embedding
        self.fourier_embedding = AF3FourierEmbedding(fourier_dim)
        
        # Project to hidden dimension
        self.time_proj = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create timestep embeddings
        
        Args:
            timesteps: Timestep indices (batch_size,)
            
        Returns:
            Time embeddings (batch_size, hidden_dim)
        """
        fourier_emb = self.fourier_embedding(timesteps)
        time_emb = self.time_proj(fourier_emb)
        return time_emb