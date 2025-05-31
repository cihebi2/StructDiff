import torch
import torch.nn as nn
import math
from typing import Optional


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings"""
    
    def __init__(self, hidden_dim: int, max_timesteps: int = 10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_timesteps = max_timesteps
        
        # Precompute embeddings
        self.register_buffer(
            'embeddings',
            self._create_sinusoidal_embeddings(max_timesteps, hidden_dim)
        )
        
        # Projection layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def _create_sinusoidal_embeddings(
        self,
        max_timesteps: int,
        hidden_dim: int
    ) -> torch.Tensor:
        """Create sinusoidal position embeddings"""
        position = torch.arange(max_timesteps).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2) * 
            -(math.log(10000.0) / hidden_dim)
        )
        
        embeddings = torch.zeros(max_timesteps, hidden_dim)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        
        return embeddings
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Tensor of shape (batch_size,) with timestep indices
            
        Returns:
            Timestep embeddings of shape (batch_size, hidden_dim)
        """
        # Get base embeddings
        embeddings = self.embeddings[timesteps]
        
        # Project through MLP
        return self.mlp(embeddings)


class ConditionEmbedding(nn.Module):
    """Embedding for peptide type conditions with dropout"""
    
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        
        # Class embeddings (including unconditional class)
        self.embeddings = nn.Embedding(num_classes + 1, hidden_dim)
        
        # Projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(
        self,
        class_labels: Optional[torch.Tensor] = None,
        force_unconditional: bool = False
    ) -> torch.Tensor:
        """
        Args:
            class_labels: Class indices of shape (batch_size,)
            force_unconditional: Force unconditional generation
            
        Returns:
            Condition embeddings of shape (batch_size, hidden_dim)
        """
        if class_labels is None or force_unconditional:
            # Use unconditional embedding
            batch_size = 1 if class_labels is None else class_labels.shape[0]
            device = self.embeddings.weight.device
            class_labels = torch.full(
                (batch_size,), self.num_classes, device=device
            )
        
        # Apply dropout during training
        if self.training and self.dropout_prob > 0:
            # Randomly set some labels to unconditional
            mask = torch.rand_like(class_labels.float()) < self.dropout_prob
            class_labels = torch.where(
                mask, 
                torch.full_like(class_labels, self.num_classes),
                class_labels
            )
        
        # Get embeddings
        embeddings = self.embeddings(class_labels)
        
        # Project
        return self.projection(embeddings)


class PositionalEncoding(nn.Module):
    """Learnable positional encoding"""
    
    def __init__(self, hidden_dim: int, max_length: int = 512):
        super().__init__()
        self.encoding = nn.Parameter(torch.randn(1, max_length, hidden_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        seq_len = x.shape[1]
        return x + self.encoding[:, :seq_len, :]
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04
