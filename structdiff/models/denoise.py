import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from einops import rearrange

from .cross_attention import CrossModalAttention
from .layers.attention import MultiHeadSelfAttention
from .layers.embeddings import TimestepEmbedding, ConditionEmbedding
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StructureAwareDenoiser(nn.Module):
    """
    Denoiser network that incorporates structure information through cross-attention
    """
    
    def __init__(
        self,
        seq_hidden_dim: int,
        struct_hidden_dim: int,
        denoiser_config: Dict
    ):
        super().__init__()
        self.hidden_dim = denoiser_config.hidden_dim
        self.num_heads = denoiser_config.num_heads
        self.num_layers = denoiser_config.num_layers
        self.dropout = denoiser_config.dropout
        
        # Input projections
        self.seq_input_proj = nn.Linear(seq_hidden_dim, self.hidden_dim)
        self.struct_input_proj = nn.Linear(struct_hidden_dim, self.hidden_dim)
        
        # Timestep and condition embeddings
        self.time_embedding = TimestepEmbedding(self.hidden_dim)
        self.condition_embedding = ConditionEmbedding(
            num_classes=4,  # antimicrobial, antifungal, antiviral, unconditioned
            hidden_dim=self.hidden_dim,
            dropout_prob=0.1
        )
        
        # Denoising blocks
        self.blocks = nn.ModuleList([
            DenoisingBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                use_cross_attention=denoiser_config.use_cross_attention
            )
            for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.output_proj = nn.Linear(self.hidden_dim, seq_hidden_dim)
        
    def forward(
        self,
        noisy_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        structure_features: Optional[torch.Tensor] = None,
        conditions: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise embeddings with structure guidance
        
        Returns:
            - Denoised embeddings
            - Cross-attention weights for visualization
        """
        batch_size, seq_len, _ = noisy_embeddings.shape
        
        # Project inputs
        x = self.seq_input_proj(noisy_embeddings)
        
        # Add timestep embedding
        time_emb = self.time_embedding(timesteps)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)
        x = x + time_emb
        
        # Add condition embedding if provided
        if conditions is not None and 'peptide_type' in conditions:
            cond_emb = self.condition_embedding(conditions['peptide_type'])
            cond_emb = cond_emb.unsqueeze(1).expand(-1, seq_len, -1)
            x = x + cond_emb
        
        # Project structure features if provided
        struct_features = None
        if structure_features is not None:
            struct_features = self.struct_input_proj(structure_features)
        
        # Apply denoising blocks
        cross_attention_weights = []
        for block in self.blocks:
            x, cross_attn = block(
                x,
                attention_mask,
                structure_features=struct_features
            )
            if cross_attn is not None:
                cross_attention_weights.append(cross_attn)
        
        # Output projection
        x = self.output_norm(x)
        denoised = self.output_proj(x)
        
        # Stack cross-attention weights
        if cross_attention_weights:
            cross_attention_weights = torch.stack(cross_attention_weights, dim=1)
        else:
            cross_attention_weights = None
        
        return denoised, cross_attention_weights


class DenoisingBlock(nn.Module):
    """
    Single denoising block with self-attention and optional cross-attention
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        use_cross_attention: bool
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadSelfAttention(
            hidden_dim, num_heads, dropout
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Cross-attention (optional)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn = CrossModalAttention(
                hidden_dim, num_heads, dropout
            )
            self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        structure_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention
        residual = x
        x = self.self_attn_norm(x)
        x = self.self_attn(x, attention_mask) + residual
        
        # Cross-attention
        cross_attn_weights = None
        if self.use_cross_attention and structure_features is not None:
            residual = x
            x_norm = self.cross_attn_norm(x)
            x_cross, cross_attn_weights = self.cross_attn(
                x_norm, structure_features, attention_mask
            )
            x = x_cross + residual
        
        # Feed-forward
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x) + residual
        
        return x, cross_attn_weights
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
