import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from einops import rearrange

from .cross_attention import CrossModalAttention
from .layers.attention import MultiHeadSelfAttention
from .layers.embeddings import TimestepEmbedding, ConditionEmbedding
from .layers.mlp import FeedForward
from .layers.alphafold3_embeddings import (
    AF3TimestepEmbedding, 
    AF3AdaptiveLayerNorm,
    AF3AdaptiveConditioning,
    AF3EnhancedConditionalLayerNorm,
    AF3ConditionalZeroInit
)
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
        
        # Enhanced AF3-style conditioning system
        self.time_embedding = AF3TimestepEmbedding(self.hidden_dim)
        
        # Multi-aspect adaptive conditioning
        self.adaptive_conditioning = AF3AdaptiveConditioning(
            hidden_dim=self.hidden_dim,
            condition_dim=self.hidden_dim // 2,  # Condition dimension
            num_condition_types=4,  # antimicrobial, antifungal, antiviral, unconditioned
            dropout=self.dropout
        )
        
        # Fallback simple condition embedding for compatibility
        self.condition_embedding = ConditionEmbedding(
            num_classes=4,
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
        
        # Enhanced output with adaptive conditioning
        self.output_norm = AF3EnhancedConditionalLayerNorm(
            self.hidden_dim, 
            condition_dim=self.hidden_dim // 2
        )
        self.output_proj = AF3ConditionalZeroInit(
            self.hidden_dim, 
            seq_hidden_dim,
            condition_dim=self.hidden_dim // 2
        )
        
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
        
        # Generate adaptive conditioning signals
        conditioning_signals = None
        if conditions is not None and 'peptide_type' in conditions:
            # Use enhanced AF3-style adaptive conditioning
            conditioning_signals = self.adaptive_conditioning(
                conditions['peptide_type'],
                strength_modifier=conditions.get('condition_strength', None)
            )
            
            # Add base condition embedding to input (for backward compatibility)
            cond_emb = self.condition_embedding(conditions['peptide_type'])
            cond_emb = cond_emb.unsqueeze(1).expand(-1, seq_len, -1)
            x = x + cond_emb
        
        # Project structure features if provided
        struct_features = None
        if structure_features is not None:
            struct_features = self.struct_input_proj(structure_features)
        
        # Apply denoising blocks with adaptive conditioning
        cross_attention_weights = []
        for block in self.blocks:
            x, cross_attn = block(
                x,
                attention_mask,
                structure_features=struct_features,
                conditioning_signals=conditioning_signals
            )
            if cross_attn is not None:
                cross_attention_weights.append(cross_attn)
        
        # Enhanced output projection with adaptive conditioning
        x = self.output_norm(x, conditioning_signals)
        denoised = self.output_proj(x, conditioning_signals)
        
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
        
        # Self-attention with enhanced normalization
        self.self_attn = MultiHeadSelfAttention(
            hidden_dim, num_heads, dropout
        )
        self.self_attn_norm = AF3EnhancedConditionalLayerNorm(
            hidden_dim, 
            condition_dim=hidden_dim // 2
        )
        
        # Cross-attention (optional) with enhanced normalization
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn = CrossModalAttention(
                hidden_dim, num_heads, dropout
            )
            self.cross_attn_norm = AF3EnhancedConditionalLayerNorm(
                hidden_dim, 
                condition_dim=hidden_dim // 2
            )
        
        # Feed-forward with GLU (Gated Linear Unit) and adaptive conditioning
        self.ffn = FeedForward(
            hidden_dim=hidden_dim,
            intermediate_dim=hidden_dim * 4,
            dropout=dropout,
            activation="silu",  # SiLU/Swish activation like AF3
            use_gate=True  # Enable GLU
        )
        
        # Enhanced FFN normalization
        self.ffn_norm = AF3EnhancedConditionalLayerNorm(
            hidden_dim,
            condition_dim=hidden_dim // 2
        )
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        structure_features: Optional[torch.Tensor] = None,
        conditioning_signals: Optional[dict] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with adaptive conditioning
        residual = x
        x = self.self_attn_norm(x, conditioning_signals)
        x = self.self_attn(x, attention_mask) + residual
        
        # Cross-attention with adaptive conditioning
        cross_attn_weights = None
        if self.use_cross_attention and structure_features is not None:
            residual = x
            x_norm = self.cross_attn_norm(x, conditioning_signals)
            x_cross, cross_attn_weights = self.cross_attn(
                x_norm, structure_features, attention_mask
            )
            x = x_cross + residual
        
        # Feed-forward with adaptive conditioning
        residual = x
        x_norm = self.ffn_norm(x, conditioning_signals)
        x = self.ffn(x_norm) + residual
        
        return x, cross_attn_weights
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
