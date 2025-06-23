import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from einops import rearrange


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between sequence and structure representations
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Sequence features (B, L, D)
            key_value: Structure features (B, L, D)
            attention_mask: Mask for valid positions (B, L)
            
        Returns:
            - Updated query features
            - Attention weights for visualization
        """
        batch_size, seq_len, _ = query.shape
        
        # Compute Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)
        
        # Reshape for multi-head attention
        Q = rearrange(Q, 'b l (h d) -> b h l d', h=self.num_heads)
        K = rearrange(K, 'b l (h d) -> b h l d', h=self.num_heads)
        V = rearrange(V, 'b l (h d) -> b h l d', h=self.num_heads)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores / self.temperature.clamp(min=0.1)  # Temperature scaling
        
        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape back
        context = rearrange(context, 'b h l d -> b l (h d)')
        
        # Output projection
        output = self.out_proj(context)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        # Average attention weights across heads for visualization
        attn_weights_avg = attn_weights.mean(dim=1)
        
        return output, attn_weights_avg


class BiDirectionalCrossAttention(nn.Module):
    """
    Bidirectional cross-attention for mutual information exchange
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        # Seq -> Struct attention
        self.seq_to_struct = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        # Struct -> Seq attention
        self.struct_to_seq = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(
        self,
        seq_features: torch.Tensor,
        struct_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Bidirectional cross-attention
        
        Returns:
            - Updated sequence features
            - Updated structure features
            - Dictionary of attention weights
        """
        # Seq attends to Struct
        seq_updated, seq_to_struct_attn = self.seq_to_struct(
            seq_features, struct_features, attention_mask
        )
        
        # Struct attends to Seq
        struct_updated, struct_to_seq_attn = self.struct_to_seq(
            struct_features, seq_features, attention_mask
        )
        
        # Fuse bidirectional information
        seq_fused = self.fusion(
            torch.cat([seq_updated, seq_features], dim=-1)
        )
        struct_fused = self.fusion(
            torch.cat([struct_updated, struct_features], dim=-1)
        )
        
        attn_weights = {
            'seq_to_struct': seq_to_struct_attn,
            'struct_to_seq': struct_to_seq_attn
        }
        
        return seq_fused, struct_fused, attn_weights
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
