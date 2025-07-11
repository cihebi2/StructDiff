import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer"""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, 'b l (three h d) -> three b h l d', 
                           three=3, h=self.num_heads)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply mask
        if attention_mask is not None:
            # Convert attention_mask to boolean if it's not already
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
            
            # Create mask for invalid positions (where attention_mask is False)
            mask = ~attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, v)
        context = rearrange(context, 'b h l d -> b l (h d)')
        
        # Output projection
        output = self.out_proj(context)
        
        return output
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
