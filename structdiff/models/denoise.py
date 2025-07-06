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
from .classifier_free_guidance import CFGConfig, ClassifierFreeGuidance, CFGTrainingMixin
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StructureAwareDenoiser(nn.Module, CFGTrainingMixin):
    """
    Denoiser network that incorporates structure information through cross-attention
    Enhanced with Classifier-Free Guidance support
    """
    
    def __init__(
        self,
        seq_hidden_dim: int,
        struct_hidden_dim: int,
        denoiser_config: Dict,
        cfg_config: Optional[CFGConfig] = None
    ):
        super().__init__()
        self.hidden_dim = denoiser_config.hidden_dim
        self.num_heads = denoiser_config.num_heads
        self.num_layers = denoiser_config.num_layers
        self.dropout = denoiser_config.dropout
        
        # Initialize CFG if provided
        if cfg_config is not None:
            CFGTrainingMixin.__init__(self, cfg_config)
            self.use_cfg = True
        else:
            self.use_cfg = False
        
        # Input projections
        self.seq_input_proj = nn.Linear(seq_hidden_dim, self.hidden_dim)
        self.struct_input_proj = nn.Linear(struct_hidden_dim, self.hidden_dim)
        
        # Enhanced AF3-style conditioning system
        self.time_embedding = AF3TimestepEmbedding(self.hidden_dim)
        
        # Multi-aspect adaptive conditioning
        # Support for unconditional token (-1) in CFG
        # Data has 3 classes (0, 1, 2), so we use 3 + 1 for unconditional
        num_condition_types = 4 if self.use_cfg else 3  # Add unconditional type for CFG
        self.adaptive_conditioning = AF3AdaptiveConditioning(
            hidden_dim=self.hidden_dim,
            condition_dim=self.hidden_dim // 2,  # Condition dimension
            num_condition_types=num_condition_types,  # antimicrobial, antifungal, antiviral, unconditioned
            dropout=self.dropout
        )
        
        # Fallback simple condition embedding for compatibility
        # Support unconditional class for CFG
        # Data has 3 classes (0, 1, 2), so we use 3 + 1 for unconditional
        num_classes = 4 if self.use_cfg else 3  # Add unconditional class for CFG
        self.condition_embedding = ConditionEmbedding(
            num_classes=num_classes,
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
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        use_cfg: bool = False,
        guidance_scale: Optional[float] = None,
        timestep_idx: Optional[int] = None,
        total_timesteps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise embeddings with structure guidance
        Enhanced with CFG support
        
        Args:
            noisy_embeddings: Noisy sequence embeddings
            timesteps: Diffusion timesteps
            attention_mask: Attention mask
            structure_features: Optional structure features
            conditions: Condition dictionary
            use_cfg: Whether to use CFG during inference
            guidance_scale: CFG guidance scale
            timestep_idx: Current timestep index for adaptive guidance
            total_timesteps: Total number of timesteps
        
        Returns:
            - Denoised embeddings
            - Cross-attention weights for visualization
        """
        # Use CFG if enabled and requested
        if self.use_cfg and use_cfg and not self.training:
            # CFG inference mode
            return self._forward_with_cfg(
                noisy_embeddings, timesteps, attention_mask,
                structure_features, conditions, guidance_scale,
                timestep_idx, total_timesteps
            )
        elif self.use_cfg and self.training:
            # CFG training mode (with condition dropout)
            return self._forward_cfg_training(
                noisy_embeddings, timesteps, attention_mask,
                structure_features, conditions
            )
        else:
            # Standard forward pass
            return self._forward_standard(
                noisy_embeddings, timesteps, attention_mask,
                structure_features, conditions
            )
    
    def _forward_standard(
        self,
        noisy_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        structure_features: Optional[torch.Tensor] = None,
        conditions: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard forward pass without CFG"""
        batch_size, seq_len, _ = noisy_embeddings.shape
        
        # Project inputs
        x = self.seq_input_proj(noisy_embeddings)
        
        # Add timestep embedding
        time_emb = self.time_embedding(timesteps)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)
        x = x + time_emb
        
        # Process conditions for CFG support
        processed_conditions = self._process_conditions_for_cfg(conditions, batch_size)
        
        # Generate adaptive conditioning signals
        conditioning_signals = None
        if processed_conditions is not None and 'peptide_type' in processed_conditions:
            # Use enhanced AF3-style adaptive conditioning
            conditioning_signals = self.adaptive_conditioning(
                processed_conditions['peptide_type'],
                strength_modifier=processed_conditions.get('condition_strength', None)
            )
            
            # Add base condition embedding to input (for backward compatibility)
            cond_emb = self.condition_embedding(processed_conditions['peptide_type'])
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
    
    def _forward_cfg_training(
        self,
        noisy_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        structure_features: Optional[torch.Tensor] = None,
        conditions: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CFG training mode with condition dropout"""
        batch_size = noisy_embeddings.shape[0]
        
        # Apply CFG condition preparation (with dropout)
        processed_conditions = self.cfg.prepare_conditions(
            conditions, batch_size, training=True
        )
        
        # Forward with processed conditions
        return self._forward_standard(
            noisy_embeddings, timesteps, attention_mask,
            structure_features, processed_conditions
        )
    
    def _forward_with_cfg(
        self,
        noisy_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor,
        structure_features: Optional[torch.Tensor] = None,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: Optional[float] = None,
        timestep_idx: Optional[int] = None,
        total_timesteps: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CFG inference mode with guided sampling"""
        batch_size = noisy_embeddings.shape[0]
        
        # Use CFG guided denoising
        def model_forward(x, t, cond):
            result, _ = self._forward_standard(x, t, attention_mask, structure_features, cond)
            return result
        
        guided_output = self.sample_with_cfg(
            model_forward,
            noisy_embeddings,
            timesteps,
            conditions,
            guidance_scale,
            timestep_idx,
            total_timesteps
        )
        
        # Return guided output with empty cross-attention weights
        return guided_output, None
    
    def _process_conditions_for_cfg(self, conditions: Optional[Dict[str, torch.Tensor]], batch_size: int) -> Dict[str, torch.Tensor]:
        """Process conditions for CFG support"""
        if conditions is None:
            return {'peptide_type': torch.full((batch_size,), -1, dtype=torch.long)}
        
        processed = conditions.copy()
        
        # Ensure peptide_type handles unconditional case (-1)
        if 'peptide_type' in processed:
            peptide_type = processed['peptide_type']
            # Map -1 (unconditional) to the last class index
            if self.use_cfg:
                peptide_type = torch.where(peptide_type == -1, 3, peptide_type)  # 3 is unconditional class (0,1,2,3)
            processed['peptide_type'] = peptide_type
        
        return processed


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
