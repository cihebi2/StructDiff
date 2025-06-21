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


class AF3AdaptiveConditioning(nn.Module):
    """
    Enhanced AlphaFold3-style adaptive conditioning system
    Provides fine-grained control over functional conditions (antimicrobial, antifungal, antiviral)
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        condition_dim: int,
        num_condition_types: int = 4,  # antimicrobial, antifungal, antiviral, unconditioned
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.num_condition_types = num_condition_types
        
        # Condition embedding with biological priors
        self.condition_embedding = nn.Embedding(num_condition_types, condition_dim)
        
        # Condition enhancement network
        self.condition_enhancer = nn.Sequential(
            nn.Linear(condition_dim, condition_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(condition_dim * 2, condition_dim),
            nn.LayerNorm(condition_dim)
        )
        
        # Fine-grained condition control
        self.strength_control = nn.Linear(condition_dim, 1)  # Control condition strength
        self.specificity_control = nn.Linear(condition_dim, hidden_dim)  # Target-specific modulation
        
        # Multi-level condition projections for different aspects
        self.charge_control = nn.Linear(condition_dim, hidden_dim // 4)
        self.hydrophobic_control = nn.Linear(condition_dim, hidden_dim // 4) 
        self.structure_control = nn.Linear(condition_dim, hidden_dim // 4)
        self.functional_control = nn.Linear(condition_dim, hidden_dim // 4)
        
        # Initialize condition embedding with biologically-inspired patterns
        self._init_condition_patterns()
        
    def _init_condition_patterns(self):
        """Initialize condition embeddings with biologically-inspired patterns"""
        with torch.no_grad():
            # Antimicrobial: positive charge and amphipathic pattern
            self.condition_embedding.weight[0].normal_(0.5, 0.1)  # Positive bias for cationic AMPs
            
            # Antifungal: balanced amphipathic pattern  
            self.condition_embedding.weight[1].normal_(0.0, 0.15)  # Neutral with variability
            
            # Antiviral: hydrophobic-positive pattern with larger peptides
            self.condition_embedding.weight[2].normal_(-0.2, 0.12)  # Slight hydrophobic bias
            
            # Unconditioned: neutral baseline
            self.condition_embedding.weight[3].zero_()
    
    def forward(self, condition_indices: torch.Tensor, strength_modifier: Optional[torch.Tensor] = None) -> dict:
        """
        Generate comprehensive adaptive conditioning signals
        
        Args:
            condition_indices: Peptide type indices (batch_size,)
            strength_modifier: Optional strength control (batch_size, 1)
            
        Returns:
            Dictionary of conditioning signals for different aspects
        """
        # Basic condition embedding
        cond_emb = self.condition_embedding(condition_indices)
        
        # Enhance condition representation
        enhanced_cond = self.condition_enhancer(cond_emb)
        
        # Apply strength modulation if provided
        if strength_modifier is not None:
            strength = torch.sigmoid(self.strength_control(enhanced_cond))
            enhanced_cond = enhanced_cond * (strength * strength_modifier.unsqueeze(-1))
        
        # Generate multi-aspect conditioning signals
        conditioning_signals = {
            'enhanced_condition': enhanced_cond,
            'charge_signal': torch.tanh(self.charge_control(enhanced_cond)),
            'hydrophobic_signal': torch.tanh(self.hydrophobic_control(enhanced_cond)),
            'structure_signal': torch.tanh(self.structure_control(enhanced_cond)),
            'functional_signal': torch.tanh(self.functional_control(enhanced_cond)),
            'global_modulation': self.specificity_control(enhanced_cond)
        }
        
        return conditioning_signals
    
    def get_adaptive_scale_and_bias(self, conditioning_signals: dict) -> tuple:
        """
        Generate adaptive scale and bias for layer normalization
        
        Args:
            conditioning_signals: Dictionary from forward() method
            
        Returns:
            scale and bias tensors for adaptive normalization
        """
        # Combine multi-aspect signals
        combined_signal = torch.cat([
            conditioning_signals['charge_signal'],
            conditioning_signals['hydrophobic_signal'], 
            conditioning_signals['structure_signal'],
            conditioning_signals['functional_signal']
        ], dim=-1)
        
        # Ensure correct dimension
        if combined_signal.size(-1) != self.hidden_dim:
            # Project to correct dimension
            if not hasattr(self, '_signal_proj'):
                self._signal_proj = nn.Linear(combined_signal.size(-1), self.hidden_dim).to(combined_signal.device)
            combined_signal = self._signal_proj(combined_signal)
        
        # Split into scale and bias
        mid_dim = self.hidden_dim // 2
        scale_signal = combined_signal[:, :mid_dim]
        bias_signal = combined_signal[:, mid_dim:mid_dim*2] if combined_signal.size(1) >= mid_dim*2 else combined_signal[:, :mid_dim]
        
        # Pad if necessary
        if scale_signal.size(1) < self.hidden_dim:
            scale_signal = torch.cat([scale_signal, scale_signal[:, :self.hidden_dim - scale_signal.size(1)]], dim=1)
        if bias_signal.size(1) < self.hidden_dim:
            bias_signal = torch.cat([bias_signal, bias_signal[:, :self.hidden_dim - bias_signal.size(1)]], dim=1)
        
        # Apply activation functions
        scale = torch.sigmoid(scale_signal)  # 0-1 range for multiplicative scaling
        bias = bias_signal * 0.1  # Small additive bias
        
        return scale, bias


class AF3EnhancedConditionalLayerNorm(nn.Module):
    """
    Enhanced AlphaFold3-style conditional layer normalization
    Uses multi-aspect conditioning for fine-grained control
    """
    
    def __init__(self, hidden_dim: int, condition_dim: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        
        # Standard layer norm (without learnable parameters)
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        
        # Base learnable parameters
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # Multi-aspect condition modulation
        self.charge_modulation = nn.Linear(condition_dim, hidden_dim)
        self.hydrophobic_modulation = nn.Linear(condition_dim, hidden_dim)
        self.functional_modulation = nn.Linear(condition_dim, hidden_dim)
        
        # Global condition control
        self.global_scale = nn.Linear(condition_dim, hidden_dim)
        self.global_bias = nn.Linear(condition_dim, hidden_dim)
        
        # Initialize condition networks for stability
        for module in [self.charge_modulation, self.hydrophobic_modulation, 
                      self.functional_modulation, self.global_scale, self.global_bias]:
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, conditioning_signals: Optional[dict] = None) -> torch.Tensor:
        """
        Apply enhanced adaptive layer normalization
        
        Args:
            x: Input tensor (batch_size, seq_len, hidden_dim)
            conditioning_signals: Multi-aspect conditioning from AF3AdaptiveConditioning
            
        Returns:
            Normalized tensor with adaptive modulation
        """
        # Standard layer normalization
        normalized = self.layer_norm(x)
        
        # Apply base parameters
        out = self.weight * normalized + self.bias
        
        if conditioning_signals is not None:
            enhanced_cond = conditioning_signals['enhanced_condition']
            
            # Multi-aspect modulation
            charge_mod = self.charge_modulation(enhanced_cond)
            hydrophobic_mod = self.hydrophobic_modulation(enhanced_cond)
            functional_mod = self.functional_modulation(enhanced_cond)
            
            # Global control
            global_scale = torch.sigmoid(self.global_scale(enhanced_cond))
            global_bias = self.global_bias(enhanced_cond)
            
            # Expand for broadcasting with sequence dimension
            if x.dim() == 3:  # (batch, seq_len, hidden_dim)
                charge_mod = charge_mod.unsqueeze(1)
                hydrophobic_mod = hydrophobic_mod.unsqueeze(1) 
                functional_mod = functional_mod.unsqueeze(1)
                global_scale = global_scale.unsqueeze(1)
                global_bias = global_bias.unsqueeze(1)
            
            # Apply multi-level conditioning
            aspect_modulation = (charge_mod + hydrophobic_mod + functional_mod) * 0.1
            out = global_scale * (out + aspect_modulation) + global_bias
        
        return out


class AF3ConditionalZeroInit(nn.Module):
    """
    Enhanced AlphaFold3-style conditional output layer with zero initialization
    Includes peptide-type specific initialization patterns
    """
    
    def __init__(self, input_dim: int, output_dim: int, condition_dim: int):
        super().__init__()
        
        # Main output projection (initialized normally)
        self.output_proj = nn.Linear(input_dim, output_dim)
        
        # Multi-level condition-dependent gating
        self.primary_gate = nn.Linear(condition_dim, output_dim)
        self.fine_gate = nn.Linear(condition_dim, output_dim)
        
        # Peptide-type specific bias
        self.type_specific_bias = nn.Linear(condition_dim, output_dim)
        
        # Initialize for stable training
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # Zero-init gating with different strategies
        nn.init.zeros_(self.primary_gate.weight)
        nn.init.constant_(self.primary_gate.bias, -2.0)  # Primary gate: small opening
        
        nn.init.zeros_(self.fine_gate.weight)  
        nn.init.constant_(self.fine_gate.bias, -3.0)  # Fine gate: very small opening
        
        nn.init.zeros_(self.type_specific_bias.weight)
        nn.init.zeros_(self.type_specific_bias.bias)
    
    def forward(self, x: torch.Tensor, conditioning_signals: Optional[dict] = None) -> torch.Tensor:
        """
        Apply enhanced conditional zero-initialized output
        
        Args:
            x: Input tensor
            conditioning_signals: Multi-aspect conditioning
            
        Returns:
            Conditionally gated output tensor
        """
        output = self.output_proj(x)
        
        if conditioning_signals is None:
            return output
        
        enhanced_cond = conditioning_signals['enhanced_condition']
        
        # Generate multi-level gates
        primary_gate = torch.sigmoid(self.primary_gate(enhanced_cond))
        fine_gate = torch.sigmoid(self.fine_gate(enhanced_cond))
        type_bias = self.type_specific_bias(enhanced_cond)
        
        # Expand gates for broadcasting if needed
        if x.dim() == 3 and enhanced_cond.dim() == 2:  # (batch, seq_len, dim)
            primary_gate = primary_gate.unsqueeze(1)
            fine_gate = fine_gate.unsqueeze(1)
            type_bias = type_bias.unsqueeze(1)
        
        # Apply hierarchical gating and bias
        gated_output = primary_gate * fine_gate * output + type_bias
        
        return gated_output