import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable activation and dropout
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        use_bias: bool = True,
        layer_norm: bool = False
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Build layers
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim
                
            if i == num_layers - 1:
                out_dim = output_dim
            else:
                out_dim = hidden_dim
            
            # Linear layer
            layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
            
            # Add layer norm if not last layer
            if layer_norm and i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
            
            # Add activation if not last layer
            if i < num_layers - 1:
                layers.append(self.activation)
                layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.1)
        }
        
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        
        return activations[name]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.mlp(x)


class FeedForward(nn.Module):
    """
    Transformer-style feed-forward network with residual connection
    """
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_gate: bool = False
    ):
        super().__init__()
        
        intermediate_dim = intermediate_dim or 4 * hidden_dim
        
        self.use_gate = use_gate
        
        if use_gate:
            # Gated Linear Unit variant
            self.w1 = nn.Linear(hidden_dim, intermediate_dim * 2)
            self.w2 = nn.Linear(intermediate_dim, hidden_dim)
        else:
            # Standard FFN
            self.w1 = nn.Linear(hidden_dim, intermediate_dim)
            self.w2 = nn.Linear(intermediate_dim, hidden_dim)
        
        self.activation = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def _get_activation(self, name: str) -> Callable:
        """Get activation function"""
        activations = {
            'relu': F.relu,
            'gelu': F.gelu,
            'silu': F.silu,
            'tanh': torch.tanh,
            'sigmoid': torch.sigmoid
        }
        
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}")
        
        return activations[name]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Output tensor of same shape
        """
        residual = x
        
        if self.use_gate:
            # Gated linear unit
            gate_and_value = self.w1(x)
            gate, value = gate_and_value.chunk(2, dim=-1)
            x = self.activation(gate) * value
        else:
            # Standard FFN
            x = self.activation(self.w1(x))
        
        x = self.dropout(x)
        x = self.w2(x)
        x = self.dropout(x)
        
        # Residual connection and layer norm
        x = self.layer_norm(x + residual)
        
        return x


class ConditionalMLP(nn.Module):
    """
    MLP with conditional input (e.g., for time/class conditioning)
    """
    
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.1,
        condition_method: str = "concat"  # concat, add, or film
    ):
        super().__init__()
        
        self.condition_method = condition_method
        
        # Adjust input dimension based on conditioning method
        if condition_method == "concat":
            mlp_input_dim = input_dim + condition_dim
        else:
            mlp_input_dim = input_dim
        
        # Main MLP
        self.mlp = MLP(
            mlp_input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            activation,
            dropout
        )
        
        # Conditioning layers for FiLM
        if condition_method == "film":
            self.scale_net = nn.Linear(condition_dim, input_dim)
            self.shift_net = nn.Linear(condition_dim, input_dim)
        elif condition_method == "add":
            self.condition_proj = nn.Linear(condition_dim, input_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with conditioning
        
        Args:
            x: Input features
            condition: Conditioning features
            
        Returns:
            Output features
        """
        if self.condition_method == "concat":
            # Concatenate condition
            x = torch.cat([x, condition], dim=-1)
            
        elif self.condition_method == "add":
            # Add projected condition
            x = x + self.condition_proj(condition)
            
        elif self.condition_method == "film":
            # Feature-wise Linear Modulation
            scale = self.scale_net(condition)
            shift = self.shift_net(condition)
            x = x * (1 + scale) + shift
        
        return self.mlp(x)
# Updated: 05/30/2025 22:59:09
