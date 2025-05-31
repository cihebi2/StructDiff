# structdiff/models/lora.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math

from ..utils.logger import get_logger

logger = get_logger(__name__)


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation"""
        # x: (batch_size, seq_len, in_features)
        dropout_x = self.lora_dropout(x)
        
        # Compute low-rank update
        # (batch_size, seq_len, rank)
        lora_out = torch.matmul(dropout_x, self.lora_A.T)
        # (batch_size, seq_len, out_features)
        lora_out = torch.matmul(lora_out, self.lora_B.T)
        
        return lora_out * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        self.original_layer = original_layer
        self.lora = LoRALayer(
            original_layer.in_features,
            original_layer.out_features,
            rank,
            alpha,
            dropout
        )
        
        # Freeze original parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA"""
        # Original forward pass
        out = self.original_layer(x)
        
        # Add LoRA adaptation
        out = out + self.lora(x)
        
        return out


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str] = None,
    rank: int = 16,
    alpha: float = 32,
    dropout: float = 0.1
) -> Dict[str, LoRALinear]:
    """
    Apply LoRA to specified modules in the model
    
    Args:
        model: Model to apply LoRA to
        target_modules: List of module names to target (e.g., ['q_proj', 'v_proj'])
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: Dropout rate
        
    Returns:
        Dictionary of LoRA modules
    """
    if target_modules is None:
        # Default: apply to attention projection layers
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    lora_modules = {}
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Create LoRA layer
                lora_layer = LoRALinear(module, rank, alpha, dropout)
                
                # Replace the original module
                parent_name, child_name = name.rsplit('.', 1)
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, lora_layer)
                
                lora_modules[name] = lora_layer
                logger.info(f"Applied LoRA to {name}")
    
    logger.info(f"Applied LoRA to {len(lora_modules)} modules")
    
    # Calculate trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")
    
    return lora_modules


def merge_lora_weights(model: nn.Module):
    """Merge LoRA weights back into the original model"""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Compute merged weights
            with torch.no_grad():
                delta_w = module.lora.scaling * torch.matmul(
                    module.lora.lora_B, module.lora.lora_A
                )
                module.original_layer.weight.data += delta_w
            
            # Replace with original layer
            parent_name, child_name = name.rsplit('.', 1)
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, child_name, module.original_layer)
            
            logger.info(f"Merged LoRA weights for {name}")


def save_lora_weights(
    model: nn.Module,
    save_path: str,
    additional_state: Optional[Dict] = None
):
    """Save only LoRA weights"""
    lora_state_dict = {}
    
    for name, param in model.named_parameters():
        if 'lora_' in name and param.requires_grad:
            lora_state_dict[name] = param.data
    
    save_dict = {
        'lora_weights': lora_state_dict,
        'model_type': type(model).__name__
    }
    
    if additional_state:
        save_dict.update(additional_state)
    
    torch.save(save_dict, save_path)
    logger.info(f"Saved LoRA weights to {save_path}")


def load_lora_weights(
    model: nn.Module,
    load_path: str,
    strict: bool = True
) -> Dict:
    """Load LoRA weights"""
    checkpoint = torch.load(load_path, map_location='cpu')
    lora_weights = checkpoint['lora_weights']
    
    # Load weights
    missing_keys = []
    unexpected_keys = []
    
    for name, param in model.named_parameters():
        if 'lora_' in name:
            if name in lora_weights:
                param.data = lora_weights[name]
            elif strict:
                missing_keys.append(name)
    
    for name in lora_weights:
        if name not in model.state_dict():
            unexpected_keys.append(name)
    
    if missing_keys or unexpected_keys:
        logger.warning(f"Missing keys: {missing_keys}")
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    logger.info(f"Loaded LoRA weights from {load_path}")
    
    return checkpoint


# Update StructDiff to use LoRA
class StructDiffWithLoRA(nn.Module):
    """StructDiff with LoRA support"""
    
    def __init__(self, config: Dict):
        super().__init__()
        # ... existing initialization ...
        
        # Apply LoRA if enabled
        if config.model.sequence_encoder.use_lora:
            self._apply_lora()
    
    def _apply_lora(self):
        """Apply LoRA to sequence encoder"""
        config = self.config.model.sequence_encoder
        
        # Target modules in ESM
        target_modules = [
            'encoder.layer.*.attention.self.query',
            'encoder.layer.*.attention.self.key',
            'encoder.layer.*.attention.self.value',
            'encoder.layer.*.attention.output.dense'
        ]
        
        # Apply LoRA
        self.lora_modules = apply_lora_to_model(
            self.sequence_encoder,
            target_modules=target_modules,
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout
        )
    
    def save_lora_checkpoint(self, save_path: str):
        """Save LoRA checkpoint"""
        save_lora_weights(
            self,
            save_path,
            additional_state={
                'config': self.config,
                'model_class': 'StructDiffWithLoRA'
            }
        )
    
    def load_lora_checkpoint(self, load_path: str):
        """Load LoRA checkpoint"""
        checkpoint = load_lora_weights(self, load_path)
        return checkpoint
# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04
