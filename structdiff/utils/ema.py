import torch
import torch.nn as nn
from typing import Optional


class EMA:
    """Exponential Moving Average for model parameters"""
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.decay = decay
        self.device = device
        
        # Create EMA parameters
        self.shadow_params = {}
        self.backup_params = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().detach()
                if device:
                    self.shadow_params[name] = self.shadow_params[name].to(device)
    
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
    
    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.backup_params[name] = param.data
                param.data = self.shadow_params[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup_params:
                param.data = self.backup_params[name]
        self.backup_params = {}
    
    def state_dict(self):
        """Get EMA state dict"""
        return {
            'shadow_params': self.shadow_params,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict):
        """Load EMA state dict"""
        self.shadow_params = state_dict['shadow_params']
        self.decay = state_dict['decay']