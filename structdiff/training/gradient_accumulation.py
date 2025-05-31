# structdiff/training/gradient_accumulation.py
import torch
import torch.nn as nn
from typing import Dict, Optional, Callable
from contextlib import contextmanager

from ..utils.logger import get_logger

logger = get_logger(__name__)


class GradientAccumulator:
    """
    Gradient accumulation manager for efficient training with large batch sizes
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int,
        max_grad_norm: Optional[float] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scaler = scaler
        
        self.step_count = 0
        self.accumulated_loss = 0.0
        
    def backward(self, loss: torch.Tensor) -> bool:
        """
        Perform backward pass with gradient accumulation
        
        Args:
            loss: Loss to backpropagate
            
        Returns:
            True if optimizer step was performed
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        self.accumulated_loss += loss.item()
        self.step_count += 1
        
        # Check if it's time to update
        if self.step_count % self.accumulation_steps == 0:
            self._optimizer_step()
            return True
        
        return False
    
    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping"""
        if self.scaler is not None:
            # Unscale gradients
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        if self.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
        else:
            grad_norm = self._compute_grad_norm()
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Log
        if self.step_count % (self.accumulation_steps * 10) == 0:
            avg_loss = self.accumulated_loss / self.accumulation_steps
            logger.info(f"Step {self.step_count}: loss={avg_loss:.4f}, grad_norm={grad_norm:.4f}")
        
        self.accumulated_loss = 0.0
    
    def _compute_grad_norm(self) -> float:
        """Compute gradient norm"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def zero_grad(self):
        """Zero gradients (only at the start of accumulation)"""
        if self.step_count % self.accumulation_steps == 0:
            self.optimizer.zero_grad()
    
    @contextmanager
    def accumulate(self):
        """Context manager for gradient accumulation"""
        # Set model to accumulate gradients
        self.zero_grad()
        yield self
        
        # Ensure final step is performed if needed
        if self.step_count % self.accumulation_steps != 0:
            self._optimizer_step()


# Update training loop to use gradient accumulation
def train_step_with_accumulation(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    accumulator: GradientAccumulator,
    config: Dict
) -> Dict[str, float]:
    """Single training step with gradient accumulation"""
    model.train()
    
    # Forward pass
    outputs = model(**batch, return_loss=True)
    loss = outputs['total_loss']
    
    # Backward pass with accumulation
    step_performed = accumulator.backward(loss)
    
    # Prepare metrics
    metrics = {
        'loss': loss.item(),
        'step_performed': step_performed
    }
    
    # Add other metrics from outputs
    for key, value in outputs.items():
        if key != 'total_loss' and isinstance(value, torch.Tensor):
            metrics[key] = value.item()
    
    return metrics


# Advanced gradient accumulation with dynamic batching
class DynamicGradientAccumulator(GradientAccumulator):
    """
    Dynamic gradient accumulation that adjusts based on memory usage
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        target_batch_size: int,
        min_accumulation_steps: int = 1,
        max_accumulation_steps: int = 8,
        memory_efficient: bool = True,
        **kwargs
    ):
        super().__init__(model, optimizer, min_accumulation_steps, **kwargs)
        
        self.target_batch_size = target_batch_size
        self.min_accumulation_steps = min_accumulation_steps
        self.max_accumulation_steps = max_accumulation_steps
        self.memory_efficient = memory_efficient
        
        # Track memory usage
        self.memory_stats = []
        
    def adjust_accumulation_steps(self, current_batch_size: int):
        """Dynamically adjust accumulation steps based on batch size"""
        # Calculate required accumulation steps
        required_steps = self.target_batch_size // current_batch_size
        
        # Clamp to allowed range
        self.accumulation_steps = max(
            self.min_accumulation_steps,
            min(required_steps, self.max_accumulation_steps)
        )
        
        logger.info(f"Adjusted accumulation steps to {self.accumulation_steps}")
    
    def backward(self, loss: torch.Tensor) -> bool:
        """Backward with memory tracking"""
        if self.memory_efficient and torch.cuda.is_available():
            # Track memory before backward
            memory_before = torch.cuda.memory_allocated()
        
        # Perform backward
        result = super().backward(loss)
        
        if self.memory_efficient and torch.cuda.is_available():
            # Track memory after backward
            memory_after = torch.cuda.memory_allocated()
            memory_used = memory_after - memory_before
            self.memory_stats.append(memory_used)
            
            # Adjust if memory usage is too high
            if len(self.memory_stats) > 10:
                avg_memory = sum(self.memory_stats[-10:]) / 10
                max_memory = torch.cuda.get_device_properties(0).total_memory
                
                if avg_memory > max_memory * 0.8:  # Using >80% memory
                    logger.warning("High memory usage detected, reducing batch size")
                    # Signal to reduce batch size
                    self.high_memory_usage = True
        
        return result


# Gradient checkpointing for memory efficiency
def apply_gradient_checkpointing(model: nn.Module, checkpoint_segments: int = 4):
    """
    Apply gradient checkpointing to transformer layers
    """
    def checkpoint_forward(module, segments):
        def custom_forward(*inputs):
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            # Checkpoint the forward pass
            return torch.utils.checkpoint.checkpoint(
                create_custom_forward(module),
                *inputs
            )
        return custom_forward
    
    # Apply to transformer layers
    if hasattr(model, 'denoiser'):
        for i, block in enumerate(model.denoiser.blocks):
            if i % checkpoint_segments == 0:
                block.forward = checkpoint_forward(block, checkpoint_segments)
                logger.info(f"Applied gradient checkpointing to block {i}")
    
    logger.info("Gradient checkpointing enabled")


# Memory-efficient training utilities
class MemoryEfficientTrainer:
    """
    Memory-efficient training with automatic optimization
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        auto_batch_size: bool = True,
        auto_accumulation: bool = True
    ):
        self.model = model
        self.config = config
        self.auto_batch_size = auto_batch_size
        self.auto_accumulation = auto_accumulation
        
        # Apply optimizations
        self._apply_memory_optimizations()
    
    def _apply_memory_optimizations(self):
        """Apply various memory optimizations"""
        # 1. Gradient checkpointing
        if self.config.get('gradient_checkpointing', False):
            apply_gradient_checkpointing(self.model)
        
        # 2. Mixed precision
        if self.config.get('use_amp', True):
            logger.info("Using automatic mixed precision")
        
        # 3. Efficient attention (if available)
        self._use_efficient_attention()
    
    def _use_efficient_attention(self):
        """Replace attention with memory-efficient implementation"""
        try:
            from torch.nn.functional import scaled_dot_product_attention
            
            # Replace attention in model
            for module in self.model.modules():
                if hasattr(module, 'attention'):
                    module._use_flash_attention = True
                    logger.info("Enabled Flash Attention")
        except ImportError:
            logger.info("Flash Attention not available")
    
    def find_optimal_batch_size(
        self,
        data_loader,
        min_batch_size: int = 1,
        max_batch_size: int = 128
    ) -> int:
        """Find optimal batch size through binary search"""
        logger.info("Finding optimal batch size...")
        
        def can_run_batch_size(batch_size):
            try:
                # Create dummy batch
                dummy_batch = next(iter(data_loader))
                
                # Adjust batch size
                for key, value in dummy_batch.items():
                    if isinstance(value, torch.Tensor):
                        dummy_batch[key] = value[:batch_size]
                
                # Try forward pass
                self.model.train()
                with torch.cuda.amp.autocast():
                    outputs = self.model(**dummy_batch, return_loss=True)
                    loss = outputs['total_loss']
                    loss.backward()
                
                # Clear
                self.model.zero_grad()
                torch.cuda.empty_cache()
                
                return True
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    return False
                raise e
        
        # Binary search
        optimal_batch_size = min_batch_size
        
        while min_batch_size <= max_batch_size:
            mid = (min_batch_size + max_batch_size) // 2
            
            if can_run_batch_size(mid):
                optimal_batch_size = mid
                min_batch_size = mid + 1
            else:
                max_batch_size = mid - 1
        
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size
# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04
