# structdiff/utils/memory_optimization.py
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any
import gc
import psutil
import GPUtil
from contextlib import contextmanager
from functools import wraps
import weakref

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MemoryMonitor:
    """Monitor and log memory usage"""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_stats = []
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage"""
        info = {}
        
        # CPU memory
        process = psutil.Process()
        info['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
        info['cpu_memory_percent'] = process.memory_percent()
        
        # GPU memory
        if self.device.type == 'cuda':
            gpu_id = self.device.index or 0
            gpus = GPUtil.getGPUs()
            if gpu_id < len(gpus):
                gpu = gpus[gpu_id]
                info['gpu_memory_mb'] = gpu.memoryUsed
                info['gpu_memory_percent'] = gpu.memoryUtil * 100
                info['gpu_memory_free_mb'] = gpu.memoryFree
            
            # PyTorch specific
            info['pytorch_allocated_mb'] = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            info['pytorch_reserved_mb'] = torch.cuda.memory_reserved(self.device) / 1024 / 1024
        
        return info
    
    def log_memory(self, tag: str = ""):
        """Log current memory usage"""
        info = self.get_memory_info()
        self.memory_stats.append({'tag': tag, **info})
        
        logger.info(f"Memory usage ({tag}):")
        for key, value in info.items():
            logger.info(f"  {key}: {value:.2f}")
    
    @contextmanager
    def track_memory(self, tag: str):
        """Context manager to track memory usage"""
        self.log_memory(f"{tag}_start")
        yield
        self.log_memory(f"{tag}_end")
        
        # Compute difference
        if len(self.memory_stats) >= 2:
            start = self.memory_stats[-2]
            end = self.memory_stats[-1]
            
            diff = {}
            for key in start:
                if key != 'tag' and key in end:
                    diff[key] = end[key] - start[key]
            
            logger.info(f"Memory change ({tag}):")
            for key, value in diff.items():
                if abs(value) > 0.1:
                    logger.info(f"  {key}: {value:+.2f}")


def optimize_model_memory(model: nn.Module) -> nn.Module:
    """Apply memory optimizations to model"""
    
    # 1. Enable gradient checkpointing for transformer blocks
    if hasattr(model, 'denoiser') and hasattr(model.denoiser, 'blocks'):
        for block in model.denoiser.blocks:
            block.gradient_checkpointing = True
    
    # 2. Use inplace operations where possible
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = True
        elif isinstance(module, nn.GELU):
            # Replace with memory-efficient version
            module.approximate = 'tanh'
    
    # 3. Share embeddings if possible
    if hasattr(model, 'sequence_encoder') and hasattr(model, 'sequence_decoder'):
        # Share embedding weights
        if hasattr(model.sequence_encoder, 'embeddings'):
            model.sequence_decoder.weight = model.sequence_encoder.embeddings.word_embeddings.weight
    
    logger.info("Applied memory optimizations to model")
    return model


class MemoryEfficientAttention(nn.Module):
    """Memory-efficient attention implementation"""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_flash_attention: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_flash_attention = use_flash_attention
        
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with memory optimization"""
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.use_flash_attention and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # Use Flash Attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Manual attention with memory optimization
            attn_output = self._manual_attention(q, k, v, attention_mask)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)
        
        return output
    
    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Manual attention with chunking for memory efficiency"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Chunk size for memory efficiency
        chunk_size = min(512, seq_len)
        
        output = torch.zeros_like(v)
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            q_chunk = q[:, :, i:end_i]
            
            attn_weights = []
            
            for j in range(0, seq_len, chunk_size):
                end_j = min(j + chunk_size, seq_len)
                k_chunk = k[:, :, j:end_j]
                
                # Compute attention scores for chunk
                scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / (head_dim ** 0.5)
                
                if mask is not None:
                    scores = scores.masked_fill(~mask[:, None, i:end_i, j:end_j], float('-inf'))
                
                attn_weights.append(scores)
            
            # Concatenate and apply softmax
            attn_weights = torch.cat(attn_weights, dim=-1)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values in chunks
            for j in range(0, seq_len, chunk_size):
                end_j = min(j + chunk_size, seq_len)
                v_chunk = v[:, :, j:end_j]
                weight_chunk = attn_weights[:, :, :, j:end_j]
                
                output[:, :, i:end_i] += torch.matmul(weight_chunk, v_chunk)
        
        return output


# Memory-efficient training utilities
@contextmanager
def memory_efficient_mode():
    """Context manager for memory-efficient operations"""
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Set memory-efficient flags
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    
    try:
        yield
    finally:
        # Restore settings
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        
        # Clear cache again
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def clear_memory():
    """Aggressively clear memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_model_memory_footprint(model: nn.Module) -> Dict[str, float]:
    """Calculate model memory footprint"""
    param_memory = 0
    buffer_memory = 0
    
    # Parameters
    for param in model.parameters():
        param_memory += param.numel() * param.element_size()
    
    # Buffers
    for buffer in model.buffers():
        buffer_memory += buffer.numel() * buffer.element_size()
    
    # Convert to MB
    param_memory_mb = param_memory / 1024 / 1024
    buffer_memory_mb = buffer_memory / 1024 / 1024
    total_memory_mb = param_memory_mb + buffer_memory_mb
    
    return {
        'param_memory_mb': param_memory_mb,
        'buffer_memory_mb': buffer_memory_mb,
        'total_memory_mb': total_memory_mb
    }


class GradientAccumulationOptimizer:
    """
    Optimizer wrapper for memory-efficient gradient accumulation
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int,
        model: nn.Module
    ):
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.model = model
        self.step_count = 0
        
        # Store gradients efficiently
        self.gradient_buffer = {}
        self._init_gradient_buffer()
    
    def _init_gradient_buffer(self):
        """Initialize gradient buffer"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.gradient_buffer[name] = torch.zeros_like(
                    param.data,
                    memory_format=torch.preserve_format
                )
    
    def accumulate_gradients(self):
        """Accumulate gradients into buffer"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.gradient_buffer[name].add_(param.grad.data)
                # Clear gradient to save memory
                param.grad = None
    
    def step(self):
        """Optimizer step with accumulated gradients"""
        self.step_count += 1
        
        if self.step_count % self.accumulation_steps == 0:
            # Copy accumulated gradients back
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.grad = self.gradient_buffer[name] / self.accumulation_steps
            
            # Optimizer step
            self.optimizer.step()
            
            # Clear buffers
            for buffer in self.gradient_buffer.values():
                buffer.zero_()
            
            return True
        
        return False
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad(set_to_none=True)  # More memory efficient


# Activation checkpointing decorator
def checkpoint_sequential(functions: List[Callable], segments: int):
    """
    Checkpoint sequential functions for memory efficiency
    """
    def forward(input):
        # Split into segments
        segment_size = len(functions) // segments
        
        def run_segment(start, end, input):
            for func in functions[start:end]:
                input = func(input)
            return input
        
        # Run with checkpointing
        for i in range(0, len(functions), segment_size):
            end = min(i + segment_size, len(functions))
            if i + segment_size < len(functions):
                # Checkpoint intermediate segments
                input = torch.utils.checkpoint.checkpoint(
                    run_segment, i, end, input
                )
            else:
                # Don't checkpoint last segment
                input = run_segment(i, end, input)
        
        return input
    
    return forward


# Memory-efficient data structures
class LazyTensor:
    """
    Lazy tensor that only materializes when needed
    """
    
    def __init__(self, shape: torch.Size, dtype: torch.dtype, device: torch.device):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._data = None
        self._generator = None
    
    def set_generator(self, generator: Callable):
        """Set generator function"""
        self._generator = generator
    
    @property
    def data(self) -> torch.Tensor:
        """Materialize tensor when accessed"""
        if self._data is None:
            if self._generator is not None:
                self._data = self._generator()
            else:
                self._data = torch.empty(self.shape, dtype=self.dtype, device=self.device)
        return self._data
    
    def __getattr__(self, name):
        """Forward attribute access to materialized tensor"""
        return getattr(self.data, name)


# Memory leak detection
class MemoryLeakDetector:
    """Detect potential memory leaks"""
    
    def __init__(self):
        self.objects = weakref.WeakValueDictionary()
        self.allocation_history = []
    
    def track_object(self, obj: Any, name: str):
        """Track object for leak detection"""
        self.objects[name] = obj
    
    def check_leaks(self) -> List[str]:
        """Check for potential leaks"""
        leaks = []
        
        # Check for retained objects
        for name, obj in self.objects.items():
            if obj is not None:
                leaks.append(f"Object '{name}' still in memory")
        
        # Check GPU memory growth
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            self.allocation_history.append(current_memory)
            
            if len(self.allocation_history) > 10:
                recent = self.allocation_history[-10:]
                if all(recent[i] <= recent[i+1] for i in range(9)):
                    leaks.append("Monotonic GPU memory growth detected")
        
        return leaks
# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
