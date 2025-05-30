from .structdiff import StructDiff
from .denoise import StructureAwareDenoiser as Denoiser
from .structure_encoder import MultiScaleStructureEncoder as StructureEncoder
from .cross_attention import CrossModalAttention, BiDirectionalCrossAttention

# For backward compatibility
BaselineDiffusion = StructDiff  # Can implement a simpler version later

__all__ = [
    "StructDiff",
    "BaselineDiffusion",
    "Denoiser",
    "StructureEncoder",
    "CrossModalAttention",
    "BiDirectionalCrossAttention",
]