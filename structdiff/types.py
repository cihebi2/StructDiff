# structdiff/types.py
"""Type definitions for StructDiff"""

from typing import (
    Dict, List, Optional, Union, Tuple, Any, 
    TypedDict, Protocol, Literal, TypeVar, Generic,
    Callable, Iterator, Sequence, Mapping
)
from typing_extensions import NotRequired
import torch
import numpy as np
from pathlib import Path


# Type aliases
TensorDict = Dict[str, torch.Tensor]
ArrayDict = Dict[str, np.ndarray]
PathLike = Union[str, Path]
Device = Union[str, torch.device]

# Type variables
T = TypeVar('T')
ModelType = TypeVar('ModelType', bound='nn.Module')


# Typed dictionaries for structured data
class SequenceData(TypedDict):
    """Type for sequence data"""
    sequence: str
    tokens: torch.Tensor
    attention_mask: torch.Tensor
    label: NotRequired[int]
    metadata: NotRequired[Dict[str, Any]]


class StructureFeatures(TypedDict):
    """Type for structure features"""
    angles: torch.Tensor  # (L, 3) - phi, psi, omega
    secondary_structure: torch.Tensor  # (L,) - 0: helix, 1: sheet, 2: coil
    distance_matrix: torch.Tensor  # (L, L)
    contact_map: torch.Tensor  # (L, L)
    plddt: NotRequired[torch.Tensor]  # (L,) - confidence scores
    sasa: NotRequired[torch.Tensor]  # (L,) - solvent accessibility
    positions: NotRequired[torch.Tensor]  # (L, 3) - 3D coordinates


class BatchData(TypedDict):
    """Type for batched data"""
    sequences: torch.Tensor  # (B, L)
    attention_mask: torch.Tensor  # (B, L)
    structures: NotRequired[StructureFeatures]
    conditions: NotRequired[Dict[str, torch.Tensor]]
    labels: NotRequired[torch.Tensor]  # (B,)


class ModelOutput(TypedDict):
    """Type for model outputs"""
    denoised_embeddings: torch.Tensor  # (B, L, D)
    cross_attention_weights: NotRequired[torch.Tensor]  # (B, H, L, L)
    total_loss: NotRequired[torch.Tensor]
    diffusion_loss: NotRequired[torch.Tensor]
    structure_loss: NotRequired[torch.Tensor]


class GenerationOutput(TypedDict):
    """Type for generation outputs"""
    sequences: List[str]
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    trajectory: NotRequired[torch.Tensor]  # (T, B, L, D)
    scores: NotRequired[torch.Tensor]  # (B,)


# Literal types for configurations
NoiseSchedule = Literal["linear", "cosine", "sqrt"]
SamplingMethod = Literal["ddpm", "ddim", "pndm"]
PeptideType = Literal["antimicrobial", "antifungal", "antiviral"]
OptimizerType = Literal["adam", "adamw", "sgd", "rmsprop"]
SchedulerType = Literal["linear", "cosine", "polynomial", "constant"]


# Protocol for model interfaces
class DiffusionModel(Protocol):
    """Protocol for diffusion models"""
    
    def forward(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        timesteps: torch.Tensor,
        structures: Optional[StructureFeatures] = None,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        return_loss: bool = True
    ) -> ModelOutput:
        ...
    
    def sample(
        self,
        batch_size: int,
        seq_length: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        guidance_scale: float = 1.0
    ) -> GenerationOutput:
        ...


class StructureEncoder(Protocol):
    """Protocol for structure encoders"""
    
    def forward(
        self,
        structures: StructureFeatures,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        ...
    
    def predict_from_sequence(
        self,
        sequence_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        ...


# Generic types for flexibility
class DataLoader(Generic[T]):
    """Generic DataLoader type"""
    
    def __iter__(self) -> Iterator[T]:
        ...
    
    def __len__(self) -> int:
        ...


# Configuration types with validation
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Model configuration with validation"""
    
    hidden_dim: int = Field(768, ge=64, le=2048)
    num_layers: int = Field(12, ge=1, le=48)
    num_attention_heads: int = Field(12, ge=1, le=32)
    dropout: float = Field(0.1, ge=0.0, le=0.9)
    
    @validator('num_attention_heads')
    def heads_divisible(cls, v: int, values: Dict[str, Any]) -> int:
        """Ensure heads divide hidden_dim evenly"""
        if 'hidden_dim' in values:
            assert values['hidden_dim'] % v == 0, \
                f"hidden_dim ({values['hidden_dim']}) must be divisible by num_attention_heads ({v})"
        return v


class TrainingConfig(BaseModel):
    """Training configuration with validation"""
    
    num_epochs: int = Field(100, ge=1)
    batch_size: int = Field(32, ge=1)
    learning_rate: float = Field(1e-4, gt=0, le=1.0)
    gradient_clip: Optional[float] = Field(None, gt=0)
    accumulation_steps: int = Field(1, ge=1)
    
    optimizer: OptimizerType = "adamw"
    scheduler: SchedulerType = "cosine"
    
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = Field(0.9999, ge=0.9, le=0.99999)


class DiffusionConfig(BaseModel):
    """Diffusion configuration with validation"""
    
    num_timesteps: int = Field(1000, ge=10, le=10000)
    noise_schedule: NoiseSchedule = "sqrt"
    beta_start: float = Field(0.0001, gt=0, lt=0.1)
    beta_end: float = Field(0.02, gt=0, lt=0.5)
    
    @validator('beta_end')
    def beta_order(cls, v: float, values: Dict[str, Any]) -> float:
        """Ensure beta_end > beta_start"""
        if 'beta_start' in values:
            assert v > values['beta_start'], \
                f"beta_end ({v}) must be greater than beta_start ({values['beta_start']})"
        return v


# Enhanced type hints for existing functions
def train_model(
    model: DiffusionModel,
    train_loader: DataLoader[BatchData],
    val_loader: DataLoader[BatchData],
    config: TrainingConfig,
    callbacks: Optional[List[Callable[[int, Dict[str, float]], None]]] = None
) -> Tuple[DiffusionModel, Dict[str, List[float]]]:
    """
    Train diffusion model
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        callbacks: Optional callbacks called after each epoch
        
    Returns:
        Trained model and training history
    """
    ...


def generate_peptides(
    model: DiffusionModel,
    num_samples: int,
    peptide_type: Optional[PeptideType] = None,
    length_range: Tuple[int, int] = (10, 30),
    temperature: float = 1.0,
    guidance_scale: float = 1.0,
    return_scores: bool = False
) -> Union[List[str], Tuple[List[str], torch.Tensor]]:
    """
    Generate peptide sequences
    
    Args:
        model: Trained diffusion model
        num_samples: Number of sequences to generate
        peptide_type: Optional peptide type for conditional generation
        length_range: Range of sequence lengths
        temperature: Sampling temperature
        guidance_scale: Classifier-free guidance scale
        return_scores: Whether to return confidence scores
        
    Returns:
        Generated sequences and optionally their scores
    """
    ...


# Complex nested types
MetricsDict = Dict[str, Union[float, Dict[str, float]]]
LossDict = Dict[Literal["total", "diffusion", "structure", "auxiliary"], torch.Tensor]
OptimizerState = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]


# Callable types with detailed signatures
LossFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    Tuple[torch.Tensor, LossDict]
]

MetricFunction = Callable[
    [List[str], Optional[List[str]]],
    MetricsDict
]

SamplingFunction = Callable[
    [torch.Tensor, int, Optional[Dict[str, Any]]],
    torch.Tensor
]


# Example of using types in a class
class TypedStructDiff(nn.Module):
    """StructDiff with full type annotations"""
    
    def __init__(
        self,
        model_config: ModelConfig,
        diffusion_config: DiffusionConfig
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.diffusion_config = diffusion_config
        
        # Initialize components
        self.sequence_encoder: nn.Module
        self.structure_encoder: StructureEncoder
        self.denoiser: nn.Module
        self.diffusion: GaussianDiffusion
    
    def forward(
        self,
        batch: BatchData,
        timesteps: torch.Tensor,
        return_loss: bool = True
    ) -> ModelOutput:
        """Type-safe forward pass"""
        output: ModelOutput = {
            'denoised_embeddings': torch.empty(0)
        }
        
        # Implementation...
        
        return output
    
    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        length: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Any
    ) -> GenerationOutput:
        """Type-safe generation"""
        output: GenerationOutput = {
            'sequences': [],
            'embeddings': torch.empty(0),
            'attention_mask': torch.empty(0)
        }
        
        # Implementation...
        
        return output


# Type stubs for external libraries
# Create structdiff/py.typed to mark package as typed
# Create structdiff/stubs/ for external library stubs

# Example stub for transformers
# structdiff/stubs/transformers.pyi
"""
from typing import Dict, Optional, Tuple
import torch

class AutoModel:
    @staticmethod
    def from_pretrained(
        model_name: str,
        trust_remote_code: bool = False,
        **kwargs
    ) -> 'PreTrainedModel': ...

class PreTrainedModel:
    config: 'PretrainedConfig'
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> 'BaseModelOutput': ...

class BaseModelOutput:
    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]]
    attentions: Optional[Tuple[torch.Tensor, ...]]
"""
# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04
