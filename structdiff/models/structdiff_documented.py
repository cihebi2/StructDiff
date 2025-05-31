# structdiff/models/structdiff_documented.py
"""
StructDiff: Structure-Aware Diffusion Model for Peptide Generation

This module implements the core StructDiff model, which combines sequence and 
structure information through a dual-stream architecture with cross-modal 
attention for generating functional peptides.

The model consists of:
1. A pre-trained sequence encoder (ESM-2) for extracting sequence features
2. A multi-scale structure encoder for processing 3D structural information
3. A denoising network with cross-attention between sequence and structure
4. A Gaussian diffusion process for iterative generation

Example:
    Basic usage for training::
    
        >>> from structdiff import StructDiff
        >>> from omegaconf import OmegaConf
        >>> 
        >>> config = OmegaConf.load("configs/default.yaml")
        >>> model = StructDiff(config)
        >>> 
        >>> # Training step
        >>> outputs = model(
        ...     sequences=batch['sequences'],
        ...     attention_mask=batch['attention_mask'],
        ...     timesteps=timesteps,
        ...     structures=batch['structures'],
        ...     return_loss=True
        ... )
        >>> loss = outputs['total_loss']
        >>> loss.backward()
    
    Generation example::
    
        >>> # Generate antimicrobial peptides
        >>> with torch.no_grad():
        ...     samples = model.sample(
        ...         batch_size=10,
        ...         seq_length=20,
        ...         conditions={'peptide_type': torch.tensor([0] * 10)},
        ...         guidance_scale=2.0
        ...     )
        >>> sequences = samples['sequences']

Attributes:
    sequence_encoder (nn.Module): Pre-trained ESM-2 model for sequence encoding
    structure_encoder (MultiScaleStructureEncoder): Structure feature extractor
    denoiser (StructureAwareDenoiser): Denoising network with cross-attention
    diffusion (GaussianDiffusion): Diffusion process handler
    
References:
    - ESM-2: Lin et al., "Language models of protein sequences at the scale 
      of evolution enable accurate structure prediction" (2022)
    - Diffusion Models: Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
    - Cross-attention: Vaswani et al., "Attention Is All You Need" (2017)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass
import warnings

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StructDiffConfig:
    """
    Configuration for StructDiff model.
    
    Attributes:
        sequence_encoder_name (str): Name of pre-trained sequence encoder
            Default: "facebook/esm2_t33_650M_UR50D"
        freeze_encoder (bool): Whether to freeze sequence encoder weights
            Default: False
        use_lora (bool): Whether to use LoRA for efficient fine-tuning
            Default: True
        structure_encoder_config (Dict): Configuration for structure encoder
        denoiser_config (Dict): Configuration for denoising network
        diffusion_config (Dict): Configuration for diffusion process
        
    Example:
        >>> config = StructDiffConfig(
        ...     sequence_encoder_name="facebook/esm2_t6_8M_UR50D",
        ...     freeze_encoder=True,
        ...     use_lora=True
        ... )
    """
    sequence_encoder_name: str = "facebook/esm2_t33_650M_UR50D"
    freeze_encoder: bool = False
    use_lora: bool = True
    structure_encoder_config: Optional[Dict] = None
    denoiser_config: Optional[Dict] = None
    diffusion_config: Optional[Dict] = None


class StructDiff(nn.Module):
    """
    Structure-Aware Diffusion Model for peptide generation.
    
    This model generates peptide sequences by combining:
    - Pre-trained protein language models for sequence understanding
    - 3D structure information through multi-scale encoding
    - Cross-modal attention for sequence-structure interaction
    - Diffusion-based iterative refinement
    
    Args:
        config (Dict): Model configuration dictionary containing:
            - model: Model architecture settings
            - diffusion: Diffusion process settings
            - training_config: Training-specific settings
            
    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If pre-trained model cannot be loaded
        
    Note:
        The model requires significant GPU memory (>8GB recommended).
        For memory-constrained environments, consider using:
        - Gradient checkpointing
        - Mixed precision training
        - Smaller batch sizes
    """
    
    def __init__(self, config: Dict):
        """
        Initialize StructDiff model.
        
        The initialization process:
        1. Loads pre-trained sequence encoder (ESM-2)
        2. Initializes structure encoder with multi-scale architecture
        3. Creates denoising network with cross-attention
        4. Sets up Gaussian diffusion process
        
        Memory optimization tips:
            - Set config.model.sequence_encoder.freeze_encoder=True to save memory
            - Use config.model.sequence_encoder.use_lora=True for efficient fine-tuning
            - Enable gradient checkpointing with config.training_config.gradient_checkpointing=True
        """
        super().__init__()
        self.config = config
        
        # Initialize components
        self._init_sequence_encoder()
        self._init_structure_encoder()
        self._init_denoiser()
        self._init_diffusion()
        
        logger.info(f"Initialized StructDiff with {self.count_parameters():,} parameters")
    
    def forward(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        timesteps: torch.Tensor,
        structures: Optional[Dict[str, torch.Tensor]] = None,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of StructDiff model.
        
        This method performs the full forward pass including:
        1. Encoding sequences with pre-trained model
        2. Adding noise according to diffusion schedule
        3. Extracting structure features (if provided)
        4. Denoising with structure guidance
        5. Computing losses (if in training mode)
        
        Args:
            sequences (torch.Tensor): Tokenized sequences of shape (B, L)
                where B is batch size and L is sequence length
            attention_mask (torch.Tensor): Binary mask of shape (B, L)
                with 1 for valid positions and 0 for padding
            timesteps (torch.Tensor): Diffusion timesteps of shape (B,)
                sampled uniformly from [0, num_timesteps)
            structures (Optional[Dict[str, torch.Tensor]]): Structure features dict:
                - 'angles': Backbone angles (B, L, 3)
                - 'secondary_structure': SS assignments (B, L)
                - 'distance_matrix': Pairwise distances (B, L, L)
                - 'contact_map': Binary contacts (B, L, L)
                - 'plddt': Confidence scores (B, L)
            conditions (Optional[Dict[str, torch.Tensor]]): Conditioning info:
                - 'peptide_type': Peptide type labels (B,)
                - 'target_property': Target property values (B, D)
            return_loss (bool): Whether to compute and return losses
                Default: True
                
        Returns:
            Dict[str, torch.Tensor]: Output dictionary containing:
                - 'denoised_embeddings': Denoised sequence embeddings (B, L, D)
                - 'cross_attention_weights': Attention weights (B, H, L, L) or None
                - 'total_loss': Total training loss (scalar) if return_loss=True
                - 'diffusion_loss': Diffusion loss component if return_loss=True
                - 'structure_loss': Structure consistency loss if applicable
                
        Raises:
            RuntimeError: If CUDA out of memory (try reducing batch size)
            ValueError: If input shapes are incompatible
            
        Example:
            >>> # Training forward pass
            >>> outputs = model(
            ...     sequences=batch['sequences'],
            ...     attention_mask=batch['attention_mask'],
            ...     timesteps=torch.randint(0, 1000, (batch_size,)),
            ...     structures=batch['structures'],
            ...     return_loss=True
            ... )
            >>> loss = outputs['total_loss']
            
            >>> # Inference forward pass
            >>> with torch.no_grad():
            ...     outputs = model(
            ...         sequences=sequences,
            ...         attention_mask=mask,
            ...         timesteps=timesteps,
            ...         return_loss=False
            ...     )
            >>> embeddings = outputs['denoised_embeddings']
            
        Note:
            During training, noise is added to the sequence embeddings
            according to the diffusion schedule. The model learns to
            denoise these embeddings conditioned on structure information.
        """
        # Implementation details...
        pass
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_length: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        target_structure: Optional[str] = None,
        guidance_scale: float = 1.0,
        sampling_method: str = "ddpm",
        num_inference_steps: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        return_trajectory: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, Union[List[str], torch.Tensor]]:
        """
        Generate peptide sequences using the diffusion model.
        
        This method implements the reverse diffusion process to generate
        new peptide sequences from random noise. The generation can be
        conditioned on various factors like peptide type or target structure.
        
        Args:
            batch_size (int): Number of sequences to generate in parallel
            seq_length (int): Length of sequences to generate (excluding special tokens)
            conditions (Optional[Dict[str, torch.Tensor]]): Conditioning information:
                - 'peptide_type': Tensor of shape (batch_size,) with type indices
                  (0: antimicrobial, 1: antifungal, 2: antiviral)
                - 'target_property': Tensor of shape (batch_size, D) with property values
            target_structure (Optional[str]): Target secondary structure string
                using 3-letter code: 'H' (helix), 'E' (sheet), 'C' (coil)
                Example: "HHHHHCCCEEEECCCC"
            guidance_scale (float): Classifier-free guidance scale (>1 for stronger guidance)
                Default: 1.0 (no guidance)
                Recommended: 2.0-3.0 for conditional generation
            sampling_method (str): Sampling algorithm to use
                Options: "ddpm" (default), "ddim" (faster), "pndm" (fastest)
            num_inference_steps (Optional[int]): Number of denoising steps
                Default: num_timesteps for DDPM, 50 for DDIM/PNDM
            temperature (float): Sampling temperature (higher = more diverse)
                Default: 1.0
            top_k (Optional[int]): Top-k filtering for discrete sampling
            top_p (Optional[float]): Nucleus (top-p) filtering
            return_trajectory (bool): Whether to return the full denoising trajectory
                Default: False (saves memory)
            progress_bar (bool): Whether to show progress bar during generation
                Default: True
                
        Returns:
            Dict containing:
                - 'sequences' (List[str]): Generated amino acid sequences
                - 'embeddings' (torch.Tensor): Final embeddings (batch_size, seq_length+2, D)
                - 'attention_mask' (torch.Tensor): Attention mask (batch_size, seq_length+2)
                - 'trajectory' (Optional[torch.Tensor]): Full denoising trajectory
                  of shape (num_steps, batch_size, seq_length+2, D) if requested
                - 'scores' (torch.Tensor): Confidence scores for each sequence (batch_size,)
                
        Raises:
            ValueError: If sampling_method is not recognized
            RuntimeError: If CUDA out of memory
            
        Example:
            >>> # Generate 10 antimicrobial peptides of length 20
            >>> samples = model.sample(
            ...     batch_size=10,
            ...     seq_length=20,
            ...     conditions={'peptide_type': torch.tensor([0] * 10)},
            ...     guidance_scale=2.0,
            ...     temperature=0.8
            ... )
            >>> sequences = samples['sequences']
            >>> print(sequences[0])  # e.g., "KRRWKWFKKLLKWFKKLLK"
            
            >>> # Generate with target secondary structure
            >>> samples = model.sample(
            ...     batch_size=5,
            ...     seq_length=15,
            ...     target_structure="HHHHHCCCCEEEEE",
            ...     guidance_scale=3.0
            ... )
            
            >>> # Fast sampling with DDIM
            >>> samples = model.sample(
            ...     batch_size=100,
            ...     seq_length=25,
            ...     sampling_method="ddim",
            ...     num_inference_steps=50  # 20x faster than DDPM
            ... )
            
        Note:
            - Generation time scales linearly with batch_size and num_inference_steps
            - Memory usage scales with batch_size * seq_length
            - For best quality, use DDPM with full timesteps
            - For faster generation, use DDIM or PNDM with fewer steps
            
        See Also:
            - :meth:`sample_with_constraints`: Generate with sequence constraints
            - :meth:`sample_diverse`: Generate diverse sequences with diversity penalty
        """
        # Implementation details...
        pass
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.
        
        Args:
            trainable_only (bool): Whether to count only trainable parameters
                Default: True
                
        Returns:
            int: Number of parameters
            
        Example:
            >>> total_params = model.count_parameters(trainable_only=False)
            >>> trainable_params = model.count_parameters(trainable_only=True)
            >>> print(f"Trainable: {trainable_params:,} / Total: {total_params:,}")
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_model(
    config_path: str,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    compile_model: bool = False
) -> StructDiff:
    """
    Create and optionally load a StructDiff model.
    
    This is a convenience function that handles:
    1. Loading configuration from YAML file
    2. Creating model instance
    3. Loading pre-trained weights (if provided)
    4. Moving model to specified device
    5. Optionally compiling with torch.compile (PyTorch 2.0+)
    
    Args:
        config_path (str): Path to configuration YAML file
        checkpoint_path (Optional[str]): Path to checkpoint file
            If provided, loads pre-trained weights
        device (Optional[torch.device]): Device to place model on
            Default: CUDA if available, else CPU
        compile_model (bool): Whether to compile model with torch.compile
            Default: False (requires PyTorch 2.0+)
            
    Returns:
        StructDiff: Initialized model ready for training or inference
        
    Raises:
        FileNotFoundError: If config_path or checkpoint_path doesn't exist
        RuntimeError: If checkpoint is incompatible with configuration
        
    Example:
        >>> # Create new model
        >>> model = create_model("configs/default.yaml")
        
        >>> # Load pre-trained model
        >>> model = create_model(
        ...     "configs/default.yaml",
        ...     checkpoint_path="checkpoints/best_model.pth",
        ...     device=torch.device("cuda:0")
        ... )
        
        >>> # Create compiled model for faster inference
        >>> model = create_model(
        ...     "configs/inference.yaml",
        ...     checkpoint_path="checkpoints/model.pth",
        ...     compile_model=True  # ~2x faster inference
        ... )
        
    See Also:
        - :func:`load_checkpoint`: Load checkpoint with additional options
        - :class:`StructDiffConfig`: Configuration dataclass
    """
    from omegaconf import OmegaConf
    
    # Load config
    config = OmegaConf.load(config_path)
    
    # Create model
    model = StructDiff(config)
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Move to device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Compile if requested
    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(model)
        logger.info("Compiled model with torch.compile")
    
    return model
# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
