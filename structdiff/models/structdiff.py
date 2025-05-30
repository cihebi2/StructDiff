import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from einops import rearrange

from .denoiser import StructureAwareDenoiser
from .structure_encoder import MultiScaleStructureEncoder
from ..diffusion.gaussian_diffusion import GaussianDiffusion
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StructDiff(nn.Module):
    """
    Structure-Aware Diffusion Model for Peptide Generation
    
    This model combines sequence and structure information through a dual-stream
    architecture with cross-modal attention for generating functional peptides.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Initialize sequence encoder (ESM-2)
        self._init_sequence_encoder()
        
        # Initialize structure encoder
        self.structure_encoder = MultiScaleStructureEncoder(
            config.model.structure_encoder
        )
        
        # Initialize denoiser with cross-attention
        self.denoiser = StructureAwareDenoiser(
            seq_hidden_dim=config.model.sequence_encoder.hidden_dim,
            struct_hidden_dim=config.model.structure_encoder.hidden_dim,
            denoiser_config=config.model.denoiser
        )
        
        # Initialize diffusion process
        self.diffusion = GaussianDiffusion(
            num_timesteps=config.model.diffusion.num_timesteps,
            noise_schedule=config.model.diffusion.noise_schedule,
            beta_start=config.model.diffusion.beta_start,
            beta_end=config.model.diffusion.beta_end
        )
        
        # Loss weights
        self.loss_weights = config.training.loss_weights
        
    def _init_sequence_encoder(self):
        """Initialize ESM-2 sequence encoder"""
        from transformers import AutoModel
        
        self.sequence_encoder = AutoModel.from_pretrained(
            self.config.model.sequence_encoder.path,
            trust_remote_code=True
        )
        
        if self.config.model.sequence_encoder.freeze:
            for param in self.sequence_encoder.parameters():
                param.requires_grad = False
                
        logger.info(f"Initialized sequence encoder: {self.config.model.sequence_encoder.name}")
    
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
        Forward pass of StructDiff
        
        Args:
            sequences: Token ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            timesteps: Diffusion timesteps of shape (batch_size,)
            structures: Optional structure features
            conditions: Optional conditioning information
            return_loss: Whether to compute and return loss
            
        Returns:
            Dictionary containing predictions and optionally losses
        """
        batch_size = sequences.shape[0]
        device = sequences.device
        
        # Get sequence embeddings
        with torch.no_grad() if self.config.model.sequence_encoder.freeze else torch.enable_grad():
            seq_embeddings = self.sequence_encoder(
                input_ids=sequences,
                attention_mask=attention_mask
            ).last_hidden_state  # (B, L, D_seq)
        
        # Add noise for diffusion training
        if return_loss:
            noise = torch.randn_like(seq_embeddings)
            noisy_embeddings, noise_level = self.diffusion.add_noise(
                seq_embeddings, timesteps, noise
            )
        else:
            noisy_embeddings = seq_embeddings
            noise = None
            noise_level = None
        
        # Extract structure features if available
        structure_features = None
        if structures is not None:
            structure_features = self.structure_encoder(
                structures, attention_mask
            )  # (B, L, D_struct)
        
        # Denoise with structure guidance
        denoised_embeddings, cross_attention_weights = self.denoiser(
            noisy_embeddings=noisy_embeddings,
            timesteps=timesteps,
            attention_mask=attention_mask,
            structure_features=structure_features,
            conditions=conditions
        )
        
        outputs = {
            'denoised_embeddings': denoised_embeddings,
            'cross_attention_weights': cross_attention_weights
        }
        
        # Compute losses if in training mode
        if return_loss and noise is not None:
            losses = self._compute_losses(
                denoised_embeddings=denoised_embeddings,
                target_embeddings=seq_embeddings,
                noise=noise,
                structure_features=structure_features,
                attention_mask=attention_mask
            )
            outputs.update(losses)
        
        return outputs
    
    def _compute_losses(
        self,
        denoised_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        noise: torch.Tensor,
        structure_features: Optional[torch.Tensor],
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-scale losses"""
        losses = {}
        
        # Sequence reconstruction loss
        seq_loss = F.mse_loss(
            denoised_embeddings[attention_mask.bool()],
            target_embeddings[attention_mask.bool()]
        )
        losses['sequence_loss'] = seq_loss * self.loss_weights.sequence
        
        # Structure consistency loss (if structure features available)
        if structure_features is not None:
            # Predict structure from denoised embeddings
            predicted_structure = self.structure_encoder.predict_from_sequence(
                denoised_embeddings, attention_mask
            )
            
            struct_loss = F.mse_loss(
                predicted_structure[attention_mask.bool()],
                structure_features[attention_mask.bool()]
            )
            losses['structure_loss'] = struct_loss * self.loss_weights.structure
        
        # Total loss
        losses['total_loss'] = sum(losses.values())
        
        return losses
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_length: int,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        target_structure: Optional[str] = None,
        guidance_scale: float = 1.0,
        return_trajectory: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Sample peptide sequences using the diffusion process
        
        Args:
            batch_size: Number of sequences to generate
            seq_length: Length of sequences to generate
            conditions: Optional conditioning information
            target_structure: Optional target secondary structure string
            guidance_scale: Scale for classifier-free guidance
            return_trajectory: Whether to return the full denoising trajectory
            
        Returns:
            Dictionary containing generated sequences and metadata
        """
        device = next(self.parameters()).device
        
        # Initialize from noise
        shape = (batch_size, seq_length + 2, self.config.model.sequence_encoder.hidden_dim)
        x_t = torch.randn(shape, device=device)
        
        # Create attention mask
        attention_mask = torch.ones(batch_size, seq_length + 2, device=device)
        
        # Parse target structure if provided
        structure_features = None
        if target_structure is not None:
            structure_features = self._parse_target_structure(
                target_structure, seq_length, batch_size, device
            )
        
        # Sampling loop
        trajectory = []
        for t in reversed(range(self.diffusion.num_timesteps)):
            timesteps = torch.full((batch_size,), t, device=device)
            
            # Denoise step
            with torch.cuda.amp.autocast():
                model_output = self.forward(
                    sequences=None,  # Not needed for sampling
                    attention_mask=attention_mask,
                    timesteps=timesteps,
                    structures={'features': structure_features} if structure_features is not None else None,
                    conditions=conditions,
                    return_loss=False
                )
            
            # Update x_t
            x_t = self.diffusion.denoise_step(
                x_t,
                t,
                model_output['denoised_embeddings'],
                guidance_scale=guidance_scale
            )
            
            if return_trajectory:
                trajectory.append(x_t.clone())
        
        # Decode to sequences
        sequences = self._decode_embeddings(x_t, attention_mask)
        
        outputs = {
            'sequences': sequences,
            'embeddings': x_t,
            'attention_mask': attention_mask
        }
        
        if return_trajectory:
            outputs['trajectory'] = torch.stack(trajectory)
        
        return outputs
    
    def _parse_target_structure(
        self,
        structure_string: str,
        seq_length: int,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Parse secondary structure string to features"""
        # Convert structure string (H: helix, E: sheet, C: coil) to one-hot
        structure_map = {'H': 0, 'E': 1, 'C': 2}
        structure_indices = [structure_map.get(s, 2) for s in structure_string]
        
        # Pad or truncate to match sequence length
        if len(structure_indices) < seq_length:
            structure_indices.extend([2] * (seq_length - len(structure_indices)))
        else:
            structure_indices = structure_indices[:seq_length]
        
        # Convert to one-hot and repeat for batch
        structure_tensor = F.one_hot(
            torch.tensor(structure_indices, device=device),
            num_classes=3
        ).float()
        
        return structure_tensor.unsqueeze(0).repeat(batch_size, 1, 1)
    
    def _decode_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[str]:
        """Decode embeddings back to sequences"""
        # This would use the ESM tokenizer and decoder
        # Placeholder implementation
        return ["GENERATED_SEQUENCE"] * embeddings.shape[0]