import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from einops import rearrange

from .denoise import StructureAwareDenoiser
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
        
        # Initialize sequence encoder (ESM-2) first to get hidden dim
        self._init_sequence_encoder()
        
        # Initialize structure encoder
        self.structure_encoder = MultiScaleStructureEncoder(
            config.model.structure_encoder
        )
        
        # Initialize denoiser with cross-attention
        self.denoiser = StructureAwareDenoiser(
            seq_hidden_dim=self.seq_hidden_dim,
            struct_hidden_dim=config.model.structure_encoder.hidden_dim,
            denoiser_config=config.model.denoiser
        )
        
        # Initialize diffusion process
        self.diffusion = GaussianDiffusion(
            num_timesteps=config.diffusion.num_timesteps,
            noise_schedule=config.diffusion.noise_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end
        )
        
        # Loss weights - handle both old and new config formats
        if hasattr(config, 'training_config') and hasattr(config.training_config, 'loss_weights'):
            self.loss_weights = config.training_config.loss_weights
        else:
            # Default weights
            self.loss_weights = type('obj', (object,), {
                'diffusion_loss': 1.0,
                'structure_consistency_loss': 0.1,
                'auxiliary_loss': 0.01
            })()
        
        # Add a simple decoder for sequence generation
        self.sequence_decoder = nn.Linear(self.seq_hidden_dim, 20)  # 20 amino acids
        
    def _init_sequence_encoder(self):
        """Initialize ESM-2 sequence encoder"""
        from transformers import AutoModel, AutoTokenizer
        
        model_name = self.config.model.sequence_encoder.pretrained_model
        
        # Use a smaller ESM model for testing if the specified one doesn't exist
        try:
            self.sequence_encoder = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            # Fallback to a known working model
            logger.warning(f"Could not load {model_name}, using facebook/esm2_t6_8M_UR50D instead")
            model_name = "facebook/esm2_t6_8M_UR50D"
            self.sequence_encoder = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get actual hidden dimension
        self.seq_hidden_dim = self.sequence_encoder.config.hidden_size
        
        # Apply LoRA if specified
        if self.config.model.sequence_encoder.get('use_lora', False):
            self._apply_lora()
        
        if self.config.model.sequence_encoder.freeze_encoder:
            for param in self.sequence_encoder.parameters():
                param.requires_grad = False
                
        logger.info(f"Initialized sequence encoder: {model_name}")
        logger.info(f"Sequence encoder hidden dim: {self.seq_hidden_dim}")
    
    def _apply_lora(self):
        """Apply LoRA to sequence encoder"""
        # Simplified LoRA implementation
        logger.info("Applying LoRA to sequence encoder")
        # This would need a proper LoRA implementation
        pass
    
    def forward(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        timesteps: torch.Tensor,
        structures: Optional[Dict[str, torch.Tensor]] = None,
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of StructDiff"""
        batch_size = sequences.shape[0]
        device = sequences.device
        
        # Get sequence embeddings
        with torch.no_grad() if self.config.model.sequence_encoder.freeze_encoder else torch.enable_grad():
            seq_outputs = self.sequence_encoder(
                input_ids=sequences,
                attention_mask=attention_mask
            )
            seq_embeddings = seq_outputs.last_hidden_state  # (B, L, D_seq)
        
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
        
        # Main diffusion loss - predict noise
        mask_expanded = attention_mask.bool().unsqueeze(-1).expand_as(denoised_embeddings)
        diffusion_loss = F.mse_loss(
            denoised_embeddings[mask_expanded],
            target_embeddings[mask_expanded]
        )
        losses['diffusion_loss'] = diffusion_loss * self.loss_weights.diffusion_loss
        
        # Structure consistency loss (if structure features available)
        if structure_features is not None and hasattr(self.loss_weights, 'structure_consistency_loss'):
            # Predict structure from denoised embeddings
            predicted_structure = self.structure_encoder.predict_from_sequence(
                denoised_embeddings, attention_mask
            )
            
            mask_expanded_struct = attention_mask.bool().unsqueeze(-1).expand_as(predicted_structure)
            struct_loss = F.mse_loss(
                predicted_structure[mask_expanded_struct],
                structure_features[mask_expanded_struct]
            )
            losses['structure_loss'] = struct_loss * self.loss_weights.structure_consistency_loss
        
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
        
        # Initialize from noise (add 2 for CLS/SEP tokens)
        shape = (batch_size, seq_length + 2, self.seq_hidden_dim)
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
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Create dummy sequences for the forward pass
            dummy_sequences = torch.zeros(batch_size, seq_length + 2, dtype=torch.long, device=device)
            
            # Denoise step
            with torch.cuda.amp.autocast(enabled=False):  # Disable for stability
                # Pass the noisy embeddings through the denoiser
                denoised_embeddings, _ = self.denoiser(
                    noisy_embeddings=x_t,
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                    structure_features=structure_features,
                    conditions=conditions
                )
            
            # Update x_t using the diffusion process
            x_t = self.diffusion.p_sample(
                denoised_embeddings,
                x_t,
                timesteps,
                clip_denoised=True
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
        
        # Pad or truncate to match sequence length (+ 2 for CLS/SEP)
        if len(structure_indices) < seq_length + 2:
            structure_indices.extend([2] * (seq_length + 2 - len(structure_indices)))
        else:
            structure_indices = structure_indices[:seq_length + 2]
        
        # Convert to tensor
        structure_tensor = torch.tensor(structure_indices, device=device)
        
        # Create feature tensor (simplified - just use embedding)
        structure_features = F.one_hot(structure_tensor, num_classes=3).float()
        
        # Expand to match hidden dimension
        structure_features = structure_features.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Project to structure encoder hidden dim
        structure_features = F.pad(
            structure_features, 
            (0, self.config.model.structure_encoder.hidden_dim - 3),
            value=0
        )
        
        return structure_features
    
    def _decode_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[str]:
        """Decode embeddings back to sequences"""
        batch_size = embeddings.shape[0]
        
        # Use the sequence decoder to get amino acid logits
        logits = self.sequence_decoder(embeddings)  # (B, L, 20)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)  # (B, L)
        
        # Convert to amino acid sequences
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        sequences = []
        
        for i in range(batch_size):
            # Get valid length (excluding padding)
            valid_length = int(attention_mask[i].sum().item())
            
            # Skip CLS/SEP tokens
            seq_indices = predictions[i, 1:valid_length-1].cpu().numpy()
            
            # Convert to string
            sequence = ''.join([amino_acids[idx] for idx in seq_indices])
            sequences.append(sequence)
        
        return sequences
# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04
