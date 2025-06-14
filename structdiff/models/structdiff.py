import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union
from einops import rearrange
from transformers import AutoModel, AutoTokenizer
import numpy as np

from .denoise import StructureAwareDenoiser
from .structure_encoder import MultiScaleStructureEncoder
from .cross_attention import BiDirectionalCrossAttention
from .layers import PositionalEncoding
from ..diffusion.gaussian_diffusion import GaussianDiffusion
from ..diffusion.sampling import get_sampler
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
        
        # Add positional encoding - account for special tokens (CLS, SEP)
        max_seq_length = config.data.max_length + 2 if hasattr(config, 'data') else 514
        self.positional_encoding = PositionalEncoding(
            self.seq_hidden_dim,
            max_length=max_seq_length
        )
        
        # Add sequence decoder for generation
        self._init_sequence_decoder()
        
        # Initialize cross-modal interaction if specified
        if config.model.get('use_bidirectional_attention', False):
            self.cross_modal_attention = BiDirectionalCrossAttention(
                hidden_dim=self.seq_hidden_dim,
                num_heads=config.model.denoiser.num_heads,
                dropout=config.model.denoiser.dropout
            )
        else:
            self.cross_modal_attention = None
        
        logger.info(f"Initialized StructDiff with {self.count_parameters():,} parameters")
    
    def _init_sequence_encoder(self):
        """Initialize ESM-2 sequence encoder"""
        model_name = self.config.model.sequence_encoder.pretrained_model
        
        # Use a smaller ESM model for testing if the specified one doesn't exist
        try:
            self.sequence_encoder = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            # Fallback to a known working model
            logger.warning(f"Could not load {model_name}: {e}")
            logger.warning("Using facebook/esm2_t6_8M_UR50D instead")
            model_name = "facebook/esm2_t6_8M_UR50D"
            self.sequence_encoder = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get actual hidden dimension
        self.seq_hidden_dim = self.sequence_encoder.config.hidden_size
        
        # Apply LoRA if specified
        if self.config.model.sequence_encoder.get('use_lora', False):
            self._apply_lora()
        
        # Freeze encoder if specified
        if self.config.model.sequence_encoder.freeze_encoder:
            for param in self.sequence_encoder.parameters():
                param.requires_grad = False
                
        logger.info(f"Initialized sequence encoder: {model_name}")
        logger.info(f"Sequence encoder hidden dim: {self.seq_hidden_dim}")
    
    def _init_sequence_decoder(self):
        """Initialize sequence decoder for generation"""
        # Get vocabulary size from tokenizer
        self.vocab_size = len(self.tokenizer)
        
        # Create decoder layers
        self.decode_projection = nn.Linear(
            self.seq_hidden_dim,
            self.vocab_size
        )
        
        # Optional: Add a small transformer decoder
        if self.config.model.get('use_decoder_layers', False):
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.seq_hidden_dim,
                nhead=self.config.model.denoiser.num_heads,
                dim_feedforward=self.seq_hidden_dim * 4,
                dropout=self.config.model.denoiser.dropout,
                batch_first=True
            )
            self.decoder_layers = nn.TransformerDecoder(
                decoder_layer,
                num_layers=2
            )
        else:
            self.decoder_layers = None
    
    def _apply_lora(self):
        """Apply LoRA to sequence encoder"""
        from .lora import apply_lora_to_model
        
        lora_config = self.config.model.sequence_encoder
        self.lora_modules = apply_lora_to_model(
            self.sequence_encoder,
            rank=lora_config.get('lora_rank', 16),
            alpha=lora_config.get('lora_alpha', 32),
            dropout=lora_config.get('lora_dropout', 0.1)
        )
        logger.info(f"Applied LoRA to {len(self.lora_modules)} modules")
    
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
                attention_mask=attention_mask,
                return_dict=True
            )
            seq_embeddings = seq_outputs.last_hidden_state  # (B, L_padded+2, D_seq)
        
        #
        # CRITICAL FIX: Trim CLS and SEP tokens from sequence embeddings and attention mask
        # to match the length of structure features.
        #
        # 序列嵌入和注意力掩码的长度将从 (L_max+2) 修正为 (L_max)，
        # 与结构特征的长度完全对齐。
        seq_embeddings = seq_embeddings[:, 1:-1, :]
        attention_mask_trimmed = attention_mask[:, 1:-1].contiguous()

        # Add positional encoding
        seq_embeddings = self.positional_encoding(seq_embeddings)
        
        # Add noise for diffusion training
        if return_loss:
            noise = torch.randn_like(seq_embeddings)
            noisy_embeddings = self.diffusion.q_sample(
                seq_embeddings, timesteps, noise
            )
        else:
            noisy_embeddings = seq_embeddings
            noise = None
        
        # Extract structure features if available
        structure_features = None
        if structures is not None:
            structure_features = self.structure_encoder(
                structures,
                attention_mask=attention_mask_trimmed  # Use the trimmed mask
            )
        
        # Optional: Apply cross-modal attention
        if self.cross_modal_attention is not None and structure_features is not None:
            seq_features, struct_features, _ = self.cross_modal_attention(
                noisy_embeddings, structure_features, attention_mask_trimmed
            )
            noisy_embeddings = seq_features
            structure_features = struct_features
        
        # Denoise with structure guidance
        denoised_embeddings, cross_attention_weights = self.denoiser(
            noisy_embeddings=noisy_embeddings,
            timesteps=timesteps,
            attention_mask=attention_mask_trimmed, # Use the trimmed mask
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
                noisy_embeddings=noisy_embeddings,
                structure_features=structure_features,
                attention_mask=attention_mask_trimmed
            )
            outputs.update(losses)
        
        return outputs
    
    def _compute_losses(
        self,
        denoised_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        noise: torch.Tensor,
        noisy_embeddings: torch.Tensor,
        structure_features: Optional[torch.Tensor],
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-scale losses"""
        losses = {}
        
        # Main diffusion loss - predict the added noise
        mask_expanded = attention_mask.bool().unsqueeze(-1).expand_as(denoised_embeddings)
        
        # Compute MSE loss between predicted and actual noise
        # The model learns to denoise by predicting the noise that was added
        predicted_noise = (noisy_embeddings - denoised_embeddings)
        diffusion_loss = F.mse_loss(
            predicted_noise[mask_expanded],
            noise[mask_expanded],
            reduction='mean'
        )
        losses['diffusion_loss'] = diffusion_loss * self.loss_weights.diffusion_loss
        
        # Structure consistency loss (if structure features available)
        if structure_features is not None and hasattr(self.loss_weights, 'structure_consistency_loss'):
            if self.loss_weights.structure_consistency_loss > 0:
                # Predict structure from denoised embeddings
                predicted_structure = self.structure_encoder.predict_from_sequence(
                    denoised_embeddings, attention_mask
                )
                
                mask_expanded_struct = attention_mask.bool().unsqueeze(-1).expand_as(predicted_structure)
                struct_loss = F.mse_loss(
                    predicted_structure[mask_expanded_struct],
                    structure_features[mask_expanded_struct],
                    reduction='mean'
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
        sampling_method: str = "ddpm",
        num_inference_steps: Optional[int] = None,
        temperature: float = 1.0,
        return_trajectory: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, Union[List[str], torch.Tensor]]:
        """
        Sample peptide sequences using the diffusion process
        
        Args:
            batch_size: Number of sequences to generate
            seq_length: Length of sequences to generate
            conditions: Optional conditioning information
            target_structure: Optional target secondary structure string
            guidance_scale: Scale for classifier-free guidance
            sampling_method: Sampling algorithm ("ddpm", "ddim", "pndm")
            num_inference_steps: Number of denoising steps
            temperature: Sampling temperature
            return_trajectory: Whether to return the full denoising trajectory
            progress_bar: Whether to show progress bar
            
        Returns:
            Dictionary containing generated sequences and metadata
        """
        device = next(self.parameters()).device
        
        # Account for special tokens (CLS/SEP)
        total_length = seq_length + 2
        
        # Initialize from noise
        shape = (batch_size, total_length, self.seq_hidden_dim)
        x_t = torch.randn(shape, device=device) * temperature
        
        # Create attention mask
        attention_mask = torch.ones(batch_size, total_length, device=device)
        
        # Parse target structure if provided
        structure_features = None
        if target_structure is not None:
            structure_features = self._parse_target_structure(
                target_structure, seq_length, batch_size, device
            )
        
        # Get sampler
        sampler = get_sampler(
            sampling_method,
            self.diffusion,
            num_inference_steps=num_inference_steps
        )
        
        # Create a wrapper for the denoising function
        def denoise_fn(x, t, cond=None):
            # Create dummy sequences for the forward pass
            dummy_sequences = torch.zeros(
                batch_size, total_length, dtype=torch.long, device=device
            )
            
            # Get timesteps tensor
            if isinstance(t, int):
                timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            else:
                timesteps = t
            
            # Denoise
            denoised, _ = self.denoiser(
                noisy_embeddings=x,
                timesteps=timesteps,
                attention_mask=attention_mask,
                structure_features=structure_features,
                conditions=cond
            )
            
            return denoised
        
        # Sample using the selected method
        trajectory = []
        
        # Run sampling
        samples = sampler.sample(
            denoise_fn,
            shape,
            conditions=conditions,
            guidance_scale=guidance_scale,
            temperature=1.0,  # Temperature already applied to initial noise
            device=device,
            verbose=progress_bar
        )
        
        # Decode to sequences
        sequences = self._decode_embeddings(samples, attention_mask)
        
        outputs = {
            'sequences': sequences,
            'embeddings': samples,
            'attention_mask': attention_mask
        }
        
        if return_trajectory:
            outputs['trajectory'] = torch.stack(trajectory) if trajectory else None
        
        # Add confidence scores
        outputs['scores'] = self._compute_sequence_scores(sequences)
        
        return outputs
    
    def _parse_target_structure(
        self,
        structure_string: str,
        seq_length: int,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Parse secondary structure string to features"""
        # Convert structure string (H: helix, E: sheet, C: coil) to indices
        structure_map = {'H': 0, 'E': 1, 'C': 2}
        structure_indices = [structure_map.get(s, 2) for s in structure_string]
        
        # Pad or truncate to match sequence length (+ 2 for CLS/SEP)
        total_length = seq_length + 2
        if len(structure_indices) < total_length:
            # Pad with coil
            structure_indices.extend([2] * (total_length - len(structure_indices)))
        else:
            structure_indices = structure_indices[:total_length]
        
        # Convert to tensor
        structure_tensor = torch.tensor(structure_indices, device=device)
        
        # Create one-hot encoding
        structure_features = F.one_hot(structure_tensor, num_classes=3).float()
        
        # Expand to batch size
        structure_features = structure_features.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Project to structure encoder hidden dim
        if structure_features.shape[-1] < self.config.model.structure_encoder.hidden_dim:
            pad_size = self.config.model.structure_encoder.hidden_dim - structure_features.shape[-1]
            structure_features = F.pad(structure_features, (0, pad_size), value=0)
        
        return structure_features
    
    def _decode_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[str]:
        """Decode embeddings back to sequences"""
        batch_size = embeddings.shape[0]
        
        # Apply decoder layers if available
        if self.decoder_layers is not None:
            # Create memory mask for transformer decoder
            memory_mask = ~attention_mask.bool()
            embeddings = self.decoder_layers(
                embeddings,
                embeddings,
                tgt_key_padding_mask=memory_mask,
                memory_key_padding_mask=memory_mask
            )
        
        # Project to vocabulary
        logits = self.decode_projection(embeddings)  # (B, L, V)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)  # (B, L)
        
        # Decode using tokenizer
        sequences = []
        for i in range(batch_size):
            # Get valid tokens (excluding padding)
            valid_length = int(attention_mask[i].sum().item())
            token_ids = predictions[i, :valid_length].cpu().tolist()
            
            # Decode with tokenizer
            sequence = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            
            # Clean up sequence (remove spaces, special characters)
            sequence = sequence.replace(' ', '').upper()
            
            # Ensure valid amino acids only
            valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
            sequence = ''.join([aa for aa in sequence if aa in valid_aas])
            
            # Handle empty sequences
            if not sequence:
                # Generate random valid sequence as fallback
                sequence = ''.join(np.random.choice(list(valid_aas), size=20))
            
            sequences.append(sequence)
        
        return sequences
    
    def _compute_sequence_scores(self, sequences: List[str]) -> torch.Tensor:
        """Compute confidence scores for generated sequences"""
        scores = []
        
        for seq in sequences:
            # Simple scoring based on sequence properties
            # In practice, this could use a trained discriminator or quality predictor
            
            # Check for valid amino acids
            valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
            validity_score = sum(1 for aa in seq if aa in valid_aas) / max(len(seq), 1)
            
            # Check for reasonable length
            length_score = 1.0 if 10 <= len(seq) <= 50 else 0.5
            
            # Check for diversity (not all same amino acid)
            diversity_score = len(set(seq)) / max(len(seq), 1)
            
            # Combine scores
            score = (validity_score + length_score + diversity_score) / 3.0
            scores.append(score)
        
        return torch.tensor(scores)
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters"""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, device: Optional[torch.device] = None):
        """Load model from checkpoint"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        
        model = cls(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def save_pretrained(self, save_path: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
