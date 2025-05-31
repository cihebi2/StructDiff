# structdiff/generation/conditional_generation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple, Callable
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GenerationConstraints:
    """Constraints for peptide generation"""
    
    # Sequence constraints
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    required_motifs: Optional[List[str]] = None
    forbidden_motifs: Optional[List[str]] = None
    fixed_positions: Optional[Dict[int, str]] = None  # {position: amino_acid}
    
    # Composition constraints
    min_charge: Optional[float] = None
    max_charge: Optional[float] = None
    min_hydrophobicity: Optional[float] = None
    max_hydrophobicity: Optional[float] = None
    required_amino_acids: Optional[Dict[str, int]] = None  # {aa: min_count}
    
    # Structure constraints
    target_secondary_structure: Optional[str] = None
    min_helix_content: Optional[float] = None
    max_helix_content: Optional[float] = None
    min_sheet_content: Optional[float] = None
    max_sheet_content: Optional[float] = None
    
    # Functional constraints
    peptide_type: Optional[str] = None
    target_activity_score: Optional[float] = None
    target_properties: Optional[Dict[str, float]] = None


class ConditionalGenerator:
    """Advanced conditional generation with multiple constraints"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        
        # Amino acid properties for constraint checking
        self.aa_properties = self._init_aa_properties()
    
    def _init_aa_properties(self) -> Dict[str, Dict[str, float]]:
        """Initialize amino acid properties"""
        return {
            'A': {'charge': 0, 'hydrophobicity': 1.8, 'helix_propensity': 1.42},
            'R': {'charge': 1, 'hydrophobicity': -4.5, 'helix_propensity': 0.98},
            'N': {'charge': 0, 'hydrophobicity': -3.5, 'helix_propensity': 0.67},
            'D': {'charge': -1, 'hydrophobicity': -3.5, 'helix_propensity': 1.01},
            'C': {'charge': 0, 'hydrophobicity': 2.5, 'helix_propensity': 0.70},
            'Q': {'charge': 0, 'hydrophobicity': -3.5, 'helix_propensity': 1.11},
            'E': {'charge': -1, 'hydrophobicity': -3.5, 'helix_propensity': 1.51},
            'G': {'charge': 0, 'hydrophobicity': -0.4, 'helix_propensity': 0.57},
            'H': {'charge': 0.5, 'hydrophobicity': -3.2, 'helix_propensity': 1.00},
            'I': {'charge': 0, 'hydrophobicity': 4.5, 'helix_propensity': 1.08},
            'L': {'charge': 0, 'hydrophobicity': 3.8, 'helix_propensity': 1.21},
            'K': {'charge': 1, 'hydrophobicity': -3.9, 'helix_propensity': 1.14},
            'M': {'charge': 0, 'hydrophobicity': 1.9, 'helix_propensity': 1.45},
            'F': {'charge': 0, 'hydrophobicity': 2.8, 'helix_propensity': 1.13},
            'P': {'charge': 0, 'hydrophobicity': -1.6, 'helix_propensity': 0.57},
            'S': {'charge': 0, 'hydrophobicity': -0.8, 'helix_propensity': 0.77},
            'T': {'charge': 0, 'hydrophobicity': -0.7, 'helix_propensity': 0.83},
            'W': {'charge': 0, 'hydrophobicity': -0.9, 'helix_propensity': 1.08},
            'Y': {'charge': 0, 'hydrophobicity': -1.3, 'helix_propensity': 0.69},
            'V': {'charge': 0, 'hydrophobicity': 4.2, 'helix_propensity': 1.06}
        }
    
    def generate_with_constraints(
        self,
        num_samples: int,
        constraints: GenerationConstraints,
        guidance_scale: float = 2.0,
        temperature: float = 1.0,
        max_attempts: int = 10,
        rejection_sampling: bool = True
    ) -> List[str]:
        """Generate peptides with multiple constraints"""
        generated_sequences = []
        attempts_per_sequence = []
        
        while len(generated_sequences) < num_samples:
            if rejection_sampling:
                # Rejection sampling approach
                sequences, attempts = self._rejection_sampling(
                    num_samples - len(generated_sequences),
                    constraints,
                    guidance_scale,
                    temperature,
                    max_attempts
                )
                generated_sequences.extend(sequences)
                attempts_per_sequence.extend(attempts)
            else:
                # Guided generation approach
                sequences = self._guided_generation(
                    num_samples - len(generated_sequences),
                    constraints,
                    guidance_scale,
                    temperature
                )
                generated_sequences.extend(sequences)
        
        if rejection_sampling:
            avg_attempts = np.mean(attempts_per_sequence)
            logger.info(f"Average attempts per sequence: {avg_attempts:.2f}")
        
        return generated_sequences[:num_samples]
    
    def _rejection_sampling(
        self,
        num_samples: int,
        constraints: GenerationConstraints,
        guidance_scale: float,
        temperature: float,
        max_attempts: int
    ) -> Tuple[List[str], List[int]]:
        """Generate using rejection sampling"""
        valid_sequences = []
        attempts_list = []
        
        # Determine length range
        min_len = constraints.min_length or 10
        max_len = constraints.max_length or 30
        
        for _ in range(num_samples):
            attempts = 0
            valid = False
            
            while not valid and attempts < max_attempts:
                attempts += 1
                
                # Sample length
                length = np.random.randint(min_len, max_len + 1)
                
                # Generate sequence
                conditions = self._prepare_conditions(constraints)
                
                with torch.no_grad():
                    samples = self.model.sample(
                        batch_size=1,
                        seq_length=length,
                        conditions=conditions,
                        guidance_scale=guidance_scale,
                        temperature=temperature
                    )
                
                sequence = samples['sequences'][0]
                
                # Check constraints
                if self._check_constraints(sequence, constraints):
                    valid = True
                    valid_sequences.append(sequence)
                    attempts_list.append(attempts)
            
            if not valid:
                logger.warning(f"Failed to generate valid sequence after {max_attempts} attempts")
        
        return valid_sequences, attempts_list
    
    def _guided_generation(
        self,
        num_samples: int,
        constraints: GenerationConstraints,
        guidance_scale: float,
        temperature: float
    ) -> List[str]:
        """Generate with guided sampling to satisfy constraints"""
        # Prepare conditions and guidance functions
        conditions = self._prepare_conditions(constraints)
        guidance_fns = self._prepare_guidance_functions(constraints)
        
        # Determine length
        if constraints.min_length and constraints.max_length:
            lengths = np.random.randint(
                constraints.min_length,
                constraints.max_length + 1,
                num_samples
            )
        else:
            lengths = [20] * num_samples  # Default length
        
        generated_sequences = []
        
        for i in range(num_samples):
            with torch.no_grad():
                # Custom sampling loop with guidance
                sequence = self._guided_sampling_loop(
                    length=lengths[i],
                    conditions=conditions,
                    guidance_fns=guidance_fns,
                    guidance_scale=guidance_scale,
                    temperature=temperature,
                    constraints=constraints
                )
                
                generated_sequences.append(sequence)
        
        return generated_sequences
    
    def _prepare_conditions(self, constraints: GenerationConstraints) -> Dict[str, torch.Tensor]:
        """Prepare conditioning tensors from constraints"""
        conditions = {}
        
        # Peptide type
        if constraints.peptide_type:
            type_map = {
                'antimicrobial': 0,
                'antifungal': 1,
                'antiviral': 2
            }
            if constraints.peptide_type in type_map:
                conditions['peptide_type'] = torch.tensor(
                    [type_map[constraints.peptide_type]],
                    device=self.device
                )
        
        # Target properties
        if constraints.target_properties:
            prop_vector = []
            for prop in ['charge', 'hydrophobicity', 'helix_content']:
                if prop in constraints.target_properties:
                    prop_vector.append(constraints.target_properties[prop])
            
            if prop_vector:
                conditions['target_properties'] = torch.tensor(
                    [prop_vector],
                    device=self.device,
                    dtype=torch.float32
                )
        
        # Target secondary structure
        if constraints.target_secondary_structure:
            # Encode as one-hot or embedding
            ss_map = {'H': 0, 'E': 1, 'C': 2}
            ss_indices = [ss_map.get(s, 2) for s in constraints.target_secondary_structure]
            conditions['target_structure'] = torch.tensor(
                ss_indices,
                device=self.device
            ).unsqueeze(0)
        
        return conditions
    
    def _prepare_guidance_functions(
        self,
        constraints: GenerationConstraints
    ) -> List[Callable]:
        """Prepare guidance functions for constrained generation"""
        guidance_fns = []
        
        # Charge guidance
        if constraints.min_charge is not None or constraints.max_charge is not None:
            def charge_guidance(embeddings, sequences):
                charges = self._compute_charge(sequences)
                loss = 0
                
                if constraints.min_charge is not None:
                    loss += F.relu(constraints.min_charge - charges).mean()
                
                if constraints.max_charge is not None:
                    loss += F.relu(charges - constraints.max_charge).mean()
                
                return loss
            
            guidance_fns.append(charge_guidance)
        
        # Hydrophobicity guidance
        if constraints.min_hydrophobicity is not None or constraints.max_hydrophobicity is not None:
            def hydrophobicity_guidance(embeddings, sequences):
                hydro = self._compute_hydrophobicity(sequences)
                loss = 0
                
                if constraints.min_hydrophobicity is not None:
                    loss += F.relu(constraints.min_hydrophobicity - hydro).mean()
                
                if constraints.max_hydrophobicity is not None:
                    loss += F.relu(hydro - constraints.max_hydrophobicity).mean()
                
                return loss
            
            guidance_fns.append(hydrophobicity_guidance)
        
        # Fixed position guidance
        if constraints.fixed_positions:
            def position_guidance(embeddings, sequences):
                loss = 0
                
                for pos, aa in constraints.fixed_positions.items():
                    if pos < sequences.shape[1]:
                        # Get predicted amino acid at position
                        pred_aa = sequences[:, pos]
                        target_aa = self._aa_to_index(aa)
                        
                        # Cross-entropy loss
                        loss += F.cross_entropy(
                            embeddings[:, pos],
                            torch.tensor([target_aa], device=embeddings.device)
                        )
                
                return loss
            
            guidance_fns.append(position_guidance)
        
        return guidance_fns
    
    def _guided_sampling_loop(
        self,
        length: int,
        conditions: Dict[str, torch.Tensor],
        guidance_fns: List[Callable],
        guidance_scale: float,
        temperature: float,
        constraints: GenerationConstraints
    ) -> str:
        """Custom sampling loop with guidance"""
        # This is a simplified version - full implementation would modify
        # the diffusion sampling process to incorporate gradients
        
        # For now, use standard sampling with post-processing
        with torch.no_grad():
            samples = self.model.sample(
                batch_size=1,
                seq_length=length,
                conditions=conditions,
                guidance_scale=guidance_scale,
                temperature=temperature
            )
        
        sequence = samples['sequences'][0]
        
        # Apply fixed positions
        if constraints.fixed_positions:
            seq_list = list(sequence)
            for pos, aa in constraints.fixed_positions.items():
                if pos < len(seq_list):
                    seq_list[pos] = aa
            sequence = ''.join(seq_list)
        
        return sequence
    
    def _check_constraints(self, sequence: str, constraints: GenerationConstraints) -> bool:
        """Check if sequence satisfies all constraints"""
        # Length constraints
        if constraints.min_length and len(sequence) < constraints.min_length:
            return False
        if constraints.max_length and len(sequence) > constraints.max_length:
            return False
        
        # Motif constraints
        if constraints.required_motifs:
            for motif in constraints.required_motifs:
                if motif not in sequence:
                    return False
        
        if constraints.forbidden_motifs:
            for motif in constraints.forbidden_motifs:
                if motif in sequence:
                    return False
        
        # Fixed positions
        if constraints.fixed_positions:
            for pos, aa in constraints.fixed_positions.items():
                if pos < len(sequence) and sequence[pos] != aa:
                    return False
        
        # Composition constraints
        charge = sum(self.aa_properties.get(aa, {}).get('charge', 0) for aa in sequence)
        
        if constraints.min_charge and charge < constraints.min_charge:
            return False
        if constraints.max_charge and charge > constraints.max_charge:
            return False
        
        # Hydrophobicity
        hydro = np.mean([
            self.aa_properties.get(aa, {}).get('hydrophobicity', 0)
            for aa in sequence
        ])
        
        if constraints.min_hydrophobicity and hydro < constraints.min_hydrophobicity:
            return False
        if constraints.max_hydrophobicity and hydro > constraints.max_hydrophobicity:
            return False
        
        # Required amino acids
        if constraints.required_amino_acids:
            aa_counts = {aa: sequence.count(aa) for aa in set(sequence)}
            for aa, min_count in constraints.required_amino_acids.items():
                if aa_counts.get(aa, 0) < min_count:
                    return False
        
        return True


class PropertyPredictor(nn.Module):
    """Predict peptide properties from embeddings"""
    
    def __init__(self, input_dim: int, num_properties: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_properties)
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict properties from embeddings"""
        # Pool over sequence dimension
        pooled = embeddings.mean(dim=1)
        return self.predictor(pooled)


class ConditionalSampler:
    """Advanced conditional sampling strategies"""
    
    def __init__(self, model: nn.Module, property_predictor: Optional[PropertyPredictor] = None):
        self.model = model
        self.property_predictor = property_predictor
    
    def sample_with_property_control(
        self,
        num_samples: int,
        target_properties: Dict[str, float],
        property_weights: Optional[Dict[str, float]] = None,
        num_iterations: int = 50,
        learning_rate: float = 0.1
    ) -> List[str]:
        """Sample with iterative property control"""
        if self.property_predictor is None:
            raise ValueError("Property predictor required for property control")
        
        # Initialize from noise
        batch_size = num_samples
        seq_length = 20  # Default
        
        # Start with random embeddings
        embeddings = torch.randn(
            batch_size, seq_length, self.model.config.model.hidden_dim,
            requires_grad=True
        )
        
        optimizer = torch.optim.Adam([embeddings], lr=learning_rate)
        
        # Property indices
        property_names = ['charge', 'hydrophobicity', 'helix_content']
        target_vector = torch.tensor([
            target_properties.get(p, 0) for p in property_names
        ])
        
        # Optimization loop
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # Predict properties
            predicted_props = self.property_predictor(embeddings)
            
            # Property loss
            property_loss = F.mse_loss(predicted_props, target_vector.expand_as(predicted_props))
            
            # Regularization to keep embeddings valid
            reg_loss = 0.01 * embeddings.norm()
            
            total_loss = property_loss + reg_loss
            total_loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                logger.info(f"Iteration {i}: property_loss={property_loss.item():.4f}")
        
        # Decode embeddings to sequences
        with torch.no_grad():
            # Project to valid sequence space
            sequences = self._decode_embeddings(embeddings.detach())
        
        return sequences
    
    def sample_diverse_set(
        self,
        num_samples: int,
        diversity_threshold: float = 0.7,
        max_attempts: int = 1000,
        conditions: Optional[Dict[str, torch.Tensor]] = None
    ) -> List[str]:
        """Sample diverse set of sequences"""
        diverse_sequences = []
        attempts = 0
        
        while len(diverse_sequences) < num_samples and attempts < max_attempts:
            # Generate candidate
            with torch.no_grad():
                samples = self.model.sample(
                    batch_size=1,
                    seq_length=np.random.randint(15, 30),
                    conditions=conditions
                )
            
            candidate = samples['sequences'][0]
            attempts += 1
            
            # Check diversity
            if self._is_diverse(candidate, diverse_sequences, diversity_threshold):
                diverse_sequences.append(candidate)
                logger.info(f"Found diverse sequence {len(diverse_sequences)}/{num_samples}")
        
        return diverse_sequences
    
    def _is_diverse(
        self,
        candidate: str,
        existing: List[str],
        threshold: float
    ) -> bool:
        """Check if candidate is diverse from existing sequences"""
        if not existing:
            return True
        
        # Compute similarities
        from difflib import SequenceMatcher
        
        similarities = []
        for seq in existing:
            similarity = SequenceMatcher(None, candidate, seq).ratio()
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        return max_similarity < threshold
    
    def _decode_embeddings(self, embeddings: torch.Tensor) -> List[str]:
        """Decode embeddings to sequences"""
        # Placeholder - would use learned decoder
        sequences = []
        
        # Simple argmax decoding
        aa_vocab = 'ACDEFGHIKLMNPQRSTVWY'
        
        # Project to vocabulary size
        vocab_size = len(aa_vocab)
        projection = nn.Linear(embeddings.shape[-1], vocab_size).to(embeddings.device)
        
        logits = projection(embeddings)
        indices = logits.argmax(dim=-1)
        
        for seq_indices in indices:
            sequence = ''.join([aa_vocab[i] for i in seq_indices])
            sequences.append(sequence)
        
        return sequences


# Attribute-guided generation
class AttributeGuidedGenerator:
    """Generate with specific attribute targets"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        
        # Attribute classifiers (would be trained separately)
        self.attribute_classifiers = {}
    
    def add_attribute_classifier(self, name: str, classifier: nn.Module):
        """Add attribute classifier"""
        self.attribute_classifiers[name] = classifier
    
    def generate_with_attributes(
        self,
        num_samples: int,
        target_attributes: Dict[str, float],
        attribute_guidance_scale: float = 1.0,
        base_guidance_scale: float = 2.0
    ) -> List[str]:
        """Generate with target attributes"""
        # Custom sampling with attribute guidance
        
        def attribute_guidance_fn(embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
            """Compute attribute guidance gradients"""
            gradients = {}
            
            for attr_name, target_value in target_attributes.items():
                if attr_name in self.attribute_classifiers:
                    classifier = self.attribute_classifiers[attr_name]
                    
                    # Get attribute prediction
                    pred = classifier(embeddings)
                    
                    # Compute gradient toward target
                    loss = F.mse_loss(pred, torch.full_like(pred, target_value))
                    
                    # Get gradient
                    grad = torch.autograd.grad(loss, embeddings, retain_graph=True)[0]
                    gradients[attr_name] = grad
            
            return gradients
        
        # Sample with attribute guidance
        generated = []
        
        for _ in range(num_samples):
            # Length
            length = np.random.randint(15, 30)
            
            # Generate with custom guidance
            with torch.enable_grad():
                # Simplified - full implementation would modify sampling loop
                samples = self.model.sample(
                    batch_size=1,
                    seq_length=length,
                    guidance_scale=base_guidance_scale
                )
            
            generated.append(samples['sequences'][0])
        
        return generated
# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
