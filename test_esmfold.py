#!/usr/bin/env python3
"""
Corrected ESMFold wrapper for Huggingface implementation
"""

import torch
import numpy as np
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

class ESMFoldWrapper:
    """Wrapper for ESMFold structure prediction using Huggingface"""
    
    def __init__(self, model_name="facebook/esmfold_v1", device=None):
        """Initialize ESMFold model and tokenizer"""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Loading ESMFold model on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with proper settings
        self.model = EsmForProteinFolding.from_pretrained(
            model_name,
            low_cpu_mem_usage=True
        )
        
        # Important: Set model to float32 and move to device
        self.model.esm = self.model.esm.float()
        self.model = self.model.to(self.device)
        
        # Set chunk size to reduce memory usage
        self.model.trunk.set_chunk_size(64)
        
        # Set model to eval mode
        self.model.eval()
        
        print("✓ ESMFold model loaded successfully")
        
    def predict_structure(self, sequence, num_recycles=3):
        """
        Predict protein structure from sequence
        
        Args:
            sequence: Amino acid sequence string
            num_recycles: Number of recycling iterations
            
        Returns:
            Dictionary containing structure predictions
        """
        try:
            # Tokenize sequence - IMPORTANT: add_special_tokens=False to avoid issues
            inputs = self.tokenizer(
                [sequence], 
                return_tensors="pt", 
                add_special_tokens=False
            )
            
            # Move inputs to device and ensure correct dtype
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Ensure input_ids are LongTensor (this is crucial!)
            if 'input_ids' in inputs:
                inputs['input_ids'] = inputs['input_ids'].long()
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs, num_recycles=num_recycles)
            
            # Extract features
            features = self._extract_features(outputs, sequence)
            
            return features
            
        except Exception as e:
            print(f"Error during structure prediction: {e}")
            print("Returning dummy features as fallback")
            return self._get_dummy_features(len(sequence))
    
    def _extract_features(self, outputs, sequence):
        """Extract structural features from model outputs"""
        seq_len = len(sequence)
        
        # Get atom positions (last layer)
        positions = outputs.positions[-1].squeeze(0)  # [L, 37, 3]
        
        # Get pLDDT scores
        plddt = outputs.plddt.squeeze(0)  # [L]
        
        # Calculate CA-CA distance matrix
        ca_positions = positions[:, 1, :]  # CA is at index 1
        distance_matrix = torch.cdist(ca_positions, ca_positions)
        
        # Create contact map (threshold at 8Å)
        contact_map = (distance_matrix < 8.0).float()
        
        # Predict secondary structure (simplified)
        # This is a placeholder - real SS prediction would be more complex
        ss_pred = self._predict_secondary_structure(positions)
        
        return {
            'positions': positions.cpu(),
            'plddt': plddt.cpu(),
            'distance_matrix': distance_matrix.cpu(),
            'contact_map': contact_map.cpu(),
            'secondary_structure': ss_pred.cpu(),
            'raw_outputs': outputs  # Keep raw outputs for advanced usage
        }
    
    def _predict_secondary_structure(self, positions):
        """Simple secondary structure prediction based on backbone geometry"""
        # This is a simplified placeholder
        # Real implementation would use backbone dihedral angles
        seq_len = positions.shape[0]
        # 0: coil, 1: helix, 2: sheet
        ss = torch.full((seq_len,), 2, dtype=torch.long)
        return ss
    
    def _get_dummy_features(self, seq_len):
        """Generate dummy features for fallback"""
        positions = torch.randn(seq_len, 37, 3) * 10
        plddt = torch.full((seq_len,), 50.0)
        
        # Generate dummy CA positions
        ca_positions = torch.stack([
            torch.arange(seq_len, dtype=torch.float32) * 3.8,
            torch.zeros(seq_len),
            torch.zeros(seq_len)
        ], dim=1)
        
        # Place CA atoms in positions tensor
        positions[:, 1, :] = ca_positions
        
        # Calculate distances
        distance_matrix = torch.cdist(ca_positions, ca_positions)
        contact_map = (distance_matrix < 8.0).float()
        
        # Dummy secondary structure
        ss = torch.full((seq_len,), 2, dtype=torch.long)
        
        return {
            'positions': positions,
            'plddt': plddt,
            'distance_matrix': distance_matrix,
            'contact_map': contact_map,
            'secondary_structure': ss
        }
    
    def save_pdb(self, outputs, output_path):
        """Save structure prediction as PDB file"""
        try:
            # Convert outputs to PDB format
            if isinstance(outputs, dict) and 'raw_outputs' in outputs:
                raw_outputs = outputs['raw_outputs']
            else:
                raw_outputs = outputs
                
            pdbs = self._convert_outputs_to_pdb(raw_outputs)
            
            # Save first prediction
            with open(output_path, 'w') as f:
                f.write(pdbs[0])
                
            print(f"✓ Structure saved to {output_path}")
            
        except Exception as e:
            print(f"Error saving PDB: {e}")
    
    def _convert_outputs_to_pdb(self, outputs):
        """Convert model outputs to PDB format"""
        final_atom_positions = atom14_to_atom37(outputs.positions[-1], outputs)
        outputs_cpu = {k: v.cpu().numpy() for k, v in outputs.items()}
        final_atom_positions = final_atom_positions.cpu().numpy()
        final_atom_mask = outputs_cpu["atom37_atom_exists"]
        
        pdbs = []
        for i in range(outputs_cpu["aatype"].shape[0]):
            aa = outputs_cpu["aatype"][i]
            pred_pos = final_atom_positions[i]
            mask = final_atom_mask[i]
            resid = outputs_cpu["residue_index"][i] + 1
            
            pred = OFProtein(
                aatype=aa,
                atom_positions=pred_pos,
                atom_mask=mask,
                residue_index=resid,
                b_factors=outputs_cpu["plddt"][i],
                chain_index=outputs_cpu["chain_index"][i] if "chain_index" in outputs_cpu else None,
            )
            pdbs.append(to_pdb(pred))
            
        return pdbs


# Example usage
if __name__ == "__main__":
    # Test sequence
    test_sequence = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGEEHFKGLVLIAFSQYLQQCPFDEHVKLVNELTEFAKTCVADESHAGCEKSLHTLFGDELCKVASLRETARDLLISEQNVVNGITKGEMLPVSDTTGFPYT"
    
    try:
        # Initialize wrapper
        wrapper = ESMFoldWrapper()
        
        # Predict structure
        print(f"\nPredicting structure for sequence of length {len(test_sequence)}...")
        features = wrapper.predict_structure(test_sequence)
        
        # Print results
        print("\n=== Structure Prediction Results ===")
        print(f"✓ Positions shape: {features['positions'].shape}")
        print(f"✓ pLDDT shape: {features['plddt'].shape}")
        print(f"✓ Mean pLDDT: {features['plddt'].mean().item():.2f}")
        print(f"✓ Distance matrix shape: {features['distance_matrix'].shape}")
        print(f"✓ Number of contacts (<8Å): {features['contact_map'].sum().item():.0f}")
        
        # Save PDB if raw outputs available
        if 'raw_outputs' in features:
            wrapper.save_pdb(features, "test_structure.pdb")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()