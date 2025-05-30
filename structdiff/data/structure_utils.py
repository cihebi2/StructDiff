import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from Bio import PDB
from Bio.PDB.DSSP import DSSP
import warnings
warnings.filterwarnings('ignore')


def extract_structure_features(
    pdb_path: str,
    sequence_length: int
) -> Dict[str, torch.Tensor]:
    """
    Extract multi-scale structure features from PDB file
    
    Returns:
        Dictionary containing:
        - angles: Backbone torsion angles (phi, psi, omega)
        - secondary_structure: SS assignments
        - distance_matrix: CA-CA distance matrix
        - contact_map: Binary contact map
        - sasa: Solvent accessible surface area
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('peptide', pdb_path)
    model = structure[0]
    
    features = {}
    
    # Extract backbone angles
    angles = extract_backbone_angles(model)
    features['angles'] = torch.tensor(angles, dtype=torch.float32)
    
    # Extract secondary structure
    ss = extract_secondary_structure(model, pdb_path)
    features['secondary_structure'] = torch.tensor(ss, dtype=torch.long)
    
    # Extract distance matrix
    dist_matrix = compute_distance_matrix(model)
    features['distance_matrix'] = torch.tensor(dist_matrix, dtype=torch.float32)
    
    # Compute contact map (8Å threshold)
    contact_map = (dist_matrix < 8.0).astype(np.float32)
    features['contact_map'] = torch.tensor(contact_map, dtype=torch.float32)
    
    # Extract SASA if possible
    try:
        sasa = compute_sasa(model)
        features['sasa'] = torch.tensor(sasa, dtype=torch.float32)
    except:
        pass
    
    return features


def predict_structure_with_esmfold(
    sequence: str,
    esmfold_model: torch.nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Predict structure features using ESMFold
    
    Args:
        sequence: Amino acid sequence
        esmfold_model: Pre-loaded ESMFold model
        
    Returns:
        Dictionary of structure features
    """
    device = next(esmfold_model.parameters()).device
    
    with torch.no_grad():
        # Predict structure
        output = esmfold_model.infer_pdb(sequence)
        
        # Extract features from predicted structure
        # This is a simplified version - actual implementation would
        # parse the output properly
        features = {}
        
        # Mock features for now
        seq_len = len(sequence)
        
        # Predicted angles (placeholder)
        features['angles'] = torch.randn(seq_len, 3, device=device)
        
        # Predicted secondary structure (placeholder)
        features['secondary_structure'] = torch.randint(
            0, 3, (seq_len,), device=device
        )
        
        # Predicted distance matrix (placeholder)
        features['distance_matrix'] = torch.rand(
            seq_len, seq_len, device=device
        ) * 20.0
        
        # Ensure symmetry
        features['distance_matrix'] = (
            features['distance_matrix'] + 
            features['distance_matrix'].T
        ) / 2
        
        # pLDDT scores
        features['plddt'] = torch.rand(seq_len, device=device) * 100
        
    return features


def extract_backbone_angles(model) -> np.ndarray:
    """Extract phi, psi, omega angles from PDB model"""
    angles = []
    
    for chain in model:
        polypeptides = PDB.PPBuilder().build_peptides(chain)
        for poly in polypeptides:
            phi_psi_list = poly.get_phi_psi_list()
            
            for i, (phi, psi) in enumerate(phi_psi_list):
                # Handle None values
                phi = phi if phi is not None else 0.0
                psi = psi if psi is not None else 0.0
                
                # Omega angle (simplified - always ~180 degrees)
                omega = np.pi
                
                angles.append([phi, psi, omega])
    
    return np.array(angles)


def extract_secondary_structure(model, pdb_path: str) -> np.ndarray:
    """
    Extract secondary structure using DSSP
    Returns: 0 for helix, 1 for sheet, 2 for coil
    """
    try:
        dssp = DSSP(model, pdb_path, dssp='mkdssp')
        
        ss_map = {
            'H': 0, 'G': 0, 'I': 0,  # Helices
            'E': 1, 'B': 1,           # Sheets
            'T': 2, 'S': 2, '-': 2    # Coils/loops
        }
        
        ss_sequence = []
        for residue in dssp.property_list:
            ss = residue[2]
            ss_sequence.append(ss_map.get(ss, 2))
        
        return np.array(ss_sequence)
    except:
        # Fallback if DSSP fails
        chain = list(model.get_chains())[0]
        num_residues = len(list(chain.get_residues()))
        return np.full(num_residues, 2)  # All coil


def compute_distance_matrix(model) -> np.ndarray:
    """Compute CA-CA distance matrix"""
    ca_atoms = []
    
    for chain in model:
        for residue in chain:
            if 'CA' in residue:
                ca_atoms.append(residue['CA'].coord)
    
    ca_coords = np.array(ca_atoms)
    n_residues = len(ca_coords)
    
    # Compute pairwise distances
    dist_matrix = np.zeros((n_residues, n_residues))
    for i in range(n_residues):
        for j in range(n_residues):
            dist_matrix[i, j] = np.linalg.norm(
                ca_coords[i] - ca_coords[j]
            )
    
    return dist_matrix


def compute_sasa(model) -> np.ndarray:
    """Compute solvent accessible surface area per residue"""
    # This would use DSSP or another tool
    # Placeholder implementation
    chain = list(model.get_chains())[0]
    num_residues = len(list(chain.get_residues()))
    return np.random.rand(num_residues) * 200  # Random SASA values


def augment_structure(
    features: Dict[str, torch.Tensor],
    noise_level: float = 0.1
) -> Dict[str, torch.Tensor]:
    """Add noise to structure features for augmentation"""
    augmented = {}
    
    for key, value in features.items():
        if key == 'secondary_structure':
            # Don't augment discrete labels
            augmented[key] = value
        elif key == 'contact_map':
            # Add some noise to contacts
            noise = torch.rand_like(value) < noise_level
            augmented[key] = (value.bool() ^ noise).float()
        else:
            # Add Gaussian noise
            noise = torch.randn_like(value) * noise_level
            augmented[key] = value + noise
    
    return augmented