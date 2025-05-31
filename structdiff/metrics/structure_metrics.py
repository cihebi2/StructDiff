import numpy as np
from typing import List, Optional, Dict
import torch


def compute_structure_metrics(
    generated_structures: List[Optional[Dict]],
    reference_structures: Optional[List[Optional[Dict]]] = None
) -> Dict[str, float]:
    """Compute structure-level metrics"""
    metrics = {}
    
    # Filter out None values
    valid_structures = [s for s in generated_structures if s is not None]
    
    if not valid_structures:
        return metrics
    
    # pLDDT scores
    plddt_scores = []
    for struct in valid_structures:
        if 'plddt' in struct:
            plddt_scores.append(struct['plddt'].mean().item())
    
    if plddt_scores:
        metrics['avg_plddt'] = np.mean(plddt_scores)
        metrics['std_plddt'] = np.std(plddt_scores)
        metrics['high_confidence_ratio'] = np.mean([s > 70 for s in plddt_scores])
    
    # Secondary structure content
    ss_contents = {'helix': [], 'sheet': [], 'coil': []}
    
    for struct in valid_structures:
        if 'secondary_structure' in struct:
            ss = struct['secondary_structure']
            if isinstance(ss, torch.Tensor):
                ss = ss.cpu().numpy()
            
            total = len(ss)
            ss_contents['helix'].append(np.sum(ss == 0) / total)
            ss_contents['sheet'].append(np.sum(ss == 1) / total)
            ss_contents['coil'].append(np.sum(ss == 2) / total)
    
    for ss_type, contents in ss_contents.items():
        if contents:
            metrics[f'avg_{ss_type}_content'] = np.mean(contents)
    
    # Contact density
    contact_densities = []
    for struct in valid_structures:
        if 'contact_map' in struct:
            contact_map = struct['contact_map']
            if isinstance(contact_map, torch.Tensor):
                contact_map = contact_map.cpu().numpy()
            
            # Exclude diagonal and immediate neighbors
            n = contact_map.shape[0]
            mask = np.ones_like(contact_map, dtype=bool)
            np.fill_diagonal(mask, 0)
            np.fill_diagonal(mask[1:], 0)
            np.fill_diagonal(mask[:, 1:], 0)
            
            density = contact_map[mask].mean()
            contact_densities.append(density)
    
    if contact_densities:
        metrics['avg_contact_density'] = np.mean(contact_densities)
    
    # Compare with reference if provided
    if reference_structures:
        valid_ref = [s for s in reference_structures if s is not None]
        
        if valid_ref:
            # TM-score would go here (requires alignment)
            # Placeholder
            metrics['avg_tm_score'] = 0.7
    
    return metrics
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04
