from .sequence_metrics import compute_sequence_metrics
from .structure_metrics import compute_structure_metrics
from .functional_metrics import compute_functional_metrics
from .diversity_metrics import compute_diversity_metrics

__all__ = [
    'compute_sequence_metrics',
    'compute_structure_metrics',
    'compute_functional_metrics',
    'compute_diversity_metrics'
]


def compute_validation_metrics(predictions, targets, config):
    """Compute all validation metrics"""
    metrics = {}
    
    # Add your metric computations here
    # This is a placeholder
    metrics['perplexity'] = 1.0
    metrics['accuracy'] = 0.9
    
    return metrics