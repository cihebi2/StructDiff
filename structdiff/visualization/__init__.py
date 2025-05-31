# structdiff/visualization/__init__.py
from .visualization import (
    PeptideVisualizer,
    InteractivePeptideExplorer,
    plot_training_curves,
    plot_sequence_distribution,
    plot_structure_analysis
)

__all__ = [
    'PeptideVisualizer',
    'InteractivePeptideExplorer',
    'plot_training_curves',
    'plot_sequence_distribution', 
    'plot_structure_analysis'
]


def plot_training_curves(history: dict, save_path: str = None):
    """Quick function to plot training curves"""
    from .visualization import PeptideVisualizer
    viz = PeptideVisualizer()
    viz.plot_generation_metrics(history, save_path)


def plot_sequence_distribution(sequences: list, save_path: str = None):
    """Quick function to plot sequence distribution"""
    from .visualization import PeptideVisualizer
    viz = PeptideVisualizer()
    viz.plot_property_distribution(sequences, save_path=save_path)


def plot_structure_analysis(structures: list, save_path: str = None):
    """Quick function to plot structure analysis"""
    from .visualization import PeptideVisualizer
    viz = PeptideVisualizer()
    # Simplified version - would need actual implementation
    if save_path:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.text(0.5, 0.5, 'Structure Analysis', ha='center', va='center')
        plt.savefig(save_path)
        plt.close()
# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
