"""
StructDiff: Structure-Aware Diffusion Model for Peptide Generation
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models import StructDiff, BaselineDiffusion
from .data import PeptideDataset, PeptideStructureCollator
from .diffusion import GaussianDiffusion

__all__ = [
    "StructDiff",
    "BaselineDiffusion", 
    "PeptideDataset",
    "PeptideStructureCollator",
    "GaussianDiffusion",
]
# Updated: 05/30/2025 22:59:09
