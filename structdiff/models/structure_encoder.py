import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from einops import rearrange, repeat

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MultiScaleStructureEncoder(nn.Module):
    """
    Multi-scale structure encoder that extracts features at different levels:
    - Residue level: phi/psi angles, solvent accessibility
    - Secondary structure level: helix/sheet/coil propensities
    - Global topology level: contact maps, distance matrices
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # ESMFold for structure prediction
        if config.use_esmfold:
            self._init_esmfold()
        
        # Feature extractors for different scales
        self.residue_encoder = ResidueFeatureEncoder(
            input_dim=10,  # phi, psi, omega, chi angles, etc.
            hidden_dim=self.hidden_dim
        )
        
        self.secondary_structure_encoder = SecondaryStructureEncoder(
            input_dim=3,  # H, E, C
            hidden_dim=self.hidden_dim
        )
        
        self.topology_encoder = TopologyEncoder(
            hidden_dim=self.hidden_dim
        )
        
        # Feature fusion
        self.fusion_layer = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def _init_esmfold(self):
        """Initialize ESMFold for structure prediction"""
        try:
            # 使用修复的 ESMFoldWrapper 而不是直接使用 ESM
            from .esmfold_wrapper import ESMFoldWrapper
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.esmfold = ESMFoldWrapper(device=device)
            
            if self.esmfold.available:
                logger.info("Initialized ESMFoldWrapper for structure prediction")
            else:
                logger.warning("ESMFoldWrapper not available")
                self.esmfold = None
                
        except Exception as e:
            logger.warning(f"Could not initialize ESMFold: {e}")
            self.esmfold = None
    
    def forward(
        self,
        structure_data: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract multi-scale structure features
        
        Args:
            structure_data: Dictionary containing structure information
            attention_mask: Attention mask for valid positions
            
        Returns:
            Structure features of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len = attention_mask.shape
        device = attention_mask.device
        
        # Extract features at different scales
        features = []
        
        # Residue-level features
        if 'angles' in structure_data:
            residue_features = self.residue_encoder(
                structure_data['angles'], attention_mask
            )
            features.append(residue_features)
        else:
            features.append(torch.zeros(batch_size, seq_len, self.hidden_dim, device=device))
        
        # Secondary structure features
        if 'secondary_structure' in structure_data:
            ss_features = self.secondary_structure_encoder(
                structure_data['secondary_structure'], attention_mask
            )
            features.append(ss_features)
        else:
            features.append(torch.zeros(batch_size, seq_len, self.hidden_dim, device=device))
        
        # Topology features
        if 'distance_matrix' in structure_data:
            topology_features = self.topology_encoder(
                structure_data['distance_matrix'], attention_mask
            )
            features.append(topology_features)
        else:
            features.append(torch.zeros(batch_size, seq_len, self.hidden_dim, device=device))
        
        # Concatenate all features
        combined_features = torch.cat(features, dim=-1)
        
        # Project to final dimension
        output_features = self.output_projection(combined_features)
        
        # Apply attention mask
        output_features = output_features * attention_mask.unsqueeze(-1)
        
        return output_features
    
    def predict_from_sequence(
        self,
        sequence_embeddings: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Predict structure features from sequence embeddings"""
        # This would use ESMFold or a learned predictor
        # Placeholder implementation
        batch_size, seq_len = attention_mask.shape
        return torch.randn(batch_size, seq_len, self.hidden_dim, device=sequence_embeddings.device)
    
    def extract_features_from_pdb(self, pdb_path: str) -> Dict[str, np.ndarray]:
        """Extract structure features from PDB file"""
        # Implementation would parse PDB and extract various features
        pass


class ResidueFeatureEncoder(nn.Module):
    """Encode residue-level structural features"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 创建一个动态的线性层，将在forward中初始化
        self.encoder = None
        
    def _init_encoder(self, actual_input_dim: int):
        """动态初始化编码器以匹配实际输入维度"""
        if self.encoder is None or self.encoder[0].in_features != actual_input_dim:
            self.encoder = nn.Sequential(
                nn.Linear(actual_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            )
            # 将编码器移动到正确的设备
            if hasattr(self, '_device'):
                self.encoder = self.encoder.to(self._device)
    
    def forward(self, angles: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 记录设备信息
        self._device = angles.device
        
        # 动态获取实际输入维度
        actual_input_dim = angles.shape[-1]
        
        # 初始化或重新初始化编码器
        self._init_encoder(actual_input_dim)
        
        # 将编码器移动到正确的设备
        self.encoder = self.encoder.to(angles.device)
        
        # 确保输入张量的形状正确
        if angles.dim() == 2:
            # 如果只有2维，添加批次维度
            angles = angles.unsqueeze(0)
        
        encoded = self.encoder(angles)
        return encoded * mask.unsqueeze(-1)


class SecondaryStructureEncoder(nn.Module):
    """Encode secondary structure information"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # 初始化一个较大的嵌入层来容纳更多的类别
        self.embedding = nn.Embedding(max(input_dim, 8), hidden_dim)  # 至少8个类别
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim // 2,
            bidirectional=True, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, ss_labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 确保标签在有效范围内
        max_label = self.embedding.num_embeddings - 1
        ss_labels = torch.clamp(ss_labels, 0, max_label)
        
        # 确保输入张量的形状正确
        if ss_labels.dim() == 1:
            # 如果只有1维，添加批次维度
            ss_labels = ss_labels.unsqueeze(0)
        
        embedded = self.embedding(ss_labels)
        lstm_out, _ = self.lstm(embedded)
        return self.norm(lstm_out) * mask.unsqueeze(-1)


class TopologyEncoder(nn.Module):
    """Encode global topology information from distance/contact matrices"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, hidden_dim, kernel_size=3, padding=1)
        ])
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, distance_matrix: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = mask.shape
        
        # Add channel dimension
        x = distance_matrix.unsqueeze(1)  # (B, 1, L, L)
        
        # Apply convolutions
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        # Global pooling for each position
        position_features = []
        for i in range(seq_len):
            # Extract features related to position i
            pos_features = x[:, :, i, :].mean(dim=-1)  # (B, hidden_dim)
            position_features.append(pos_features)
        
        features = torch.stack(position_features, dim=1)  # (B, L, hidden_dim)
        
        return self.projection(features) * mask.unsqueeze(-1)
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
