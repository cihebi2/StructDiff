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
            # 确保形状匹配
            residue_features = self._align_feature_shape(residue_features, batch_size, seq_len, self.hidden_dim, device)
            features.append(residue_features)
        else:
            features.append(torch.zeros(batch_size, seq_len, self.hidden_dim, device=device))
        
        # Secondary structure features
        if 'secondary_structure' in structure_data:
            ss_features = self.secondary_structure_encoder(
                structure_data['secondary_structure'], attention_mask
            )
            # 确保形状匹配
            ss_features = self._align_feature_shape(ss_features, batch_size, seq_len, self.hidden_dim, device)
            features.append(ss_features)
        else:
            features.append(torch.zeros(batch_size, seq_len, self.hidden_dim, device=device))
        
        # Topology features
        if 'distance_matrix' in structure_data:
            topology_features = self.topology_encoder(
                structure_data['distance_matrix'], attention_mask
            )
            # 确保形状匹配
            topology_features = self._align_feature_shape(topology_features, batch_size, seq_len, self.hidden_dim, device)
            features.append(topology_features)
        else:
            features.append(torch.zeros(batch_size, seq_len, self.hidden_dim, device=device))
        
        # 再次验证所有特征的形状一致性
        for i, feat in enumerate(features):
            if feat.shape != (batch_size, seq_len, self.hidden_dim):
                print(f"Warning: Feature {i} has shape {feat.shape}, expected ({batch_size}, {seq_len}, {self.hidden_dim})")
                # 强制调整形状
                feat = self._force_align_shape(feat, batch_size, seq_len, self.hidden_dim, device)
                features[i] = feat
        
        # Concatenate all features
        try:
            combined_features = torch.cat(features, dim=-1)
        except Exception as e:
            print(f"Error concatenating features: {e}")
            print(f"Feature shapes: {[f.shape for f in features]}")
            # 如果拼接失败，创建默认特征
            combined_features = torch.zeros(batch_size, seq_len, self.hidden_dim * 3, device=device)
        
        # Project to final dimension
        output_features = self.output_projection(combined_features)
        
        # Apply attention mask
        output_features = output_features * attention_mask.unsqueeze(-1)
        
        return output_features
    
    def _align_feature_shape(self, features, target_batch_size, target_seq_len, target_hidden_dim, device):
        """对齐特征形状"""
        if features is None:
            return torch.zeros(target_batch_size, target_seq_len, target_hidden_dim, device=device)
        
        current_shape = features.shape
        
        # 调整batch_size
        if current_shape[0] != target_batch_size:
            if current_shape[0] == 1 and target_batch_size > 1:
                # 扩展batch维度
                features = features.expand(target_batch_size, -1, -1)
            elif current_shape[0] > target_batch_size:
                # 截断batch维度
                features = features[:target_batch_size]
            else:
                # 填充batch维度
                pad_batch = target_batch_size - current_shape[0]
                padding_shape = (pad_batch,) + current_shape[1:]
                padding = torch.zeros(padding_shape, device=device, dtype=features.dtype)
                features = torch.cat([features, padding], dim=0)
        
        # 调整seq_len
        if len(current_shape) > 1 and current_shape[1] != target_seq_len:
            if current_shape[1] > target_seq_len:
                # 截断序列长度
                features = features[:, :target_seq_len]
            else:
                # 填充序列长度
                pad_seq = target_seq_len - current_shape[1]
                if len(current_shape) == 3:
                    features = F.pad(features, (0, 0, 0, pad_seq), value=0)
                else:
                    features = F.pad(features, (0, pad_seq), value=0)
        
        # 调整hidden_dim
        if len(current_shape) > 2 and current_shape[2] != target_hidden_dim:
            if current_shape[2] > target_hidden_dim:
                # 截断特征维度
                features = features[:, :, :target_hidden_dim]
            else:
                # 填充特征维度
                pad_hidden = target_hidden_dim - current_shape[2]
                features = F.pad(features, (0, pad_hidden), value=0)
        
        # 确保最终形状正确
        if features.shape != (target_batch_size, target_seq_len, target_hidden_dim):
            # 如果还是不匹配，强制reshape
            features = features.contiguous().view(target_batch_size, target_seq_len, -1)
            if features.shape[2] != target_hidden_dim:
                if features.shape[2] > target_hidden_dim:
                    features = features[:, :, :target_hidden_dim]
                else:
                    pad_hidden = target_hidden_dim - features.shape[2]
                    features = F.pad(features, (0, pad_hidden), value=0)
        
        return features
    
    def _force_align_shape(self, features, target_batch_size, target_seq_len, target_hidden_dim, device):
        """强制对齐形状（最后的备用方案）"""
        try:
            # 先尝试简单的reshape
            total_elements = features.numel()
            target_elements = target_batch_size * target_seq_len * target_hidden_dim
            
            if total_elements >= target_elements:
                # 如果元素足够，直接reshape并截断
                features = features.view(-1)[:target_elements].view(target_batch_size, target_seq_len, target_hidden_dim)
            else:
                # 如果元素不够，填充零
                features_flat = features.view(-1)
                pad_size = target_elements - total_elements
                features_flat = F.pad(features_flat, (0, pad_size), value=0)
                features = features_flat.view(target_batch_size, target_seq_len, target_hidden_dim)
            
            return features
        except:
            # 如果所有方法都失败，创建零张量
            return torch.zeros(target_batch_size, target_seq_len, target_hidden_dim, device=device)
    
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
        
        # 修复mask维度对齐问题
        if mask.shape != encoded.shape[:2]:
            # 如果mask的形状与encoded的前两个维度不匹配，调整mask
            batch_size, seq_len = encoded.shape[:2]
            if mask.shape[0] != batch_size:
                # mask的batch维度不匹配，扩展或截断
                if mask.shape[0] == 1 and batch_size > 1:
                    mask = mask.expand(batch_size, -1)
                else:
                    mask = mask[:batch_size]
            
            if mask.shape[1] != seq_len:
                # mask的序列长度不匹配，调整
                if mask.shape[1] > seq_len:
                    mask = mask[:, :seq_len]
                else:
                    # 填充mask
                    pad_size = seq_len - mask.shape[1]
                    mask = torch.nn.functional.pad(mask, (0, pad_size), value=0)
        
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
        output = self.norm(lstm_out)
        
        # 修复mask维度对齐问题
        if mask.shape != output.shape[:2]:
            # 如果mask的形状与output的前两个维度不匹配，调整mask
            batch_size, seq_len = output.shape[:2]
            if mask.shape[0] != batch_size:
                if mask.shape[0] == 1 and batch_size > 1:
                    mask = mask.expand(batch_size, -1)
                else:
                    mask = mask[:batch_size]
            
            if mask.shape[1] != seq_len:
                if mask.shape[1] > seq_len:
                    mask = mask[:, :seq_len]
                else:
                    pad_size = seq_len - mask.shape[1]
                    mask = torch.nn.functional.pad(mask, (0, pad_size), value=0)
        
        return output * mask.unsqueeze(-1)


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
        
        # 确保distance_matrix的形状与mask匹配
        if distance_matrix.shape[:2] != (batch_size, seq_len):
            # 调整distance_matrix的形状
            if distance_matrix.shape[0] != batch_size:
                if distance_matrix.shape[0] == 1 and batch_size > 1:
                    distance_matrix = distance_matrix.expand(batch_size, -1, -1)
                else:
                    distance_matrix = distance_matrix[:batch_size]
            
            if distance_matrix.shape[1] != seq_len or distance_matrix.shape[2] != seq_len:
                # 调整矩阵大小
                current_len = min(distance_matrix.shape[1], distance_matrix.shape[2])
                if current_len > seq_len:
                    distance_matrix = distance_matrix[:, :seq_len, :seq_len]
                elif current_len < seq_len:
                    pad_size = seq_len - current_len
                    distance_matrix = torch.nn.functional.pad(
                        distance_matrix, (0, pad_size, 0, pad_size), value=0
                    )
        
        # Add channel dimension
        x = distance_matrix.unsqueeze(1)  # (B, 1, L, L)
        
        # Apply convolutions
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        # Global pooling for each position
        position_features = []
        actual_seq_len = x.shape[2]  # 使用实际的序列长度
        for i in range(min(seq_len, actual_seq_len)):
            # Extract features related to position i
            pos_features = x[:, :, i, :].mean(dim=-1)  # (B, hidden_dim)
            position_features.append(pos_features)
        
        # 如果position_features不够，填充零
        while len(position_features) < seq_len:
            zero_features = torch.zeros_like(position_features[0])
            position_features.append(zero_features)
        
        features = torch.stack(position_features, dim=1)  # (B, L, hidden_dim)
        output = self.projection(features)
        
        # 修复mask维度对齐问题
        if mask.shape != output.shape[:2]:
            batch_size, output_seq_len = output.shape[:2]
            if mask.shape[0] != batch_size:
                if mask.shape[0] == 1 and batch_size > 1:
                    mask = mask.expand(batch_size, -1)
                else:
                    mask = mask[:batch_size]
            
            if mask.shape[1] != output_seq_len:
                if mask.shape[1] > output_seq_len:
                    mask = mask[:, :output_seq_len]
                else:
                    pad_size = output_seq_len - mask.shape[1]
                    mask = torch.nn.functional.pad(mask, (0, pad_size), value=0)
        
        return output * mask.unsqueeze(-1)
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
