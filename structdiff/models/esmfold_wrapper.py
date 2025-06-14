# structdiff/models/esmfold_wrapper.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
import tempfile
import os

try:
    from transformers import AutoTokenizer, EsmForProteinFolding
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from Bio.PDB import PDBParser, DSSP
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

from ..utils.logger import get_logger

logger = get_logger(__name__)

# 应用 ESMFold 全局补丁（参考 fix_esmfold.py）
def apply_esmfold_global_patch():
    """应用全局 ESMFold one_hot 补丁"""
    if hasattr(apply_esmfold_global_patch, '_applied'):
        return  # 只应用一次
    
    # 全局修复：重写 one_hot 函数
    _original_one_hot = F.one_hot

    def safe_one_hot(input, num_classes=-1):
        """
        安全的 one_hot 函数，自动转换为 LongTensor
        """
        if hasattr(input, 'dtype') and input.dtype != torch.long:
            input = input.long()
        return _original_one_hot(input, num_classes)

    # 应用全局补丁
    F.one_hot = safe_one_hot
    torch.nn.functional.one_hot = safe_one_hot
    
    apply_esmfold_global_patch._applied = True
    logger.info("✓ ESMFold 全局补丁已应用")


class ESMFoldWrapper(nn.Module):
    """Wrapper for ESMFold structure prediction using Huggingface transformers"""
    
    def __init__(self, device: torch.device = None, model_name: str = "facebook/esmfold_v1"):
        super().__init__()
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.available = False  # 默认设置为 False
        self.model = None
        self.tokenizer = None
        
        # 首先检查 transformers 库是否可用
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers library is required for ESMFold")
            return
        
        # 应用全局补丁
        apply_esmfold_global_patch()
        
        # Try to load ESMFold model and tokenizer from Huggingface
        logger.info(f"尝试加载 ESMFold 模型: {model_name}...")
        try:
            # 使用与 fix_esmfold.py 相同的加载方式
            print("加载 ESMFold 模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = EsmForProteinFolding.from_pretrained(
                model_name,
                low_cpu_mem_usage=True
            )
            
            # 配置模型（参考 fix_esmfold.py）
            self.model.esm = self.model.esm.float()
            self.model = self.model.to(self.device)
            self.model.trunk.set_chunk_size(64)
            self.model.eval()
            
            # Disable gradients for ESMFold
            for param in self.model.parameters():
                param.requires_grad = False
                
            logger.info(f"✓ ESMFold 成功加载到 {self.device}")
            print("✓ ESMFold 模型加载完成")
            self.available = True
            
        except Exception as e:
            logger.error(f"无法初始化 ESMFold: {e}")
            logger.info("ESMFold 将被禁用，使用虚拟结构预测")
            print(f"❌ ESMFold 初始化失败: {e}")
            self.available = False
    
    @torch.no_grad()
    def predict_structure(
        self, 
        sequence: str,
        num_recycles: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict 3D structure from sequence using Huggingface ESMFold
        
        Args:
            sequence: Amino acid sequence
            num_recycles: Number of recycling iterations (not used in HF implementation)
            
        Returns:
            Dictionary containing:
            - positions: 3D coordinates (L, 37, 3)
            - plddt: Per-residue confidence scores (L,)
            - distogram: Predicted distance distribution
            - secondary_structure: Predicted secondary structure
        """
        logger.debug(f"预测长度为 {len(sequence)} 的序列...")
        
        # If ESMFold is not available, return dummy features immediately
        if not self.available:
            logger.debug("ESMFold not available, returning dummy features")
            return self._create_dummy_features(sequence)
        
        try:
            # 使用与 fix_esmfold.py 相同的方式
            inputs = self.tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 额外的类型检查（参考 fix_esmfold.py）
            for key in inputs:
                if hasattr(inputs[key], 'dtype') and inputs[key].dtype in [torch.int32, torch.int16, torch.int8]:
                    inputs[key] = inputs[key].long()
            
            # 运行预测
            outputs = self.model(**inputs)
            
            # Extract features
            features = self._extract_features(outputs, sequence)
            
            logger.debug("Structure prediction completed successfully")
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting features: {e}")
            return self._create_dummy_features(sequence)
    
    def _create_dummy_features(self, sequence: str) -> Dict[str, torch.Tensor]:
        """Create dummy features when prediction fails"""
        seq_len = len(sequence)
        
        # Create random but reasonable coordinates - 确保在正确设备上
        positions = torch.randn(seq_len, 37, 3, device=self.device) * 2.0
        
        # Make it look like a somewhat extended structure
        for i in range(seq_len):
            positions[i, 1] = torch.tensor([i * 3.8, 0.0, 0.0], device=self.device)  # CA atoms along x-axis
        
        features = {
            'positions': positions,
            'plddt': torch.ones(seq_len, device=self.device) * 50.0,  # Medium confidence
            'distance_matrix': self._compute_distance_matrix(positions[:, 1]),  # CA positions
            'contact_map': torch.zeros(seq_len, seq_len, device=self.device),
            'angles': torch.zeros(seq_len, 3, device=self.device),
            'secondary_structure': torch.full((seq_len,), 2, device=self.device)  # All coil
        }
        
        # Compute contact map
        features['contact_map'] = (features['distance_matrix'] < 8.0).float()
        
        return features
    
    def _extract_features(
        self, 
        output: Dict,
        sequence: str
    ) -> Dict[str, torch.Tensor]:
        """Extract structure features from ESMFold output"""
        try:
            # ESMFold的输出通常包含多个回收周期的结果，我们需要最后一个周期的最终预测。
            # 原始形状可能是 (num_recycles, batch, seq_len, n_atoms, 3)
            # 我们选择最后一个回收周期 ([-1]) 并移除批次维度 ([0])
            
            # 提取并清理 positions
            # 原始形状: (num_recycles, 1, seq_len, 37, 3)
            positions = output['positions'][-1].squeeze(0)  # -> (seq_len, 37, 3)
            
            # 提取并清理 pLDDT
            # 原始形状: (num_recycles, 1, seq_len) -> plddt
            # 还有一个 per-atom 的 plddt: (num_recycles, 1, seq_len, 37) -> ptms
            plddt = output['plddt'][-1].squeeze(0) # -> (seq_len,)
            
            # 确保所有张量在正确设备上
            positions = positions.to(self.device)
            plddt = plddt.to(self.device)
            
            seq_len = positions.shape[0]
            n_atoms = positions.shape[1] if len(positions.shape) > 1 else 1
            
            logger.debug(f"ESMFold output shapes - positions: {positions.shape}, plddt: {plddt.shape}")
            
            # 安全地提取CA原子位置
            if n_atoms > 1:
                ca_positions = positions[:, 1]  # CA atoms (index 1)
            else:
                # 如果只有一个原子，使用第一个原子作为CA
                ca_positions = positions[:, 0] if len(positions.shape) > 1 else positions
                logger.warning(f"Only {n_atoms} atoms available, using first atom as CA")
            
            # Compute distance matrix (CA-CA distances)
            distance_matrix = self._compute_distance_matrix(ca_positions)
            
            # Compute contact map (contacts < 8Å)
            contact_map = (distance_matrix < 8.0).float()
            
            # Compute proper dihedral angles instead of using raw coordinates
            if n_atoms >= 3:  # 需要至少N, CA, C三个原子
                angles = self._compute_dihedral_angles(positions)  # (seq_len, 4) - phi, psi, omega, chi1
            else:
                # 如果原子数不足，创建虚拟角度
                angles = torch.zeros(seq_len, 4, device=self.device)
                logger.warning(f"Insufficient atoms ({n_atoms}) for dihedral calculation, using dummy angles")
            
            # Pad angles to expected 10 dimensions
            angles_padded = torch.zeros(seq_len, 10, device=self.device)
            angles_padded[:, :angles.shape[1]] = angles  # Copy available angles
            
            # Secondary structure prediction (dummy for now)
            secondary_structure = torch.zeros(seq_len, dtype=torch.long, device=self.device)
            
            features = {
                'positions': positions,
                'plddt': plddt,
                'distance_matrix': distance_matrix,
                'contact_map': contact_map,
                'angles': angles_padded,  # 现在是正确的10维角度特征
                'secondary_structure': secondary_structure,
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting features: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return self._create_dummy_features(sequence)
    
    def _compute_dihedral_angles(self, positions: torch.Tensor) -> torch.Tensor:
        """计算蛋白质骨架二面角"""
        seq_len = positions.shape[0]
        n_atoms = positions.shape[1] if len(positions.shape) > 1 else 1
        angles = torch.zeros(seq_len, 4, device=self.device)  # phi, psi, omega, chi1
        
        try:
            # 检查是否有足够的原子
            if n_atoms < 3:
                logger.warning(f"Insufficient atoms ({n_atoms}) for backbone dihedral calculation")
                return angles
            
            # Extract backbone atoms (N, CA, C)
            N = positions[:, 0]   # N atoms
            CA = positions[:, 1]  # CA atoms  
            C = positions[:, 2]   # C atoms
            
            for i in range(1, seq_len - 1):
                try:
                    # Phi angle: C(i-1) - N(i) - CA(i) - C(i)
                    if i > 0:
                        phi = self._dihedral_angle(C[i-1], N[i], CA[i], C[i])
                        if torch.isfinite(phi):
                            angles[i, 0] = phi
                    
                    # Psi angle: N(i) - CA(i) - C(i) - N(i+1)
                    if i < seq_len - 1:
                        psi = self._dihedral_angle(N[i], CA[i], C[i], N[i+1])
                        if torch.isfinite(psi):
                            angles[i, 1] = psi
                    
                    # Omega angle: CA(i-1) - C(i-1) - N(i) - CA(i)  
                    if i > 0:
                        omega = self._dihedral_angle(CA[i-1], C[i-1], N[i], CA[i])
                        if torch.isfinite(omega):
                            angles[i, 2] = omega
                            
                except Exception as e:
                    logger.debug(f"Error computing angle for residue {i}: {e}")
                    continue
                    
            # 将角度归一化到 [-π, π] 并转换为 [-1, 1]
            angles = torch.tanh(angles)  # 平滑归一化
            
        except Exception as e:
            logger.warning(f"Error computing dihedral angles: {e}")
            # 返回随机但合理的角度
            angles = torch.randn(seq_len, 4, device=self.device) * 0.1
            
        return angles
    
    def _dihedral_angle(self, p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
        """计算四个点定义的二面角"""
        try:
            # 计算向量
            b1 = p2 - p1
            b2 = p3 - p2
            b3 = p4 - p3
            
            # 归一化
            b1_norm = F.normalize(b1, dim=-1)
            b2_norm = F.normalize(b2, dim=-1)
            b3_norm = F.normalize(b3, dim=-1)
            
            # 计算法向量
            n1 = torch.cross(b1_norm, b2_norm)
            n2 = torch.cross(b2_norm, b3_norm)
            
            # 归一化法向量
            n1_norm = F.normalize(n1, dim=-1)
            n2_norm = F.normalize(n2, dim=-1)
            
            # 计算二面角
            cos_angle = torch.clamp(torch.sum(n1_norm * n2_norm, dim=-1), -1.0, 1.0)
            angle = torch.acos(cos_angle)
            
            # 确定符号
            sign = torch.sign(torch.sum(n1_norm * b3_norm, dim=-1))
            angle = angle * sign
            
            return angle
            
        except Exception as e:
            # 返回随机角度作为后备
            return torch.randn(1, device=self.device).squeeze() * 0.1
    
    def _compute_distance_matrix(
        self, 
        ca_positions: torch.Tensor
    ) -> torch.Tensor:
        """Compute CA-CA distance matrix"""
        # Compute pairwise distances
        diff = ca_positions.unsqueeze(0) - ca_positions.unsqueeze(1)
        distances = torch.norm(diff, dim=-1)
        
        return distances
    
    def _predict_secondary_structure(
        self, 
        positions: torch.Tensor,
        sequence: str
    ) -> torch.Tensor:
        """Predict secondary structure from 3D coordinates"""
        if not BIOPYTHON_AVAILABLE:
            logger.warning("BioPython not available. Using fallback for secondary structure.")
            # Fallback: all coil - 确保在正确设备上
            return torch.full((len(sequence),), 2, device=self.device)
        
        # Save to temporary PDB file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            self._write_pdb(f, positions, sequence)
            pdb_path = f.name
        
        try:
            # Use DSSP for secondary structure assignment
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('peptide', pdb_path)
            model = structure[0]
            
            dssp = DSSP(model, pdb_path, dssp='mkdssp')
            
            # Convert DSSP output to simple categories
            ss_map = {
                'H': 0, 'G': 0, 'I': 0,  # Helix
                'E': 1, 'B': 1,           # Sheet
                'T': 2, 'S': 2, '-': 2    # Coil
            }
            
            ss_sequence = []
            for residue in dssp.property_list:
                ss = residue[2]
                ss_sequence.append(ss_map.get(ss, 2))
            
            return torch.tensor(ss_sequence, device=self.device)  # 确保在正确设备上
            
        except Exception as e:
            logger.warning(f"DSSP failed: {e}. Using fallback.")
            # Fallback: all coil - 确保在正确设备上
            return torch.full((len(sequence),), 2, device=self.device)
        
        finally:
            # Clean up
            if os.path.exists(pdb_path):
                os.remove(pdb_path)
    
    def fold_sequence(self, sequence: str) -> Dict:
        """
        为PeptideEvaluator提供的接口方法
        """
        if not self.available:
            return None
        
        try:
            result = self.predict_structure(sequence)
            return {
                'plddt': result.get('plddt', torch.zeros(len(sequence), device=self.device))
            }
        except Exception as e:
            logger.warning(f"序列折叠失败: {e}")
            return None
    
    def _write_pdb(
        self, 
        file_handle,
        positions: torch.Tensor,
        sequence: str
    ):
        """Write coordinates to PDB format"""
        atom_types = ['N', 'CA', 'C', 'O']  # Backbone atoms
        
        atom_idx = 1
        for res_idx, aa in enumerate(sequence):
            for atom_idx_in_res, atom_type in enumerate(atom_types):
                if atom_idx_in_res < positions.shape[1]:
                    x, y, z = positions[res_idx, atom_idx_in_res].tolist()
                    
                    file_handle.write(
                        f"ATOM  {atom_idx:5d}  {atom_type:<3s} {aa:3s} A"
                        f"{res_idx+1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}"
                        f"  1.00  0.00           {atom_type[0]:>2s}\n"
                    )
                    atom_idx += 1


# Update structure_utils.py to use ESMFoldWrapper
def predict_structure_with_esmfold_v2(
    sequence: str,
    esmfold_wrapper: Optional[ESMFoldWrapper] = None,
    device: Optional[torch.device] = None
) -> Dict[str, torch.Tensor]:
    """
    Predict structure using ESMFold with proper feature extraction
    
    Args:
        sequence: Amino acid sequence
        esmfold_wrapper: Pre-initialized wrapper (optional)
        device: Device to use
        
    Returns:
        Dictionary of structural features
    """
    if esmfold_wrapper is None:
        esmfold_wrapper = ESMFoldWrapper(device)
    
    # Predict structure
    features = esmfold_wrapper.predict_structure(sequence)
    
    # Add derived features
    seq_len = len(sequence)
    
    # Compute solvent accessibility (simplified)
    features['sasa'] = torch.rand(seq_len) * 200  # Placeholder
    
    # Compute residue depth
    ca_positions = features['positions'][:, 1]  # CA atoms
    center = ca_positions.mean(dim=0)
    features['residue_depth'] = torch.norm(ca_positions - center, dim=-1)
    
    # Compute local structure features
    features['local_backbone_rmsd'] = compute_local_backbone_rmsd(
        features['positions']
    )
    
    return features


def compute_local_backbone_rmsd(
    positions: torch.Tensor,
    window_size: int = 5
) -> torch.Tensor:
    """Compute local backbone RMSD for each residue"""
    seq_len = positions.shape[0]
    rmsd_values = torch.zeros(seq_len)
    
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        
        if end - start < 3:
            continue
        
        # Get local backbone coordinates
        local_coords = positions[start:end, :4]  # N, CA, C, O
        
        # Compute RMSD from ideal geometry (simplified)
        rmsd_values[i] = torch.std(local_coords.reshape(-1, 3)).item()
    
    return rmsd_values
# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
