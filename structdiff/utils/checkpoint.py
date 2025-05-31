import os
import torch
from typing import Dict, Optional
import shutil


class CheckpointManager:
    """Manage model checkpoints"""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(
        self,
        state_dict: Dict,
        epoch: int,
        is_best: bool = False
    ):
        """Save checkpoint"""
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pth"
        )
        torch.save(state_dict, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            shutil.copy(checkpoint_path, best_path)
        
        # Save latest
        latest_path = os.path.join(self.checkpoint_dir, "latest.pth")
        shutil.copy(checkpoint_path, latest_path)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones"""
        checkpoints = []
        for f in os.listdir(self.checkpoint_dir):
            if f.startswith("checkpoint_epoch_") and f.endswith(".pth"):
                epoch = int(f.split("_")[2].split(".")[0])
                checkpoints.append((epoch, f))
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x[0])
        
        # Remove old checkpoints
        while len(checkpoints) > self.max_checkpoints:
            _, filename = checkpoints.pop(0)
            os.remove(os.path.join(self.checkpoint_dir, filename))
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04
