import logging
import sys
from typing import Optional


def setup_logger(
    log_file: Optional[str] = None,
    level: int = logging.INFO
):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Override existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger for module"""
    return logging.getLogger(name)
# Updated: 05/30/2025 22:59:09

# Updated: 05/31/2025 15:11:07

# Updated: 05/31/2025 15:14:04

# Updated: 05/31/2025 23:30:00

# Updated: 05/31/2025 23:30:18
