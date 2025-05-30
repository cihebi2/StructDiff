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