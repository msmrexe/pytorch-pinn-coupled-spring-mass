"""
Utility functions for logging and device configuration.
"""

import logging
import sys
import torch
import os

def setup_logging(log_file: str = 'logs/pinn.log'):
    """
    Sets up logging to both console and a file.
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging setup complete.")

def get_device() -> torch.device:
    """
    Determines and returns the available device (CUDA or CPU).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    return device
