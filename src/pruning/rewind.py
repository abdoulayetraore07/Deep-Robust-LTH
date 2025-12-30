"""
Weight rewinding utilities
"""

import torch
import torch.nn as nn
from typing import Dict
from pathlib import Path


def save_initial_weights(model: nn.Module, filepath: str) -> None:
    """
    Save initial weights (theta_0) before training
    
    Args:
        model: Neural network
        filepath: Path to save weights
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"Initial weights saved to {filepath}")


def rewind_to_initialization(
    model: nn.Module,
    init_weights_path: str,
    mask: Dict[str, torch.Tensor]
) -> None:
    """
    Rewind model to initial weights and apply pruning mask
    
    Args:
        model: Neural network
        init_weights_path: Path to initial weights
        mask: Pruning mask to apply
    """
    # Load initial weights
    init_state = torch.load(init_weights_path)
    model.load_state_dict(init_state)
    
    # Apply mask
    from .magnitude import apply_mask
    apply_mask(model, mask)
    
    print(f"Model rewound to initialization from {init_weights_path}")


def rewind_to_epoch(
    model: nn.Module,
    checkpoint_path: str,
    mask: Dict[str, torch.Tensor],
    epoch: int = 0
) -> None:
    """
    Rewind model to specific epoch (for iterative pruning)
    
    Args:
        model: Neural network
        checkpoint_path: Path to checkpoint
        mask: Pruning mask
        epoch: Epoch to rewind to
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    from .magnitude import apply_mask
    apply_mask(model, mask)
    
    print(f"Model rewound to epoch {epoch}")