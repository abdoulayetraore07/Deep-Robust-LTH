"""
Mask utilities for pruning
"""

import torch
import torch.nn as nn
from typing import Dict


class MaskApplier:
    """
    Apply pruning masks during training
    """
    
    def __init__(self, mask: Dict[str, torch.Tensor]):
        """
        Initialize mask applier
        
        Args:
            mask: Dictionary of {parameter_name: binary_mask}
        """
        self.mask = mask
    
    def apply(self, model: nn.Module) -> None:
        """
        Apply mask to model weights (zeroing pruned weights)
        
        Args:
            model: Neural network
        """
        for name, param in model.named_parameters():
            if name in self.mask:
                param.data *= self.mask[name]
                # Also zero gradients of pruned weights
                if param.grad is not None:
                    param.grad *= self.mask[name]
    
    def register_hooks(self, model: nn.Module) -> None:
        """
        Register hooks to automatically apply mask after each gradient computation
        
        Args:
            model: Neural network
        """
        for name, param in model.named_parameters():
            if name in self.mask:
                def make_hook(mask):
                    def hook(grad):
                        return grad * mask
                    return hook
                param.register_hook(make_hook(self.mask[name]))