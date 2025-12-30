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
            mask: Dictionary of {layer_name: binary_mask}
        """
        self.mask = mask
    
    def apply(self, model: nn.Module) -> None:
        """
        Apply mask to model weights (zeroing pruned weights)
        
        Args:
            model: Neural network
        """
        for name, module in model.named_modules():
            if name in self.mask and isinstance(module, nn.Linear):
                module.weight.data *= self.mask[name]
                
                # Also zero gradients of pruned weights
                if module.weight.grad is not None:
                    module.weight.grad *= self.mask[name]
    
    def register_hooks(self, model: nn.Module) -> None:
        """
        Register hooks to automatically apply mask after each optimizer step
        
        Args:
            model: Neural network
        """
        for name, module in model.named_modules():
            if name in self.mask and isinstance(module, nn.Linear):
                def make_hook(mask):
                    def hook(grad):
                        return grad * mask
                    return hook
                
                module.weight.register_hook(make_hook(self.mask[name]))