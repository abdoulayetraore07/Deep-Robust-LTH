"""
Mask Management for Lottery Ticket Hypothesis

Implements binary masks for network pruning following Frankle & Carlin (2019).

A mask is a binary tensor (0 or 1) that multiplies the weights:
    W_effective = W * mask

Key operations:
- Create masks from weights
- Apply masks to model
- Save/load masks
- Compute sparsity statistics
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import os
import copy


class MaskManager:
    """
    Manages binary masks for pruning.
    
    Each mask corresponds to a weight tensor in the model.
    Masks are stored as dictionaries: {layer_name: mask_tensor}
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize mask manager.
        
        Args:
            model: Neural network model
        """
        self.model = model
        self.masks: Dict[str, torch.Tensor] = {}
        
        # Initialize all masks to ones (no pruning)
        self._init_masks()
    
    def _init_masks(self):
        """Initialize masks to all ones for each weight tensor."""
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                self.masks[name] = torch.ones_like(param.data)
    
    def apply_masks(self):
        """Apply current masks to model weights (zero out pruned weights)."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    param.data.mul_(self.masks[name])
    
    def register_mask_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Register forward hooks to apply masks during forward pass.
        
        This ensures pruned weights stay zero even after gradient updates.
        
        Returns:
            List of hook handles (for removal)
        """
        handles = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Find corresponding mask
                weight_name = f"{name}.weight" if name else "weight"
                
                # Search for the mask
                mask_key = None
                for key in self.masks.keys():
                    if key.endswith('.weight') and name in key:
                        mask_key = key
                        break
                
                if mask_key is not None:
                    mask = self.masks[mask_key]
                    
                    def make_hook(m):
                        def hook(module, input, output):
                            # Apply mask after forward
                            module.weight.data.mul_(m)
                        return hook
                    
                    handle = module.register_forward_hook(make_hook(mask))
                    handles.append(handle)
        
        return handles
    
    def register_gradient_hooks(self) -> List[torch.utils.hooks.RemovableHandle]:
        """
        Register backward hooks to zero gradients for pruned weights.
        
        This prevents pruned weights from being updated during training.
        
        Returns:
            List of hook handles
        """
        handles = []
        
        for name, param in self.model.named_parameters():
            if name in self.masks:
                mask = self.masks[name]
                
                def make_hook(m):
                    def hook(grad):
                        return grad * m
                    return hook
                
                handle = param.register_hook(make_hook(mask))
                handles.append(handle)
        
        return handles
    
    def update_mask(self, name: str, new_mask: torch.Tensor):
        """
        Update mask for a specific layer.
        
        Args:
            name: Layer name
            new_mask: New binary mask tensor
        """
        if name not in self.masks:
            raise ValueError(f"Unknown layer: {name}")
        
        if new_mask.shape != self.masks[name].shape:
            raise ValueError(f"Mask shape mismatch: {new_mask.shape} vs {self.masks[name].shape}")
        
        self.masks[name] = new_mask.to(self.masks[name].device)
    
    def get_mask(self, name: str) -> torch.Tensor:
        """Get mask for a specific layer."""
        if name not in self.masks:
            raise ValueError(f"Unknown layer: {name}")
        return self.masks[name]
    
    def get_sparsity(self) -> Dict[str, float]:
        """
        Compute sparsity (fraction of zeros) for each layer and overall.
        
        Returns:
            Dictionary with per-layer and total sparsity
        """
        stats = {}
        total_zeros = 0
        total_params = 0
        
        for name, mask in self.masks.items():
            n_zeros = (mask == 0).sum().item()
            n_params = mask.numel()
            sparsity = n_zeros / n_params
            
            stats[name] = {
                'sparsity': sparsity,
                'zeros': n_zeros,
                'total': n_params,
                'remaining': n_params - n_zeros
            }
            
            total_zeros += n_zeros
            total_params += n_params
        
        stats['total'] = {
            'sparsity': total_zeros / total_params if total_params > 0 else 0,
            'zeros': total_zeros,
            'total': total_params,
            'remaining': total_params - total_zeros
        }
        
        return stats
    
    def get_total_sparsity(self) -> float:
        """Get overall sparsity as a single float."""
        stats = self.get_sparsity()
        return stats['total']['sparsity']
    
    def get_remaining_weights(self) -> int:
        """Get total number of non-zero weights."""
        stats = self.get_sparsity()
        return stats['total']['remaining']
    
    def save(self, filepath: str):
        """
        Save masks to file.
        
        Args:
            filepath: Path to save masks
        """
        # Convert to CPU for saving
        masks_cpu = {name: mask.cpu() for name, mask in self.masks.items()}
        torch.save(masks_cpu, filepath)
    
    def load(self, filepath: str, device: Optional[torch.device] = None):
        """
        Load masks from file.
        
        Args:
            filepath: Path to load masks from
            device: Device to load masks to
        """
        masks_loaded = torch.load(filepath, map_location='cpu')
        
        for name, mask in masks_loaded.items():
            if name in self.masks:
                if device is not None:
                    mask = mask.to(device)
                self.masks[name] = mask
    
    def to(self, device: torch.device):
        """Move all masks to device."""
        self.masks = {name: mask.to(device) for name, mask in self.masks.items()}
        return self
    
    def clone(self) -> 'MaskManager':
        """Create a deep copy of the mask manager."""
        new_manager = MaskManager.__new__(MaskManager)
        new_manager.model = self.model
        new_manager.masks = {name: mask.clone() for name, mask in self.masks.items()}
        return new_manager
    
    def reset(self):
        """Reset all masks to ones (no pruning)."""
        self._init_masks()
    
    def combine_with(self, other: 'MaskManager', operation: str = 'and') -> 'MaskManager':
        """
        Combine with another mask manager.
        
        Args:
            other: Another MaskManager
            operation: 'and' (intersection) or 'or' (union)
            
        Returns:
            New MaskManager with combined masks
        """
        combined = self.clone()
        
        for name in self.masks:
            if name in other.masks:
                if operation == 'and':
                    combined.masks[name] = self.masks[name] * other.masks[name]
                elif operation == 'or':
                    combined.masks[name] = torch.clamp(self.masks[name] + other.masks[name], max=1)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
        
        return combined
    
    def summary(self) -> str:
        """Get a summary string of mask statistics."""
        stats = self.get_sparsity()
        
        lines = [
            "Mask Summary",
            "=" * 50
        ]
        
        for name, layer_stats in stats.items():
            if name != 'total':
                lines.append(
                    f"  {name}: {layer_stats['sparsity']:.2%} sparse "
                    f"({layer_stats['remaining']}/{layer_stats['total']} remaining)"
                )
        
        lines.append("-" * 50)
        lines.append(
            f"  TOTAL: {stats['total']['sparsity']:.2%} sparse "
            f"({stats['total']['remaining']}/{stats['total']['total']} remaining)"
        )
        lines.append("=" * 50)
        
        return "\n".join(lines)


def create_mask_from_weights(
    model: nn.Module,
    threshold: float = 0.0
) -> Dict[str, torch.Tensor]:
    """
    Create masks based on weight magnitudes.
    
    Args:
        model: Neural network
        threshold: Weights with |w| <= threshold are masked (set to 0)
        
    Returns:
        Dictionary of masks
    """
    masks = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            masks[name] = (param.data.abs() > threshold).float()
    
    return masks


def create_random_mask(
    model: nn.Module,
    sparsity: float,
    seed: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    """
    Create random masks with given sparsity.
    
    Args:
        model: Neural network
        sparsity: Fraction of weights to prune (0 to 1)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of masks
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    masks = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            mask = torch.ones_like(param.data)
            n_prune = int(sparsity * param.numel())
            
            # Random indices to prune
            indices = torch.randperm(param.numel())[:n_prune]
            mask.view(-1)[indices] = 0
            
            masks[name] = mask
    
    return masks


def apply_mask_to_model(model: nn.Module, masks: Dict[str, torch.Tensor]):
    """
    Apply masks directly to model weights.
    
    Args:
        model: Neural network
        masks: Dictionary of masks
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.data.mul_(masks[name].to(param.device))


def count_parameters(model: nn.Module, masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, int]:
    """
    Count model parameters (total and remaining after masking).
    
    Args:
        model: Neural network
        masks: Optional masks
        
    Returns:
        Dictionary with parameter counts
    """
    total = 0
    remaining = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            n = param.numel()
            total += n
            
            if masks is not None and name in masks:
                remaining += masks[name].sum().item()
            else:
                remaining += n
    
    return {
        'total': int(total),
        'remaining': int(remaining),
        'pruned': int(total - remaining),
        'sparsity': (total - remaining) / total if total > 0 else 0
    }