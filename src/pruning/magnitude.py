"""
Magnitude-based pruning
"""

import torch
import torch.nn as nn
from typing import Dict
from pathlib import Path


def magnitude_pruning(
    model: nn.Module,
    sparsity: float,
    exclude_output: bool = True
) -> Dict[str, torch.Tensor]:
    """
    One-shot global magnitude-based pruning
    
    Args:
        model: Neural network
        sparsity: Target sparsity (fraction of weights to prune)
        exclude_output: Whether to exclude output layer from pruning
        
    Returns:
        mask: Dictionary of binary masks for each layer
    """
    # Collect all weights
    all_weights = []
    weight_names = []
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Exclude output layer if specified
            if exclude_output and 'network.8' in name:  # Last layer
                continue
            all_weights.append(param.data.abs().flatten())
            weight_names.append(name)
    
    # Concatenate all weights
    all_weights_flat = torch.cat(all_weights)
    
    # Compute threshold
    num_weights = len(all_weights_flat)
    k = int(sparsity * num_weights)
    threshold = torch.kthvalue(all_weights_flat, k)[0]
    
    # Create masks
    masks = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            if exclude_output and 'network.8' in name:
                # Keep output layer unpruned
                masks[name] = torch.ones_like(param.data)
            else:
                masks[name] = (param.data.abs() >= threshold).float()
    
    return masks


def apply_mask(model: nn.Module, mask: Dict[str, torch.Tensor]) -> None:
    """
    Apply pruning mask to model weights
    
    Args:
        model: Neural network
        mask: Dictionary of binary masks
    """
    for name, param in model.named_parameters():
        if name in mask:
            param.data *= mask[name].to(param.device)


def get_sparsity(model: nn.Module) -> float:
    """
    Compute current sparsity of model
    
    Args:
        model: Neural network
        
    Returns:
        sparsity: Fraction of zero weights
    """
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param.data == 0).sum().item()
    
    return zero_params / total_params


def save_mask(mask: Dict[str, torch.Tensor], filepath: str) -> None:
    """
    Save pruning mask to file
    
    Args:
        mask: Dictionary of binary masks
        filepath: Path to save mask
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(mask, filepath)


def load_mask(filepath: str) -> Dict[str, torch.Tensor]:
    """
    Load pruning mask from file
    
    Args:
        filepath: Path to mask file
        
    Returns:
        mask: Dictionary of binary masks
    """
    return torch.load(filepath)


# ✅ FONCTION MANQUANTE
def rewind_weights(
    model: nn.Module,
    init_weights_path: str,
    mask: Dict[str, torch.Tensor]
) -> None:
    """
    Rewind model to initial weights and apply mask
    
    This is the core operation for Lottery Ticket Hypothesis:
    - Load weights from initialization (theta_0)
    - Apply pruning mask
    
    Args:
        model: Neural network
        init_weights_path: Path to initial weights
        mask: Pruning mask to apply
    """
    # Load initial weights
    init_state = torch.load(init_weights_path)
    model.load_state_dict(init_state)
    
    # Apply mask
    apply_mask(model, mask)
    
    print(f"Model rewound to {init_weights_path} with sparsity {get_sparsity(model):.2%}")


def iterative_pruning(
    model: nn.Module,
    train_fn: callable,
    target_sparsity: float,
    num_iterations: int = 5,
    exclude_output: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Iterative magnitude pruning (standard LTH protocol)
    
    Args:
        model: Neural network
        train_fn: Function that trains the model
        target_sparsity: Target sparsity
        num_iterations: Number of pruning iterations
        exclude_output: Whether to exclude output layer
        
    Returns:
        final_mask: Final pruning mask
    """
    # Compute per-iteration sparsity
    per_iter_sparsity = 1 - (1 - target_sparsity) ** (1 / num_iterations)
    
    current_mask = {name: torch.ones_like(param) 
                   for name, param in model.named_parameters() if 'weight' in name}
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Train
        train_fn(model)
        
        # Prune
        new_mask = magnitude_pruning(model, per_iter_sparsity, exclude_output)
        
        # Combine with existing mask
        for name in current_mask:
            if name in new_mask:
                current_mask[name] *= new_mask[name]
        
        # Apply combined mask
        apply_mask(model, current_mask)
        
        print(f"Current sparsity: {get_sparsity(model):.2%}")
    
    return current_mask