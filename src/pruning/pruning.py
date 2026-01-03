"""
Pruning Module using PyTorch's native torch.nn.utils.prune

This replaces the custom MaskManager, magnitude pruning, and rewind systems
with PyTorch's battle-tested pruning API.

Key advantages:
- Masks are stored as module buffers (saved with model automatically)
- Forward hooks are handled correctly by PyTorch
- Gradients for pruned weights are automatically zeroed
- No manual hook registration needed

Replaces: masks.py, magnitude.py, rewind.py
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Tuple, Callable
from copy import deepcopy


class PruningManager:
    """
    Manages pruning operations using PyTorch's native API.
    
    Usage:
        pm = PruningManager(model)
        pm.save_initial_weights()  # For LTH rewinding
        
        # Train model...
        
        pm.prune_by_magnitude(sparsity=0.8)  # Prune 80%
        pm.rewind_to_initial()  # Rewind unpruned weights to initial values
        
        # Retrain...
        
        pm.make_permanent()  # Optional: fuse masks into weights
    """
    
    def __init__(self, model: nn.Module, exclude_layers: Optional[List[str]] = None):
        """
        Initialize pruning manager.
        
        Args:
            model: The neural network to prune
            exclude_layers: List of layer name patterns to exclude (e.g., ['output', 'bias'])
        """
        self.model = model
        self.exclude_layers = exclude_layers or []
        self.initial_weights: Dict[str, torch.Tensor] = {}
        self._pruned_params: List[Tuple[nn.Module, str]] = []
    
    def save_initial_weights(self) -> None:
        """
        Save initial weights for LTH rewinding.
        Call this BEFORE training the dense model.
        """
        self.initial_weights.clear()
        for name, param in self.model.named_parameters():
            if self._should_prune(name):
                # Store on same device as parameter
                self.initial_weights[name] = param.data.clone()
        print(f"[Pruning] Saved initial weights for {len(self.initial_weights)} parameters")
    
    def _should_prune(self, name: str) -> bool:
        """Check if a parameter should be pruned."""
        # Only prune weights, not biases
        if 'bias' in name:
            return False
        if 'weight' not in name:
            return False
        # Check exclusion patterns
        for pattern in self.exclude_layers:
            if pattern in name:
                return False
        return True
    
    def _get_prunable_modules(self) -> List[Tuple[nn.Module, str]]:
        """Get list of (module, param_name) tuples that can be pruned."""
        prunable = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                full_name = f"{name}.weight" if name else "weight"
                if self._should_prune(full_name):
                    prunable.append((module, 'weight'))
        return prunable
    
    def prune_by_magnitude(self, sparsity: float) -> Dict[str, float]:
        """
        Prune weights globally by magnitude.
        
        Uses L1 unstructured pruning: removes the smallest magnitude weights
        across ALL layers (global pruning).
        
        Args:
            sparsity: Fraction of weights to prune (0.8 = remove 80%)
            
        Returns:
            Dictionary of per-layer sparsities
        """
        if not 0 < sparsity < 1:
            raise ValueError(f"Sparsity must be in (0, 1), got {sparsity}")
        
        # Get prunable parameters
        prunable = self._get_prunable_modules()
        
        if not prunable:
            raise ValueError("No prunable modules found!")
        
        # Apply global unstructured pruning
        prune.global_unstructured(
            prunable,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )
        
        # Store for later reference
        self._pruned_params = prunable
        
        # Compute per-layer sparsities
        sparsities = self.get_sparsity()
        
        print(f"[Pruning] Applied {sparsity:.1%} global magnitude pruning")
        print(f"[Pruning] Actual sparsity: {sparsities['total']:.2%}")
        
        return sparsities
    
    def prune_layer_wise(self, sparsity: float) -> Dict[str, float]:
        """
        Prune each layer independently to the same sparsity.
        
        Args:
            sparsity: Fraction of weights to prune per layer
            
        Returns:
            Dictionary of per-layer sparsities
        """
        prunable = self._get_prunable_modules()
        
        for module, param_name in prunable:
            prune.l1_unstructured(module, param_name, amount=sparsity)
        
        self._pruned_params = prunable
        
        sparsities = self.get_sparsity()
        print(f"[Pruning] Applied {sparsity:.1%} layer-wise magnitude pruning")
        print(f"[Pruning] Actual sparsity: {sparsities['total']:.2%}")
        
        return sparsities
    
    def rewind_to_initial(self) -> None:
        """
        Rewind unpruned weights to their initial values (LTH protocol).
        
        This implements the "rewinding" step from the Lottery Ticket Hypothesis:
        - Pruned weights stay at zero (enforced by mask)
        - Unpruned weights are reset to their initial values θ₀
        """
        if not self.initial_weights:
            raise RuntimeError("No initial weights saved! Call save_initial_weights() first.")
        
        if not self._pruned_params:
            raise RuntimeError("Model not pruned yet! Call prune_by_magnitude() first.")
        
        with torch.no_grad():
            for module, param_name in self._pruned_params:
                # Get the mask
                mask = getattr(module, f"{param_name}_mask")
                # Get the original weight tensor (before masking)
                weight_orig = getattr(module, f"{param_name}_orig")
                
                # Find the corresponding initial weight
                for name, mod in self.model.named_modules():
                    if mod is module:
                        full_name = f"{name}.{param_name}" if name else param_name
                        # Try both with and without _orig suffix
                        init_key = full_name
                        if init_key not in self.initial_weights:
                            init_key = f"{full_name}_orig"
                        if init_key not in self.initial_weights:
                            # Search for matching key
                            for key in self.initial_weights:
                                if param_name in key and name in key:
                                    init_key = key
                                    break
                        
                        if init_key in self.initial_weights:
                            initial = self.initial_weights[init_key].to(weight_orig.device)
                            # Copy initial weights, mask will be applied automatically by PyTorch
                            weight_orig.data.copy_(initial)
                        break
        
        # Verify sparsity is maintained
        self._verify_sparsity()
        
        print("[Pruning] Rewound to initial weights with masks applied")
    
    def _verify_sparsity(self) -> None:
        """Verify that pruned weights are actually zero."""
        for module, param_name in self._pruned_params:
            # Get the effective weight (with mask applied)
            weight = getattr(module, param_name)
            mask = getattr(module, f"{param_name}_mask")
            
            # Check that masked positions are zero
            masked_weights = weight.data[mask == 0]
            if masked_weights.numel() > 0:
                max_violation = masked_weights.abs().max().item()
                if max_violation > 1e-8:
                    print(f"[WARNING] Pruning violation detected: max={max_violation:.2e}")
    
    def get_sparsity(self) -> Dict[str, float]:
        """
        Get actual sparsity of weights (not just mask).
        
        Returns:
            Dictionary with per-layer and total sparsity
        """
        sparsities = {}
        total_zeros = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and 'mask' not in name and '_orig' not in name:
                zeros = (param.data.abs() < 1e-8).sum().item()
                total = param.numel()
                sparsities[name] = zeros / total
                total_zeros += zeros
                total_params += total
        
        # Also check effective weights for pruned modules
        for module, param_name in self._pruned_params:
            weight = getattr(module, param_name)
            zeros = (weight.data.abs() < 1e-8).sum().item()
            total = weight.numel()
            
            for n, m in self.model.named_modules():
                if m is module:
                    full_name = f"{n}.{param_name}" if n else param_name
                    sparsities[full_name] = zeros / total
                    break
            
            # Update totals (avoid double counting)
            if total_params == 0:
                total_zeros += zeros
                total_params += total
        
        sparsities['total'] = total_zeros / total_params if total_params > 0 else 0.0
        return sparsities
    
    def get_mask_sparsity(self) -> Dict[str, float]:
        """Get sparsity based on masks (what PyTorch thinks is pruned)."""
        sparsities = {}
        total_zeros = 0
        total_params = 0
        
        for module, param_name in self._pruned_params:
            if hasattr(module, f"{param_name}_mask"):
                mask = getattr(module, f"{param_name}_mask")
                zeros = (mask == 0).sum().item()
                total = mask.numel()
                
                # Get full name
                for n, m in self.model.named_modules():
                    if m is module:
                        full_name = f"{n}.{param_name}" if n else param_name
                        sparsities[full_name] = zeros / total
                        break
                
                total_zeros += zeros
                total_params += total
        
        sparsities['total'] = total_zeros / total_params if total_params > 0 else 0.0
        return sparsities
    
    def make_permanent(self) -> None:
        """
        Make pruning permanent by removing masks and hooks.
        
        After this, the pruned weights are simply zeros in the weight tensor,
        with no mask overhead. The model can be saved normally.
        
        WARNING: After calling this, you cannot rewind or change the pruning!
        """
        for module, param_name in self._pruned_params:
            if prune.is_pruned(module):
                prune.remove(module, param_name)
        
        self._pruned_params = []
        print("[Pruning] Made pruning permanent (masks removed)")
    
    def is_pruned(self) -> bool:
        """Check if the model has active pruning."""
        return len(self._pruned_params) > 0
    
    def verify_integrity(self) -> bool:
        """
        Verify that pruning is working correctly.
        
        Checks:
        1. Mask sparsity matches actual weight sparsity
        2. All masked weights are exactly zero
        
        Returns:
            True if pruning is intact, False otherwise
        """
        if not self._pruned_params:
            return True  # No pruning to verify
        
        mask_sparsity = self.get_mask_sparsity()
        actual_sparsity = self.get_sparsity()
        
        # Check sparsity match
        if abs(mask_sparsity.get('total', 0) - actual_sparsity.get('total', 0)) > 0.01:
            print(f"[ERROR] Sparsity mismatch: mask={mask_sparsity['total']:.2%}, "
                  f"actual={actual_sparsity['total']:.2%}")
            return False
        
        # Check masked weights are zero
        for module, param_name in self._pruned_params:
            if hasattr(module, f"{param_name}_mask"):
                mask = getattr(module, f"{param_name}_mask")
                weight = getattr(module, param_name)
                
                masked_weights = weight.data[mask == 0]
                if masked_weights.numel() > 0:
                    max_val = masked_weights.abs().max().item()
                    if max_val > 1e-8:
                        print(f"[ERROR] Masked weights not zero: max={max_val:.2e}")
                        return False
        
        return True
    
    def get_pruned_params(self) -> List[Tuple[nn.Module, str]]:
        """Get list of pruned (module, param_name) tuples."""
        return self._pruned_params.copy()
    
    def summary(self) -> str:
        """Get a summary of pruning state."""
        if not self._pruned_params:
            return "Model not pruned"
        
        mask_sp = self.get_mask_sparsity()
        actual_sp = self.get_sparsity()
        
        lines = [
            "Pruning Summary",
            "=" * 40,
            f"Pruned layers: {len(self._pruned_params)}",
            f"Mask sparsity: {mask_sp.get('total', 0):.2%}",
            f"Actual sparsity: {actual_sp.get('total', 0):.2%}",
            f"Initial weights saved: {len(self.initial_weights) > 0}",
            f"Integrity check: {'PASS' if self.verify_integrity() else 'FAIL'}",
            "=" * 40
        ]
        return "\n".join(lines)


def iterative_magnitude_pruning(
    model: nn.Module,
    train_fn: Callable[[nn.Module], Dict],
    target_sparsity: float,
    n_rounds: int = 5,
    rewind: bool = True,
    exclude_layers: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[nn.Module, PruningManager, Dict]:
    """
    Iterative Magnitude Pruning (IMP) as described in the LTH paper.
    
    Protocol:
    1. Train dense network to completion
    2. Prune p% of remaining weights
    3. Rewind unpruned weights to initial values
    4. Repeat until target sparsity reached
    
    Args:
        model: The model to prune
        train_fn: Function that takes model and trains it (returns metrics dict)
        target_sparsity: Final target sparsity (e.g., 0.9 for 90%)
        n_rounds: Number of pruning rounds
        rewind: Whether to rewind to initial weights (True for LTH)
        exclude_layers: Layers to exclude from pruning
        verbose: Print progress
        
    Returns:
        Tuple of (pruned_model, pruning_manager, results_dict)
    """
    pm = PruningManager(model, exclude_layers=exclude_layers)
    
    # Save initial weights for rewinding
    pm.save_initial_weights()
    
    # Calculate per-round pruning rate
    # After n rounds of pruning p each, remaining = (1-p)^n = 1 - target_sparsity
    # So p = 1 - (1 - target_sparsity)^(1/n)
    remaining = 1 - target_sparsity
    per_round_prune = 1 - (remaining ** (1 / n_rounds))
    
    results = {
        'rounds': [],
        'target_sparsity': target_sparsity,
        'n_rounds': n_rounds,
        'per_round_prune': per_round_prune
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Iterative Magnitude Pruning")
        print(f"Target sparsity: {target_sparsity:.1%}")
        print(f"Rounds: {n_rounds}")
        print(f"Per-round pruning: {per_round_prune:.1%}")
        print(f"{'='*60}\n")
    
    # Initial training (dense)
    if verbose:
        print("Round 0: Training dense model...")
    metrics = train_fn(model)
    results['rounds'].append({
        'round': 0,
        'sparsity': 0.0,
        'metrics': metrics
    })
    
    current_sparsity = 0.0
    
    for round_idx in range(1, n_rounds + 1):
        # Calculate cumulative sparsity for this round
        current_sparsity = 1 - (1 - per_round_prune) ** round_idx
        
        if verbose:
            print(f"\nRound {round_idx}/{n_rounds}: Pruning to {current_sparsity:.1%} sparsity...")
        
        # Prune
        pm.prune_by_magnitude(current_sparsity)
        
        # Rewind if using LTH protocol
        if rewind:
            pm.rewind_to_initial()
        
        # Retrain
        if verbose:
            print(f"Retraining sparse model...")
        metrics = train_fn(model)
        
        # Verify integrity
        if not pm.verify_integrity():
            print("[WARNING] Pruning integrity check failed!")
        
        results['rounds'].append({
            'round': round_idx,
            'sparsity': current_sparsity,
            'metrics': metrics
        })
    
    results['final_sparsity'] = pm.get_sparsity()
    
    return model, pm, results


def one_shot_prune(
    model: nn.Module,
    sparsity: float,
    exclude_layers: Optional[List[str]] = None
) -> PruningManager:
    """
    Apply one-shot magnitude pruning.
    
    Args:
        model: Model to prune
        sparsity: Target sparsity
        exclude_layers: Layers to exclude
        
    Returns:
        PruningManager instance
    """
    pm = PruningManager(model, exclude_layers=exclude_layers)
    pm.prune_by_magnitude(sparsity)
    return pm


def create_pruning_manager(
    model: nn.Module,
    config: Dict
) -> PruningManager:
    """
    Factory function to create a PruningManager from config.
    
    Args:
        model: The model to manage
        config: Configuration dictionary with optional 'pruning' section
        
    Returns:
        Configured PruningManager
    """
    pruning_config = config.get('pruning', {})
    exclude_layers = pruning_config.get('exclude_layers', [])
    
    # Add common exclusions
    if pruning_config.get('exclude_output', True):
        # Exclude the last layer (output layer)
        exclude_layers.append('layers.' + str(len(list(model.modules())) - 1))
    
    pm = PruningManager(model, exclude_layers=exclude_layers)
    
    return pm
