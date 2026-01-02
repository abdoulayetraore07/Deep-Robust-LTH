"""
Iterative Magnitude Pruning (IMP)

Implements the pruning algorithm from Frankle & Carlin (2019) "The Lottery Ticket Hypothesis".

Algorithm:
1. Train network to completion
2. Prune p% of smallest magnitude weights globally
3. Reset remaining weights to initial values
4. Repeat until target sparsity

Key insight: The "winning ticket" is the combination of:
- The sparse mask (which connections survive)
- The initial weights (the starting point matters!)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import numpy as np
from copy import deepcopy

from src.pruning.masks import MaskManager, apply_mask_to_model


class MagnitudePruner:
    """
    Iterative Magnitude Pruning for finding Lottery Tickets.
    
    Prunes weights based on their absolute magnitude,
    either globally (across all layers) or locally (per layer).
    """
    
    def __init__(
        self,
        model: nn.Module,
        pruning_rate: float = 0.2,
        pruning_strategy: str = 'global',
        exclude_layers: Optional[List[str]] = None
    ):
        """
        Initialize pruner.
        
        Args:
            model: Neural network to prune
            pruning_rate: Fraction of remaining weights to prune each iteration (e.g., 0.2 = 20%)
            pruning_strategy: 'global' (prune across all layers) or 'local' (prune per layer)
            exclude_layers: Layer names to exclude from pruning (e.g., ['layers.6'] for output layer)
        """
        self.model = model
        self.pruning_rate = pruning_rate
        self.pruning_strategy = pruning_strategy.lower()
        self.exclude_layers = exclude_layers or []
        
        assert self.pruning_strategy in ['global', 'local'], \
            f"Strategy must be 'global' or 'local', got {pruning_strategy}"
        
        # Initialize mask manager
        self.mask_manager = MaskManager(model)
        
        # Track pruning history
        self.pruning_history = []
    
    def _get_prunable_weights(self) -> Dict[str, torch.Tensor]:
        """Get weights that can be pruned (excluding specified layers)."""
        weights = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' not in name or not param.requires_grad:
                continue
            
            # Check if layer should be excluded
            excluded = False
            for exclude in self.exclude_layers:
                if exclude in name:
                    excluded = True
                    break
            
            if not excluded:
                weights[name] = param.data
        
        return weights
    
    def _compute_threshold_global(self, prune_fraction: float) -> float:
        """
        Compute global magnitude threshold for pruning.
        
        Args:
            prune_fraction: Fraction of current remaining weights to prune
            
        Returns:
            Threshold value (prune weights with |w| < threshold)
        """
        # Collect all non-zero weights
        all_weights = []
        
        weights = self._get_prunable_weights()
        for name, weight in weights.items():
            mask = self.mask_manager.get_mask(name)
            # Only consider currently active weights
            active_weights = weight[mask == 1].abs()
            all_weights.append(active_weights.flatten())
        
        all_weights = torch.cat(all_weights)
        
        if len(all_weights) == 0:
            return 0.0
        
        # Compute threshold as the prune_fraction percentile
        k = int(prune_fraction * len(all_weights))
        if k == 0:
            return 0.0
        
        # Get k-th smallest value
        threshold = torch.kthvalue(all_weights, k).values.item()
        
        return threshold
    
    def _compute_thresholds_local(self, prune_fraction: float) -> Dict[str, float]:
        """
        Compute per-layer magnitude thresholds for pruning.
        
        Args:
            prune_fraction: Fraction of current remaining weights to prune per layer
            
        Returns:
            Dictionary of thresholds per layer
        """
        thresholds = {}
        
        weights = self._get_prunable_weights()
        for name, weight in weights.items():
            mask = self.mask_manager.get_mask(name)
            active_weights = weight[mask == 1].abs()
            
            if len(active_weights) == 0:
                thresholds[name] = 0.0
                continue
            
            k = int(prune_fraction * len(active_weights))
            if k == 0:
                thresholds[name] = 0.0
            else:
                thresholds[name] = torch.kthvalue(active_weights.flatten(), k).values.item()
        
        return thresholds
    
    def prune_once(self, prune_fraction: Optional[float] = None) -> Dict[str, float]:
        """
        Perform one round of magnitude pruning.
        
        Args:
            prune_fraction: Override default pruning rate
            
        Returns:
            Pruning statistics
        """
        if prune_fraction is None:
            prune_fraction = self.pruning_rate
        
        sparsity_before = self.mask_manager.get_total_sparsity()
        
        weights = self._get_prunable_weights()
        
        if self.pruning_strategy == 'global':
            threshold = self._compute_threshold_global(prune_fraction)
            
            for name, weight in weights.items():
                mask = self.mask_manager.get_mask(name)
                # Prune weights below threshold (only among currently active)
                new_mask = mask * (weight.abs() >= threshold).float()
                self.mask_manager.update_mask(name, new_mask)
        
        else:  # local
            thresholds = self._compute_thresholds_local(prune_fraction)
            
            for name, weight in weights.items():
                mask = self.mask_manager.get_mask(name)
                threshold = thresholds[name]
                new_mask = mask * (weight.abs() >= threshold).float()
                self.mask_manager.update_mask(name, new_mask)
        
        # Apply masks to model
        self.mask_manager.apply_masks()
        
        sparsity_after = self.mask_manager.get_total_sparsity()
        
        stats = {
            'sparsity_before': sparsity_before,
            'sparsity_after': sparsity_after,
            'weights_pruned': sparsity_after - sparsity_before,
            'remaining_fraction': 1 - sparsity_after
        }
        
        self.pruning_history.append(stats)
        
        return stats
    
    def prune_to_sparsity(self, target_sparsity: float) -> Dict[str, float]:
        """
        Prune iteratively until reaching target sparsity.
        
        Args:
            target_sparsity: Target sparsity level (e.g., 0.9 for 90% sparse)
            
        Returns:
            Final pruning statistics
        """
        current_sparsity = self.mask_manager.get_total_sparsity()
        iteration = 0
        
        while current_sparsity < target_sparsity:
            # Calculate how much more to prune
            remaining_current = 1 - current_sparsity
            remaining_target = 1 - target_sparsity
            
            if remaining_current <= 0:
                break
            
            # Prune fraction of remaining weights
            stats = self.prune_once()
            current_sparsity = stats['sparsity_after']
            iteration += 1
            
            # Safety check
            if iteration > 100:
                print(f"Warning: Reached 100 iterations, stopping at sparsity {current_sparsity:.2%}")
                break
        
        return {
            'final_sparsity': current_sparsity,
            'iterations': iteration,
            'target_sparsity': target_sparsity
        }
    
    def get_masks(self) -> Dict[str, torch.Tensor]:
        """Get current masks."""
        return self.mask_manager.masks.copy()
    
    def set_masks(self, masks: Dict[str, torch.Tensor]):
        """Set masks from external source."""
        for name, mask in masks.items():
            if name in self.mask_manager.masks:
                self.mask_manager.update_mask(name, mask)
        self.mask_manager.apply_masks()
    
    def reset_masks(self):
        """Reset all masks to ones (no pruning)."""
        self.mask_manager.reset()
        self.pruning_history = []
    
    def get_sparsity(self) -> float:
        """Get current sparsity level."""
        return self.mask_manager.get_total_sparsity()
    
    def summary(self) -> str:
        """Get summary of current pruning state."""
        return self.mask_manager.summary()


class IterativeMagnitudePruning:
    """
    Full Iterative Magnitude Pruning pipeline for Lottery Ticket Hypothesis.
    
    Implements the complete IMP algorithm:
    1. Save initial weights
    2. Train to completion
    3. Prune smallest weights
    4. Rewind to initial weights
    5. Repeat
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict
    ):
        """
        Initialize IMP pipeline.
        
        Args:
            model: Neural network
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        
        # Pruning config
        pruning_config = config.get('pruning', {})
        self.pruning_rate = pruning_config.get('rate', 0.2)
        self.target_sparsity = pruning_config.get('target_sparsity', 0.9)
        self.n_rounds = pruning_config.get('n_rounds', 10)
        self.strategy = pruning_config.get('strategy', 'global')
        self.exclude_output = pruning_config.get('exclude_output', True)
        
        # Determine layers to exclude
        exclude_layers = []
        if self.exclude_output:
            # Exclude the last linear layer (output layer)
            for name, _ in model.named_parameters():
                if 'weight' in name:
                    last_layer_name = name
            if last_layer_name:
                exclude_layers.append(last_layer_name.replace('.weight', ''))
        
        # Initialize pruner
        self.pruner = MagnitudePruner(
            model=model,
            pruning_rate=self.pruning_rate,
            pruning_strategy=self.strategy,
            exclude_layers=exclude_layers
        )
        
        # Storage for initial weights (for rewinding)
        self.initial_weights: Optional[Dict[str, torch.Tensor]] = None
        
        # Track tickets found
        self.tickets: List[Dict] = []
    
    def save_initial_weights(self):
        """Save initial weights for later rewinding."""
        self.initial_weights = {}
        for name, param in self.model.named_parameters():
            self.initial_weights[name] = param.data.clone()
        print("[IMP] Saved initial weights")
    
    def rewind_weights(self):
        """Rewind weights to initial values (keeping current mask)."""
        if self.initial_weights is None:
            raise RuntimeError("No initial weights saved. Call save_initial_weights() first.")
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.initial_weights:
                    param.data.copy_(self.initial_weights[name])
        
        # Apply current masks to rewound weights
        self.pruner.mask_manager.apply_masks()
        
        print(f"[IMP] Rewound weights to initialization (sparsity: {self.pruner.get_sparsity():.2%})")
    
    def prune_round(self) -> Dict:
        """
        Perform one round of pruning.
        
        Returns:
            Pruning statistics
        """
        stats = self.pruner.prune_once()
        return stats
    
    def get_current_ticket(self) -> Dict:
        """
        Get current "ticket" (mask + initial weights).
        
        Returns:
            Dictionary with mask and statistics
        """
        return {
            'masks': {name: mask.clone() for name, mask in self.pruner.get_masks().items()},
            'sparsity': self.pruner.get_sparsity(),
            'remaining_fraction': 1 - self.pruner.get_sparsity()
        }
    
    def save_ticket(self, filepath: str):
        """Save current ticket to file."""
        ticket = self.get_current_ticket()
        
        if self.initial_weights is not None:
            ticket['initial_weights'] = {
                name: w.cpu() for name, w in self.initial_weights.items()
            }
        
        torch.save(ticket, filepath)
        print(f"[IMP] Saved ticket to {filepath}")
    
    def load_ticket(self, filepath: str, device: torch.device):
        """
        Load ticket and apply to model.
        
        Args:
            filepath: Path to ticket file
            device: Device to load to
        """
        ticket = torch.load(filepath, map_location=device)
        
        # Load masks
        for name, mask in ticket['masks'].items():
            if name in self.pruner.mask_manager.masks:
                self.pruner.mask_manager.update_mask(name, mask.to(device))
        
        # Load initial weights if present
        if 'initial_weights' in ticket:
            self.initial_weights = {
                name: w.to(device) for name, w in ticket['initial_weights'].items()
            }
            self.rewind_weights()
        else:
            self.pruner.mask_manager.apply_masks()
        
        print(f"[IMP] Loaded ticket from {filepath} (sparsity: {self.pruner.get_sparsity():.2%})")
    
    def compute_sparsity_schedule(self) -> List[float]:
        """
        Compute sparsity levels for each pruning round.
        
        Using iterative pruning: after n rounds with rate p,
        remaining fraction = (1-p)^n
        
        Returns:
            List of target sparsities for each round
        """
        schedule = []
        remaining = 1.0
        
        for round_idx in range(self.n_rounds):
            remaining *= (1 - self.pruning_rate)
            sparsity = 1 - remaining
            schedule.append(sparsity)
            
            if sparsity >= self.target_sparsity:
                break
        
        return schedule
    
    def summary(self) -> str:
        """Get summary of IMP state."""
        lines = [
            "Iterative Magnitude Pruning Summary",
            "=" * 50,
            f"Pruning rate: {self.pruning_rate:.1%}",
            f"Target sparsity: {self.target_sparsity:.1%}",
            f"Strategy: {self.strategy}",
            f"Current sparsity: {self.pruner.get_sparsity():.2%}",
            f"Initial weights saved: {self.initial_weights is not None}",
            f"Rounds completed: {len(self.pruner.pruning_history)}",
            "=" * 50
        ]
        return "\n".join(lines)


def create_pruner(model: nn.Module, config: Dict) -> IterativeMagnitudePruning:
    """
    Factory function to create IMP pipeline.
    
    Args:
        model: Neural network
        config: Configuration dictionary
        
    Returns:
        IterativeMagnitudePruning instance
    """
    imp = IterativeMagnitudePruning(model, config)
    
    print(f"[Pruning] Created IMP (rate={imp.pruning_rate:.1%}, "
          f"target={imp.target_sparsity:.1%}, strategy={imp.strategy})")
    
    return imp