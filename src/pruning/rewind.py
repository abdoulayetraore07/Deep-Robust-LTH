"""
Weight Rewinding for Lottery Ticket Hypothesis

Implements weight rewinding strategies from:
- Frankle & Carlin (2019): Rewind to initialization
- Frankle et al. (2020): "Late Rewinding" - rewind to early training checkpoint

Key insight: The original LTH rewinds to random initialization (epoch 0),
but "Late Rewinding" to epoch k (e.g., k=1-5) often finds better tickets,
especially for larger models.

Rewinding strategies:
1. 'init': Rewind to random initialization (original LTH)
2. 'late': Rewind to early checkpoint (epoch k)
3. 'learning_rate': Rewind learning rate schedule only
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple
from copy import deepcopy
import os


class WeightRewinder:
    """
    Manages weight rewinding for Lottery Ticket experiments.
    
    Stores checkpoints at various points during training
    and allows rewinding to any saved checkpoint.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize rewinder.
        
        Args:
            model: Neural network model
        """
        self.model = model
        
        # Storage for weight checkpoints
        # Key: checkpoint name (e.g., 'init', 'epoch_1', 'epoch_5')
        # Value: dict of {param_name: param_tensor}
        self.checkpoints: Dict[str, Dict[str, torch.Tensor]] = {}
        
        # Track which checkpoint was used for last rewind
        self.last_rewind_checkpoint: Optional[str] = None
    
    def save_checkpoint(self, name: str):
        """
        Save current weights as a checkpoint.
        
        Args:
            name: Checkpoint name (e.g., 'init', 'epoch_1')
        """
        checkpoint = {}
        for param_name, param in self.model.named_parameters():
            checkpoint[param_name] = param.data.clone().cpu()
        
        self.checkpoints[name] = checkpoint
    
    def save_init(self):
        """Save initial weights (convenience method)."""
        self.save_checkpoint('init')
    
    def save_epoch(self, epoch: int):
        """Save weights at specific epoch."""
        self.save_checkpoint(f'epoch_{epoch}')
    
    def rewind_to(
        self, 
        name: str,
        mask: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Rewind weights to a saved checkpoint.
        
        Args:
            name: Checkpoint name to rewind to
            mask: Optional mask to apply after rewinding
        """
        if name not in self.checkpoints:
            raise ValueError(f"Checkpoint '{name}' not found. Available: {list(self.checkpoints.keys())}")
        
        checkpoint = self.checkpoints[name]
        
        with torch.no_grad():
            for param_name, param in self.model.named_parameters():
                if param_name in checkpoint:
                    param.data.copy_(checkpoint[param_name].to(param.device))
        
        # Apply mask if provided
        if mask is not None:
            with torch.no_grad():
                for param_name, param in self.model.named_parameters():
                    if param_name in mask:
                        param.data.mul_(mask[param_name].to(param.device))
        
        self.last_rewind_checkpoint = name
    
    def rewind_to_init(self, mask: Optional[Dict[str, torch.Tensor]] = None):
        """Rewind to initial weights."""
        self.rewind_to('init', mask)
    
    def rewind_to_epoch(self, epoch: int, mask: Optional[Dict[str, torch.Tensor]] = None):
        """Rewind to specific epoch."""
        self.rewind_to(f'epoch_{epoch}', mask)
    
    def has_checkpoint(self, name: str) -> bool:
        """Check if a checkpoint exists."""
        return name in self.checkpoints
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        return list(self.checkpoints.keys())
    
    def delete_checkpoint(self, name: str):
        """Delete a checkpoint to free memory."""
        if name in self.checkpoints:
            del self.checkpoints[name]
    
    def clear_all(self):
        """Clear all checkpoints."""
        self.checkpoints.clear()
        self.last_rewind_checkpoint = None
    
    def save_to_file(self, filepath: str):
        """
        Save all checkpoints to file.
        
        Args:
            filepath: Path to save checkpoints
        """
        torch.save({
            'checkpoints': self.checkpoints,
            'last_rewind': self.last_rewind_checkpoint
        }, filepath)
    
    def load_from_file(self, filepath: str):
        """
        Load checkpoints from file.
        
        Args:
            filepath: Path to load from
        """
        data = torch.load(filepath, map_location='cpu')
        self.checkpoints = data['checkpoints']
        self.last_rewind_checkpoint = data.get('last_rewind')
    
    def get_checkpoint_size(self) -> Dict[str, int]:
        """Get memory size of each checkpoint in bytes."""
        sizes = {}
        for name, checkpoint in self.checkpoints.items():
            size = sum(t.numel() * t.element_size() for t in checkpoint.values())
            sizes[name] = size
        return sizes
    
    def summary(self) -> str:
        """Get summary of rewinder state."""
        lines = [
            "Weight Rewinder Summary",
            "=" * 50,
            f"Checkpoints saved: {len(self.checkpoints)}",
        ]
        
        for name in self.checkpoints:
            n_params = sum(t.numel() for t in self.checkpoints[name].values())
            lines.append(f"  - {name}: {n_params:,} parameters")
        
        if self.last_rewind_checkpoint:
            lines.append(f"Last rewind: {self.last_rewind_checkpoint}")
        
        lines.append("=" * 50)
        return "\n".join(lines)


class LateRewinder(WeightRewinder):
    """
    Specialized rewinder for Late Rewinding (Frankle et al., 2020).
    
    Late rewinding rewinds to epoch k instead of epoch 0,
    which often finds better tickets for larger models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        rewind_epoch: int = 1,
        save_frequency: int = 1
    ):
        """
        Initialize late rewinder.
        
        Args:
            model: Neural network
            rewind_epoch: Epoch to rewind to (default: 1)
            save_frequency: How often to save checkpoints during training
        """
        super().__init__(model)
        
        self.rewind_epoch = rewind_epoch
        self.save_frequency = save_frequency
        self.current_epoch = 0
    
    def on_epoch_end(self, epoch: int):
        """
        Call this at the end of each epoch during initial training.
        
        Args:
            epoch: Current epoch number (0-indexed)
        """
        self.current_epoch = epoch
        
        # Always save init
        if epoch == 0 and not self.has_checkpoint('init'):
            # This should have been saved before training
            pass
        
        # Save at rewind epoch
        if epoch == self.rewind_epoch:
            self.save_epoch(epoch)
        
        # Optionally save at regular intervals
        if self.save_frequency > 0 and epoch % self.save_frequency == 0:
            self.save_epoch(epoch)
    
    def rewind(self, mask: Optional[Dict[str, torch.Tensor]] = None):
        """
        Rewind to the configured rewind epoch.
        
        Args:
            mask: Optional mask to apply
        """
        checkpoint_name = f'epoch_{self.rewind_epoch}'
        
        if not self.has_checkpoint(checkpoint_name):
            # Fall back to init if rewind epoch not saved
            if self.has_checkpoint('init'):
                print(f"[LateRewinder] Checkpoint {checkpoint_name} not found, using 'init'")
                checkpoint_name = 'init'
            else:
                raise ValueError(f"No valid checkpoint found for rewinding")
        
        self.rewind_to(checkpoint_name, mask)


class RewindScheduler:
    """
    Manages the complete rewind schedule for multi-round LTH experiments.
    
    Handles:
    - Initial weight saving
    - Checkpoint saving during training
    - Rewinding between pruning rounds
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict
    ):
        """
        Initialize rewind scheduler.
        
        Args:
            model: Neural network
            config: Configuration with rewind settings
        """
        self.model = model
        self.config = config
        
        # Rewind config
        rewind_config = config.get('rewind', {})
        self.strategy = rewind_config.get('strategy', 'init')  # 'init' or 'late'
        self.rewind_epoch = rewind_config.get('epoch', 0)
        
        # Initialize appropriate rewinder
        if self.strategy == 'late' and self.rewind_epoch > 0:
            self.rewinder = LateRewinder(model, rewind_epoch=self.rewind_epoch)
        else:
            self.rewinder = WeightRewinder(model)
        
        # Track pruning rounds
        self.current_round = 0
    
    def before_training(self):
        """Call before starting initial training."""
        self.rewinder.save_init()
        print(f"[Rewind] Saved initial weights (strategy: {self.strategy})")
    
    def on_epoch_end(self, epoch: int):
        """Call at end of each training epoch."""
        if isinstance(self.rewinder, LateRewinder):
            self.rewinder.on_epoch_end(epoch)
    
    def before_pruning_round(self, round_idx: int):
        """Call before each pruning round."""
        self.current_round = round_idx
    
    def after_pruning(self, mask: Dict[str, torch.Tensor]):
        """
        Call after pruning to rewind weights.
        
        Args:
            mask: Pruning mask to apply
        """
        if self.strategy == 'late' and isinstance(self.rewinder, LateRewinder):
            self.rewinder.rewind(mask)
        else:
            self.rewinder.rewind_to_init(mask)
        
        print(f"[Rewind] Rewound to {self.strategy} "
              f"(epoch {self.rewind_epoch if self.strategy == 'late' else 0})")
    
    def save_state(self, filepath: str):
        """Save rewinder state to file."""
        state = {
            'strategy': self.strategy,
            'rewind_epoch': self.rewind_epoch,
            'current_round': self.current_round,
            'checkpoints': self.rewinder.checkpoints
        }
        torch.save(state, filepath)
    
    def load_state(self, filepath: str):
        """Load rewinder state from file."""
        state = torch.load(filepath, map_location='cpu')
        self.strategy = state['strategy']
        self.rewind_epoch = state['rewind_epoch']
        self.current_round = state['current_round']
        self.rewinder.checkpoints = state['checkpoints']
    
    def summary(self) -> str:
        """Get summary of rewind scheduler."""
        lines = [
            "Rewind Scheduler Summary",
            "=" * 50,
            f"Strategy: {self.strategy}",
            f"Rewind epoch: {self.rewind_epoch}",
            f"Current round: {self.current_round}",
            f"Checkpoints: {self.rewinder.list_checkpoints()}",
            "=" * 50
        ]
        return "\n".join(lines)


def create_rewind_scheduler(model: nn.Module, config: Dict) -> RewindScheduler:
    """
    Factory function to create RewindScheduler.
    
    Args:
        model: Neural network
        config: Configuration dictionary
        
    Returns:
        RewindScheduler instance
    """
    scheduler = RewindScheduler(model, config)
    
    print(f"[Rewind] Created scheduler (strategy={scheduler.strategy}, "
          f"rewind_epoch={scheduler.rewind_epoch})")
    
    return scheduler


def compute_weight_distance(
    weights1: Dict[str, torch.Tensor],
    weights2: Dict[str, torch.Tensor],
    norm: str = 'l2'
) -> float:
    """
    Compute distance between two weight configurations.
    
    Useful for analyzing how far weights move from initialization.
    
    Args:
        weights1: First weight dict
        weights2: Second weight dict
        norm: 'l2' or 'linf'
        
    Returns:
        Distance value
    """
    total_distance = 0.0
    total_params = 0
    
    for name in weights1:
        if name in weights2:
            diff = weights1[name].float() - weights2[name].float()
            
            if norm == 'l2':
                total_distance += (diff ** 2).sum().item()
            else:  # linf
                total_distance = max(total_distance, diff.abs().max().item())
            
            total_params += diff.numel()
    
    if norm == 'l2':
        return (total_distance / total_params) ** 0.5 if total_params > 0 else 0.0
    else:
        return total_distance


def analyze_weight_movement(
    model: nn.Module,
    initial_weights: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Analyze how weights have moved from initialization.
    
    Args:
        model: Current model
        initial_weights: Initial weight values
        
    Returns:
        Movement statistics
    """
    current_weights = {name: param.data.cpu() for name, param in model.named_parameters()}
    
    return {
        'l2_distance': compute_weight_distance(current_weights, initial_weights, 'l2'),
        'linf_distance': compute_weight_distance(current_weights, initial_weights, 'linf')
    }