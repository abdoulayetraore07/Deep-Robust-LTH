"""
Pruning module for Lottery Ticket Hypothesis experiments.
Uses PyTorch native pruning (torch.nn.utils.prune).
"""
from .pruning import (
    PruningManager,
    iterative_magnitude_pruning,
    one_shot_prune,
    create_pruning_manager
)

__all__ = [
    'PruningManager',
    'iterative_magnitude_pruning',
    'one_shot_prune',
    'create_pruning_manager'
]