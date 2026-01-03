"""
Pruning module for Lottery Ticket Hypothesis experiments.

Uses PyTorch native pruning (torch.nn.utils.prune).
"""

from .pruning import (
    PruningManager,
    iterative_magnitude_pruning,
    find_winning_ticket
)

__all__ = [
    'PruningManager',
    'iterative_magnitude_pruning', 
    'find_winning_ticket'
]