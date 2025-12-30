"""
Visualization utilities for plots and figures
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Training Curves",
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        title: Plot title
        save_path: Path to save figure
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_pnl_distribution(
    pnl: np.ndarray,
    title: str = "P&L Distribution",
    save_path: Optional[str] = None
) -> None:
    """
    Plot P&L distribution histogram
    
    Args:
        pnl: P&L values
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    plt.hist(pnl, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(np.mean(pnl), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(pnl):.4f}')
    plt.axvline(np.median(pnl), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(pnl):.4f}')
    plt.xlabel('P&L')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_convergence_comparison(
    results: Dict[str, List[float]],
    title: str = "Convergence Comparison",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    save_path: Optional[str] = None
) -> None:
    """
    Plot convergence comparison for multiple models
    
    Args:
        results: Dictionary of {model_name: losses}
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 6))
    
    for name, losses in results.items():
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, linewidth=2, label=name, marker='o', markersize=4, markevery=max(1, len(losses)//10))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_robustness_comparison(
    model_names: List[str],
    clean_cvars: List[float],
    fgsm_cvars: List[float],
    pgd_cvars: List[float],
    title: str = "Robustness Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Plot robustness comparison across models
    
    Args:
        model_names: List of model names
        clean_cvars: Clean CVaR values
        fgsm_cvars: FGSM CVaR values
        pgd_cvars: PGD CVaR values
        title: Plot title
        save_path: Path to save figure
    """
    x = np.arange(len(model_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.bar(x - width, clean_cvars, width, label='Clean', color='blue', alpha=0.8)
    ax.bar(x, fgsm_cvars, width, label='FGSM', color='orange', alpha=0.8)
    ax.bar(x + width, pgd_cvars, width, label='PGD-10', color='red', alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('CVaR (lower = better)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_sparsity_performance(
    sparsities: List[float],
    cvars: List[float],
    baseline_cvar: float,
    title: str = "Performance vs Sparsity",
    save_path: Optional[str] = None
) -> None:
    """
    Plot performance as function of sparsity
    
    Args:
        sparsities: Sparsity levels (0-1)
        cvars: CVaR values
        baseline_cvar: Baseline dense model CVaR
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    sparsities_pct = [s * 100 for s in sparsities]
    
    plt.plot(sparsities_pct, cvars, 'o-', linewidth=2, markersize=8, color='blue', label='Sparse Model')
    plt.axhline(baseline_cvar, color='red', linestyle='--', linewidth=2, label='Dense Baseline')
    
    plt.xlabel('Sparsity (%)')
    plt.ylabel('CVaR (lower = better)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance bar chart
    
    Args:
        feature_names: List of feature names
        importances: Importance values
        title: Plot title
        save_path: Path to save figure
    """
    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_features)), sorted_importances, color='steelblue', alpha=0.8)
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Importance')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()