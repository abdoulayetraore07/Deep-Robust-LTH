"""
Visualization Utilities for Deep Hedging

Provides plotting functions for:
1. Training curves (loss, metrics over epochs)
2. P&L distributions
3. Delta comparison (model vs baseline)
4. Pruning analysis (performance vs sparsity)
5. Adversarial robustness visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import torch


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2563eb',
    'secondary': '#16a34a', 
    'tertiary': '#dc2626',
    'quaternary': '#9333ea',
    'clean': '#2563eb',
    'adversarial': '#dc2626',
    'baseline': '#6b7280'
}


def plot_training_curves(
    history: List[Dict[str, float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot training curves from history.
    
    Args:
        history: List of metric dictionaries per epoch
        metrics: Which metrics to plot (default: loss metrics)
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ['train_loss', 'val_loss']
    
    # Filter to available metrics
    available_metrics = []
    for m in metrics:
        if any(m in h for h in history):
            available_metrics.append(m)
    
    if not available_metrics:
        print("No metrics found to plot")
        return None
    
    n_metrics = len(available_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    epochs = [h.get('epoch', i+1) for i, h in enumerate(history)]
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        values = [h.get(metric, np.nan) for h in history]
        
        ax.plot(epochs, values, color=COLORS['primary'], linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # Mark best value
        valid_values = [(e, v) for e, v in zip(epochs, values) if not np.isnan(v)]
        if valid_values:
            if 'loss' in metric:
                best_idx = np.argmin([v for _, v in valid_values])
            else:
                best_idx = np.argmax([v for _, v in valid_values])
            best_epoch, best_val = valid_values[best_idx]
            ax.axvline(best_epoch, color=COLORS['secondary'], linestyle='--', alpha=0.5)
            ax.scatter([best_epoch], [best_val], color=COLORS['secondary'], s=100, zorder=5)
    
    # Hide unused axes
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return fig


def plot_pnl_distribution(
    pnl: Union[np.ndarray, torch.Tensor],
    pnl_baseline: Optional[Union[np.ndarray, torch.Tensor]] = None,
    labels: Tuple[str, str] = ('Deep Hedging', 'Delta Hedging'),
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot P&L distribution comparison.
    
    Args:
        pnl: Model P&L values
        pnl_baseline: Optional baseline P&L for comparison
        labels: Labels for model and baseline
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if isinstance(pnl, torch.Tensor):
        pnl = pnl.cpu().numpy()
    if pnl_baseline is not None and isinstance(pnl_baseline, torch.Tensor):
        pnl_baseline = pnl_baseline.cpu().numpy()
    
    pnl = pnl.flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1 = axes[0]
    ax1.hist(pnl, bins=50, alpha=0.7, color=COLORS['primary'], 
             label=labels[0], density=True)
    
    if pnl_baseline is not None:
        pnl_baseline = pnl_baseline.flatten()
        ax1.hist(pnl_baseline, bins=50, alpha=0.5, color=COLORS['baseline'],
                 label=labels[1], density=True)
    
    ax1.axvline(0, color='black', linestyle='-', linewidth=1)
    ax1.axvline(np.mean(pnl), color=COLORS['primary'], linestyle='--', 
                label=f'{labels[0]} Mean: {np.mean(pnl):.4f}')
    
    if pnl_baseline is not None:
        ax1.axvline(np.mean(pnl_baseline), color=COLORS['baseline'], linestyle='--',
                    label=f'{labels[1]} Mean: {np.mean(pnl_baseline):.4f}')
    
    ax1.set_xlabel('P&L')
    ax1.set_ylabel('Density')
    ax1.set_title('P&L Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2 = axes[1]
    data = [pnl]
    tick_labels = [labels[0]]
    
    if pnl_baseline is not None:
        data.append(pnl_baseline)
        tick_labels.append(labels[1])
    
    bp = ax2.boxplot(data, labels=tick_labels, patch_artist=True)
    colors = [COLORS['primary'], COLORS['baseline']]
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('P&L')
    ax2.set_title('P&L Comparison')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved P&L distribution to {save_path}")
    
    return fig


def plot_delta_comparison(
    deltas_model: Union[np.ndarray, torch.Tensor],
    deltas_baseline: Union[np.ndarray, torch.Tensor],
    S: Optional[Union[np.ndarray, torch.Tensor]] = None,
    n_paths: int = 5,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Compare model deltas with baseline (e.g., BS delta).
    
    Args:
        deltas_model: Model's hedging positions (n_paths, n_steps)
        deltas_baseline: Baseline positions
        S: Optional stock prices for context
        n_paths: Number of paths to plot
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if isinstance(deltas_model, torch.Tensor):
        deltas_model = deltas_model.cpu().numpy()
    if isinstance(deltas_baseline, torch.Tensor):
        deltas_baseline = deltas_baseline.cpu().numpy()
    if S is not None and isinstance(S, torch.Tensor):
        S = S.cpu().numpy()
    
    n_paths = min(n_paths, len(deltas_model))
    n_steps = deltas_model.shape[1]
    time = np.arange(n_steps)
    
    if S is not None:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        ax1, ax2 = axes
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0], figsize[1]//2))
        ax2 = None
    
    # Plot deltas
    for i in range(n_paths):
        alpha = 0.7 if i == 0 else 0.3
        ax1.plot(time, deltas_model[i], color=COLORS['primary'], alpha=alpha,
                 label='Deep Hedging' if i == 0 else None)
        ax1.plot(time, deltas_baseline[i], color=COLORS['baseline'], alpha=alpha,
                 linestyle='--', label='BS Delta' if i == 0 else None)
    
    ax1.set_ylabel('Delta')
    ax1.set_title('Hedging Positions Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot stock prices if provided
    if ax2 is not None and S is not None:
        for i in range(n_paths):
            alpha = 0.7 if i == 0 else 0.3
            ax2.plot(time, S[i], color=COLORS['secondary'], alpha=alpha)
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Stock Price')
        ax2.set_title('Stock Price Paths')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved delta comparison to {save_path}")
    
    return fig


def plot_sparsity_performance(
    sparsities: List[float],
    metrics: Dict[str, List[float]],
    baseline_metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot performance metrics vs sparsity level (for LTH analysis).
    
    Args:
        sparsities: List of sparsity levels
        metrics: Dict mapping metric_name -> list of values at each sparsity
        baseline_metrics: Optional baseline values to show as horizontal lines
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    remaining = [1 - s for s in sparsities]  # Convert to remaining weights %
    
    colors_list = list(COLORS.values())
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx]
        
        ax.plot(remaining, values, 'o-', color=colors_list[idx % len(colors_list)],
                linewidth=2, markersize=8, label=metric_name)
        
        # Add baseline if provided
        if baseline_metrics and metric_name in baseline_metrics:
            ax.axhline(baseline_metrics[metric_name], color=COLORS['baseline'],
                      linestyle='--', label=f'Dense baseline')
        
        ax.set_xlabel('Remaining Weights (%)')
        ax.set_ylabel(metric_name.replace('_', ' ').title())
        ax.set_title(f'{metric_name.replace("_", " ").title()} vs Sparsity')
        ax.set_xscale('log')
        ax.invert_xaxis()  # Higher sparsity (fewer weights) on right
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    
    # Hide unused axes
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sparsity performance to {save_path}")
    
    return fig


def plot_adversarial_comparison(
    pnl_clean: Union[np.ndarray, torch.Tensor],
    pnl_adversarial: Union[np.ndarray, torch.Tensor],
    attack_name: str = 'PGD',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Compare clean and adversarial P&L distributions.
    
    Args:
        pnl_clean: P&L on clean data
        pnl_adversarial: P&L on adversarial data
        attack_name: Name of attack for title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if isinstance(pnl_clean, torch.Tensor):
        pnl_clean = pnl_clean.cpu().numpy()
    if isinstance(pnl_adversarial, torch.Tensor):
        pnl_adversarial = pnl_adversarial.cpu().numpy()
    
    pnl_clean = pnl_clean.flatten()
    pnl_adversarial = pnl_adversarial.flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Distribution comparison
    ax1 = axes[0]
    ax1.hist(pnl_clean, bins=50, alpha=0.7, color=COLORS['clean'],
             label='Clean', density=True)
    ax1.hist(pnl_adversarial, bins=50, alpha=0.7, color=COLORS['adversarial'],
             label=f'Adversarial ({attack_name})', density=True)
    
    ax1.axvline(np.mean(pnl_clean), color=COLORS['clean'], linestyle='--', linewidth=2)
    ax1.axvline(np.mean(pnl_adversarial), color=COLORS['adversarial'], linestyle='--', linewidth=2)
    
    ax1.set_xlabel('P&L')
    ax1.set_ylabel('Density')
    ax1.set_title(f'P&L Distribution: Clean vs {attack_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metrics comparison
    ax2 = axes[1]
    
    metrics_clean = {
        'Mean': np.mean(pnl_clean),
        'Std': np.std(pnl_clean),
        'CVaR 5%': np.mean(np.sort(pnl_clean)[:int(0.05 * len(pnl_clean))]),
        'Min': np.min(pnl_clean)
    }
    
    metrics_adv = {
        'Mean': np.mean(pnl_adversarial),
        'Std': np.std(pnl_adversarial),
        'CVaR 5%': np.mean(np.sort(pnl_adversarial)[:int(0.05 * len(pnl_adversarial))]),
        'Min': np.min(pnl_adversarial)
    }
    
    x = np.arange(len(metrics_clean))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, list(metrics_clean.values()), width,
                    label='Clean', color=COLORS['clean'], alpha=0.7)
    bars2 = ax2.bar(x + width/2, list(metrics_adv.values()), width,
                    label=f'Adversarial', color=COLORS['adversarial'], alpha=0.7)
    
    ax2.set_ylabel('Value')
    ax2.set_title('Metrics Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(metrics_clean.keys()))
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved adversarial comparison to {save_path}")
    
    return fig


def plot_robustness_heatmap(
    sparsities: List[float],
    epsilons: List[float],
    robustness_matrix: np.ndarray,
    metric_name: str = 'Robustness Gap',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot heatmap of robustness across sparsity and epsilon levels.
    
    Args:
        sparsities: List of sparsity levels
        epsilons: List of adversarial epsilon values
        robustness_matrix: Matrix of shape (len(sparsities), len(epsilons))
        metric_name: Name of metric for colorbar
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(robustness_matrix, aspect='auto', cmap='RdYlGn_r')
    
    # Labels
    ax.set_xticks(np.arange(len(epsilons)))
    ax.set_yticks(np.arange(len(sparsities)))
    ax.set_xticklabels([f'{e:.2f}' for e in epsilons])
    ax.set_yticklabels([f'{s:.0%}' for s in sparsities])
    
    ax.set_xlabel('Adversarial ε')
    ax.set_ylabel('Sparsity')
    ax.set_title(f'{metric_name} Heatmap')
    
    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(metric_name, rotation=-90, va='bottom')
    
    # Add text annotations
    for i in range(len(sparsities)):
        for j in range(len(epsilons)):
            text = ax.text(j, i, f'{robustness_matrix[i, j]:.3f}',
                          ha='center', va='center', color='black', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved robustness heatmap to {save_path}")
    
    return fig


def plot_lottery_ticket_summary(
    results: Dict[float, Dict[str, float]],
    baseline_results: Dict[str, float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    Comprehensive summary plot for Lottery Ticket experiments.
    
    Args:
        results: Dict mapping sparsity -> metrics
        baseline_results: Dense model metrics
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    sparsities = sorted(results.keys())
    remaining = [1 - s for s in sparsities]
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    # Plot 1: Sharpe Ratio vs Sparsity
    ax1 = fig.add_subplot(gs[0, 0])
    sharpe_values = [results[s].get('sharpe_ratio', 0) for s in sparsities]
    ax1.semilogx(remaining, sharpe_values, 'o-', color=COLORS['primary'],
                 linewidth=2, markersize=8)
    ax1.axhline(baseline_results.get('sharpe_ratio', 0), color=COLORS['baseline'],
                linestyle='--', label='Dense Model')
    ax1.set_xlabel('Remaining Weights')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Sharpe Ratio vs Sparsity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Plot 2: CVaR vs Sparsity
    ax2 = fig.add_subplot(gs[0, 1])
    cvar_values = [results[s].get('cvar_05', 0) for s in sparsities]
    ax2.semilogx(remaining, cvar_values, 'o-', color=COLORS['tertiary'],
                 linewidth=2, markersize=8)
    ax2.axhline(baseline_results.get('cvar_05', 0), color=COLORS['baseline'],
                linestyle='--', label='Dense Model')
    ax2.set_xlabel('Remaining Weights')
    ax2.set_ylabel('CVaR 5%')
    ax2.set_title('CVaR vs Sparsity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    # Plot 3: Robustness Gap vs Sparsity (if available)
    ax3 = fig.add_subplot(gs[1, 0])
    if 'robustness_mean_gap' in results[sparsities[0]]:
        gap_values = [results[s].get('robustness_mean_gap', 0) for s in sparsities]
        ax3.semilogx(remaining, gap_values, 'o-', color=COLORS['quaternary'],
                     linewidth=2, markersize=8)
        ax3.axhline(baseline_results.get('robustness_mean_gap', 0), color=COLORS['baseline'],
                    linestyle='--', label='Dense Model')
        ax3.set_xlabel('Remaining Weights')
        ax3.set_ylabel('Robustness Gap')
        ax3.set_title('Adversarial Robustness vs Sparsity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.invert_xaxis()
    else:
        ax3.text(0.5, 0.5, 'No robustness data', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Adversarial Robustness vs Sparsity')
    
    # Plot 4: Summary Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Find winning ticket
    best_sparsity = max(sparsities, key=lambda s: results[s].get('sharpe_ratio', 0))
    
    summary_text = f"""
    LOTTERY TICKET SUMMARY
    {'='*40}
    
    Dense Model Performance:
      Sharpe Ratio: {baseline_results.get('sharpe_ratio', 0):.4f}
      CVaR 5%: {baseline_results.get('cvar_05', 0):.4f}
    
    Best Sparse Model:
      Sparsity: {best_sparsity:.1%}
      Remaining Weights: {(1-best_sparsity)*100:.1f}%
      Sharpe Ratio: {results[best_sparsity].get('sharpe_ratio', 0):.4f}
      CVaR 5%: {results[best_sparsity].get('cvar_05', 0):.4f}
    
    Maximum Sparsity Tested: {max(sparsities):.1%}
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved lottery ticket summary to {save_path}")
    
    return fig


def save_all_figures(
    figures: Dict[str, plt.Figure],
    output_dir: str,
    format: str = 'png'
):
    """
    Save multiple figures to a directory.
    
    Args:
        figures: Dict mapping name -> figure
        output_dir: Output directory
        format: Image format (png, pdf, svg)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, fig in figures.items():
        filepath = output_path / f"{name}.{format}"
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved {filepath}")
    
    print(f"All figures saved to {output_dir}")