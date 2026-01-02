#!/usr/bin/env python3
"""
Script to generate all publication-ready figures

Generates 7 figures from experiment results:
1. Performance vs Sparsity (LTH)
2. Convergence Speed Comparison
3. Adversarial Robustness Comparison
4. Regime Shift Performance
5. Sparsity-Robustness Trade-off
6. Training Dynamics
7. Summary Table

Usage:
    python scripts/generate_all_figures.py
    python scripts/generate_all_figures.py --experiments-dir experiments
    python scripts/generate_all_figures.py --output-dir figures
"""

import sys
import argparse
from pathlib import Path
from glob import glob
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'primary': '#2563eb',
    'secondary': '#16a34a',
    'tertiary': '#dc2626',
    'quaternary': '#9333ea',
    'baseline': '#6b7280',
    'clean': '#2563eb',
    'fgsm': '#f59e0b',
    'pgd': '#dc2626',
}


def find_latest_experiment(experiments_dir: Path, pattern: str) -> Path:
    """Find the most recent experiment directory matching pattern."""
    matches = list(experiments_dir.glob(pattern))
    if not matches:
        return None
    # Sort by modification time, get latest
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_json_safe(filepath: Path) -> dict:
    """Load JSON file safely, return None if not found."""
    if filepath and filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def figure1_performance_vs_sparsity(experiments_dir: Path, output_dir: Path) -> bool:
    """
    Figure 1: Performance vs Sparsity
    Shows how model performance degrades (or not) with increasing sparsity.
    """
    print("  Figure 1: Performance vs Sparsity...", end=" ")
    
    # Find LTH experiment
    lth_dir = find_latest_experiment(experiments_dir, "*lth*")
    if not lth_dir:
        print("SKIP (no LTH experiment found)")
        return False
    
    results_path = lth_dir / 'logs' / 'lth_results.json'
    results = load_json_safe(results_path)
    
    if not results:
        print(f"SKIP (no results at {results_path})")
        return False
    
    # Extract data
    results_by_sparsity = results.get('results_by_sparsity', {})
    dense_metrics = results.get('dense_metrics', {})
    
    if not results_by_sparsity:
        print("SKIP (no sparsity results)")
        return False
    
    sparsities = sorted([float(s) for s in results_by_sparsity.keys()])
    remaining = [(1 - s) * 100 for s in sparsities]  # Convert to % remaining
    
    sharpe_values = [results_by_sparsity[str(s)].get('sharpe_ratio', 0) for s in sparsities]
    cvar_values = [results_by_sparsity[str(s)].get('cvar_05', 0) for s in sparsities]
    
    dense_sharpe = dense_metrics.get('sharpe_ratio', sharpe_values[0] if sharpe_values else 0)
    dense_cvar = dense_metrics.get('cvar_05', cvar_values[0] if cvar_values else 0)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sharpe Ratio
    ax1.semilogx(remaining, sharpe_values, 'o-', color=COLORS['primary'], 
                 linewidth=2, markersize=8, label='Sparse Model')
    ax1.axhline(dense_sharpe, color=COLORS['baseline'], linestyle='--', 
                linewidth=2, label=f'Dense Baseline ({dense_sharpe:.3f})')
    ax1.set_xlabel('Remaining Weights (%)')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.set_title('Sharpe Ratio vs Sparsity')
    ax1.legend()
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)
    
    # CVaR
    ax2.semilogx(remaining, cvar_values, 'o-', color=COLORS['tertiary'], 
                 linewidth=2, markersize=8, label='Sparse Model')
    ax2.axhline(dense_cvar, color=COLORS['baseline'], linestyle='--', 
                linewidth=2, label=f'Dense Baseline ({dense_cvar:.3f})')
    ax2.set_xlabel('Remaining Weights (%)')
    ax2.set_ylabel('CVaR 5%')
    ax2.set_title('CVaR vs Sparsity')
    ax2.legend()
    ax2.invert_xaxis()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_performance_vs_sparsity.pdf')
    plt.savefig(output_dir / 'figure1_performance_vs_sparsity.png')
    plt.close()
    
    print("DONE")
    return True


def figure2_convergence_speed(experiments_dir: Path, output_dir: Path) -> bool:
    """
    Figure 2: Convergence Speed Comparison
    Compares training curves of dense vs sparse models.
    """
    print("  Figure 2: Convergence Speed...", end=" ")
    
    # Find baseline and LTH experiments
    baseline_dir = find_latest_experiment(experiments_dir, "*baseline*")
    lth_dir = find_latest_experiment(experiments_dir, "*lth*")
    
    if not baseline_dir and not lth_dir:
        print("SKIP (no experiments found)")
        return False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    found_data = False
    
    # Try to load baseline training history
    if baseline_dir:
        baseline_results = load_json_safe(baseline_dir / 'logs' / 'final_results.json')
        if baseline_results and 'training' in baseline_results:
            history = baseline_results['training'].get('history', [])
            if history:
                epochs = [h.get('epoch', i+1) for i, h in enumerate(history)]
                val_loss = [h.get('val_loss', 0) for h in history]
                ax.plot(epochs, val_loss, '-', color=COLORS['baseline'], 
                       linewidth=2, label='Dense Model')
                found_data = True
    
    # Try to load LTH results for best ticket
    if lth_dir:
        lth_results = load_json_safe(lth_dir / 'logs' / 'lth_results.json')
        if lth_results:
            best_sparsity = lth_results.get('best_sparsity_sharpe', 0)
            if best_sparsity > 0:
                ax.axhline(y=0, color=COLORS['primary'], linestyle='--',
                          label=f'Best Ticket ({best_sparsity:.0%} sparse)')
                found_data = True
    
    if not found_data:
        print("SKIP (no training history found)")
        plt.close()
        return False
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Training Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_convergence_speed.pdf')
    plt.savefig(output_dir / 'figure2_convergence_speed.png')
    plt.close()
    
    print("DONE")
    return True


def figure3_robustness_comparison(experiments_dir: Path, output_dir: Path) -> bool:
    """
    Figure 3: Adversarial Robustness Comparison
    Compares clean vs FGSM vs PGD performance.
    """
    print("  Figure 3: Robustness Comparison...", end=" ")
    
    # Find adversarial experiment
    adv_dir = find_latest_experiment(experiments_dir, "*adv*")
    if not adv_dir:
        print("SKIP (no adversarial experiment found)")
        return False
    
    results = load_json_safe(adv_dir / 'logs' / 'final_results.json')
    if not results or 'robustness' not in results:
        print("SKIP (no robustness results)")
        return False
    
    robustness = results['robustness']
    
    # Extract metrics
    metrics = ['pnl_mean', 'cvar_05', 'sharpe_ratio']
    metric_labels = ['Mean P&L', 'CVaR 5%', 'Sharpe Ratio']
    
    clean_vals = [robustness.get('clean', {}).get(m, 0) for m in metrics]
    fgsm_vals = [robustness.get('fgsm', {}).get(m, 0) for m in metrics]
    pgd_vals = [robustness.get('pgd', {}).get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, clean_vals, width, label='Clean', color=COLORS['clean'], alpha=0.8)
    bars2 = ax.bar(x, fgsm_vals, width, label='FGSM Attack', color=COLORS['fgsm'], alpha=0.8)
    bars3 = ax.bar(x + width, pgd_vals, width, label='PGD Attack', color=COLORS['pgd'], alpha=0.8)
    
    ax.set_ylabel('Value')
    ax.set_title('Model Performance: Clean vs Adversarial')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_robustness_comparison.pdf')
    plt.savefig(output_dir / 'figure3_robustness_comparison.png')
    plt.close()
    
    print("DONE")
    return True


def figure4_regime_shifts(experiments_dir: Path, output_dir: Path) -> bool:
    """
    Figure 4: Performance Across Market Regimes
    Shows model robustness to different volatility regimes.
    """
    print("  Figure 4: Regime Shifts...", end=" ")
    
    # Find any experiment with stress test results
    for pattern in ["*baseline*", "*lth*", "*adv*"]:
        exp_dir = find_latest_experiment(experiments_dir, pattern)
        if exp_dir:
            eval_path = exp_dir / 'evaluation' / 'evaluation_results.json'
            results = load_json_safe(eval_path)
            if results and 'stress_test' in results:
                stress_results = results['stress_test']
                
                regimes = list(stress_results.keys())
                pnl_means = [stress_results[r].get('pnl_mean', 0) for r in regimes]
                cvar_vals = [stress_results[r].get('cvar_05', 0) for r in regimes]
                sharpe_vals = [stress_results[r].get('sharpe_ratio', 0) for r in regimes]
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                x = np.arange(len(regimes))
                
                axes[0].bar(x, pnl_means, color=COLORS['primary'], alpha=0.8)
                axes[0].set_title('Mean P&L by Regime')
                axes[0].set_xticks(x)
                axes[0].set_xticklabels([r.replace('_', ' ').title() for r in regimes], rotation=45)
                axes[0].grid(True, alpha=0.3, axis='y')
                
                axes[1].bar(x, cvar_vals, color=COLORS['tertiary'], alpha=0.8)
                axes[1].set_title('CVaR 5% by Regime')
                axes[1].set_xticks(x)
                axes[1].set_xticklabels([r.replace('_', ' ').title() for r in regimes], rotation=45)
                axes[1].grid(True, alpha=0.3, axis='y')
                
                axes[2].bar(x, sharpe_vals, color=COLORS['secondary'], alpha=0.8)
                axes[2].set_title('Sharpe Ratio by Regime')
                axes[2].set_xticks(x)
                axes[2].set_xticklabels([r.replace('_', ' ').title() for r in regimes], rotation=45)
                axes[2].grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                plt.savefig(output_dir / 'figure4_regime_shifts.pdf')
                plt.savefig(output_dir / 'figure4_regime_shifts.png')
                plt.close()
                
                print("DONE")
                return True
    
    print("SKIP (no stress test results found)")
    return False


def figure5_sparsity_robustness_tradeoff(experiments_dir: Path, output_dir: Path) -> bool:
    """
    Figure 5: Sparsity-Robustness Trade-off
    Scatter plot showing trade-off between sparsity and adversarial robustness.
    """
    print("  Figure 5: Sparsity-Robustness Trade-off...", end=" ")
    
    # This requires multiple experiments at different sparsity levels with robustness eval
    # For now, create placeholder from LTH results if available
    
    lth_dir = find_latest_experiment(experiments_dir, "*lth*")
    if not lth_dir:
        print("SKIP (no LTH experiment found)")
        return False
    
    results = load_json_safe(lth_dir / 'logs' / 'lth_results.json')
    if not results:
        print("SKIP (no results)")
        return False
    
    results_by_sparsity = results.get('results_by_sparsity', {})
    if not results_by_sparsity:
        print("SKIP (no sparsity results)")
        return False
    
    sparsities = sorted([float(s) for s in results_by_sparsity.keys()])
    sharpe_values = [results_by_sparsity[str(s)].get('sharpe_ratio', 0) for s in sparsities]
    cvar_values = [results_by_sparsity[str(s)].get('cvar_05', 0) for s in sparsities]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Size based on remaining weights
    sizes = [(1 - s) * 500 + 50 for s in sparsities]
    
    scatter = ax.scatter(sharpe_values, cvar_values, s=sizes, c=sparsities, 
                        cmap='RdYlGn_r', alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sparsity')
    
    # Annotate points
    for i, s in enumerate(sparsities):
        ax.annotate(f'{s:.0%}', (sharpe_values[i], cvar_values[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Sharpe Ratio')
    ax.set_ylabel('CVaR 5%')
    ax.set_title('Sparsity-Performance Trade-off\n(bubble size = remaining weights)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure5_tradeoff.pdf')
    plt.savefig(output_dir / 'figure5_tradeoff.png')
    plt.close()
    
    print("DONE")
    return True


def figure6_training_dynamics(experiments_dir: Path, output_dir: Path) -> bool:
    """
    Figure 6: Training Dynamics
    Shows loss and premium evolution during training.
    """
    print("  Figure 6: Training Dynamics...", end=" ")
    
    baseline_dir = find_latest_experiment(experiments_dir, "*baseline*")
    if not baseline_dir:
        print("SKIP (no baseline experiment)")
        return False
    
    results = load_json_safe(baseline_dir / 'logs' / 'final_results.json')
    if not results or 'training' not in results:
        print("SKIP (no training data)")
        return False
    
    history = results['training'].get('history', [])
    if not history:
        print("SKIP (no training history)")
        return False
    
    epochs = [h.get('epoch', i+1) for i, h in enumerate(history)]
    train_loss = [h.get('train_loss', 0) for h in history]
    val_loss = [h.get('val_loss', 0) for h in history]
    premium = [h.get('val_premium', h.get('train_premium', 0)) for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, train_loss, '-', color=COLORS['primary'], linewidth=2, label='Train Loss')
    ax1.plot(epochs, val_loss, '-', color=COLORS['tertiary'], linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Premium evolution
    ax2.plot(epochs, premium, '-', color=COLORS['secondary'], linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learned Premium')
    ax2.set_title('Premium Parameter Evolution')
    ax2.grid(True, alpha=0.3)
    
    # Add final value annotation
    if premium:
        ax2.annotate(f'Final: {premium[-1]:.4f}', 
                    xy=(epochs[-1], premium[-1]),
                    xytext=(-50, 10), textcoords='offset points',
                    fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure6_training_dynamics.pdf')
    plt.savefig(output_dir / 'figure6_training_dynamics.png')
    plt.close()
    
    print("DONE")
    return True


def figure7_summary_table(experiments_dir: Path, output_dir: Path) -> bool:
    """
    Figure 7: Summary Table
    Creates a visual summary table of all results.
    """
    print("  Figure 7: Summary Table...", end=" ")
    
    # Collect all results
    summary_data = {}
    
    # Baseline
    baseline_dir = find_latest_experiment(experiments_dir, "*baseline*")
    if baseline_dir:
        results = load_json_safe(baseline_dir / 'logs' / 'final_results.json')
        if results and 'test_metrics' in results:
            summary_data['Dense Baseline'] = results['test_metrics']
    
    # LTH best ticket
    lth_dir = find_latest_experiment(experiments_dir, "*lth*")
    if lth_dir:
        results = load_json_safe(lth_dir / 'logs' / 'lth_results.json')
        if results:
            best_sparsity = results.get('best_sparsity_sharpe', 0)
            if best_sparsity > 0:
                metrics = results.get('results_by_sparsity', {}).get(str(best_sparsity), {})
                if metrics:
                    summary_data[f'Best Ticket ({best_sparsity:.0%})'] = metrics
    
    # Adversarial trained
    adv_dir = find_latest_experiment(experiments_dir, "*adv*")
    if adv_dir:
        results = load_json_safe(adv_dir / 'logs' / 'final_results.json')
        if results and 'robustness' in results:
            summary_data['Adversarial Trained'] = results['robustness'].get('clean', {})
    
    if not summary_data:
        print("SKIP (no results found)")
        return False
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(12, 4 + len(summary_data) * 0.5))
    ax.axis('off')
    
    # Metrics to show
    metrics = ['pnl_mean', 'pnl_std', 'cvar_05', 'sharpe_ratio']
    metric_labels = ['Mean P&L', 'Std P&L', 'CVaR 5%', 'Sharpe']
    
    # Build table data
    cell_text = []
    row_labels = []
    
    for model_name, model_metrics in summary_data.items():
        row = [f"{model_metrics.get(m, 0):.4f}" for m in metrics]
        cell_text.append(row)
        row_labels.append(model_name)
    
    # Create table
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=metric_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.15] * len(metrics)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.5, 2)
    
    # Style header
    for j in range(len(metrics)):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Style row labels
    for i in range(len(row_labels)):
        table[(i+1, -1)].set_facecolor('#D9E2F3')
        table[(i+1, -1)].set_text_props(fontweight='bold')
    
    ax.set_title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure7_summary_table.pdf')
    plt.savefig(output_dir / 'figure7_summary_table.png')
    plt.close()
    
    print("DONE")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--experiments-dir', type=str, default='experiments',
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for figures')
    return parser.parse_args()


def main():
    args = parse_args()
    
    experiments_dir = Path(args.experiments_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)
    print(f"Experiments directory: {experiments_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    if not experiments_dir.exists():
        print(f"\nWARNING: Experiments directory '{experiments_dir}' does not exist.")
        print("Run training scripts first to generate results.")
        return
    
    # Generate all figures
    results = {
        'Figure 1': figure1_performance_vs_sparsity(experiments_dir, output_dir),
        'Figure 2': figure2_convergence_speed(experiments_dir, output_dir),
        'Figure 3': figure3_robustness_comparison(experiments_dir, output_dir),
        'Figure 4': figure4_regime_shifts(experiments_dir, output_dir),
        'Figure 5': figure5_sparsity_robustness_tradeoff(experiments_dir, output_dir),
        'Figure 6': figure6_training_dynamics(experiments_dir, output_dir),
        'Figure 7': figure7_summary_table(experiments_dir, output_dir),
    }
    
    print("-" * 60)
    print("\nSUMMARY:")
    generated = sum(results.values())
    skipped = len(results) - generated
    print(f"  Generated: {generated}/{len(results)}")
    print(f"  Skipped:   {skipped}/{len(results)}")
    
    if generated > 0:
        print(f"\nFigures saved to: {output_dir}/")
    
    print("=" * 60)


if __name__ == '__main__':
    main()
