"""
Script to generate all publication-ready figures
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def generate_all_figures():
    """
    Generate all 7 publication-ready figures
    """
    figures_dir = Path('figures')
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating publication figures...")
    
    # Figure 1: Performance vs Sparsity (from Phase 2)
    try:
        with open('experiments/pruning/sparsity_results.json', 'r') as f:
            sparsity_results = json.load(f)
        
        sparsities = [float(k) for k in sparsity_results.keys()]
        cvars = [sparsity_results[str(s)]['final_cvar'] for s in sparsities]
        
        with open('experiments/baseline/metrics.json', 'r') as f:
            baseline_metrics = json.load(f)
        baseline_cvar = baseline_metrics['cvar_005']
        
        plt.figure(figsize=(10, 6))
        plt.plot([s*100 for s in sparsities], cvars, 'o-', linewidth=2, markersize=8, label='Sparse Model')
        plt.axhline(baseline_cvar, color='red', linestyle='--', linewidth=2, label='Dense Baseline')
        plt.xlabel('Sparsity (%)')
        plt.ylabel('CVaR (lower = better)')
        plt.title('Performance vs Sparsity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(figures_dir / 'figure1_performance_vs_sparsity.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Figure 1: Performance vs Sparsity - DONE")
    except Exception as e:
        print(f"  Figure 1: FAILED - {e}")
    
    # Figure 2: Convergence Speed (from Phase 2)
    print("  Figure 2: Convergence Speed - SKIP (requires training curves)")
    
    # Figure 3: Robustness Comparison (from Phase 3)
    try:
        with open('experiments/adversarial/attack_results.json', 'r') as f:
            attack_results = json.load(f)
        
        model_names = list(attack_results.keys())
        clean_cvars = [attack_results[m]['clean']['cvar_005'] for m in model_names]
        fgsm_cvars = [attack_results[m]['fgsm']['cvar_005'] for m in model_names]
        pgd_cvars = [attack_results[m]['pgd10']['cvar_005'] for m in model_names]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, clean_cvars, width, label='Clean', color='blue', alpha=0.8)
        ax.bar(x, fgsm_cvars, width, label='FGSM', color='orange', alpha=0.8)
        ax.bar(x + width, pgd_cvars, width, label='PGD-10', color='red', alpha=0.8)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('CVaR (lower = better)')
        ax.set_title('Robustness Comparison: Clean vs FGSM vs PGD')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'figure3_robustness_comparison.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Figure 3: Robustness Comparison - DONE")
    except Exception as e:
        print(f"  Figure 3: FAILED - {e}")
    
    # Figure 4: Regime Shifts (from Phase 4)
    try:
        with open('experiments/regime_shifts/results.json', 'r') as f:
            regime_results = json.load(f)
        
        regimes = ['calm', 'high_vol', 'extreme']
        model_names = list(regime_results.keys())
        
        x = np.arange(len(regimes))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, model_name in enumerate(model_names):
            cvars = [regime_results[model_name][r]['cvar_005'] for r in regimes]
            ax.bar(x + i*width, cvars, width, label=model_name)
        
        ax.set_xlabel('Market Regime')
        ax.set_ylabel('CVaR')
        ax.set_title('Performance Across Market Regimes')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(['Calm', 'High Vol', 'Extreme'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'figure4_regime_shifts.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("  Figure 4: Regime Shifts - DONE")
    except Exception as e:
        print(f"  Figure 4: FAILED - {e}")
    
    # Figure 5: Comparison Table (from Phase 5)
    try:
        with open('experiments/adversarial_training/comparison.json', 'r') as f:
            comparison = json.load(f)
        
        print("  Figure 5: Comparison Table - Data available in JSON")
    except Exception as e:
        print(f"  Figure 5: FAILED - {e}")
    
    # Figure 6: Trade-off Scatter (from Phase 5)
    print("  Figure 6: Trade-off Scatter - SKIP (requires multiple runs)")
    
    # Figure 7: Feature Importance (from Phase 6)
    print("  Figure 7: Feature Importance - SKIP (requires interpretation notebook)")
    
    print("\nFigure generation complete")


if __name__ == '__main__':
    generate_all_figures()