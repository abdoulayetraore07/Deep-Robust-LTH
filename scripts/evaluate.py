#!/usr/bin/env python3
"""
Evaluation Script for Deep Hedging Models

Usage:
    python scripts/evaluate.py --checkpoint path/to/best.pt
    python scripts/evaluate.py --checkpoint path/to/best.pt --adversarial
    python scripts/evaluate.py --checkpoint path/to/best.pt --stress-test
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from src.utils.config import load_config, get_device
from src.utils.visualization import (
    plot_pnl_distribution, plot_delta_comparison,
    plot_adversarial_comparison, save_all_figures
)
from src.data.heston import get_or_generate_dataset, simulate_heston
from src.data.preprocessor import create_dataloaders, compute_features
from src.models.deep_hedging import create_model
from src.models.losses import create_loss_function
from src.models.trainer import load_trained_model
from src.attacks.fgsm import create_fgsm_attack
from src.attacks.pgd import create_pgd_attack
from src.evaluation.baselines import (
    DeltaHedgingBaseline, evaluate_all_baselines, print_baseline_comparison
)
from src.evaluation.metrics import compute_all_metrics, print_metrics, compare_models


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Deep Hedging Model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config (default: load from checkpoint dir)')
    parser.add_argument('--adversarial', action='store_true',
                        help='Run adversarial evaluation')
    parser.add_argument('--stress-test', action='store_true',
                        help='Run stress test with different market regimes')
    parser.add_argument('--n-paths', type=int, default=None,
                        help='Override number of test paths')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default=None)
    
    return parser.parse_args()


def evaluate_standard(model, loss_fn, test_loader, config, device):
    """Standard evaluation on test set."""
    heston_config = config['data']['heston']
    K, T = heston_config.get('K', 100.0), config['data']['T']
    dt = T / config['data']['n_steps']
    
    model.eval()
    all_pnls, all_deltas = [], []
    all_S, all_Z = [], []
    
    with torch.no_grad():
        for S, v, Z in test_loader:
            S, v, Z = S.to(device), v.to(device), Z.to(device)
            features = compute_features(S, v, K, T, dt)
            deltas, y = model(features, S)
            pnl = loss_fn.compute_pnl(deltas, S, Z, dt)
            
            all_pnls.append(pnl.cpu())
            all_deltas.append(deltas.cpu())
            all_S.append(S.cpu())
            all_Z.append(Z.cpu())
    
    return {
        'pnl': torch.cat(all_pnls).numpy(),
        'deltas': torch.cat(all_deltas).numpy(),
        'S': torch.cat(all_S).numpy(),
        'Z': torch.cat(all_Z).numpy(),
        'premium': float(model.y.item())
    }


def evaluate_adversarial(model, loss_fn, test_loader, config, device):
    """Adversarial robustness evaluation."""
    heston_config = config['data']['heston']
    K, T = heston_config.get('K', 100.0), config['data']['T']
    dt = T / config['data']['n_steps']
    
    fgsm = create_fgsm_attack(model, loss_fn, config)
    pgd = create_pgd_attack(model, loss_fn, config)
    
    clean_pnls, fgsm_pnls, pgd_pnls = [], [], []
    
    model.eval()
    for S, v, Z in test_loader:
        S, v, Z = S.to(device), v.to(device), Z.to(device)
        features = compute_features(S, v, K, T, dt)
        
        # Clean
        with torch.no_grad():
            deltas, y = model(features, S)
            clean_pnls.append(loss_fn.compute_pnl(deltas, S, Z, dt).cpu())
        
        # FGSM
        features_fgsm, _ = fgsm.attack(features, S, Z, dt)
        with torch.no_grad():
            deltas, y = model(features_fgsm, S)
            fgsm_pnls.append(loss_fn.compute_pnl(deltas, S, Z, dt).cpu())
        
        # PGD
        features_pgd, _ = pgd.attack(features, S, Z, dt)
        with torch.no_grad():
            deltas, y = model(features_pgd, S)
            pgd_pnls.append(loss_fn.compute_pnl(deltas, S, Z, dt).cpu())
    
    return {
        'clean': torch.cat(clean_pnls).numpy(),
        'fgsm': torch.cat(fgsm_pnls).numpy(),
        'pgd': torch.cat(pgd_pnls).numpy()
    }


def evaluate_stress_test(model, loss_fn, config, device, n_paths=10000):
    """Stress test under different market regimes."""
    heston_config = config['data']['heston']
    K, T = heston_config.get('K', 100.0), config['data']['T']
    n_steps = config['data']['n_steps']
    dt = T / n_steps
    
    regimes = {
        'normal': {'v0': 0.04, 'kappa': 2.0, 'theta': 0.04, 'xi': 0.3},
        'high_vol': {'v0': 0.09, 'kappa': 2.0, 'theta': 0.09, 'xi': 0.5},
        'crisis': {'v0': 0.16, 'kappa': 1.0, 'theta': 0.16, 'xi': 0.8},
        'low_vol': {'v0': 0.01, 'kappa': 3.0, 'theta': 0.01, 'xi': 0.2}
    }
    
    results = {}
    model.eval()
    
    for regime_name, params in regimes.items():
        print(f"  Testing regime: {regime_name}")
        
        # Generate paths for this regime
        regime_config = heston_config.copy()
        regime_config.update(params)
        
        S, v = simulate_heston(
            n_paths=n_paths,
            n_steps=n_steps,
            S0=regime_config.get('S0', 100.0),
            v0=params['v0'],
            kappa=params['kappa'],
            theta=params['theta'],
            xi=params['xi'],
            rho=regime_config.get('rho', -0.7),
            r=regime_config.get('r', 0.05),
            T=T
        )
        
        # Compute payoff
        option_type = config.get('option_type', 'call')
        if option_type == 'call':
            Z = np.maximum(S[:, -1] - K, 0)
        else:
            Z = np.maximum(K - S[:, -1], 0)
        
        # Convert to tensors
        S_t = torch.FloatTensor(S).to(device)
        v_t = torch.FloatTensor(v).to(device)
        Z_t = torch.FloatTensor(Z).to(device)
        
        # Evaluate
        features = compute_features(S_t, v_t, K, T, dt)
        
        with torch.no_grad():
            deltas, y = model(features, S_t)
            pnl = loss_fn.compute_pnl(deltas, S_t, Z_t, dt)
        
        results[regime_name] = compute_all_metrics(pnl.cpu().numpy())
    
    return results


def main():
    args = parse_args()
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        # Try to load from checkpoint directory
        config_path = checkpoint_path.parent.parent / 'config.yaml'
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            print("Error: Config not found. Provide --config")
            sys.exit(1)
    
    if args.device:
        config['device'] = args.device
    
    device = get_device(config)
    print(f"Device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"Loading model from {checkpoint_path}")
    model = create_model(config)
    model, info = load_trained_model(model, str(checkpoint_path), device)
    loss_fn = create_loss_function(config)
    
    print(f"Model trained for {info['epoch']} epochs")
    print(f"Sparsity: {model.get_sparsity():.2%}")
    print(f"Learned premium: {model.y.item():.4f}")
    
    # Load test data
    cache_dir = config.get('caching', {}).get('directory', 'cache')
    S_test, v_test, Z_test = get_or_generate_dataset(config, 'test', cache_dir)
    
    if args.n_paths and args.n_paths < len(S_test):
        S_test = S_test[:args.n_paths]
        v_test = v_test[:args.n_paths]
        Z_test = Z_test[:args.n_paths]
    
    batch_size = config.get('training', {}).get('batch_size', 256)
    _, _, test_loader = create_dataloaders(
        S_test, v_test, Z_test,
        S_test, v_test, Z_test,
        S_test, v_test, Z_test,
        batch_size=batch_size
    )
    
    figures = {}
    all_results = {}
    
    # =========================================================================
    # STANDARD EVALUATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("STANDARD EVALUATION")
    print("=" * 60)
    
    std_results = evaluate_standard(model, loss_fn, test_loader, config, device)
    std_metrics = compute_all_metrics(std_results['pnl'])
    print_metrics(std_metrics, "Test Set Performance")
    all_results['standard'] = std_metrics
    
    # Baseline comparison
    baseline_results = evaluate_all_baselines(S_test, v_test, Z_test, config)
    print_baseline_comparison(baseline_results)
    
    # P&L distribution plot
    heston_config = config['data']['heston']
    delta_baseline = DeltaHedgingBaseline(
        heston_config.get('K', 100.0),
        config['data']['T'],
        heston_config.get('r', 0.02)
    )
    baseline_pnl = delta_baseline.compute_pnl(
        S_test, Z_test,
        delta_baseline.compute_deltas(S_test, v_test, dt=config['data']['T']/config['data']['n_steps'])
    )
    
    figures['pnl_distribution'] = plot_pnl_distribution(
        std_results['pnl'], baseline_pnl,
        save_path=str(output_dir / 'pnl_distribution.png')
    )
    
    # =========================================================================
    # ADVERSARIAL EVALUATION
    # =========================================================================
    if args.adversarial:
        print("\n" + "=" * 60)
        print("ADVERSARIAL EVALUATION")
        print("=" * 60)
        
        adv_results = evaluate_adversarial(model, loss_fn, test_loader, config, device)
        
        clean_metrics = compute_all_metrics(adv_results['clean'])
        fgsm_metrics = compute_all_metrics(adv_results['fgsm'])
        pgd_metrics = compute_all_metrics(adv_results['pgd'])
        
        print("\nClean:")
        print(f"  Mean P&L: {clean_metrics['pnl_mean']:.4f}, Sharpe: {clean_metrics['sharpe_ratio']:.4f}")
        print("\nFGSM Attack:")
        print(f"  Mean P&L: {fgsm_metrics['pnl_mean']:.4f}, Sharpe: {fgsm_metrics['sharpe_ratio']:.4f}")
        print(f"  Gap: {clean_metrics['pnl_mean'] - fgsm_metrics['pnl_mean']:.4f}")
        print("\nPGD Attack:")
        print(f"  Mean P&L: {pgd_metrics['pnl_mean']:.4f}, Sharpe: {pgd_metrics['sharpe_ratio']:.4f}")
        print(f"  Gap: {clean_metrics['pnl_mean'] - pgd_metrics['pnl_mean']:.4f}")
        
        all_results['adversarial'] = {
            'clean': clean_metrics,
            'fgsm': fgsm_metrics,
            'pgd': pgd_metrics
        }
        
        figures['adversarial'] = plot_adversarial_comparison(
            adv_results['clean'], adv_results['pgd'],
            save_path=str(output_dir / 'adversarial_comparison.png')
        )
    
    # =========================================================================
    # STRESS TEST
    # =========================================================================
    if args.stress_test:
        print("\n" + "=" * 60)
        print("STRESS TEST (Market Regimes)")
        print("=" * 60)
        
        stress_results = evaluate_stress_test(model, loss_fn, config, device)
        
        for regime, metrics in stress_results.items():
            print(f"\n{regime.upper()}:")
            print(f"  Mean P&L: {metrics['pnl_mean']:.4f}")
            print(f"  Sharpe: {metrics['sharpe_ratio']:.4f}")
            print(f"  CVaR 5%: {metrics['cvar_05']:.4f}")
        
        all_results['stress_test'] = stress_results
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    import json
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()