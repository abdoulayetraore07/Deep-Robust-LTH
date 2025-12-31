"""
Evaluation metrics for deep hedging
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Any
from tqdm import tqdm

from ..models.losses import compute_pnl
from ..attacks.fgsm import fgsm_attack
from ..attacks.pgd import pgd_attack


def compute_all_metrics(
    model: nn.Module,
    test_loader: DataLoader,
    config: Dict[str, Any],
    K: float,
    T: float,
    dt: float,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute comprehensive metrics on test set
    
    Args:
        model: Deep Hedging network
        test_loader: Test data loader
        config: Configuration dictionary
        K: Strike price
        T: Time to maturity
        dt: Time step
        device: Device to use
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_pnl = []
    all_hedging_errors = []
    all_trading_volumes = []
    all_payoffs = []
    
    with torch.no_grad():
        for S, v, Z in tqdm(test_loader, desc="Computing metrics"):
            S = S.to(device)
            v = v.to(device)
            Z = Z.to(device)
            
            # Compute features
            features = _compute_features_batch(S, v, K, T, dt, device)
            
            # Forward pass
            delta = model(features)
            
            # P&L
            pnl = compute_pnl(S, delta, Z, c_prop=config['data']['transaction_cost']['c_prop'])
            all_pnl.append(pnl.cpu().numpy())
            
            # Hedging error: |Z - integral(delta dS)|
            dS = S[:, 1:] - S[:, :-1]
            trading_pnl = (delta[:, :-1] * dS).sum(dim=1)
            hedging_error = torch.abs(Z - trading_pnl)
            all_hedging_errors.append(hedging_error.cpu().numpy())
            
            # Trading volume
            delta_changes = torch.abs(delta[:, 1:] - delta[:, :-1])
            trading_volume = delta_changes.sum(dim=1)
            all_trading_volumes.append(trading_volume.cpu().numpy())
            
            # Payoffs
            all_payoffs.append(Z.cpu().numpy())
    
    # Concatenate all batches
    all_pnl = np.concatenate(all_pnl)
    all_hedging_errors = np.concatenate(all_hedging_errors)
    all_trading_volumes = np.concatenate(all_trading_volumes)
    all_payoffs = np.concatenate(all_payoffs)
    
    # Compute metrics
    metrics = {
        'mean_pnl': float(np.mean(all_pnl)),
        'std_pnl': float(np.std(all_pnl)),
        'sharpe_ratio': float(np.mean(all_pnl) / np.std(all_pnl)) if np.std(all_pnl) > 0 else 0.0,
        'cvar_005': float(np.mean(np.sort(all_pnl)[:int(0.05 * len(all_pnl))])),
        'cvar_010': float(np.mean(np.sort(all_pnl)[:int(0.10 * len(all_pnl))])),
        'max_drawdown': float(np.min(all_pnl)),
        'hedging_error_rmse': float(np.sqrt(np.mean(all_hedging_errors ** 2))),
        'hedging_error_mae': float(np.mean(all_hedging_errors)),
        'total_trading_volume': float(np.mean(all_trading_volumes)),
        'mean_payoff': float(np.mean(all_payoffs)),
    }
    
    return metrics


def evaluate_robustness(
    model: nn.Module,
    test_loader: DataLoader,
    config: Dict[str, Any],
    K: float,
    T: float,
    dt: float,
    device: str = 'cuda'
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model robustness under adversarial attacks
    
    Args:
        model: Deep Hedging network
        test_loader: Test data loader
        config: Configuration dictionary
        K: Strike price
        T: Time to maturity
        dt: Time step
        device: Device to use
        
    Returns:
        Dictionary with clean, FGSM, and PGD metrics
    """
    results = {}
    
    # Clean evaluation
    print("Evaluating on clean data...")
    results['clean'] = compute_all_metrics(model, test_loader, config, K, T, dt, device)
    
    # FGSM attack
    print("Evaluating on FGSM adversarial examples...")
    results['fgsm'] = _evaluate_under_attack(
        model, test_loader, config, K, T, dt, device, attack_type='fgsm'
    )
    
    # PGD-10 attack
    print("Evaluating on PGD-10 adversarial examples...")
    results['pgd10'] = _evaluate_under_attack(
        model, test_loader, config, K, T, dt, device, attack_type='pgd', num_steps=10
    )
    
    # PGD-20 attack
    print("Evaluating on PGD-20 adversarial examples...")
    results['pgd20'] = _evaluate_under_attack(
        model, test_loader, config, K, T, dt, device, attack_type='pgd', num_steps=20
    )
    
    # Compute robustness gaps
    results['robustness_gap_fgsm'] = results['fgsm']['cvar_005'] - results['clean']['cvar_005']
    results['robustness_gap_pgd10'] = results['pgd10']['cvar_005'] - results['clean']['cvar_005']
    results['robustness_gap_pgd20'] = results['pgd20']['cvar_005'] - results['clean']['cvar_005']
    
    return results


def _evaluate_under_attack(
    model: nn.Module,
    test_loader: DataLoader,
    config: Dict[str, Any],
    K: float,
    T: float,
    dt: float,
    device: str,
    attack_type: str = 'fgsm',
    num_steps: int = 10
) -> Dict[str, float]:
    """
    Evaluate model under adversarial attack
    """
    model.eval()
    
    all_pnl = []
    all_hedging_errors = []
    all_trading_volumes = []
    
    for S, v, Z in tqdm(test_loader, desc=f"Attack: {attack_type}"):
        S = S.to(device)
        v = v.to(device)
        Z = Z.to(device)
        
        # Generate adversarial examples
        if attack_type == 'fgsm':
            S_adv, v_adv = fgsm_attack(
                model, S, v, Z,
                lambda s, vv: _compute_features_batch(s, vv, K, T, dt, device),
                config,
                config['attacks']['fgsm']['epsilon_S'],
                config['attacks']['fgsm']['epsilon_v']
            )
        elif attack_type == 'pgd':
            S_adv, v_adv = pgd_attack(
                model, S, v, Z,
                lambda s, vv: _compute_features_batch(s, vv, K, T, dt, device),
                config,
                config['attacks']['pgd']['epsilon_S'],
                config['attacks']['pgd']['epsilon_v'],
                config['attacks']['pgd']['alpha_S'],
                config['attacks']['pgd']['alpha_v'],
                num_steps
            )
        
        # Evaluate on adversarial examples
        with torch.no_grad():
            features_adv = _compute_features_batch(S_adv, v_adv, K, T, dt, device)
            delta = model(features_adv)
            
            pnl = compute_pnl(S_adv, delta, Z, c_prop=config['data']['transaction_cost']['c_prop'])
            all_pnl.append(pnl.cpu().numpy())
            
            dS = S_adv[:, 1:] - S_adv[:, :-1]
            trading_pnl = (delta[:, :-1] * dS).sum(dim=1)
            hedging_error = torch.abs(Z - trading_pnl)
            all_hedging_errors.append(hedging_error.cpu().numpy())
            
            delta_changes = torch.abs(delta[:, 1:] - delta[:, :-1])
            trading_volume = delta_changes.sum(dim=1)
            all_trading_volumes.append(trading_volume.cpu().numpy())
    
    # Concatenate
    all_pnl = np.concatenate(all_pnl)
    all_hedging_errors = np.concatenate(all_hedging_errors)
    all_trading_volumes = np.concatenate(all_trading_volumes)
    
    # Compute metrics
    metrics = {
        'mean_pnl': float(np.mean(all_pnl)),
        'std_pnl': float(np.std(all_pnl)),
        'sharpe_ratio': float(np.mean(all_pnl) / np.std(all_pnl)) if np.std(all_pnl) > 0 else 0.0,
        'cvar_005': float(np.mean(np.sort(all_pnl)[:int(0.05 * len(all_pnl))])),
        'cvar_010': float(np.mean(np.sort(all_pnl)[:int(0.10 * len(all_pnl))])),
        'max_drawdown': float(np.min(all_pnl)),
        'hedging_error_rmse': float(np.sqrt(np.mean(all_hedging_errors ** 2))),
        'hedging_error_mae': float(np.mean(all_hedging_errors)),
        'total_trading_volume': float(np.mean(all_trading_volumes)),
    }
    
    return metrics

def _compute_features_batch(
    S: torch.Tensor,
    v: torch.Tensor,
    K: float,
    T: float,
    dt: float,
    device: str
) -> torch.Tensor:
    """
    Compute features for a batch (helper function)
    """
    from ..data.preprocessor import compute_features
    
    # Compute direct sur GPU
    features = compute_features(S, v, K, T, dt)
    
    return features

