"""
Evaluation Metrics for Deep Hedging

Comprehensive metrics for evaluating hedging performance:
1. P&L Statistics (mean, std, min, max, median)
2. Risk Metrics (VaR, CVaR, Maximum Drawdown)
3. Performance Ratios (Sharpe, Sortino, Calmar)
4. Hedging Quality (tracking error, hedge effectiveness)
5. Robustness Metrics (adversarial gap, worst-case performance)
"""

import numpy as np
import torch
from typing import Dict, Union, Optional, Tuple, List
from scipy import stats


def compute_pnl_statistics(
    pnl: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute basic P&L statistics.
    
    Args:
        pnl: P&L values (n_paths,)
        
    Returns:
        Dictionary of statistics
    """
    if isinstance(pnl, torch.Tensor):
        pnl = pnl.cpu().numpy()
    
    pnl = pnl.flatten()
    
    return {
        'mean': float(np.mean(pnl)),
        'std': float(np.std(pnl)),
        'min': float(np.min(pnl)),
        'max': float(np.max(pnl)),
        'median': float(np.median(pnl)),
        'skewness': float(stats.skew(pnl)),
        'kurtosis': float(stats.kurtosis(pnl)),
        'iqr': float(np.percentile(pnl, 75) - np.percentile(pnl, 25))
    }


def compute_var(
    pnl: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.05
) -> float:
    """
    Compute Value at Risk (VaR).
    
    VaR_α is the α-quantile of the P&L distribution.
    For α=0.05, this is the 5th percentile (worst 5% of outcomes).
    
    Args:
        pnl: P&L values
        alpha: Confidence level (default: 0.05 for 95% VaR)
        
    Returns:
        VaR value (negative means loss)
    """
    if isinstance(pnl, torch.Tensor):
        pnl = pnl.cpu().numpy()
    
    return float(np.percentile(pnl.flatten(), alpha * 100))


def compute_cvar(
    pnl: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.05
) -> float:
    """
    Compute Conditional Value at Risk (CVaR / Expected Shortfall).
    
    CVaR_α is the expected value of P&L given that P&L < VaR_α.
    This is the average of the worst α% of outcomes.
    
    Args:
        pnl: P&L values
        alpha: Confidence level
        
    Returns:
        CVaR value
    """
    if isinstance(pnl, torch.Tensor):
        pnl = pnl.cpu().numpy()
    
    pnl = pnl.flatten()
    var = np.percentile(pnl, alpha * 100)
    cvar = pnl[pnl <= var].mean()
    
    return float(cvar) if not np.isnan(cvar) else float(var)


def compute_sharpe_ratio(
    pnl: Union[np.ndarray, torch.Tensor],
    target_return: float = 0.0
) -> float:
    """
    Compute Sharpe Ratio.
    
    For hedging, target_return is typically 0 (perfect hedge has zero P&L).
    
    Args:
        pnl: P&L values
        target_return: Target return (default: 0)
        
    Returns:
        Sharpe ratio
    """
    if isinstance(pnl, torch.Tensor):
        pnl = pnl.cpu().numpy()
    
    pnl = pnl.flatten()
    excess_return = np.mean(pnl) - target_return
    std = np.std(pnl)
    
    return float(excess_return / std) if std > 1e-8 else 0.0


def compute_sortino_ratio(
    pnl: Union[np.ndarray, torch.Tensor],
    target_return: float = 0.0
) -> float:
    """
    Compute Sortino Ratio (uses downside deviation).
    
    Better than Sharpe for asymmetric distributions.
    
    Args:
        pnl: P&L values
        target_return: Target return
        
    Returns:
        Sortino ratio
    """
    if isinstance(pnl, torch.Tensor):
        pnl = pnl.cpu().numpy()
    
    pnl = pnl.flatten()
    excess_return = np.mean(pnl) - target_return
    
    # Downside deviation (only negative deviations)
    downside = pnl[pnl < target_return] - target_return
    downside_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 0.0
    
    return float(excess_return / downside_std) if downside_std > 1e-8 else 0.0


def compute_max_drawdown(
    pnl: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute Maximum Drawdown.
    
    For a sequence of P&L values, this is the largest peak-to-trough decline.
    
    Args:
        pnl: P&L values (can be cumulative or per-path)
        
    Returns:
        Maximum drawdown (negative value)
    """
    if isinstance(pnl, torch.Tensor):
        pnl = pnl.cpu().numpy()
    
    pnl = pnl.flatten()
    
    # Cumulative P&L
    cumulative = np.cumsum(pnl) if pnl.ndim == 1 else pnl
    
    # Running maximum
    running_max = np.maximum.accumulate(cumulative)
    
    # Drawdown
    drawdown = cumulative - running_max
    
    return float(np.min(drawdown))


def compute_calmar_ratio(
    pnl: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute Calmar Ratio (return / max drawdown).
    
    Args:
        pnl: P&L values
        
    Returns:
        Calmar ratio
    """
    if isinstance(pnl, torch.Tensor):
        pnl = pnl.cpu().numpy()
    
    pnl = pnl.flatten()
    mean_return = np.mean(pnl)
    max_dd = abs(compute_max_drawdown(pnl))
    
    return float(mean_return / max_dd) if max_dd > 1e-8 else 0.0


def compute_hedge_effectiveness(
    pnl_hedged: Union[np.ndarray, torch.Tensor],
    pnl_unhedged: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute hedge effectiveness ratio.
    
    HE = 1 - Var(hedged) / Var(unhedged)
    
    HE = 1 means perfect hedge, HE = 0 means no improvement.
    
    Args:
        pnl_hedged: P&L with hedging
        pnl_unhedged: P&L without hedging
        
    Returns:
        Hedge effectiveness (0 to 1)
    """
    if isinstance(pnl_hedged, torch.Tensor):
        pnl_hedged = pnl_hedged.cpu().numpy()
    if isinstance(pnl_unhedged, torch.Tensor):
        pnl_unhedged = pnl_unhedged.cpu().numpy()
    
    var_hedged = np.var(pnl_hedged)
    var_unhedged = np.var(pnl_unhedged)
    
    if var_unhedged < 1e-8:
        return 0.0
    
    return float(1 - var_hedged / var_unhedged)


def compute_tracking_error(
    deltas_model: Union[np.ndarray, torch.Tensor],
    deltas_target: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute tracking error between model deltas and target deltas.
    
    Args:
        deltas_model: Model's hedging positions (n_paths, n_steps)
        deltas_target: Target positions (e.g., BS delta)
        
    Returns:
        Tracking error statistics
    """
    if isinstance(deltas_model, torch.Tensor):
        deltas_model = deltas_model.cpu().numpy()
    if isinstance(deltas_target, torch.Tensor):
        deltas_target = deltas_target.cpu().numpy()
    
    error = deltas_model - deltas_target
    
    return {
        'mean_error': float(np.mean(error)),
        'std_error': float(np.std(error)),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'mae': float(np.mean(np.abs(error))),
        'max_error': float(np.max(np.abs(error)))
    }


def compute_robustness_metrics(
    pnl_clean: Union[np.ndarray, torch.Tensor],
    pnl_adversarial: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute robustness metrics comparing clean and adversarial performance.
    
    Args:
        pnl_clean: P&L on clean data
        pnl_adversarial: P&L on adversarial data
        
    Returns:
        Robustness metrics
    """
    if isinstance(pnl_clean, torch.Tensor):
        pnl_clean = pnl_clean.cpu().numpy()
    if isinstance(pnl_adversarial, torch.Tensor):
        pnl_adversarial = pnl_adversarial.cpu().numpy()
    
    pnl_clean = pnl_clean.flatten()
    pnl_adversarial = pnl_adversarial.flatten()
    
    mean_clean = np.mean(pnl_clean)
    mean_adv = np.mean(pnl_adversarial)
    
    cvar_clean = compute_cvar(pnl_clean)
    cvar_adv = compute_cvar(pnl_adversarial)
    
    return {
        'mean_clean': float(mean_clean),
        'mean_adversarial': float(mean_adv),
        'mean_gap': float(mean_clean - mean_adv),
        'mean_gap_pct': float((mean_clean - mean_adv) / abs(mean_clean)) if abs(mean_clean) > 1e-8 else 0.0,
        'cvar_clean': cvar_clean,
        'cvar_adversarial': cvar_adv,
        'cvar_gap': float(cvar_clean - cvar_adv),
        'worst_case_clean': float(np.min(pnl_clean)),
        'worst_case_adversarial': float(np.min(pnl_adversarial)),
        'robustness_ratio': float(mean_adv / mean_clean) if abs(mean_clean) > 1e-8 else 0.0
    }


def compute_sparsity_performance(
    results_by_sparsity: Dict[float, Dict[str, float]]
) -> Dict[str, float]:
    """
    Analyze performance across different sparsity levels.
    
    For Lottery Ticket Hypothesis analysis.
    
    Args:
        results_by_sparsity: Dict mapping sparsity -> performance metrics
        
    Returns:
        Summary statistics
    """
    sparsities = sorted(results_by_sparsity.keys())
    
    if len(sparsities) == 0:
        return {}
    
    # Find best performing sparsity
    best_sparsity = max(sparsities, key=lambda s: results_by_sparsity[s].get('sharpe_ratio', 0))
    
    # Find maximum sparsity with acceptable performance
    baseline_sharpe = results_by_sparsity[sparsities[0]].get('sharpe_ratio', 0)
    threshold = 0.95 * baseline_sharpe  # 95% of baseline performance
    
    max_acceptable_sparsity = 0.0
    for s in sparsities:
        if results_by_sparsity[s].get('sharpe_ratio', 0) >= threshold:
            max_acceptable_sparsity = s
    
    return {
        'best_sparsity': best_sparsity,
        'best_sharpe': results_by_sparsity[best_sparsity].get('sharpe_ratio', 0),
        'max_acceptable_sparsity': max_acceptable_sparsity,
        'performance_at_90_sparse': results_by_sparsity.get(0.9, {}).get('sharpe_ratio', None),
        'num_sparsity_levels': len(sparsities)
    }


def compute_all_metrics(
    pnl: Union[np.ndarray, torch.Tensor],
    pnl_unhedged: Optional[Union[np.ndarray, torch.Tensor]] = None,
    deltas_model: Optional[Union[np.ndarray, torch.Tensor]] = None,
    deltas_target: Optional[Union[np.ndarray, torch.Tensor]] = None,
    pnl_adversarial: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Dict[str, float]:
    """
    Compute all available metrics.
    
    Args:
        pnl: P&L values
        pnl_unhedged: P&L without hedging (for hedge effectiveness)
        deltas_model: Model's deltas (for tracking error)
        deltas_target: Target deltas (for tracking error)
        pnl_adversarial: Adversarial P&L (for robustness)
        
    Returns:
        Comprehensive metrics dictionary
    """
    metrics = {}
    
    # Basic statistics
    pnl_stats = compute_pnl_statistics(pnl)
    for key, value in pnl_stats.items():
        metrics[f'pnl_{key}'] = value
    
    # Risk metrics
    metrics['var_05'] = compute_var(pnl, 0.05)
    metrics['var_01'] = compute_var(pnl, 0.01)
    metrics['cvar_05'] = compute_cvar(pnl, 0.05)
    metrics['cvar_01'] = compute_cvar(pnl, 0.01)
    
    # Performance ratios
    metrics['sharpe_ratio'] = compute_sharpe_ratio(pnl)
    metrics['sortino_ratio'] = compute_sortino_ratio(pnl)
    metrics['calmar_ratio'] = compute_calmar_ratio(pnl)
    metrics['max_drawdown'] = compute_max_drawdown(pnl)
    
    # Hedge effectiveness
    if pnl_unhedged is not None:
        metrics['hedge_effectiveness'] = compute_hedge_effectiveness(pnl, pnl_unhedged)
    
    # Tracking error
    if deltas_model is not None and deltas_target is not None:
        tracking = compute_tracking_error(deltas_model, deltas_target)
        for key, value in tracking.items():
            metrics[f'tracking_{key}'] = value
    
    # Robustness
    if pnl_adversarial is not None:
        robustness = compute_robustness_metrics(pnl, pnl_adversarial)
        for key, value in robustness.items():
            metrics[f'robustness_{key}'] = value
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """Pretty print metrics."""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print('=' * 60)
    
    # Group metrics
    groups = {
        'P&L Statistics': ['pnl_mean', 'pnl_std', 'pnl_min', 'pnl_max', 'pnl_median'],
        'Risk Metrics': ['var_05', 'var_01', 'cvar_05', 'cvar_01', 'max_drawdown'],
        'Performance Ratios': ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'hedge_effectiveness'],
        'Tracking Error': ['tracking_rmse', 'tracking_mae', 'tracking_max_error'],
        'Robustness': ['robustness_mean_gap', 'robustness_cvar_gap', 'robustness_ratio']
    }
    
    for group_name, keys in groups.items():
        group_metrics = {k: v for k, v in metrics.items() if any(k.startswith(key.split('_')[0]) or k == key for key in keys)}
        
        if group_metrics:
            print(f"\n{group_name}:")
            print('-' * 40)
            for key, value in group_metrics.items():
                if value is not None:
                    print(f"  {key:<30} {value:>12.6f}")
    
    print('=' * 60 + '\n')


def compare_models(
    results: Dict[str, Dict[str, float]],
    metric_keys: Optional[List[str]] = None
) -> str:
    """
    Create comparison table for multiple models.
    
    Args:
        results: Dict mapping model_name -> metrics
        metric_keys: Which metrics to include (default: main ones)
        
    Returns:
        Formatted comparison table string
    """
    if metric_keys is None:
        metric_keys = ['pnl_mean', 'pnl_std', 'cvar_05', 'sharpe_ratio']
    
    # Header
    lines = ["\nModel Comparison", "=" * 80]
    
    header = f"{'Model':<25}"
    for key in metric_keys:
        short_key = key.replace('pnl_', '').replace('robustness_', 'rob_')[:12]
        header += f" {short_key:>12}"
    lines.append(header)
    lines.append("-" * 80)
    
    # Rows
    for model_name, metrics in results.items():
        row = f"{model_name:<25}"
        for key in metric_keys:
            value = metrics.get(key, float('nan'))
            row += f" {value:>12.4f}"
        lines.append(row)
    
    lines.append("=" * 80)
    
    return "\n".join(lines)