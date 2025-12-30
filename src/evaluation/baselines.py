"""
Baseline hedging strategies for comparison
"""

import numpy as np
from scipy.stats import norm


def black_scholes_delta(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    Compute Black-Scholes Delta for European call option
    
    Args:
        S: Stock price (batch, n_steps)
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility (batch, n_steps)
        t: Time array (n_steps,)
        
    Returns:
        delta: Black-Scholes delta (batch, n_steps)
    """
    batch_size, n_steps = S.shape
    delta = np.zeros((batch_size, n_steps))
    
    for i in range(n_steps):
        tau = T - t[i]  # Time to maturity
        
        if tau > 1e-6:
            d1 = (np.log(S[:, i] / K) + (r + 0.5 * sigma[:, i]**2) * tau) / (sigma[:, i] * np.sqrt(tau))
            delta[:, i] = norm.cdf(d1)
        else:
            # At maturity
            delta[:, i] = (S[:, i] >= K).astype(float)
    
    return delta


def delta_hedging_baseline(
    S: np.ndarray,
    v: np.ndarray,
    Z: np.ndarray,
    K: float,
    T: float,
    r: float,
    dt: float,
    c_prop: float = 0.001
) -> dict:
    """
    Evaluate Black-Scholes delta hedging baseline
    
    Args:
        S: Stock prices (n_paths, n_steps)
        v: Variances (n_paths, n_steps)
        Z: Payoffs (n_paths,)
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        dt: Time step
        c_prop: Proportional transaction cost
        
    Returns:
        Dictionary of metrics
    """
    n_paths, n_steps = S.shape
    
    # Time array
    t = np.arange(n_steps) * dt
    
    # Volatility (sqrt of variance)
    sigma = np.sqrt(v)
    
    # Compute BS delta
    delta = black_scholes_delta(S, K, T, r, sigma, t)
    
    # Compute P&L
    # Trading P&L: sum of delta_t * (S_{t+1} - S_t)
    dS = S[:, 1:] - S[:, :-1]
    trading_pnl = (delta[:, :-1] * dS).sum(axis=1)
    
    # Transaction costs
    delta_changes = np.abs(delta[:, 1:] - delta[:, :-1])
    costs = c_prop * delta_changes.sum(axis=1)
    
    # Final P&L
    pnl = - Z + trading_pnl - costs
    
    # Hedging error
    hedging_error = np.abs(Z - trading_pnl)
    
    # Trading volume
    trading_volume = delta_changes.sum(axis=1)
    
    # Compute metrics
    metrics = {
        'mean_pnl': float(np.mean(pnl)),
        'std_pnl': float(np.std(pnl)),
        'sharpe_ratio': float(np.mean(pnl) / np.std(pnl)) if np.std(pnl) > 0 else 0.0,
        'cvar_005': float(np.mean(np.sort(pnl)[:int(0.05 * len(pnl))])),
        'cvar_010': float(np.mean(np.sort(pnl)[:int(0.10 * len(pnl))])),
        'max_drawdown': float(np.min(pnl)),
        'hedging_error_rmse': float(np.sqrt(np.mean(hedging_error ** 2))),
        'hedging_error_mae': float(np.mean(hedging_error)),
        'total_trading_volume': float(np.mean(trading_volume)),
    }
    
    return metrics


def no_hedging_baseline(
    Z: np.ndarray
) -> dict:
    """
    Evaluate no-hedging baseline (just hold the option)
    
    Args:
        Z: Payoffs (n_paths,)
        
    Returns:
        Dictionary of metrics
    """
    # P&L = just the payoff (no hedging)
    pnl = - Z
    
    # Compute metrics
    metrics = {
        'mean_pnl': float(np.mean(pnl)),
        'std_pnl': float(np.std(pnl)),
        'sharpe_ratio': float(np.mean(pnl) / np.std(pnl)) if np.std(pnl) > 0 else 0.0,
        'cvar_005': float(np.mean(np.sort(pnl)[:int(0.05 * len(pnl))])),
        'cvar_010': float(np.mean(np.sort(pnl)[:int(0.10 * len(pnl))])),
        'max_drawdown': float(np.min(pnl)),
        'hedging_error_rmse': 0.0,
        'hedging_error_mae': 0.0,
        'total_trading_volume': 0.0,
    }
    
    return metrics