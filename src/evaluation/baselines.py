"""
Baseline Hedging Strategies

Implements baseline strategies for comparison with Deep Hedging:
1. Delta Hedging (Black-Scholes) - The standard benchmark
2. No Hedging - Just hold the option position
3. Static Hedging - Fixed delta throughout

These baselines help evaluate whether the deep hedging model
actually learns something useful beyond classical approaches.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union
from scipy.stats import norm


def black_scholes_delta(
    S: Union[np.ndarray, torch.Tensor],
    K: float,
    T: float,
    r: float,
    sigma: Union[float, np.ndarray, torch.Tensor],
    t: Union[float, np.ndarray, torch.Tensor] = 0.0,
    option_type: str = 'call'
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute Black-Scholes delta.
    
    Delta = N(d1) for call, N(d1) - 1 for put
    
    Args:
        S: Spot price(s)
        K: Strike price
        T: Total time to maturity
        r: Risk-free rate
        sigma: Volatility (can be array for local vol)
        t: Current time (default: 0)
        option_type: 'call' or 'put'
        
    Returns:
        Delta value(s)
    """
    is_torch = isinstance(S, torch.Tensor)
    
    if is_torch:
        return _bs_delta_torch(S, K, T, r, sigma, t, option_type)
    else:
        return _bs_delta_numpy(S, K, T, r, sigma, t, option_type)


def _bs_delta_numpy(
    S: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
    option_type: str
) -> np.ndarray:
    """NumPy implementation of BS delta."""
    tau = np.maximum(T - t, 1e-8)  # Time to maturity
    
    # Handle sigma as array or scalar
    if isinstance(sigma, np.ndarray):
        vol = sigma
    else:
        vol = sigma * np.ones_like(S)
    
    d1 = (np.log(S / K) + (r + 0.5 * vol ** 2) * tau) / (vol * np.sqrt(tau))
    
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    return delta


def _bs_delta_torch(
    S: torch.Tensor,
    K: float,
    T: float,
    r: float,
    sigma: Union[float, torch.Tensor],
    t: Union[float, torch.Tensor],
    option_type: str
) -> torch.Tensor:
    """PyTorch implementation of BS delta."""
    if isinstance(t, (int, float)):
        tau = T - t
    else:
        tau = T - t
    
    tau = torch.clamp(tau, min=1e-8) if isinstance(tau, torch.Tensor) else max(tau, 1e-8)
    
    # Handle sigma
    if isinstance(sigma, (int, float)):
        vol = sigma
    else:
        vol = sigma
    
    if isinstance(tau, torch.Tensor):
        sqrt_tau = torch.sqrt(tau)
    else:
        sqrt_tau = np.sqrt(tau)
    
    d1 = (torch.log(S / K) + (r + 0.5 * vol ** 2) * tau) / (vol * sqrt_tau)
    
    # Use normal CDF
    normal = torch.distributions.Normal(0, 1)
    
    if option_type.lower() == 'call':
        delta = normal.cdf(d1)
    else:
        delta = normal.cdf(d1) - 1
    
    return delta


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call'
) -> float:
    """
    Compute Black-Scholes option price.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        
    Returns:
        Option price
    """
    if T <= 0:
        # At expiry
        if option_type.lower() == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price


class DeltaHedgingBaseline:
    """
    Black-Scholes Delta Hedging baseline.
    
    At each time step, hold delta shares of the underlying
    to hedge the option position.
    """
    
    def __init__(
        self,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        use_implied_vol: bool = False
    ):
        """
        Initialize delta hedging baseline.
        
        Args:
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            use_implied_vol: If True, use instantaneous vol; else use constant vol
        """
        self.K = K
        self.T = T
        self.r = r
        self.option_type = option_type
        self.use_implied_vol = use_implied_vol
    
    def compute_deltas(
        self,
        S: Union[np.ndarray, torch.Tensor],
        v: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sigma: float = 0.2,
        dt: Optional[float] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute delta hedging positions for all paths and time steps.
        
        Args:
            S: Stock prices (n_paths, n_steps)
            v: Variances (n_paths, n_steps) - used if use_implied_vol=True
            sigma: Constant volatility (used if use_implied_vol=False)
            dt: Time step size
            
        Returns:
            deltas: Hedging positions (n_paths, n_steps)
        """
        is_torch = isinstance(S, torch.Tensor)
        
        if is_torch:
            n_paths, n_steps = S.shape
            device = S.device
            dtype = S.dtype
            
            deltas = torch.zeros(n_paths, n_steps, device=device, dtype=dtype)
            
            for t_idx in range(n_steps):
                t = t_idx * dt if dt is not None else t_idx * (self.T / n_steps)
                
                if self.use_implied_vol and v is not None:
                    vol = torch.sqrt(torch.clamp(v[:, t_idx], min=1e-8))
                else:
                    vol = sigma
                
                deltas[:, t_idx] = black_scholes_delta(
                    S[:, t_idx], self.K, self.T, self.r, vol, t, self.option_type
                )
        else:
            n_paths, n_steps = S.shape
            deltas = np.zeros((n_paths, n_steps))
            
            for t_idx in range(n_steps):
                t = t_idx * dt if dt is not None else t_idx * (self.T / n_steps)
                
                if self.use_implied_vol and v is not None:
                    vol = np.sqrt(np.maximum(v[:, t_idx], 1e-8))
                else:
                    vol = sigma
                
                deltas[:, t_idx] = black_scholes_delta(
                    S[:, t_idx], self.K, self.T, self.r, vol, t, self.option_type
                )
        
        return deltas
    
    def compute_pnl(
        self,
        S: Union[np.ndarray, torch.Tensor],
        Z: Union[np.ndarray, torch.Tensor],
        deltas: Union[np.ndarray, torch.Tensor],
        transaction_cost: float = 0.0
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute P&L for delta hedging strategy.
        
        PnL = hedging_gains - payoff - transaction_costs
        
        Args:
            S: Stock prices (n_paths, n_steps)
            Z: Option payoff (n_paths,)
            deltas: Hedging positions (n_paths, n_steps)
            transaction_cost: Proportional transaction cost
            
        Returns:
            pnl: P&L per path (n_paths,)
        """
        is_torch = isinstance(S, torch.Tensor)
        
        if is_torch:
            # Stock price changes
            dS = S[:, 1:] - S[:, :-1]
            
            # Hedging gains: Σ δ_{t-1} * dS_t
            delta_prev = deltas[:, :-1]
            hedging_gains = (delta_prev * dS).sum(dim=1)
            
            # Transaction costs
            if transaction_cost > 0:
                initial_trade = torch.abs(deltas[:, 0]) * S[:, 0]
                delta_changes = torch.abs(deltas[:, 1:] - deltas[:, :-1])
                subsequent_trades = (delta_changes * S[:, 1:]).sum(dim=1)
                final_trade = torch.abs(deltas[:, -1]) * S[:, -1]
                total_tc = transaction_cost * (initial_trade + subsequent_trades + final_trade)
            else:
                total_tc = 0.0
            
            pnl = hedging_gains - Z - total_tc
        else:
            dS = S[:, 1:] - S[:, :-1]
            delta_prev = deltas[:, :-1]
            hedging_gains = (delta_prev * dS).sum(axis=1)
            
            if transaction_cost > 0:
                initial_trade = np.abs(deltas[:, 0]) * S[:, 0]
                delta_changes = np.abs(deltas[:, 1:] - deltas[:, :-1])
                subsequent_trades = (delta_changes * S[:, 1:]).sum(axis=1)
                final_trade = np.abs(deltas[:, -1]) * S[:, -1]
                total_tc = transaction_cost * (initial_trade + subsequent_trades + final_trade)
            else:
                total_tc = 0.0
            
            pnl = hedging_gains - Z - total_tc
        
        return pnl
    
    def evaluate(
        self,
        S: Union[np.ndarray, torch.Tensor],
        v: Optional[Union[np.ndarray, torch.Tensor]],
        Z: Union[np.ndarray, torch.Tensor],
        sigma: float = 0.2,
        dt: Optional[float] = None,
        transaction_cost: float = 0.0
    ) -> Dict[str, float]:
        """
        Full evaluation of delta hedging baseline.
        
        Args:
            S: Stock prices
            v: Variances (optional)
            Z: Option payoffs
            sigma: Constant volatility
            dt: Time step
            transaction_cost: Transaction cost
            
        Returns:
            Dictionary of metrics
        """
        # Compute deltas
        deltas = self.compute_deltas(S, v, sigma, dt)
        
        # Compute P&L
        pnl = self.compute_pnl(S, Z, deltas, transaction_cost)
        
        # Convert to numpy for statistics
        if isinstance(pnl, torch.Tensor):
            pnl_np = pnl.cpu().numpy()
        else:
            pnl_np = pnl
        
        # Compute metrics
        mean_pnl = float(np.mean(pnl_np))
        std_pnl = float(np.std(pnl_np))
        
        # CVaR at 5%
        sorted_pnl = np.sort(pnl_np)
        var_idx = int(0.05 * len(sorted_pnl))
        cvar_05 = float(np.mean(sorted_pnl[:max(var_idx, 1)]))
        
        # Sharpe ratio (assuming mean should be 0 for perfect hedge)
        sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
        
        return {
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'min_pnl': float(np.min(pnl_np)),
            'max_pnl': float(np.max(pnl_np)),
            'cvar_05': cvar_05,
            'sharpe_ratio': sharpe,
            'median_pnl': float(np.median(pnl_np))
        }


class NoHedgingBaseline:
    """
    No hedging baseline - just hold the naked option position.
    
    This is a lower bound on performance.
    """
    
    def __init__(self, option_type: str = 'call'):
        self.option_type = option_type
    
    def evaluate(
        self,
        S: Union[np.ndarray, torch.Tensor],
        Z: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate no hedging baseline.
        
        P&L = -Z (we sold the option and did nothing)
        """
        if isinstance(Z, torch.Tensor):
            pnl = -Z.cpu().numpy()
        else:
            pnl = -Z
        
        mean_pnl = float(np.mean(pnl))
        std_pnl = float(np.std(pnl))
        
        sorted_pnl = np.sort(pnl)
        var_idx = int(0.05 * len(sorted_pnl))
        cvar_05 = float(np.mean(sorted_pnl[:max(var_idx, 1)]))
        
        return {
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'min_pnl': float(np.min(pnl)),
            'max_pnl': float(np.max(pnl)),
            'cvar_05': cvar_05,
            'sharpe_ratio': mean_pnl / std_pnl if std_pnl > 0 else 0.0
        }


class StaticHedgingBaseline:
    """
    Static hedging baseline - hold a fixed delta throughout.
    
    Useful to see if dynamic hedging adds value.
    """
    
    def __init__(self, fixed_delta: float = 0.5):
        self.fixed_delta = fixed_delta
    
    def evaluate(
        self,
        S: Union[np.ndarray, torch.Tensor],
        Z: Union[np.ndarray, torch.Tensor],
        transaction_cost: float = 0.0
    ) -> Dict[str, float]:
        """Evaluate static hedging baseline."""
        is_torch = isinstance(S, torch.Tensor)
        
        if is_torch:
            n_paths, n_steps = S.shape
            deltas = torch.full((n_paths, n_steps), self.fixed_delta, device=S.device)
            
            dS = S[:, 1:] - S[:, :-1]
            hedging_gains = (deltas[:, :-1] * dS).sum(dim=1)
            
            # Only initial and final trades for static hedge
            if transaction_cost > 0:
                total_tc = transaction_cost * self.fixed_delta * (S[:, 0] + S[:, -1])
            else:
                total_tc = 0.0
            
            pnl = (hedging_gains - Z - total_tc).cpu().numpy()
        else:
            n_paths, n_steps = S.shape
            deltas = np.full((n_paths, n_steps), self.fixed_delta)
            
            dS = S[:, 1:] - S[:, :-1]
            hedging_gains = (deltas[:, :-1] * dS).sum(axis=1)
            
            if transaction_cost > 0:
                total_tc = transaction_cost * self.fixed_delta * (S[:, 0] + S[:, -1])
            else:
                total_tc = 0.0
            
            pnl = hedging_gains - Z - total_tc
        
        mean_pnl = float(np.mean(pnl))
        std_pnl = float(np.std(pnl))
        
        sorted_pnl = np.sort(pnl)
        var_idx = int(0.05 * len(sorted_pnl))
        cvar_05 = float(np.mean(sorted_pnl[:max(var_idx, 1)]))
        
        return {
            'mean_pnl': mean_pnl,
            'std_pnl': std_pnl,
            'min_pnl': float(np.min(pnl)),
            'max_pnl': float(np.max(pnl)),
            'cvar_05': cvar_05,
            'sharpe_ratio': mean_pnl / std_pnl if std_pnl > 0 else 0.0,
            'fixed_delta': self.fixed_delta
        }


def evaluate_all_baselines(
    S: Union[np.ndarray, torch.Tensor],
    v: Optional[Union[np.ndarray, torch.Tensor]],
    Z: Union[np.ndarray, torch.Tensor],
    config: Dict
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all baseline strategies.
    
    Args:
        S: Stock prices
        v: Variances
        Z: Option payoffs
        config: Configuration dictionary
        
    Returns:
        Dictionary of results for each baseline
    """
  
    heston_config = config['data']['heston']
    K = heston_config.get('K', 100.0)
    T = config['data']['T']
    r = heston_config.get('r', 0.05)
    sigma = np.sqrt(heston_config.get('v0', 0.04))  # Initial vol as proxy
    n_steps = config['data']['n_steps']
    dt = T / n_steps
    
    option_type = config.get('option_type', 'call')
    transaction_cost = config.get('transaction_cost', 0.0)
    
    results = {}
    
    # Delta Hedging (constant vol)
    delta_baseline = DeltaHedgingBaseline(K, T, r, option_type, use_implied_vol=False)
    results['delta_hedging_const_vol'] = delta_baseline.evaluate(
        S, v, Z, sigma, dt, transaction_cost
    )
    
    # Delta Hedging (instantaneous vol)
    delta_baseline_iv = DeltaHedgingBaseline(K, T, r, option_type, use_implied_vol=True)
    results['delta_hedging_inst_vol'] = delta_baseline_iv.evaluate(
        S, v, Z, sigma, dt, transaction_cost
    )
    
    # No Hedging
    no_hedge = NoHedgingBaseline(option_type)
    results['no_hedging'] = no_hedge.evaluate(S, Z)
    
    # Static Hedging (delta = 0.5)
    static_05 = StaticHedgingBaseline(0.5)
    results['static_delta_0.5'] = static_05.evaluate(S, Z, transaction_cost)
    
    # Static Hedging (delta = 1.0 for call)
    if option_type == 'call':
        static_10 = StaticHedgingBaseline(1.0)
        results['static_delta_1.0'] = static_10.evaluate(S, Z, transaction_cost)
    
    return results


def print_baseline_comparison(results: Dict[str, Dict[str, float]]):
    """Pretty print baseline comparison."""
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)
    
    header = f"{'Strategy':<30} {'Mean P&L':>12} {'Std P&L':>12} {'CVaR 5%':>12} {'Sharpe':>10}"
    print(header)
    print("-" * 80)
    
    for name, metrics in results.items():
        row = (
            f"{name:<30} "
            f"{metrics['mean_pnl']:>12.4f} "
            f"{metrics['std_pnl']:>12.4f} "
            f"{metrics['cvar_05']:>12.4f} "
            f"{metrics['sharpe_ratio']:>10.4f}"
        )
        print(row)
    
    print("=" * 80 + "\n")