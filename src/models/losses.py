"""
Loss functions for Deep Hedging
"""

import torch
import torch.nn as nn

def compute_pnl(
    S: torch.Tensor,
    delta: torch.Tensor,
    Z: torch.Tensor,
    y: torch.Tensor,
    c_prop: float = 0.001
) -> torch.Tensor:
    """
    Compute P&L for a hedging strategy
    
    P&L = y - Z + (sum of trading P&L) - (transaction costs)
    
    Args:
        S: Stock prices (batch, n_steps)
        delta: Hedging positions (batch, n_steps)
        Z: Payoffs at maturity (batch,)
        y: Learned premium (scalar)
        c_prop: Proportional transaction cost
        
    Returns:
        pnl: Final P&L (batch,)
    """
    batch_size, n_steps = S.shape
    
    # Trading P&L: sum of delta_t * (S_{t+1} - S_t)
    dS = S[:, 1:] - S[:, :-1]  # (batch, n_steps-1)
    trading_pnl = (delta[:, :-1] * dS).sum(dim=1)  # (batch,)
    
    # Transaction costs: sum of c_prop * |delta_{t+1} - delta_t|
    delta_changes = torch.abs(delta[:, 1:] - delta[:, :-1])  # (batch, n_steps-1)
    costs = c_prop * delta_changes.sum(dim=1)  # (batch,)
    
    # Final P&L: y - payoff + trading P&L - costs
    pnl = y - Z + trading_pnl - costs
    
    return pnl


def cvar_loss(
    pnl: torch.Tensor,
    alpha: float = 0.05
) -> torch.Tensor:
    """
    Conditional Value at Risk (CVaR) loss
    
    CVaR is the expected value of the worst alpha% cases
    We minimize -CVaR to maximize the expected worst-case P&L
    
    Args:
        pnl: P&L values (batch,)
        alpha: Quantile level (0.05 = 5% worst cases)
        
    Returns:
        cvar: Scalar CVaR loss
    """
    # Sort P&L (ascending = worst to best)
    sorted_pnl, _ = torch.sort(pnl)
    
    # Take worst alpha% cases
    n = len(pnl)
    k = max(1, int(alpha * n))
    worst_pnl = sorted_pnl[:k]
    
    # CVaR = mean of worst cases
    # We return -CVaR to maximize P&L (minimize loss)
    cvar = -worst_pnl.mean()
    
    return cvar


def entropic_risk_measure(
    pnl: torch.Tensor,
    lambda_risk: float = 1.0
) -> torch.Tensor:
    """
    Entropic risk measure (alternative to CVaR)
    
    U(X) = -1/lambda * log E[exp(-lambda * X)]
    
    Args:
        pnl: P&L values (batch,)
        lambda_risk: Risk aversion parameter
        
    Returns:
        risk: Scalar risk measure
    """
    risk = -torch.logsumexp(-lambda_risk * pnl, dim=0) / lambda_risk + torch.log(torch.tensor(len(pnl), dtype=pnl.dtype))
    return -risk  # Negative to minimize


def mean_variance_loss(
    pnl: torch.Tensor,
    lambda_var: float = 0.5
) -> torch.Tensor:
    """
    Mean-variance utility
    
    U(X) = E[X] - lambda/2 * Var[X]
    
    Args:
        pnl: P&L values (batch,)
        lambda_var: Risk aversion parameter
        
    Returns:
        loss: Scalar loss
    """
    mean_pnl = pnl.mean()
    var_pnl = pnl.var()
    
    utility = mean_pnl - (lambda_var / 2) * var_pnl
    
    return -utility  # Negative to minimize