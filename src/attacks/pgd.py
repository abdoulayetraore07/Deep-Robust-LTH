"""
Projected Gradient Descent (PGD) attack
"""

import torch
import torch.nn as nn
from typing import Tuple
from ..models.losses import compute_pnl, cvar_loss


def pgd_attack(
    model: nn.Module,
    S: torch.Tensor,
    v: torch.Tensor,
    Z: torch.Tensor,
    features_fn,
    config: dict,
    epsilon_S: float = 0.05,
    epsilon_v: float = 0.5,
    alpha_S: float = 0.01,
    alpha_v: float = 0.1,
    num_steps: int = 10,
    random_start: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PGD attack on price and volatility
    
    Iteratively perturbs inputs to maximize CVaR loss
    Uses MULTIPLICATIVE perturbations for financial data
    
    Args:
        model: Hedging network
        S: Stock prices (batch, n_steps)
        v: Variances (batch, n_steps)
        Z: Payoffs (batch,)
        features_fn: Function to compute features
        config: Configuration dict
        epsilon_S: Max relative perturbation on price (fraction)
        epsilon_v: Max relative perturbation on variance (fraction)
        alpha_S: Step size for price perturbation
        alpha_v: Step size for volatility perturbation
        num_steps: Number of PGD iterations
        random_start: Whether to start from random perturbation
        
    Returns:
        S_adv: Adversarial prices (batch, n_steps)
        v_adv: Adversarial variances (batch, n_steps)
    """
    # Initialize perturbations (relative perturbations)
    if random_start:
        delta_S = torch.zeros_like(S).uniform_(-epsilon_S, epsilon_S)
        delta_v = torch.zeros_like(v).uniform_(-epsilon_v, epsilon_v)
    else:
        delta_S = torch.zeros_like(S)
        delta_v = torch.zeros_like(v)
    
    # PGD iterations
    for step in range(num_steps):
        # Zero gradients explicitly
        if delta_S.grad is not None:
            delta_S.grad.zero_()
        if delta_v.grad is not None:
            delta_v.grad.zero_()
        
        # Enable gradients
        delta_S.requires_grad = True
        delta_v.requires_grad = True
        
        # Apply perturbations (multiplicative)
        S_adv = S * (1 + delta_S)
        v_adv = v * (1 + delta_v)
        
        # Forward pass
        features = features_fn(S_adv, v_adv)
        delta_hedge = model(features)
        
        # Compute P&L
        pnl = compute_pnl(S_adv, delta_hedge, Z, c_prop=config['data']['transaction_cost']['c_prop'])
        
        # Loss: maximize CVaR loss = minimize P&L
        loss = -cvar_loss(pnl, alpha=config['training']['cvar_alpha'])
        
        # Backward pass
        loss.backward()
        
        # Gradient ascent step
        with torch.no_grad():
            delta_S = delta_S + alpha_S * delta_S.grad.sign()
            delta_v = delta_v + alpha_v * delta_v.grad.sign()
            
            # Project into epsilon-ball
            delta_S = torch.clamp(delta_S, -epsilon_S, epsilon_S)
            delta_v = torch.clamp(delta_v, -epsilon_v, epsilon_v)
            
            # Ensure validity (price and variance > 0)
            S_temp = S * (1 + delta_S)
            v_temp = v * (1 + delta_v)
            S_temp = torch.clamp(S_temp, 1e-6, None)
            v_temp = torch.clamp(v_temp, 1e-6, None)
            
            # Recompute deltas to respect positivity constraint
            delta_S = S_temp / S - 1
            delta_v = v_temp / v - 1
        
        # Detach for next iteration
        delta_S = delta_S.detach()
        delta_v = delta_v.detach()
    
    # Final adversarial examples
    S_adv = S * (1 + delta_S)
    v_adv = v * (1 + delta_v)
    
    return S_adv, v_adv