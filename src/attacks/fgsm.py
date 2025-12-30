"""
Fast Gradient Sign Method (FGSM) attack
"""

import torch
import torch.nn as nn
from typing import Tuple
from ..models.losses import compute_pnl, cvar_loss


def fgsm_attack(
    model: nn.Module,
    S: torch.Tensor,
    v: torch.Tensor,
    Z: torch.Tensor,
    features_fn,
    config: dict,
    epsilon_S: float = 0.02,
    epsilon_v: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FGSM attack on price and volatility
    
    Perturbs inputs to maximize CVaR loss (worst-case P&L)
    Uses MULTIPLICATIVE perturbations for financial data
    
    Args:
        model: Hedging network
        S: Stock prices (batch, n_steps)
        v: Variances (batch, n_steps)
        Z: Payoffs (batch,)
        features_fn: Function to compute features
        config: Configuration dict
        epsilon_S: Max relative perturbation on price (fraction, e.g. 0.02 = 2%)
        epsilon_v: Max relative perturbation on variance (fraction, e.g. 0.2 = 20%)
        
    Returns:
        S_adv: Adversarial prices (batch, n_steps)
        v_adv: Adversarial variances (batch, n_steps)
    """
    # Clone and enable gradients
    S_adv = S.clone().detach().requires_grad_(True)
    v_adv = v.clone().detach().requires_grad_(True)
    
    # Forward pass
    features = features_fn(S_adv, v_adv)
    delta = model(features)
    
    # Compute P&L
    pnl = compute_pnl(S_adv, delta, Z, c_prop=config['data']['transaction_cost']['c_prop'])
    
    # Loss: maximize CVaR loss = minimize P&L
    loss = -cvar_loss(pnl, alpha=config['training']['cvar_alpha'])
    
    # Backward pass
    loss.backward()
    
    # FGSM step: MULTIPLICATIVE perturbation (finance-adapted)
    # Formula: x_adv = x * (1 + epsilon * sign(grad))
    S_grad_sign = S_adv.grad.sign()
    v_grad_sign = v_adv.grad.sign()
    
    S_adv = S * (1 + epsilon_S * S_grad_sign)
    v_adv = v * (1 + epsilon_v * v_grad_sign)
    
    # Clip to ensure validity
    # 1. Stay within epsilon-ball
    S_adv = torch.clamp(S_adv, S * (1 - epsilon_S), S * (1 + epsilon_S))
    v_adv = torch.clamp(v_adv, v * (1 - epsilon_v), v * (1 + epsilon_v))
    
    # 2. Ensure positivity (financial constraints)
    S_adv = torch.clamp(S_adv, 1e-6, None)
    v_adv = torch.clamp(v_adv, 1e-6, None)
    
    return S_adv.detach(), v_adv.detach()