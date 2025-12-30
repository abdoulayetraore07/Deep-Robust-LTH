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
    
    Args:
        model: Hedging network
        S: Stock prices (batch, n_steps)
        v: Variances (batch, n_steps)
        Z: Payoffs (batch,)
        features_fn: Function to compute features
        config: Configuration dict
        epsilon_S: Max perturbation on price (fraction)
        epsilon_v: Max perturbation on volatility (fraction)
        
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
    
    # FGSM step: perturb in direction of gradient sign
    S_grad_sign = S_adv.grad.sign()
    v_grad_sign = v_adv.grad.sign()
    
    S_adv = S_adv + epsilon_S * S * S_grad_sign
    v_adv = v_adv + epsilon_v * v * v_grad_sign
    
    # Clip to ensure validity
    S_adv = torch.clamp(S_adv, S * (1 - epsilon_S), S * (1 + epsilon_S))
    S_adv = torch.clamp(S_adv, 1e-6, None)  # Price > 0
    
    v_adv = torch.clamp(v_adv, v * (1 - epsilon_v), v * (1 + epsilon_v))
    v_adv = torch.clamp(v_adv, 1e-6, None)  # Variance > 0
    
    return S_adv.detach(), v_adv.detach()