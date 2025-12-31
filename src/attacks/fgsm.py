"""
FGSM (Fast Gradient Sign Method) attack
"""

import torch
from typing import Callable
from ..models.losses import compute_pnl, cvar_loss


def fgsm_attack(
    model: torch.nn.Module,
    S: torch.Tensor,
    v: torch.Tensor,
    Z: torch.Tensor,
    feature_fn: Callable,
    config: dict,
    epsilon_S: float,
    epsilon_v: float
) -> tuple:
    """
    FGSM attack on stock price and variance
    
    Args:
        model: Deep Hedging network
        S: Stock prices (batch, n_steps)
        v: Variances (batch, n_steps)
        Z: Payoffs (batch,)
        feature_fn: Function to compute features from S, v
        config: Configuration dictionary
        epsilon_S: Max perturbation for S (fraction)
        epsilon_v: Max perturbation for v (fraction)
        
    Returns:
        S_adv: Adversarial stock prices
        v_adv: Adversarial variances
    """
    # Clone and require gradients
    S_adv = S.clone().detach().requires_grad_(True)
    v_adv = v.clone().detach().requires_grad_(True)
    
    # Forward pass
    features = feature_fn(S_adv, v_adv)
    delta, y = model(features)
    
    # Compute loss
    pnl = compute_pnl(S_adv, delta, Z, y, c_prop=config['data']['transaction_cost']['c_prop'])
    loss = cvar_loss(pnl, alpha=config['training']['cvar_alpha'])
    
    # Backward pass
    loss.backward()
    
    # Get gradients
    grad_S = S_adv.grad.data
    grad_v = v_adv.grad.data
    
    # FGSM update
    S_adv = S_adv.detach() + epsilon_S * S * grad_S.sign()
    v_adv = v_adv.detach() + epsilon_v * v * grad_v.sign()
    
    # Project back to epsilon-ball
    S_adv = torch.clamp(S_adv, min=S * (1 - epsilon_S), max=S * (1 + epsilon_S))
    v_adv = torch.clamp(v_adv, min=v * (1 - epsilon_v), max=v * (1 + epsilon_v))
    
    # Ensure positivity of variance
    v_adv = torch.clamp(v_adv, min=1e-6)
    
    return S_adv.detach(), v_adv.detach()