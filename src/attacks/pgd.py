"""
PGD (Projected Gradient Descent) attack
"""

import torch
from typing import Callable
from ..models.losses import compute_pnl, cvar_loss


def pgd_attack(
    model: torch.nn.Module,
    S: torch.Tensor,
    v: torch.Tensor,
    Z: torch.Tensor,
    feature_fn: Callable,
    config: dict,
    epsilon_S: float,
    epsilon_v: float,
    alpha_S: float,
    alpha_v: float,
    num_steps: int = 10
) -> tuple:
    """
    PGD attack on stock price and variance
    
    Args:
        model: Deep Hedging network
        S: Stock prices (batch, n_steps)
        v: Variances (batch, n_steps)
        Z: Payoffs (batch,)
        feature_fn: Function to compute features from S, v
        config: Configuration dictionary
        epsilon_S: Max perturbation for S (fraction)
        epsilon_v: Max perturbation for v (fraction)
        alpha_S: Step size for S
        alpha_v: Step size for v
        num_steps: Number of PGD iterations
        
    Returns:
        S_adv: Adversarial stock prices
        v_adv: Adversarial variances
    """
    # Initialize adversarial examples
    S_adv = S.clone().detach()
    v_adv = v.clone().detach()
    
    for _ in range(num_steps):
        # Require gradients
        S_adv.requires_grad = True
        v_adv.requires_grad = True
        
        # Forward pass
        features = feature_fn(S_adv, v_adv)
        delta, y = model(features)
        
        # Compute loss
        pnl = compute_pnl(S_adv, delta, Z, y, c_prop=config['data']['transaction_cost']['c_prop'])
        loss = cvar_loss(pnl, alpha=config['training']['cvar_alpha'])
        
        # Zero gradients
        model.zero_grad()
        if S_adv.grad is not None:
            S_adv.grad.zero_()
        if v_adv.grad is not None:
            v_adv.grad.zero_()
        
        # Backward pass
        loss.backward()
        
        # Get gradients
        grad_S = S_adv.grad.data
        grad_v = v_adv.grad.data
        
        # PGD update
        S_adv = S_adv.detach() + alpha_S * S * grad_S.sign()
        v_adv = v_adv.detach() + alpha_v * v * grad_v.sign()
        
        # Project back to epsilon-ball
        S_adv = torch.clamp(S_adv, min=S * (1 - epsilon_S), max=S * (1 + epsilon_S))
        v_adv = torch.clamp(v_adv, min=v * (1 - epsilon_v), max=v * (1 + epsilon_v))
        
        # Ensure positivity of variance
        v_adv = torch.clamp(v_adv, min=1e-6)
    
    return S_adv.detach(), v_adv.detach()