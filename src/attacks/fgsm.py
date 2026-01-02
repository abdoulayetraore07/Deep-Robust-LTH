"""
Fast Gradient Sign Method (FGSM) Attack

Implements FGSM from Goodfellow et al. (2014) adapted for Deep Hedging.

Key insight: We perturb the INPUT FEATURES, not the stock prices directly.
This simulates adversarial market conditions where the agent receives
slightly misleading signals about the market state.

The attack finds the worst-case feature perturbation within an ε-ball
that maximizes the loss (minimizes hedging performance).
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional


class FGSM:
    """
    Fast Gradient Sign Method for Deep Hedging.
    
    Perturbs input features to maximize the loss:
        features_adv = features + ε * sign(∇_features L)
    
    This finds adversarial market conditions that hurt hedging performance.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        epsilon: float = 0.1,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None
    ):
        """
        Initialize FGSM attack.
        
        Args:
            model: DeepHedgingNetwork
            loss_fn: Loss function (OCELoss)
            epsilon: Maximum perturbation magnitude (L∞ norm)
            clip_min: Minimum value for clipped features
            clip_max: Maximum value for clipped features
        """
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    def attack(
        self,
        features: torch.Tensor,
        S: torch.Tensor,
        Z: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Generate adversarial features using FGSM.
        
        Args:
            features: Exogenous market features (batch, n_steps, n_features)
            S: Stock prices (batch, n_steps) - NOT perturbed
            Z: Option payoff (batch,)
            dt: Time step
            
        Returns:
            features_adv: Adversarial features (batch, n_steps, n_features)
            info: Attack statistics
        """
        # Enable gradient computation for features
        features_adv = features.clone().detach().requires_grad_(True)
        
        # Forward pass
        deltas, y = self.model(features_adv, S)
        
        # Compute loss (we want to MAXIMIZE this)
        loss, _ = self.loss_fn(deltas, S, Z, y, dt)
        
        # Backward pass to get gradients
        loss.backward()
        
        # Get gradient sign
        grad_sign = features_adv.grad.sign()
        
        # Apply perturbation (maximize loss → add gradient)
        features_adv = features + self.epsilon * grad_sign
        
        # Clip if bounds specified
        if self.clip_min is not None or self.clip_max is not None:
            features_adv = torch.clamp(
                features_adv,
                min=self.clip_min if self.clip_min is not None else float('-inf'),
                max=self.clip_max if self.clip_max is not None else float('inf')
            )
        
        # Compute attack statistics
        with torch.no_grad():
            perturbation = features_adv - features
            info = {
                'perturbation_linf': perturbation.abs().max().item(),
                'perturbation_l2': perturbation.norm(p=2).item() / features.numel() ** 0.5,
                'perturbation_mean': perturbation.abs().mean().item(),
                'original_loss': loss.item()
            }
            
            # Compute loss on adversarial features
            deltas_adv, y_adv = self.model(features_adv.detach(), S)
            loss_adv, _ = self.loss_fn(deltas_adv, S, Z, y_adv, dt)
            info['adversarial_loss'] = loss_adv.item()
            info['loss_increase'] = loss_adv.item() - loss.item()
        
        return features_adv.detach(), info
    
    def attack_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        compute_features_fn,
        K: float,
        T: float,
        dt: float,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Attack an entire batch and return aggregate statistics.
        
        Args:
            dataloader: DataLoader yielding (S, v, Z)
            compute_features_fn: Function to compute features from (S, v)
            K: Strike price
            T: Time to maturity
            dt: Time step
            device: torch device
            
        Returns:
            Aggregate attack statistics
        """
        total_loss_clean = 0.0
        total_loss_adv = 0.0
        n_batches = 0
        
        self.model.eval()
        
        for S, v, Z in dataloader:
            S = S.to(device)
            v = v.to(device)
            Z = Z.to(device)
            
            # Compute features
            features = compute_features_fn(S, v, K, T, dt)
            
            # Clean forward
            with torch.no_grad():
                deltas_clean, y_clean = self.model(features, S)
                loss_clean, _ = self.loss_fn(deltas_clean, S, Z, y_clean, dt)
                total_loss_clean += loss_clean.item()
            
            # Adversarial attack
            features_adv, info = self.attack(features, S, Z, dt)
            total_loss_adv += info['adversarial_loss']
            n_batches += 1
        
        return {
            'clean_loss': total_loss_clean / n_batches,
            'adversarial_loss': total_loss_adv / n_batches,
            'loss_increase': (total_loss_adv - total_loss_clean) / n_batches,
            'robustness_gap': (total_loss_adv - total_loss_clean) / max(abs(total_loss_clean / n_batches), 1e-8)
        }


class TargetedFGSM(FGSM):
    """
    Targeted FGSM - perturb towards a specific target loss.
    
    Instead of maximizing loss, minimize distance to target.
    Useful for controlled adversarial training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        epsilon: float = 0.1,
        target_loss: float = 0.0,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None
    ):
        super().__init__(model, loss_fn, epsilon, clip_min, clip_max)
        self.target_loss = target_loss
    
    def attack(
        self,
        features: torch.Tensor,
        S: torch.Tensor,
        Z: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Generate adversarial features targeting a specific loss."""
        features_adv = features.clone().detach().requires_grad_(True)
        
        deltas, y = self.model(features_adv, S)
        loss, _ = self.loss_fn(deltas, S, Z, y, dt)
        
        # Minimize |loss - target|
        target_tensor = torch.tensor(self.target_loss, device=loss.device)
        loss_diff = (loss - target_tensor).abs()
        loss_diff.backward()
        
        # Move towards target
        grad_sign = features_adv.grad.sign()
        if loss.item() < self.target_loss:
            # Loss too low, increase it
            features_adv = features + self.epsilon * grad_sign
        else:
            # Loss too high, decrease it
            features_adv = features - self.epsilon * grad_sign
        
        if self.clip_min is not None or self.clip_max is not None:
            features_adv = torch.clamp(
                features_adv,
                min=self.clip_min if self.clip_min is not None else float('-inf'),
                max=self.clip_max if self.clip_max is not None else float('inf')
            )
        
        with torch.no_grad():
            info = {
                'original_loss': loss.item(),
                'target_loss': self.target_loss
            }
            deltas_adv, y_adv = self.model(features_adv.detach(), S)
            loss_adv, _ = self.loss_fn(deltas_adv, S, Z, y_adv, dt)
            info['adversarial_loss'] = loss_adv.item()
        
        return features_adv.detach(), info


def create_fgsm_attack(
    model: nn.Module,
    loss_fn: nn.Module,
    config: Dict
) -> FGSM:
    """
    Factory function to create FGSM attack.
    
    Args:
        model: DeepHedgingNetwork
        loss_fn: Loss function
        config: Configuration dictionary
        
    Returns:
        FGSM attack instance
    """
    attack_config = config.get('adversarial', {}).get('fgsm', {})
    
    epsilon = attack_config.get('epsilon', 0.1)
    clip_min = attack_config.get('clip_min', None)
    clip_max = attack_config.get('clip_max', None)
    
    attack = FGSM(
        model=model,
        loss_fn=loss_fn,
        epsilon=epsilon,
        clip_min=clip_min,
        clip_max=clip_max
    )
    
    print(f"[Attack] Created FGSM (ε={epsilon})")
    
    return attack