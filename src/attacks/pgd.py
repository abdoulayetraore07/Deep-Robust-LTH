"""
Projected Gradient Descent (PGD) Attack

Implements PGD from Madry et al. (2018) adapted for Deep Hedging.

PGD is an iterative version of FGSM:
    1. Start from random point within ε-ball
    2. Take multiple small FGSM steps
    3. Project back onto ε-ball after each step

This is considered the strongest first-order attack and is the
gold standard for adversarial robustness evaluation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional
import numpy as np


class PGD:
    """
    Projected Gradient Descent Attack for Deep Hedging.
    
    Iteratively perturbs input features to maximize the loss:
        x_{t+1} = Π_{ε}(x_t + α * sign(∇_x L))
    
    Where Π_{ε} projects back onto the ε-ball around the original input.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        num_steps: int = 10,
        random_start: bool = True,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        norm: str = 'linf'
    ):
        """
        Initialize PGD attack.
        
        Args:
            model: DeepHedgingNetwork
            loss_fn: Loss function (OCELoss)
            epsilon: Maximum perturbation magnitude (ball radius)
            alpha: Step size for each iteration
            num_steps: Number of PGD iterations
            random_start: Whether to start from random point in ε-ball
            clip_min: Minimum value for clipped features
            clip_max: Maximum value for clipped features
            norm: Norm type ('linf' or 'l2')
        """
        self.model = model
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.random_start = random_start
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.norm = norm.lower()
        
        assert self.norm in ['linf', 'l2'], f"Norm must be 'linf' or 'l2', got {norm}"
    
    def _random_init(self, features: torch.Tensor) -> torch.Tensor:
        """Initialize with random perturbation within ε-ball."""
        if self.norm == 'linf':
            # Uniform random in [-ε, ε]
            delta = torch.empty_like(features).uniform_(-self.epsilon, self.epsilon)
        else:  # l2
            # Random direction, random magnitude up to ε
            delta = torch.randn_like(features)
            delta_norm = delta.view(delta.shape[0], -1).norm(p=2, dim=1, keepdim=True)
            delta_norm = delta_norm.view(delta.shape[0], 1, 1)
            delta = delta / (delta_norm + 1e-8)
            magnitude = torch.empty(delta.shape[0], 1, 1, device=delta.device).uniform_(0, self.epsilon)
            delta = delta * magnitude
        
        return delta
    
    def _project(self, delta: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Project perturbation onto ε-ball and apply value clipping."""
        if self.norm == 'linf':
            # Clamp each element to [-ε, ε]
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        else:  # l2
            # Project onto L2 ball
            delta_norm = delta.view(delta.shape[0], -1).norm(p=2, dim=1, keepdim=True)
            delta_norm = delta_norm.view(delta.shape[0], 1, 1)
            factor = torch.min(torch.ones_like(delta_norm), self.epsilon / (delta_norm + 1e-8))
            delta = delta * factor
        
        # Apply value clipping if specified
        if self.clip_min is not None or self.clip_max is not None:
            perturbed = original + delta
            perturbed = torch.clamp(
                perturbed,
                min=self.clip_min if self.clip_min is not None else float('-inf'),
                max=self.clip_max if self.clip_max is not None else float('inf')
            )
            delta = perturbed - original
        
        return delta
    
    def attack(
        self,
        features: torch.Tensor,
        S: torch.Tensor,
        Z: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Generate adversarial features using PGD.
        
        Args:
            features: Exogenous market features (batch, n_steps, n_features)
            S: Stock prices (batch, n_steps) - NOT perturbed
            Z: Option payoff (batch,)
            dt: Time step
            
        Returns:
            features_adv: Adversarial features
            info: Attack statistics
        """
        # Store original for projection
        original_features = features.clone().detach()
        
        # Initialize perturbation
        if self.random_start:
            delta = self._random_init(features)
        else:
            delta = torch.zeros_like(features)
        
        delta = self._project(delta, original_features)
        
        # Track best adversarial example
        best_loss = float('-inf')
        best_delta = delta.clone()
        
        # Compute clean loss for reference
        with torch.no_grad():
            deltas_clean, y_clean = self.model(features, S)
            loss_clean, _ = self.loss_fn(deltas_clean, S, Z, y_clean, dt)
            clean_loss_val = loss_clean.item()
        
        # PGD iterations
        for step in range(self.num_steps):
            # Current adversarial features
            features_adv = original_features + delta
            features_adv.requires_grad_(True)
            
            # Forward pass
            deltas_pred, y = self.model(features_adv, S)
            loss, _ = self.loss_fn(deltas_pred, S, Z, y, dt)
            
            # Track best
            if loss.item() > best_loss:
                best_loss = loss.item()
                best_delta = delta.clone()
            
            # Backward pass
            loss.backward()
            
            # Get gradient
            grad = features_adv.grad.detach()
            
            # Update delta based on norm type
            if self.norm == 'linf':
                delta = delta + self.alpha * grad.sign()
            else:  # l2
                grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1, keepdim=True)
                grad_norm = grad_norm.view(grad.shape[0], 1, 1)
                grad_normalized = grad / (grad_norm + 1e-8)
                delta = delta + self.alpha * grad_normalized
            
            # Project back onto ε-ball
            delta = self._project(delta, original_features)
            delta = delta.detach()
        
        # Use best adversarial example found
        features_adv = original_features + best_delta
        
        # Compute final statistics
        with torch.no_grad():
            deltas_adv, y_adv = self.model(features_adv, S)
            loss_adv, _ = self.loss_fn(deltas_adv, S, Z, y_adv, dt)
            
            perturbation = best_delta
            info = {
                'perturbation_linf': perturbation.abs().max().item(),
                'perturbation_l2': perturbation.norm(p=2).item() / features.numel() ** 0.5,
                'perturbation_mean': perturbation.abs().mean().item(),
                'original_loss': clean_loss_val,
                'adversarial_loss': loss_adv.item(),
                'loss_increase': loss_adv.item() - clean_loss_val,
                'num_steps': self.num_steps,
                'epsilon': self.epsilon,
                'alpha': self.alpha
            }
        
        return features_adv.detach(), info
    
    def attack_with_features(
        self,
        features: torch.Tensor,
        S: torch.Tensor,
        Z: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Generate adversarial features using PGD.
        
        This is an alias for attack() with explicit naming to clarify
        that the input features should already have gradient tracking
        if computed via compute_features_differentiable().
        
        Called by AdversarialTrainer._generate_adversarial().
        
        Args:
            features: Exogenous market features (batch, n_steps, n_features)
                     Can have requires_grad=True for gradient flow
            S: Stock prices (batch, n_steps)
            Z: Option payoff (batch,)
            dt: Time step
            
        Returns:
            features_adv: Adversarial features (batch, n_steps, n_features)
            info: Attack statistics
        """
        return self.attack(features, S, Z, dt)
    
    def attack_with_restarts(
        self,
        features: torch.Tensor,
        S: torch.Tensor,
        Z: torch.Tensor,
        dt: float,
        num_restarts: int = 5
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        PGD with multiple random restarts (stronger attack).
        
        Args:
            features: Input features
            S: Stock prices
            Z: Option payoff
            dt: Time step
            num_restarts: Number of random restarts
            
        Returns:
            Best adversarial features across all restarts
        """
        best_adv = None
        best_loss = float('-inf')
        best_info = None
        
        for restart in range(num_restarts):
            features_adv, info = self.attack(features, S, Z, dt)
            
            if info['adversarial_loss'] > best_loss:
                best_loss = info['adversarial_loss']
                best_adv = features_adv
                best_info = info
        
        best_info['num_restarts'] = num_restarts
        
        return best_adv, best_info
    
    def attack_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
        compute_features_fn,
        K: float,
        T: float,
        dt: float,
        device: torch.device,
        num_restarts: int = 1
    ) -> Dict[str, float]:
        """
        Attack an entire dataset and return aggregate statistics.
        
        Args:
            dataloader: DataLoader yielding (S, v, Z)
            compute_features_fn: Function to compute features
            K: Strike price
            T: Time to maturity
            dt: Time step
            device: torch device
            num_restarts: Number of restarts per batch
            
        Returns:
            Aggregate attack statistics
        """
        total_loss_clean = 0.0
        total_loss_adv = 0.0
        total_perturbation = 0.0
        n_batches = 0
        
        self.model.eval()
        
        for S, v, Z in dataloader:
            S = S.to(device)
            v = v.to(device)
            Z = Z.to(device)
            
            features = compute_features_fn(S, v, K, T, dt)
            
            # Clean loss
            with torch.no_grad():
                deltas_clean, y_clean = self.model(features, S)
                loss_clean, _ = self.loss_fn(deltas_clean, S, Z, y_clean, dt)
                total_loss_clean += loss_clean.item()
            
            # Adversarial attack
            if num_restarts > 1:
                _, info = self.attack_with_restarts(features, S, Z, dt, num_restarts)
            else:
                _, info = self.attack(features, S, Z, dt)
            
            total_loss_adv += info['adversarial_loss']
            total_perturbation += info['perturbation_linf']
            n_batches += 1
        
        avg_clean = total_loss_clean / n_batches
        avg_adv = total_loss_adv / n_batches
        
        return {
            'clean_loss': avg_clean,
            'adversarial_loss': avg_adv,
            'loss_increase': avg_adv - avg_clean,
            'robustness_gap': (avg_adv - avg_clean) / max(abs(avg_clean), 1e-8),
            'avg_perturbation_linf': total_perturbation / n_batches,
            'epsilon': self.epsilon,
            'num_steps': self.num_steps,
            'num_restarts': num_restarts
        }


class PGDTrainer:
    """
    Adversarial training using PGD.
    
    Implements the robust optimization:
        min_θ E_{(x,y)} [max_{δ∈Δ} L(f_θ(x + δ), y)]
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        pgd_attack: PGD,
        mix_ratio: float = 0.5
    ):
        """
        Initialize PGD trainer.
        
        Args:
            model: DeepHedgingNetwork
            loss_fn: Loss function
            pgd_attack: PGD attack instance
            mix_ratio: Ratio of adversarial examples (0.5 = 50% clean, 50% adversarial)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.attack = pgd_attack
        self.mix_ratio = mix_ratio
    
    def train_step(
        self,
        features: torch.Tensor,
        S: torch.Tensor,
        Z: torch.Tensor,
        dt: float,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single adversarial training step.
        
        Args:
            features: Input features
            S: Stock prices
            Z: Option payoff
            dt: Time step
            optimizer: Optimizer
            
        Returns:
            Training metrics
        """
        self.model.train()
        batch_size = features.shape[0]
        
        # Decide which samples get adversarial perturbation
        n_adv = int(batch_size * self.mix_ratio)
        
        if n_adv > 0:
            # Generate adversarial examples for subset
            self.model.eval()  # Eval mode for attack
            features_adv, attack_info = self.attack.attack(
                features[:n_adv], S[:n_adv], Z[:n_adv], dt
            )
            self.model.train()  # Back to train mode
            
            # Combine clean and adversarial
            features_mixed = torch.cat([features_adv, features[n_adv:]], dim=0)
        else:
            features_mixed = features
        
        # Forward pass
        deltas, y = self.model(features_mixed, S)
        loss, loss_info = self.loss_fn(deltas, S, Z, y, dt)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'pnl_mean': loss_info['pnl_mean'].item(),
            'premium_y': y.item(),
            'n_adversarial': n_adv,
            'n_clean': batch_size - n_adv
        }


def create_pgd_attack(
    model: nn.Module,
    loss_fn: nn.Module,
    config: Dict
) -> PGD:
    """
    Factory function to create PGD attack.
    
    Args:
        model: DeepHedgingNetwork
        loss_fn: Loss function
        config: Configuration dictionary
        
    Returns:
        PGD attack instance
    """
    attack_config = config.get('adversarial', {}).get('pgd', {})
    
    epsilon = attack_config.get('epsilon', 0.1)
    alpha = attack_config.get('alpha', 0.01)
    num_steps = attack_config.get('num_steps', 10)
    random_start = attack_config.get('random_start', True)
    clip_min = attack_config.get('clip_min', None)
    clip_max = attack_config.get('clip_max', None)
    norm = attack_config.get('norm', 'linf')
    
    attack = PGD(
        model=model,
        loss_fn=loss_fn,
        epsilon=epsilon,
        alpha=alpha,
        num_steps=num_steps,
        random_start=random_start,
        clip_min=clip_min,
        clip_max=clip_max,
        norm=norm
    )
    
    print(f"[Attack] Created PGD (ε={epsilon}, α={alpha}, steps={num_steps}, norm={norm})")
    
    return attack
