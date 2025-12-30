"""
Adversarial training for Deep Hedging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict

from ..models.trainer import Trainer
from .fgsm import fgsm_attack
from .pgd import pgd_attack


class AdversarialTrainer(Trainer):
    """
    Trainer for adversarial training
    
    Extends the base Trainer to generate adversarial examples during training
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        attack_type: str = 'pgd',
        device: str = 'cuda',
        mask: Optional[Dict[str, torch.Tensor]] = None  
    ):
        """
        Initialize adversarial trainer
        
        Args:
            model: Neural network
            config: Configuration dictionary
            attack_type: Type of attack ('fgsm' or 'pgd')
            device: Device to use
            mask: Optional pruning mask  
        """
        super().__init__(model, config, device, mask)  
        
        self.attack_type = attack_type
        
        # Attack parameters
        if attack_type == 'fgsm':
            self.epsilon_S = config['attacks']['fgsm']['epsilon_S']
            self.epsilon_v = config['attacks']['fgsm']['epsilon_v']
        elif attack_type == 'pgd':
            self.epsilon_S = config['attacks']['pgd']['epsilon_S']
            self.epsilon_v = config['attacks']['pgd']['epsilon_v']
            self.alpha_S = config['attacks']['pgd']['alpha_S']
            self.alpha_v = config['attacks']['pgd']['alpha_v']
            self.num_steps = config['attacks']['pgd']['num_steps']
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        K: float,
        T: float,
        dt: float
    ) -> float:
        """
        Train for one epoch with adversarial examples
        
        Args:
            train_loader: Training data loader
            K: Strike price
            T: Time to maturity
            dt: Time step
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for S, v, Z in train_loader:
            S = S.to(self.device)
            v = v.to(self.device)
            Z = Z.to(self.device)
            
            # Generate adversarial examples
            if self.attack_type == 'fgsm':
                S_adv, v_adv = fgsm_attack(
                    self.model, S, v, Z,
                    lambda s, vv: self._compute_features_batch(s, vv, K, T, dt),
                    self.config,
                    self.epsilon_S,
                    self.epsilon_v
                )
            else:  # pgd
                S_adv, v_adv = pgd_attack(
                    self.model, S, v, Z,
                    lambda s, vv: self._compute_features_batch(s, vv, K, T, dt),
                    self.config,
                    self.epsilon_S,
                    self.epsilon_v,
                    self.alpha_S,
                    self.alpha_v,
                    self.num_steps
                )
            
            # Compute features on adversarial examples
            features_adv = self._compute_features_batch(S_adv, v_adv, K, T, dt)
            
            # Forward pass
            delta = self.model(features_adv)
            
            # Compute P&L on adversarial examples
            from ..models.losses import compute_pnl, cvar_loss
            pnl = compute_pnl(S_adv, delta, Z, c_prop=self.config['data']['transaction_cost']['c_prop'])
            
            # Compute loss
            loss = cvar_loss(pnl, alpha=self.criterion_alpha)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Re-apply mask
            if self.mask is not None:
                self._apply_mask()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    
    def fit_with_warmup(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        K: float,
        T: float,
        dt: float,
        epochs: int,
        lr_start: float,
        lr_end: float,
        warmup_epochs: int = 10
    ) -> float:
        """
        Train with learning rate warmup (for PGD retraining phase)
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            K: Strike price
            T: Time to maturity
            dt: Time step
            epochs: Total number of epochs
            lr_start: Starting learning rate
            lr_end: Ending learning rate
            warmup_epochs: Number of warmup epochs
            
        Returns:
            Best validation loss
        """
        # Override epochs
        original_epochs = self.config['training']['epochs']
        self.config['training']['epochs'] = epochs
        
        # Warmup schedule: lr_start -> lr_end (linear)
        for epoch in range(epochs):
            # Adjust learning rate
            if epoch < warmup_epochs:
                # Linear warmup
                lr = lr_start + (lr_end - lr_start) * (epoch / warmup_epochs)
            else:
                # Constant after warmup
                lr = lr_end
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Train
            train_loss = self.train_epoch(train_loader, K, T, dt)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader, K, T, dt)
            self.val_losses.append(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} (lr={lr:.6f}): train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            # Log metrics
            if self.logger:
                self.logger.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': lr
                }, step=epoch)
            
            # Save best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('experiments/adversarial_training/best_model.pt')
        
        # Close logger
        if self.logger:
            self.logger.close()
            
        # Restore original epochs
        self.config['training']['epochs'] = original_epochs
        
        return self.best_val_loss