"""
Adversarial training for Deep Hedging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
from ..models.losses import cvar_loss, compute_pnl
from ..attacks.fgsm import fgsm_attack
from ..attacks.pgd import pgd_attack
from pathlib import Path


class AdversarialTrainer:
    """
    Trainer for adversarial Deep Hedging
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        attack_type: str = 'fgsm',
        device: str = 'cuda',
        mask: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Initialize adversarial trainer
        
        Args:
            model: Deep Hedging network
            config: Configuration dictionary
            attack_type: Type of attack ('fgsm' or 'pgd')
            device: Device to use
            mask: Optional pruning mask
        """
        self.model = model.to(device)
        self.config = config
        self.attack_type = attack_type
        self.device = device
        self.mask = mask
        
        # Optimizer
        lr = config['training']['learning_rate']
        weight_decay = config['training'].get('weight_decay', 0)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Loss function
        self.criterion_alpha = config['training'].get('cvar_alpha', 0.05)
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Early stopping
        self.patience = config['training'].get('patience', 20)
        self.patience_counter = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        K: float,
        T: float,
        dt: float
    ) -> float:
        """
        Train for one epoch with adversarial examples
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
                    self.config['attacks']['fgsm']['epsilon_S'],
                    self.config['attacks']['fgsm']['epsilon_v']
                )
            elif self.attack_type == 'pgd':
                S_adv, v_adv = pgd_attack(
                    self.model, S, v, Z,
                    lambda s, vv: self._compute_features_batch(s, vv, K, T, dt),
                    self.config,
                    self.config['attacks']['pgd']['epsilon_S'],
                    self.config['attacks']['pgd']['epsilon_v'],
                    self.config['attacks']['pgd']['alpha_S'],
                    self.config['attacks']['pgd']['alpha_v'],
                    self.config['attacks']['pgd']['num_steps']
                )
            
            # Forward pass on adversarial examples
            features = self._compute_features_batch(S_adv, v_adv, K, T, dt)
            delta, y = self.model(features)
            
            # Compute loss
            pnl = compute_pnl(S_adv, delta, Z, y, c_prop=self.config['data']['transaction_cost']['c_prop'])
            loss = cvar_loss(pnl, alpha=self.criterion_alpha)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Re-apply mask if using sparse network
            if self.mask is not None:
                self._apply_mask()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(
        self,
        val_loader: DataLoader,
        K: float,
        T: float,
        dt: float
    ) -> float:
        """
        Validate on clean data
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for S, v, Z in val_loader:
                S = S.to(self.device)
                v = v.to(self.device)
                Z = Z.to(self.device)
                
                # Compute features
                features = self._compute_features_batch(S, v, K, T, dt)
                
                # Forward pass
                delta, y = self.model(features)
                
                # Compute loss
                pnl = compute_pnl(S, delta, Z, y, c_prop=self.config['data']['transaction_cost']['c_prop'])
                loss = cvar_loss(pnl, alpha=self.criterion_alpha)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        K: float,
        T: float,
        dt: float
    ) -> float:
        """
        Train with adversarial examples
        """
        num_epochs = self.config['training']['epochs']
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, K, T, dt)
            self.train_losses.append(train_loss)
            
            val_loss = self.validate(val_loader, K, T, dt)
            self.val_losses.append(val_loss)
            
            # Print progress
            y_value = self.model.y.item()
            print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, y={y_value:.6f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(f'experiments/adversarial_{self.attack_type}/best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self.best_val_loss
    
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
        warmup_epochs: int
    ) -> float:
        """
        Train with learning rate warmup
        """
        for epoch in range(epochs):
            # Learning rate warmup
            if epoch < warmup_epochs:
                lr = lr_start + (lr_end - lr_start) * (epoch / warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
            
            train_loss = self.train_epoch(train_loader, K, T, dt)
            self.train_losses.append(train_loss)
            
            val_loss = self.validate(val_loader, K, T, dt)
            self.val_losses.append(val_loss)
            
            # Print progress
            y_value = self.model.y.item()
            print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, y={y_value:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f'experiments/adversarial_{self.attack_type}/best_model.pt')
        
        return self.best_val_loss
    
    def _compute_features_batch(
        self,
        S: torch.Tensor,
        v: torch.Tensor,
        K: float,
        T: float,
        dt: float
    ) -> torch.Tensor:
        """
        Compute features for a batch
        """
        from ..data.preprocessor import compute_features
        features = compute_features(S, v, K, T, dt)
        return features
    
    def _apply_mask(self):
        """
        Apply pruning mask
        """
        for name, param in self.model.named_parameters():
            if name in self.mask:
                param.data *= self.mask[name].to(self.device)
    
    def save_checkpoint(self, filepath: str):
        """
        Save model checkpoint
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), filepath)