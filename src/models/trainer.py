"""
Training infrastructure for Deep Hedging
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
from ..models.losses import cvar_loss
from ..utils.logging import setup_logger
from pathlib import Path 


def get_optimizer(model, training_config):
    """Create optimizer from config"""
    optimizer_name = training_config['optimizer_name']
    lr = training_config['learning_rate']
    weight_decay = training_config.get('weight_decay', 0)
    
    if optimizer_name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = training_config.get('momentum', 0.9)
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_lr_schedule(optimizer, training_config):
    """Create learning rate scheduler from config"""
    scheduler_type = training_config.get('lr_scheduler', None)
    
    if scheduler_type is None:
        return None
    elif scheduler_type == 'cosine':
        epochs = training_config['epochs']
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'step':
        step_size = training_config.get('lr_step_size', 30)
        gamma = training_config.get('lr_gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    


class Trainer:
    """
    Trainer for Deep Hedging networks
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: str = 'cuda',
        mask: Optional[Dict[str, torch.Tensor]] = None  
    ):
        """
        Initialize trainer
        
        Args:
            model: Neural network
            config: Configuration dictionary
            device: Device to use
            mask: Optional pruning mask (for training sparse networks)  
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.mask = mask 
        
        # Optimizer and scheduler
        self.optimizer = get_optimizer(model, config['training'])
        self.lr_scheduler = get_lr_schedule(self.optimizer, config['training'])
        
        # Loss function
        self.criterion_alpha = config['training'].get('cvar_alpha', 0.05)
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Early stopping
        self.patience = config['training'].get('patience', 20)
        self.patience_counter = 0

        # Logger pour TensorBoard
        if 'logging' in config and 'experiment_name' in config:
            self.logger = setup_logger(config)
        else:
            self.logger = None
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        K: float,
        T: float,
        dt: float
    ) -> float:
        """
        Train for one epoch
        
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
            
            # Compute features
            features = self._compute_features_batch(S, v, K, T, dt)
            
            # Forward pass
            delta = self.model(features)
            
            # Compute P&L
            from ..models.losses import compute_pnl
            pnl = compute_pnl(S, delta, Z, c_prop=self.config['data']['transaction_cost']['c_prop'])
            
            # Compute loss
            loss = cvar_loss(pnl, alpha=self.criterion_alpha)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Re-apply mask after optimizer step
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
        Validate the model
        
        Args:
            val_loader: Validation data loader
            K: Strike price
            T: Time to maturity
            dt: Time step
            
        Returns:
            Average validation loss
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
                delta = self.model(features)
                
                # Compute P&L
                from ..models.losses import compute_pnl
                pnl = compute_pnl(S, delta, Z, c_prop=self.config['data']['transaction_cost']['c_prop'])
                
                # Compute loss
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
        Train the model with early stopping
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            K: Strike price
            T: Time to maturity
            dt: Time step
            
        Returns:
            Best validation loss
        """
        num_epochs = self.config['training']['epochs']
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, K, T, dt)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader, K, T, dt)
            self.val_losses.append(val_loss)
            
            # Learning rate scheduler
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            # Log metrics to TensorBoard
            if self.logger:
                self.logger.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, step=epoch)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint('experiments/baseline/best_model.pt')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Close logger
        if self.logger:
            self.logger.close()

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
        
        Args:
            S: Stock prices (batch, n_steps)
            v: Variances (batch, n_steps)
            K: Strike price
            T: Time to maturity
            dt: Time step
            
        Returns:
            features: (batch, n_steps, 8)
        """
        from ..data.preprocessor import compute_features
        
        # Convert to numpy
        S_np = S.cpu().numpy()
        v_np = v.cpu().numpy()
        
        # Compute features
        features_np = compute_features(S_np, v_np, K, T, dt)
        
        # Convert back to tensor
        features = torch.from_numpy(features_np).float().to(self.device)
        
        return features
    
    def _apply_mask(self):
        """
        Apply pruning mask to model weights
        
        This ensures that pruned weights remain at zero after optimizer updates
        """
        for name, param in self.model.named_parameters():
            if name in self.mask:
                param.data *= self.mask[name].to(self.device)
    
    def save_checkpoint(self, filepath: str):
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
    
    def load_checkpoint(self, filepath: str):
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))