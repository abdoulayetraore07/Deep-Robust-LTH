"""
Training Loop for Deep Hedging

Implements the training loop with:
- OCE loss optimization
- Checkpointing (save/load)
- Early stopping
- Learning rate scheduling
- Comprehensive logging
- Optional pruning integrity verification
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List, Any, TYPE_CHECKING
from pathlib import Path
import numpy as np

from src.data.preprocessor import compute_features, N_EXOGENOUS_FEATURES
from src.models.deep_hedging import DeepHedgingNetwork
from src.models.losses import OCELoss

# Conditional import for type hints only (avoids circular imports)
if TYPE_CHECKING:
    from src.pruning.pruning import PruningManager


class Trainer:
    """
    Trainer for Deep Hedging models.
    
    Handles:
    - Training loop with temporal forward pass
    - Validation
    - Checkpointing (save/load)
    - Early stopping
    - Learning rate scheduling
    - Logging
    - Optional pruning integrity verification
    """
    
    def __init__(
        self,
        model: DeepHedgingNetwork,
        loss_fn: nn.Module,
        config: Dict,
        device: torch.device,
        pruning_manager: Optional['PruningManager'] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: DeepHedgingNetwork instance
            loss_fn: Loss function (OCELoss or similar)
            config: Configuration dictionary
            device: torch device
            pruning_manager: Optional PruningManager for integrity verification
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.pruning_manager = pruning_manager
        
        # Extract training config
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 100)
        self.lr = train_config.get('learning_rate', 1e-3)
        self.weight_decay = train_config.get('weight_decay', 0.0)
        self.grad_clip = train_config.get('clip_grad_norm', train_config.get('gradient_clip', 1.0))
        self.patience = train_config.get('patience', 20)
        self.min_delta = train_config.get('min_delta', 1e-6)
        
        # Loss type for appropriate logging
        self.loss_type = train_config.get('loss_type', 'cvar')
        
        # Alpha for CVaR monitoring (independent of loss function)
        # This ensures we can always compute CVaR metrics for comparison,
        # regardless of which loss function is used for training
        self.monitoring_alpha = train_config.get('cvar_alpha', 0.05)
        
        # Extract Heston config for dt calculation
        heston_config = config['data']['heston']
        self.T = config['data']['T']
        self.n_steps = config['data']['n_steps']
        self.dt = self.T / self.n_steps
        self.K = heston_config.get('K', 100.0)
        
        # Checkpointing config
        checkpoint_config = config.get('checkpointing', {})
        self.checkpoint_enabled = checkpoint_config.get('enabled', True)
        self.checkpoint_dir = checkpoint_config.get('directory', 'checkpoints')
        self.save_freq = checkpoint_config.get('save_freq', 10)
        self.keep_last = checkpoint_config.get('keep_last', 3)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler (optional)
        scheduler_config = train_config.get('scheduler', {})
        lr_scheduler_type = train_config.get('lr_scheduler', None)

        if scheduler_config.get('enabled', False):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        elif lr_scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=train_config.get('min_lr', 1e-6)
            )
        elif lr_scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config.get('lr_step_size', 50),
                gamma=train_config.get('lr_gamma', 0.5)
            )
        else:
            self.scheduler = None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = []
        
        # Create checkpoint directory
        if self.checkpoint_enabled:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader yielding (S, v, Z) batches
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_pnl = 0.0
        total_premium = 0.0
        n_batches = 0
        
        for batch_idx, (S, v, Z) in enumerate(train_loader):
            # Move to device
            S = S.to(self.device)
            v = v.to(self.device)
            Z = Z.to(self.device)
            
            # Compute exogenous features on GPU
            features = compute_features(S, v, self.K, self.T, self.dt)
            
            # Forward pass with temporal loop
            deltas, y = self.model(features, S)
            
            # Compute loss
            loss, info = self.loss_fn(deltas, S, Z, y, self.dt)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Verify pruning integrity if pruning manager is active
            # PyTorch's pruning hooks automatically maintain sparsity,
            # but we verify periodically for safety
            if self.pruning_manager is not None and batch_idx == 0:
                if not self.pruning_manager.verify_integrity():
                    print(f"[WARNING] Pruning integrity violation at epoch {self.current_epoch}, batch {batch_idx}")
            
            # Accumulate metrics
            total_loss += loss.item()
            total_pnl += info['pnl_mean'].item()
            total_premium += info['premium_y'].item()
            n_batches += 1
        
        return {
            'train_loss': total_loss / n_batches,
            'train_pnl_mean': total_pnl / n_batches,
            'train_premium': total_premium / n_batches
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        This method is loss-agnostic: it works with any loss function
        (OCELoss, EntropicRiskLoss, or future losses) by:
        1. Using monitoring_alpha from config for CVaR computation
        2. Using .get() for optional info fields
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_pnl = 0.0
        total_pnl_std = 0.0
        total_premium = 0.0
        n_batches = 0
        
        all_pnls = []
        
        for S, v, Z in val_loader:
            S = S.to(self.device)
            v = v.to(self.device)
            Z = Z.to(self.device)
            
            features = compute_features(S, v, self.K, self.T, self.dt)
            deltas, y = self.model(features, S)
            loss, info = self.loss_fn(deltas, S, Z, y, self.dt)
            
            total_loss += loss.item()
            total_pnl += info['pnl_mean'].item()
            total_pnl_std += info['pnl_std'].item()
            total_premium += info['premium_y'].item()
            n_batches += 1
            
            # Collect P&Ls for CVaR calculation
            pnl = self.loss_fn.compute_pnl(deltas, S, Z, self.dt)
            all_pnls.append(pnl.cpu())
        
        # Compute CVaR from all P&Ls using monitoring_alpha (from config)
        # This works regardless of which loss function is used
        all_pnls = torch.cat(all_pnls)
        alpha = self.monitoring_alpha
        sorted_pnls, _ = torch.sort(all_pnls)
        var_idx = int(alpha * len(sorted_pnls))
        cvar = sorted_pnls[:max(var_idx, 1)].mean().item()
        
        return {
            'val_loss': total_loss / n_batches,
            'val_pnl_mean': total_pnl / n_batches,
            'val_pnl_std': total_pnl_std / n_batches,
            'val_premium': total_premium / n_batches,
            'val_cvar': cvar
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 0
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            start_epoch: Starting epoch (for resumption)
            
        Returns:
            Dictionary with training results
        """
        self.current_epoch = start_epoch
        start_time = time.time()
        
        # Get sparsity info if pruning is active
        sparsity_info = ""
        if self.pruning_manager is not None and self.pruning_manager.is_pruned():
            sparsity = self.pruning_manager.get_sparsity().get('total', 0)
            sparsity_info = f", Sparsity: {sparsity:.1%}"
        
        print(f"\n{'='*60}")
        print(f"Starting training from epoch {start_epoch + 1}")
        print(f"Device: {self.device}")
        print(f"Loss type: {self.loss_type.upper()}{sparsity_info}")
        print(f"Epochs: {self.epochs}, LR: {self.lr}")
        print(f"Early stopping patience: {self.patience}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics, 'epoch': epoch + 1}
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            metrics['epoch_time'] = time.time() - epoch_start
            
            # Add sparsity to metrics if pruning is active
            if self.pruning_manager is not None and self.pruning_manager.is_pruned():
                metrics['sparsity'] = self.pruning_manager.get_sparsity().get('total', 0)
            
            self.training_history.append(metrics)
            
            # Check for improvement
            if val_metrics['val_loss'] < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_metrics['val_loss']
                self.epochs_without_improvement = 0
                
                # Save best model
                if self.checkpoint_enabled:
                    self.save_checkpoint('best')
                
                improved = " *"
            else:
                self.epochs_without_improvement += 1
                improved = ""
            
            # Logging - adapted based on loss type
            # For both loss types, we show PnL Mean and Std which are the key metrics
            # Premium is shown for OCE (learned y), for Entropic it's the implied premium (-pnl_mean)
            sparsity_str = ""
            if self.pruning_manager is not None and self.pruning_manager.is_pruned():
                sparsity_str = f" | Sp: {metrics.get('sparsity', 0):.1%}"
            
            print(f"Epoch {epoch + 1:3d}/{self.epochs} | "
                  f"Train Loss: {train_metrics['train_loss']:.6f} | "
                  f"Val Loss: {val_metrics['val_loss']:.6f} | "
                  f"PnL Mean: {val_metrics['val_pnl_mean']:.4f} | "
                  f"PnL Std: {val_metrics['val_pnl_std']:.4f} | "
                  f"CVaR: {val_metrics['val_cvar']:.4f}{sparsity_str} | "
                  f"Time: {metrics['epoch_time']:.1f}s{improved}")
            
            # Periodic checkpoint
            if self.checkpoint_enabled and (epoch + 1) % self.save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}')
                self._cleanup_old_checkpoints()
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final checkpoint
        if self.checkpoint_enabled:
            self.save_checkpoint('latest')
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"Training completed in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Print final price estimate based on loss type
        if self.training_history:
            final_pnl_mean = self.training_history[-1]['val_pnl_mean']
            final_premium = self.training_history[-1]['val_premium']
            if self.loss_type == 'entropic':
                print(f"Implied price (from -PnL Mean): {-final_pnl_mean:.4f}")
            else:
                print(f"Learned premium y: {final_premium:.4f}")
            
            # Print final sparsity if pruning is active
            if self.pruning_manager is not None and self.pruning_manager.is_pruned():
                final_sparsity = self.training_history[-1].get('sparsity', 0)
                print(f"Final sparsity: {final_sparsity:.2%}")
        print(f"{'='*60}\n")
        
        return {
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch + 1,
            'total_time': total_time,
            'history': self.training_history
        }
    
    def save_checkpoint(self, name: str):
        """
        Save a checkpoint.
        
        Args:
            name: Checkpoint name (e.g., 'best', 'latest', 'epoch_50')
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'training_history': self.training_history,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = os.path.join(self.checkpoint_dir, f'{name}.pt')
        torch.save(checkpoint, path)
        
        # Also save training history as JSON for easy inspection
        if name == 'best' or name == 'latest':
            history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
    
    def load_checkpoint(self, name: str) -> int:
        """
        Load a checkpoint.
        
        Args:
            name: Checkpoint name
            
        Returns:
            Epoch number to resume from
        """
        path = os.path.join(self.checkpoint_dir, f'{name}.pt')
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.epochs_without_improvement = checkpoint['epochs_without_improvement']
        self.training_history = checkpoint.get('training_history', [])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        resume_epoch = checkpoint['epoch'] + 1
        
        print(f"[Trainer] Loaded checkpoint '{name}' (epoch {checkpoint['epoch'] + 1})")
        print(f"[Trainer] Best val loss: {self.best_val_loss:.6f}")
        
        return resume_epoch
    
    def _cleanup_old_checkpoints(self):
        """Remove old epoch checkpoints, keeping only the last N."""
        import glob
        
        pattern = os.path.join(self.checkpoint_dir, 'epoch_*.pt')
        checkpoints = sorted(glob.glob(pattern))
        
        # Keep best, latest, and last N epoch checkpoints
        if len(checkpoints) > self.keep_last:
            for ckpt in checkpoints[:-self.keep_last]:
                os.remove(ckpt)


def check_existing_checkpoint(checkpoint_dir: str, checkpoint_type: str = 'best') -> Optional[str]:
    """
    Check if a checkpoint exists.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        checkpoint_type: Type of checkpoint ('best', 'latest', etc.)
        
    Returns:
        Path to checkpoint if exists, None otherwise
    """
    path = os.path.join(checkpoint_dir, f'{checkpoint_type}.pt')
    return path if os.path.exists(path) else None


def load_trained_model(
    model: DeepHedgingNetwork,
    checkpoint_path: str,
    device: torch.device
) -> Tuple[DeepHedgingNetwork, Dict]:
    """
    Load a trained model from checkpoint.
    
    Args:
        model: Model instance (architecture must match)
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded model
        info: Checkpoint info (epoch, val_loss, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    info = {
        'epoch': checkpoint['epoch'] + 1,
        'best_val_loss': checkpoint['best_val_loss'],
        'training_history': checkpoint.get('training_history', [])
    }
    
    print(f"[Model] Loaded trained model from {checkpoint_path}")
    print(f"[Model] Trained for {info['epoch']} epochs, val_loss: {info['best_val_loss']:.6f}")
    
    return model, info


def create_trainer(
    model: DeepHedgingNetwork,
    loss_fn: nn.Module,
    config: Dict,
    device: torch.device,
    experiment_dir: Optional[str] = None,
    pruning_manager: Optional['PruningManager'] = None
) -> Trainer:
    """
    Factory function to create a Trainer.
    
    Args:
        model: DeepHedgingNetwork instance
        loss_fn: Loss function
        config: Configuration dictionary
        device: torch device
        experiment_dir: Override checkpoint directory
        pruning_manager: Optional PruningManager for integrity verification
        
    Returns:
        Trainer instance
    """
    # Override checkpoint directory if experiment_dir provided
    if experiment_dir is not None:
        if 'checkpointing' not in config:
            config['checkpointing'] = {}
        config['checkpointing']['directory'] = os.path.join(experiment_dir, 'checkpoints')
    
    trainer = Trainer(model, loss_fn, config, device, pruning_manager=pruning_manager)
    
    return trainer
