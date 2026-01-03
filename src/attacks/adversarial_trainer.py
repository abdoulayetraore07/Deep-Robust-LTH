"""
Adversarial Training for Deep Hedging

Implements adversarial training following Madry et al. (2018):
    min_θ E_{(x,y)} [max_{δ∈Δ} L(f_θ(x + δ), y)]

Supports multiple training modes:
1. Standard training (no adversarial)
2. FGSM adversarial training (fast, weak)
3. PGD adversarial training (slow, strong)
4. Curriculum adversarial training (increasing ε)
5. Mixed training (clean + adversarial batches)

Key insight from "Boosting Tickets" paper:
- Use FGSM to find tickets (fast, captures structure)
- Use PGD to train final model (strong robustness)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List, Any, TYPE_CHECKING
import os
import time
import json
from pathlib import Path

from src.data.preprocessor import compute_features, compute_features_differentiable
from src.attacks.fgsm import FGSM
from src.attacks.pgd import PGD

# Conditional import for type hints only (avoids circular imports)
if TYPE_CHECKING:
    from src.pruning.pruning import PruningManager


class AdversarialTrainer:
    """
    Adversarial Trainer for Deep Hedging models.
    
    Supports multiple adversarial training strategies:
    - 'none': Standard training
    - 'fgsm': FGSM adversarial training
    - 'pgd': PGD adversarial training
    - 'curriculum': Gradually increasing ε
    - 'mixed': Mix of clean and adversarial examples
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: Dict,
        device: torch.device,
        pruning_manager: Optional['PruningManager'] = None
    ):
        """
        Initialize adversarial trainer.
        
        Args:
            model: DeepHedgingNetwork
            loss_fn: Loss function (OCELoss)
            config: Configuration dictionary
            device: torch device
            pruning_manager: Optional PruningManager for integrity verification
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.config = config
        self.device = device
        self.pruning_manager = pruning_manager
        
        # Training config
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 100)
        self.lr = train_config.get('learning_rate', 1e-3)
        self.weight_decay = train_config.get('weight_decay', 0.0)
        self.grad_clip = train_config.get('gradient_clip', 1.0)
        self.patience = train_config.get('patience', 20)
        
        # Loss type for appropriate logging
        self.loss_type = train_config.get('loss_type', 'cvar')
        
        # Alpha for CVaR monitoring (independent of loss function)
        # This ensures we can always compute CVaR metrics for comparison,
        # regardless of which loss function is used for training
        self.monitoring_alpha = train_config.get('cvar_alpha', 0.05)
        
        # Adversarial config
        adv_config = config.get('adversarial', {})
        self.adv_mode = adv_config.get('mode', 'none')  # none, fgsm, pgd, curriculum, mixed
        self.mix_ratio = adv_config.get('mix_ratio', 0.5)  # Ratio of adversarial examples
        
        
        # Heston config for feature computation
        heston_config = config['data']['heston']
        self.T = config['data']['T']
        self.n_steps = config['data']['n_steps']
        self.dt = self.T / self.n_steps
        self.K = heston_config.get('K', 100.0)
        
        # Checkpointing
        checkpoint_config = config.get('checkpointing', {})
        self.checkpoint_enabled = checkpoint_config.get('enabled', True)
        self.checkpoint_dir = checkpoint_config.get('directory', 'checkpoints')
        self.save_freq = checkpoint_config.get('save_freq', 10)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Initialize attacks based on mode
        self.fgsm_attack = None
        self.pgd_attack = None
        self._init_attacks(adv_config)
        
        # Curriculum schedule (for curriculum mode)
        self.curriculum_schedule = adv_config.get('curriculum', {
            'start_epsilon': 0.01,
            'end_epsilon': 0.1,
            'warmup_epochs': 10
        })
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = []
        
        # Create checkpoint directory
        if self.checkpoint_enabled:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def _init_attacks(self, adv_config: Dict):
        """Initialize attack objects based on configuration."""
        # FGSM
        fgsm_config = adv_config.get('fgsm', {})
        self.fgsm_attack = FGSM(
            model=self.model,
            loss_fn=self.loss_fn,
            epsilon=fgsm_config.get('epsilon', 0.1),
            clip_min=fgsm_config.get('clip_min', None),
            clip_max=fgsm_config.get('clip_max', None)
        )
        
        # PGD
        pgd_config = adv_config.get('pgd', {})
        self.pgd_attack = PGD(
            model=self.model,
            loss_fn=self.loss_fn,
            epsilon=pgd_config.get('epsilon', 0.1),
            alpha=pgd_config.get('alpha', 0.01),
            num_steps=pgd_config.get('num_steps', 10),
            random_start=pgd_config.get('random_start', True),
            clip_min=pgd_config.get('clip_min', None),
            clip_max=pgd_config.get('clip_max', None),
            norm=pgd_config.get('norm', 'linf')
        )
    
    def _get_current_epsilon(self) -> float:
        """Get current epsilon for curriculum training."""
        if self.adv_mode != 'curriculum':
            return self.pgd_attack.epsilon
        
        schedule = self.curriculum_schedule
        start_eps = schedule['start_epsilon']
        end_eps = schedule['end_epsilon']
        warmup = schedule['warmup_epochs']
        
        if self.current_epoch < warmup:
            # Linear warmup
            progress = self.current_epoch / warmup
            return start_eps + progress * (end_eps - start_eps)
        else:
            return end_eps
    
    def _generate_adversarial(
        self,
        S: torch.Tensor,
        v: torch.Tensor,
        Z: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate adversarial features based on current mode.
        
        IMPORTANT: This method computes features internally using 
        compute_features_differentiable() to ensure gradients can flow
        back through the feature computation for FGSM/PGD attacks.
        
        Args:
            S: Stock prices (requires_grad will be set internally)
            v: Variance (requires_grad will be set internally)
            Z: Option payoff
            
        Returns:
            Adversarial features (n_paths, n_steps, n_features)
        """
        self.model.eval()  # Eval mode for attack generation
        
        # Create copies with gradient tracking for adversarial perturbation
        S_adv = S.clone().detach().requires_grad_(True)
        v_adv = v.clone().detach().requires_grad_(True)
        
        # Compute features with gradient flow
        features = compute_features_differentiable(S_adv, v_adv, self.K, self.T, self.dt)
        
        if self.adv_mode == 'fgsm':
            features_adv, _ = self.fgsm_attack.attack_with_features(
                features, S_adv, Z, self.dt
            )
        
        elif self.adv_mode in ['pgd', 'mixed']:
            features_adv, _ = self.pgd_attack.attack_with_features(
                features, S_adv, Z, self.dt
            )
        
        elif self.adv_mode == 'curriculum':
            # Update epsilon based on curriculum
            current_eps = self._get_current_epsilon()
            self.pgd_attack.epsilon = current_eps
            self.pgd_attack.alpha = current_eps / 4  # Adjust step size
            features_adv, _ = self.pgd_attack.attack_with_features(
                features, S_adv, Z, self.dt
            )
        
        else:
            features_adv = features.detach()
        
        self.model.train()  # Back to train mode
        return features_adv.detach()  # Detach for training step
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch with adversarial training.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_loss_clean = 0.0
        total_loss_adv = 0.0
        total_pnl = 0.0
        total_pnl_std = 0.0
        total_premium = 0.0
        n_batches = 0
        
        for batch_idx, (S, v, Z) in enumerate(train_loader):
            S = S.to(self.device)
            v = v.to(self.device)
            Z = Z.to(self.device)
            
            # Compute features (standard, no gradient needed for clean pass)
            features = compute_features(S, v, self.K, self.T, self.dt)
            
            # Forward pass (clean)
            deltas, y = self.model(features, S)
            loss_clean, info = self.loss_fn(deltas, S, Z, y, self.dt)
            
            total_loss_clean += loss_clean.item()
            total_pnl += info['pnl_mean'].item()
            total_pnl_std += info['pnl_std'].item()
            total_premium += info['premium_y'].item()
            
            # Determine training loss based on mode
            if self.adv_mode == 'none':
                loss = loss_clean
            
            elif self.adv_mode == 'mixed':
                # Mixed training: half clean, half adversarial
                if batch_idx % 2 == 0:
                    loss = loss_clean
                else:
                    features_adv = self._generate_adversarial(S, v, Z)
                    deltas_adv, y_adv = self.model(features_adv, S)
                    loss_adv, _ = self.loss_fn(deltas_adv, S, Z, y_adv, self.dt)
                    loss = loss_adv
                    total_loss_adv += loss_adv.item()
            
            else:
                # Full adversarial training (fgsm, pgd, curriculum)
                features_adv = self._generate_adversarial(S, v, Z)
                deltas_adv, y_adv = self.model(features_adv, S)
                loss_adv, _ = self.loss_fn(deltas_adv, S, Z, y_adv, self.dt)
                loss = loss_adv
                total_loss_adv += loss_adv.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Verify pruning integrity if pruning manager is active
            if self.pruning_manager is not None and batch_idx == 0:
                if not self.pruning_manager.verify_integrity():
                    print(f"[WARNING] Pruning integrity violation at epoch {self.current_epoch}, batch {batch_idx}")
            
            total_loss += loss.item()
            n_batches += 1
        
        metrics = {
            'train_loss': total_loss / n_batches,
            'train_loss_clean': total_loss_clean / n_batches,
            'train_pnl_mean': total_pnl / n_batches,
            'train_pnl_std': total_pnl_std / n_batches,
            'train_premium': total_premium / n_batches
        }
        
        if total_loss_adv > 0:
            metrics['train_loss_adv'] = total_loss_adv / max(n_batches // 2, 1)
        
        return metrics
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        include_adversarial: bool = True
    ) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            include_adversarial: Whether to compute adversarial metrics
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        total_loss_clean = 0.0
        total_loss_adv = 0.0
        total_pnl = 0.0
        total_pnl_std = 0.0
        total_premium = 0.0
        n_batches = 0
        
        all_pnls_clean = []
        all_pnls_adv = []
        
        for S, v, Z in val_loader:
            S = S.to(self.device)
            v = v.to(self.device)
            Z = Z.to(self.device)
            
            features = compute_features(S, v, self.K, self.T, self.dt)
            
            # Clean evaluation
            deltas, y = self.model(features, S)
            loss_clean, info = self.loss_fn(deltas, S, Z, y, self.dt)
            
            total_loss_clean += loss_clean.item()
            total_pnl += info['pnl_mean'].item()
            total_pnl_std += info['pnl_std'].item()
            total_premium += info['premium_y'].item()
            
            pnl_clean = self.loss_fn.compute_pnl(deltas, S, Z, self.dt)
            all_pnls_clean.append(pnl_clean.cpu())
            
            # Adversarial evaluation (need to temporarily enable gradients)
            if include_adversarial and self.adv_mode != 'none':
                with torch.enable_grad():
                    features_adv = self._generate_adversarial(S, v, Z)
                deltas_adv, y_adv = self.model(features_adv, S)
                loss_adv, _ = self.loss_fn(deltas_adv, S, Z, y_adv, self.dt)
                
                total_loss_adv += loss_adv.item()
                
                pnl_adv = self.loss_fn.compute_pnl(deltas_adv, S, Z, self.dt)
                all_pnls_adv.append(pnl_adv.cpu())
            
            n_batches += 1
        
        # Compute CVaR using monitoring_alpha (from config, not from loss)
        alpha = self.monitoring_alpha
        all_pnls_clean = torch.cat(all_pnls_clean)
        sorted_pnls, _ = torch.sort(all_pnls_clean)
        var_idx = int(alpha * len(sorted_pnls))
        cvar_clean = sorted_pnls[:max(var_idx, 1)].mean().item()
        
        metrics = {
            'val_loss': total_loss_clean / n_batches,
            'val_pnl_mean': total_pnl / n_batches,
            'val_pnl_std': total_pnl_std / n_batches,
            'val_premium': total_premium / n_batches,
            'val_cvar': cvar_clean
        }
        
        if include_adversarial and self.adv_mode != 'none' and len(all_pnls_adv) > 0:
            metrics['val_loss_adv'] = total_loss_adv / n_batches
            metrics['val_robustness_gap'] = metrics['val_loss_adv'] - metrics['val_loss']
            
            all_pnls_adv = torch.cat(all_pnls_adv)
            sorted_pnls_adv, _ = torch.sort(all_pnls_adv)
            cvar_adv = sorted_pnls_adv[:max(var_idx, 1)].mean().item()
            metrics['val_cvar_adv'] = cvar_adv
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 0
    ) -> Dict[str, Any]:
        """
        Full adversarial training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            start_epoch: Starting epoch (for resumption)
            
        Returns:
            Training results
        """
        self.current_epoch = start_epoch
        start_time = time.time()
        
        # Get sparsity info if pruning is active
        sparsity_info = ""
        if self.pruning_manager is not None and self.pruning_manager.is_pruned():
            sparsity = self.pruning_manager.get_sparsity().get('total', 0)
            sparsity_info = f", Sparsity: {sparsity:.1%}"
        
        print(f"\n{'='*70}")
        print(f"Starting ADVERSARIAL training from epoch {start_epoch + 1}")
        print(f"Mode: {self.adv_mode.upper()}")
        print(f"Loss type: {self.loss_type.upper()}{sparsity_info}")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
        
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics, 'epoch': epoch + 1}
            metrics['lr'] = self.optimizer.param_groups[0]['lr']
            metrics['epoch_time'] = time.time() - epoch_start
            
            # Add sparsity to metrics if pruning is active
            if self.pruning_manager is not None and self.pruning_manager.is_pruned():
                metrics['sparsity'] = self.pruning_manager.get_sparsity().get('total', 0)
            
            self.training_history.append(metrics)
            
            # Check for improvement (use clean val loss for model selection)
            if val_metrics['val_loss'] < self.best_val_loss - 1e-6:
                self.best_val_loss = val_metrics['val_loss']
                self.epochs_without_improvement = 0
                
                if self.checkpoint_enabled:
                    self.save_checkpoint('best')
                improved = " *"
            else:
                self.epochs_without_improvement += 1
                improved = ""
            
            # Logging - show PnL Mean and Std for both loss types
            sparsity_str = ""
            if self.pruning_manager is not None and self.pruning_manager.is_pruned():
                sparsity_str = f" | Sp: {metrics.get('sparsity', 0):.1%}"
            
            log_str = (
                f"Epoch {epoch + 1:3d}/{self.epochs} | "
                f"Loss: {train_metrics['train_loss']:.4f} | "
                f"Val: {val_metrics['val_loss']:.4f} | "
                f"PnL: {val_metrics['val_pnl_mean']:.4f} | "
                f"Std: {val_metrics['val_pnl_std']:.4f}"
            )
            
            if 'val_loss_adv' in val_metrics:
                log_str += f" | Val_Adv: {val_metrics['val_loss_adv']:.4f}"
                log_str += f" | Gap: {val_metrics['val_robustness_gap']:.4f}"
            
            if self.adv_mode == 'curriculum':
                log_str += f" | ε: {self._get_current_epsilon():.4f}"
            
            log_str += f"{sparsity_str} | Time: {metrics['epoch_time']:.1f}s{improved}"
            print(log_str)
            
            # Periodic checkpoint
            if self.checkpoint_enabled and (epoch + 1) % self.save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}')
            
            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final
        if self.checkpoint_enabled:
            self.save_checkpoint('latest')
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"Adversarial training completed in {total_time:.1f}s")
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
        print(f"{'='*70}\n")
        
        return {
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.current_epoch + 1,
            'total_time': total_time,
            'history': self.training_history
        }
    
    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'training_history': self.training_history,
            'config': self.config,
            'adv_mode': self.adv_mode
        }
        
        path = os.path.join(self.checkpoint_dir, f'{name}.pt')
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, name: str) -> int:
        """Load checkpoint and return resume epoch."""
        path = os.path.join(self.checkpoint_dir, f'{name}.pt')
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.epochs_without_improvement = checkpoint['epochs_without_improvement']
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"[AdvTrainer] Loaded checkpoint '{name}' (epoch {checkpoint['epoch'] + 1})")
        return checkpoint['epoch'] + 1


def create_adversarial_trainer(
    model: nn.Module,
    loss_fn: nn.Module,
    config: Dict,
    device: torch.device,
    experiment_dir: Optional[str] = None,
    pruning_manager: Optional['PruningManager'] = None
) -> AdversarialTrainer:
    """
    Factory function to create AdversarialTrainer.
    
    Args:
        model: DeepHedgingNetwork
        loss_fn: Loss function
        config: Configuration
        device: torch device
        experiment_dir: Override checkpoint directory
        pruning_manager: Optional PruningManager for integrity verification
        
    Returns:
        AdversarialTrainer instance
    """
    if experiment_dir is not None:
        if 'checkpointing' not in config:
            config['checkpointing'] = {}
        config['checkpointing']['directory'] = os.path.join(experiment_dir, 'checkpoints')
    
    trainer = AdversarialTrainer(model, loss_fn, config, device, pruning_manager=pruning_manager)
    
    adv_mode = config.get('adversarial', {}).get('mode', 'none')
    print(f"[AdvTrainer] Created with mode: {adv_mode.upper()}")
    
    return trainer
