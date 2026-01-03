#!/usr/bin/env python3
"""
Lottery Ticket Hypothesis Training Script

Implements Iterative Magnitude Pruning (IMP) from Frankle & Carlin (2019).

Usage:
    python scripts/train_pruning.py --config configs/config.yaml
    python scripts/train_pruning.py --config configs/config.yaml --target-sparsity 0.9
    python scripts/train_pruning.py --config configs/config.yaml --n-rounds 10

Algorithm:
    1. Train dense network to completion
    2. Prune p% of smallest magnitude weights
    3. Rewind remaining weights to initialization
    4. Repeat until target sparsity
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import random
import json

from src.utils.config import (
    load_config, get_device, get_experiment_dir,
    save_config_to_experiment
)
from src.utils.logging import create_logger
from src.utils.visualization import plot_sparsity_performance, plot_lottery_ticket_summary
from src.data.heston import get_or_generate_dataset
from src.data.preprocessor import create_dataloaders, compute_features
from src.models.deep_hedging import create_model
from src.models.losses import create_loss_function
from src.models.trainer import Trainer, create_trainer
from src.pruning.pruning import PruningManager
from src.evaluation.metrics import compute_all_metrics, print_metrics


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Lottery Ticket Pruning')
    
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--target-sparsity', type=float, default=None,
                        help='Target sparsity level (e.g., 0.9 for 90%)')
    parser.add_argument('--n-rounds', type=int, default=None,
                        help='Number of pruning rounds')
    parser.add_argument('--pruning-rate', type=float, default=None,
                        help='Fraction to prune each round (e.g., 0.2)')
    parser.add_argument('--rewind-epoch', type=int, default=None,
                        help='Epoch to rewind to (0 for init, >0 for late rewinding)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save-tickets', action='store_true',
                        help='Save ticket at each sparsity level')
    
    return parser.parse_args()


def evaluate_model(model, loss_fn, test_loader, config, device):
    """Evaluate model and return metrics."""
    heston_config = config['data']['heston']
    K = heston_config.get('K', 100.0)
    T = config['data']['T']
    n_steps = config['data']['n_steps']
    dt = T / n_steps

    model.eval()
    all_pnls = []
    
    with torch.no_grad():
        for S, v, Z in test_loader:
            S, v, Z = S.to(device), v.to(device), Z.to(device)
            features = compute_features(S, v, K, T, dt)
            deltas, y = model(features, S)
            pnl = loss_fn.compute_pnl(deltas, S, Z, dt)
            all_pnls.append(pnl.cpu())
    
    all_pnls = torch.cat(all_pnls).numpy()
    metrics = compute_all_metrics(all_pnls)
    metrics['learned_premium'] = float(model.y.item())
    
    return metrics


def train_one_round(
    model, loss_fn, train_loader, val_loader,
    config, device, round_idx, logger, pruning_manager=None
):
    """Train model for one pruning round."""
    # Get learning rate based on round
    # Round 0: use standard LR for initial training
    # Round > 0: use pruning LR (lower) for fine-tuning
    training_config = config.get('training', {})
    
    if round_idx == 0:
        lr = training_config.get('learning_rate', 1e-3)
    else:
        # Use pruning LR (lower for stability during retraining)
        lr = config.get('pruning', {}).get('retrain_lr', 
             training_config.get('pruning_lr', 1e-4))
    
    # Update config for this round
    round_config = config.copy()
    round_config['training'] = training_config.copy()
    round_config['training']['learning_rate'] = lr
    
    # Reduce epochs for retraining rounds (optional)
    if round_idx > 0:
        retrain_epochs = config.get('pruning', {}).get('retrain_epochs', 
                         training_config.get('epochs', 100))
        round_config['training']['epochs'] = retrain_epochs
    
    # Create trainer with pruning manager for integrity verification
    trainer = create_trainer(
        model=model,
        loss_fn=loss_fn,
        config=round_config,
        device=device,
        pruning_manager=pruning_manager
    )
    
    logger.info(f"  Training with LR={lr}, epochs={round_config['training']['epochs']}")
    
    results = trainer.train(train_loader, val_loader)
    
    return results


def main():
    """Main LTH pruning function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with CLI args
    if args.target_sparsity is not None:
        config.setdefault('pruning', {})['target_sparsity'] = args.target_sparsity
    if args.n_rounds is not None:
        config.setdefault('pruning', {})['n_rounds'] = args.n_rounds
    if args.pruning_rate is not None:
        config.setdefault('pruning', {})['rate'] = args.pruning_rate
    if args.rewind_epoch is not None:
        config.setdefault('rewind', {})['epoch'] = args.rewind_epoch
        config['rewind']['strategy'] = 'late' if args.rewind_epoch > 0 else 'init'
    if args.device:
        config['device'] = args.device
    if args.seed:
        config['seed'] = args.seed
    
    # Update experiment name
    config['experiment_name'] = config.get('experiment_name', 'deep_hedging') + '_lth'
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Device
    device = get_device(config)
    print(f"Device: {device}")
    
    # Pruning config
    pruning_config = config.get('pruning', {})
    target_sparsity = pruning_config.get('target_sparsity', 0.9)
    n_rounds = pruning_config.get('n_rounds', 10)
    pruning_rate = pruning_config.get('rate', 0.2)
    
    print(f"Target sparsity: {target_sparsity:.0%}")
    print(f"Pruning rounds: {n_rounds}")
    print(f"Pruning rate per round: {pruning_rate:.0%}")
    
    # Experiment directory
    experiment_dir = get_experiment_dir(config)
    save_config_to_experiment(config)
    tickets_dir = experiment_dir / 'tickets'
    tickets_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    logger = create_logger(
        config['experiment_name'],
        config=config,
        log_dir=str(experiment_dir / 'logs')
    )
    
    # =========================================================================
    # DATA
    # =========================================================================
    logger.info("Loading data...")
    cache_dir = config.get('caching', {}).get('directory', 'cache')
    
    S_train, v_train, Z_train = get_or_generate_dataset(config, 'train', cache_dir)
    S_val, v_val, Z_val = get_or_generate_dataset(config, 'val', cache_dir)
    S_test, v_test, Z_test = get_or_generate_dataset(config, 'test', cache_dir)
    
    batch_size = config.get('training', {}).get('batch_size', 256)
    train_loader, val_loader, test_loader = create_dataloaders(
        S_train, v_train, Z_train,
        S_val, v_val, Z_val,
        S_test, v_test, Z_test,
        batch_size=batch_size
    )
    
    # =========================================================================
    # MODEL & PRUNING SETUP
    # =========================================================================
    logger.info("Creating model...")
    model = create_model(config)
    model = model.to(device)
    loss_fn = create_loss_function(config)
    
    # Initialize pruning manager with PyTorch native pruning
    exclude_layers = pruning_config.get('exclude_layers', [])
    if pruning_config.get('exclude_output', True):
        exclude_layers.append('layers.' + str(len(model.layers) - 1))
    
    pruning_manager = PruningManager(model, exclude_layers=exclude_layers)
    
    # Save initial weights for LTH rewinding
    pruning_manager.save_initial_weights()
    
    # Results storage
    results_by_sparsity = {}
    
    # =========================================================================
    # ITERATIVE MAGNITUDE PRUNING
    # =========================================================================
    logger.info("=" * 60)
    logger.info("ITERATIVE MAGNITUDE PRUNING")
    logger.info("=" * 60)
    
    current_sparsity = 0.0
    
    for round_idx in range(n_rounds + 1):  # +1 for initial dense training
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {round_idx}/{n_rounds}")
        logger.info(f"Current sparsity: {current_sparsity:.2%}")
        logger.info(f"Remaining weights: {(1-current_sparsity)*100:.1f}%")
        logger.info("=" * 60)
        
        # Train
        logger.info("Training...")
        train_results = train_one_round(
            model, loss_fn, train_loader, val_loader,
            config, device, round_idx, logger,
            pruning_manager=pruning_manager if round_idx > 0 else None
        )
        
        # Evaluate
        logger.info("Evaluating...")
        metrics = evaluate_model(model, loss_fn, test_loader, config, device)
        metrics['sparsity'] = current_sparsity
        metrics['remaining_weights'] = 1 - current_sparsity
        metrics['round'] = round_idx
        metrics['train_loss'] = train_results['best_val_loss']
        
        # Verify pruning integrity if pruned
        if round_idx > 0:
            if pruning_manager.verify_integrity():
                logger.info("  Pruning integrity: PASS")
            else:
                logger.warning("  Pruning integrity: FAIL")
            
            # Get actual sparsity
            actual_sparsity = pruning_manager.get_sparsity()
            metrics['actual_sparsity'] = actual_sparsity['total']
            logger.info(f"  Actual sparsity: {actual_sparsity['total']:.2%}")
        
        results_by_sparsity[current_sparsity] = metrics
        
        logger.log_pruning_round(round_idx, current_sparsity, {
            'sharpe': metrics['sharpe_ratio'],
            'cvar_05': metrics['cvar_05'],
            'mean_pnl': metrics['pnl_mean']
        })
        
        # Save ticket if requested
        if args.save_tickets:
            ticket_path = tickets_dir / f'ticket_sparsity_{current_sparsity:.2f}.pt'
            # With torch.nn.utils.prune, masks are saved with model.state_dict()
            torch.save({
                'model_state_dict': model.state_dict(),
                'sparsity': current_sparsity,
                'metrics': metrics,
                'round': round_idx
            }, ticket_path)
            logger.info(f"  Saved ticket to {ticket_path}")
        
        # Check if target sparsity reached
        if current_sparsity >= target_sparsity:
            logger.info(f"Target sparsity {target_sparsity:.0%} reached!")
            break
        
        # Prune and rewind
        if round_idx < n_rounds:
            # Calculate cumulative sparsity for next round
            # Using: remaining = (1 - rate)^(round+1), so sparsity = 1 - remaining
            next_sparsity = 1 - (1 - pruning_rate) ** (round_idx + 1)
            next_sparsity = min(next_sparsity, target_sparsity)
            
            logger.info(f"Pruning to {next_sparsity:.2%} sparsity...")
            pruning_manager.prune_by_magnitude(next_sparsity)
            
            # Rewind weights to initial values (LTH protocol)
            logger.info("Rewinding weights to initialization...")
            pruning_manager.rewind_to_initial()
            
            current_sparsity = next_sparsity
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("LOTTERY TICKET ANALYSIS")
    logger.info("=" * 60)
    
    # Find best ticket
    sparsities = sorted(results_by_sparsity.keys())
    
    # Best by Sharpe ratio
    best_sparsity_sharpe = max(sparsities, 
                               key=lambda s: results_by_sparsity[s]['sharpe_ratio'])
    
    # Best by CVaR
    best_sparsity_cvar = max(sparsities,
                             key=lambda s: results_by_sparsity[s]['cvar_05'])
    
    # Dense baseline (sparsity = 0)
    dense_metrics = results_by_sparsity.get(0.0, results_by_sparsity[sparsities[0]])
    
    # Maximum sparsity with >= 95% of dense performance
    threshold = 0.95 * dense_metrics['sharpe_ratio']
    max_efficient_sparsity = 0.0
    for s in sparsities:
        if results_by_sparsity[s]['sharpe_ratio'] >= threshold:
            max_efficient_sparsity = s
    
    logger.info(f"\nDense Model (0% sparse):")
    logger.info(f"  Sharpe: {dense_metrics['sharpe_ratio']:.4f}")
    logger.info(f"  CVaR 5%: {dense_metrics['cvar_05']:.4f}")
    
    logger.info(f"\nBest Ticket by Sharpe ({best_sparsity_sharpe:.0%} sparse):")
    logger.info(f"  Sharpe: {results_by_sparsity[best_sparsity_sharpe]['sharpe_ratio']:.4f}")
    
    logger.info(f"\nBest Ticket by CVaR ({best_sparsity_cvar:.0%} sparse):")
    logger.info(f"  CVaR 5%: {results_by_sparsity[best_sparsity_cvar]['cvar_05']:.4f}")
    
    logger.info(f"\nMax Efficient Sparsity (>= 95% dense Sharpe):")
    logger.info(f"  Sparsity: {max_efficient_sparsity:.0%}")
    logger.info(f"  Remaining weights: {(1-max_efficient_sparsity)*100:.1f}%")
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)
    
    # Save final ticket (best by Sharpe)
    final_ticket_path = experiment_dir / 'best_ticket.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'sparsity': best_sparsity_sharpe,
        'metrics': results_by_sparsity[best_sparsity_sharpe],
        'initial_weights': pruning_manager.initial_weights
    }, final_ticket_path)
    logger.info(f"Saved best ticket to {final_ticket_path}")
    
    # Save all results
    final_results = {
        'results_by_sparsity': {str(k): v for k, v in results_by_sparsity.items()},
        'best_sparsity_sharpe': best_sparsity_sharpe,
        'best_sparsity_cvar': best_sparsity_cvar,
        'max_efficient_sparsity': max_efficient_sparsity,
        'dense_metrics': dense_metrics,
        'config': config
    }
    
    logger.save_results(final_results, 'lth_results.json')
    
    # Generate plots
    logger.info("Generating plots...")
    
    # Performance vs sparsity
    metrics_to_plot = {
        'sharpe_ratio': [results_by_sparsity[s]['sharpe_ratio'] for s in sparsities],
        'cvar_05': [results_by_sparsity[s]['cvar_05'] for s in sparsities],
        'pnl_mean': [results_by_sparsity[s]['pnl_mean'] for s in sparsities]
    }
    
    fig1 = plot_sparsity_performance(
        sparsities, metrics_to_plot,
        baseline_metrics=dense_metrics,
        save_path=str(experiment_dir / 'sparsity_performance.png')
    )
    
    # LTH summary
    fig2 = plot_lottery_ticket_summary(
        results_by_sparsity, dense_metrics,
        save_path=str(experiment_dir / 'lth_summary.png')
    )
    
    logger.finalize()
    
    print(f"\nLottery Ticket search complete!")
    print(f"Results: {experiment_dir}")
    print(f"Best ticket: {best_sparsity_sharpe:.0%} sparse, "
          f"Sharpe={results_by_sparsity[best_sparsity_sharpe]['sharpe_ratio']:.4f}")


if __name__ == '__main__':
    main()
