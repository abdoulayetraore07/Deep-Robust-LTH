#!/usr/bin/env python3
"""
Training Script for Deep Hedging

Usage:
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --config configs/config.yaml --force-retrain
    python scripts/train.py --config configs/config.yaml --resume

Options:
    --config: Path to configuration file
    --force-retrain: Force retraining even if model exists
    --resume: Resume training from latest checkpoint
    --device: Override device (cpu/cuda)
    --seed: Override random seed
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import random

from src.utils.config import (
    load_config, 
    get_device, 
    get_experiment_dir,
    check_existing_model,
    should_skip_training,
    save_config_to_experiment
)
from src.utils.logging import create_logger
from src.data.heston import get_or_generate_dataset
from src.data.preprocessor import create_dataloaders
from src.models.deep_hedging import create_model
from src.models.losses import create_loss_function
from src.models.trainer import Trainer, load_trained_model, check_existing_checkpoint
from src.evaluation.baselines import DeltaHedgingBaseline, evaluate_all_baselines, print_baseline_comparison
from src.evaluation.metrics import compute_all_metrics, print_metrics


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Deep Hedging Model')
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retraining even if model exists'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from latest checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Override device (cpu/cuda)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override random seed'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Override experiment name'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.device is not None:
        config['device'] = args.device
    if args.seed is not None:
        config['seed'] = args.seed
    if args.experiment_name is not None:
        config['experiment_name'] = args.experiment_name
    
    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    # Get device
    device = get_device(config)
    print(f"Using device: {device}")
    
    # Get experiment directory
    experiment_dir = get_experiment_dir(config)
    print(f"Experiment directory: {experiment_dir}")
    
    # Check if model already exists
    if not args.force_retrain and not args.resume:
        model_status = check_existing_model(config, 'baseline')
        if model_status['exists'] and model_status['training_complete']:
            print(f"\n[INFO] Trained model already exists at {model_status['checkpoint_path']}")
            print("[INFO] Use --force-retrain to retrain or --resume to continue training")
            print("[INFO] Loading existing model for evaluation...")
            
            # Load and evaluate existing model
            model = create_model(config)
            model, info = load_trained_model(model, model_status['checkpoint_path'], device)
            
            # Quick evaluation
            print(f"\nModel trained for {info['epoch']} epochs")
            print(f"Best validation loss: {info['best_val_loss']:.6f}")
            return
    
    # Create experiment directory and save config
    save_config_to_experiment(config)
    
    # Initialize logger
    logger = create_logger(
        config.get('experiment_name', 'deep_hedging'),
        config=config,
        log_dir=str(experiment_dir / 'logs')
    )
    
    # =========================================================================
    # DATA GENERATION / LOADING
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Data Generation")
    logger.info("=" * 60)
    
    cache_dir = config.get('caching', {}).get('directory', 'cache')
    
    # Generate or load datasets
    logger.info("Loading/generating training data...")
    S_train, v_train, Z_train = get_or_generate_dataset(
        config, 'train', cache_dir, force_regenerate=False
    )
    logger.info(f"  Training: {S_train.shape[0]} paths, {S_train.shape[1]} steps")
    
    logger.info("Loading/generating validation data...")
    S_val, v_val, Z_val = get_or_generate_dataset(
        config, 'val', cache_dir, force_regenerate=False
    )
    logger.info(f"  Validation: {S_val.shape[0]} paths")
    
    logger.info("Loading/generating test data...")
    S_test, v_test, Z_test = get_or_generate_dataset(
        config, 'test', cache_dir, force_regenerate=False
    )
    logger.info(f"  Test: {S_test.shape[0]} paths")
    
    # Create dataloaders
    batch_size = config.get('training', {}).get('batch_size', 256)
    train_loader, val_loader, test_loader = create_dataloaders(
        S_train, v_train, Z_train,
        S_val, v_val, Z_val,
        S_test, v_test, Z_test,
        batch_size=batch_size
    )
    logger.info(f"Created dataloaders with batch_size={batch_size}")
    
    # =========================================================================
    # BASELINE EVALUATION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Baseline Evaluation")
    logger.info("=" * 60)
    
    # Evaluate baselines on test set
    baseline_results = evaluate_all_baselines(
        S_test, v_test, Z_test, config
    )
    print_baseline_comparison(baseline_results)
    
    # Log baseline results
    for name, metrics in baseline_results.items():
        logger.log_metrics(metrics, prefix=f"Baseline: {name}")
    
    # =========================================================================
    # MODEL CREATION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Model Creation")
    logger.info("=" * 60)
    
    model = create_model(config)
    model = model.to(device)
    logger.log_model_summary(model.summary())
    
    # Create loss function
    loss_fn = create_loss_function(config)
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 4: Training")
    logger.info("=" * 60)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        device=device
    )
    
    # Set checkpoint directory to experiment directory
    trainer.checkpoint_dir = str(experiment_dir / 'checkpoints')
    Path(trainer.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Resume if requested
    start_epoch = 0
    if args.resume:
        checkpoint_path = check_existing_checkpoint(trainer.checkpoint_dir, 'latest')
        if checkpoint_path:
            start_epoch = trainer.load_checkpoint('latest')
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            logger.warning("No checkpoint found for resuming, starting from scratch")
    
    # Train
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=start_epoch
    )
    
    # Log training results
    logger.info(f"Training completed in {training_results['total_time']:.1f}s")
    logger.info(f"Best validation loss: {training_results['best_val_loss']:.6f}")
    logger.info(f"Final epoch: {training_results['final_epoch']}")
    
    # =========================================================================
    # EVALUATION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 5: Final Evaluation")
    logger.info("=" * 60)
    
    # Load best model
    trainer.load_checkpoint('best')
    model.eval()
    
    # Evaluate on test set
    from src.data.preprocessor import compute_features
    
    heston_config = config['data']['heston']
    K = heston_config.get('K', 100.0)
    T = config['data']['T']
    n_steps = config['data']['n_steps']
    dt = T / n_steps
    
    # Get test predictions
    all_pnls = []
    all_deltas = []
    
    with torch.no_grad():
        for S, v, Z in test_loader:
            S = S.to(device)
            v = v.to(device)
            Z = Z.to(device)
            
            features = compute_features(S, v, K, T, dt)
            deltas, y = model(features, S)
            pnl = loss_fn.compute_pnl(deltas, S, Z, dt)
            
            all_pnls.append(pnl.cpu())
            all_deltas.append(deltas.cpu())
    
    all_pnls = torch.cat(all_pnls).numpy()
    all_deltas = torch.cat(all_deltas).numpy()
    
    # Compute baseline deltas for comparison
    delta_baseline = DeltaHedgingBaseline(K, T, heston_config.get('r', 0.02), 'call')
    baseline_deltas = delta_baseline.compute_deltas(
        S_test, v_test, 
        sigma=np.sqrt(heston_config.get('v_0', 0.0175)),
        dt=dt
    )
    
    # Compute all metrics
    test_metrics = compute_all_metrics(
        pnl=all_pnls,
        pnl_unhedged=-Z_test,  # Unhedged = just the payoff
        deltas_model=all_deltas,
        deltas_target=baseline_deltas
    )
    
    print_metrics(test_metrics, title="Test Set Evaluation")
    logger.log_metrics(test_metrics, prefix="Test")
    
    # =========================================================================
    # COMPARISON WITH BASELINES
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 6: Model vs Baselines Comparison")
    logger.info("=" * 60)
    
    comparison = {
        'Deep Hedging': {
            'mean_pnl': test_metrics['pnl_mean'],
            'std_pnl': test_metrics['pnl_std'],
            'cvar_05': test_metrics['cvar_05'],
            'sharpe_ratio': test_metrics['sharpe_ratio']
        }
    }
    
    for name, metrics in baseline_results.items():
        comparison[name] = {
            'mean_pnl': metrics['mean_pnl'],
            'std_pnl': metrics['std_pnl'],
            'cvar_05': metrics['cvar_05'],
            'sharpe_ratio': metrics['sharpe_ratio']
        }
    
    logger.log_comparison_table(comparison)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    logger.info("=" * 60)
    logger.info("STEP 7: Saving Results")
    logger.info("=" * 60)
    
    # Save final results
    final_results = {
        'training': training_results,
        'test_metrics': test_metrics,
        'baselines': baseline_results,
        'learned_premium': float(model.y.item()),
        'model_sparsity': model.get_sparsity(),
        'config': config
    }
    
    logger.save_results(final_results, 'final_results.json')
    
    # Finalize logger
    logger.finalize()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Best model: {experiment_dir / 'checkpoints' / 'best.pt'}")
    print(f"Results: {experiment_dir / 'logs' / 'final_results.json'}")
    print("=" * 60)


if __name__ == '__main__':
    main()