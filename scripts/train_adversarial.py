#!/usr/bin/env python3
"""
Adversarial Training Script for Deep Hedging

Usage:
    python scripts/train_adversarial.py --config configs/config.yaml
    python scripts/train_adversarial.py --config configs/config.yaml --mode pgd
    python scripts/train_adversarial.py --config configs/config.yaml --mode curriculum
    python scripts/train_adversarial.py --config configs/config.yaml --pruned-model path/to/ticket.pt

Options:
    --config: Path to configuration file
    --mode: Adversarial mode (fgsm/pgd/curriculum/mixed)
    --epsilon: Override adversarial epsilon
    --force-retrain: Force retraining
    --resume: Resume from checkpoint
    --pruned-model: Path to pruned model (lottery ticket) to adversarially train
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

from src.utils.config import (
    load_config, get_device, get_experiment_dir,
    check_existing_model, save_config_to_experiment
)
from src.utils.logging import create_logger
from src.data.heston import get_or_generate_dataset
from src.data.preprocessor import create_dataloaders, compute_features
from src.models.deep_hedging import create_model
from src.models.losses import create_loss_function
from src.models.trainer import load_trained_model, check_existing_checkpoint
from src.attacks.adversarial_trainer import create_adversarial_trainer
from src.attacks.pgd import create_pgd_attack
from src.pruning.pruning import PruningManager
from src.evaluation.baselines import evaluate_all_baselines
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
    parser = argparse.ArgumentParser(description='Adversarial Training for Deep Hedging')
    
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['fgsm', 'pgd', 'curriculum', 'mixed'],
                        help='Adversarial training mode')
    parser.add_argument('--epsilon', type=float, default=None,
                        help='Override adversarial epsilon')
    parser.add_argument('--force-retrain', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--load-baseline', type=str, default=None,
                        help='Path to pretrained baseline model to finetune')
    parser.add_argument('--pruned-model', type=str, default=None,
                        help='Path to pruned model (lottery ticket) to adversarially train')
    
    return parser.parse_args()


def load_pruned_model(model, checkpoint_path, device):
    """
    Load a pruned model (lottery ticket) from checkpoint.
    
    Args:
        model: Model instance (architecture must match)
        checkpoint_path: Path to ticket checkpoint
        device: Device to load on
        
    Returns:
        model: Loaded model with pruning masks
        pruning_manager: PruningManager instance
        info: Checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state (includes pruning masks as buffers)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Create pruning manager and detect pruned layers
    pruning_manager = PruningManager(model)
    
    # Reconstruct pruned params list from model state
    # PyTorch's prune creates weight_orig and weight_mask buffers
    import torch.nn.utils.prune as prune
    for name, module in model.named_modules():
        if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
            pruning_manager._pruned_params.append((module, 'weight'))
    
    # Get info
    info = {
        'sparsity': checkpoint.get('sparsity', 0.0),
        'metrics': checkpoint.get('metrics', {}),
        'round': checkpoint.get('round', -1)
    }
    
    # Verify integrity
    if pruning_manager.is_pruned():
        actual_sparsity = pruning_manager.get_sparsity()
        print(f"[Model] Loaded pruned model from {checkpoint_path}")
        print(f"[Model] Sparsity: {actual_sparsity['total']:.2%}")
        if pruning_manager.verify_integrity():
            print(f"[Model] Pruning integrity: PASS")
        else:
            print(f"[Model] Pruning integrity: FAIL - weights may have been corrupted")
    else:
        print(f"[Model] Loaded model from {checkpoint_path} (no pruning detected)")
        pruning_manager = None
    
    return model, pruning_manager, info


def evaluate_robustness(
    model, loss_fn, test_loader, config, device, logger
):
    """Evaluate model robustness against adversarial attacks."""
    from src.attacks.fgsm import create_fgsm_attack
    from src.attacks.pgd import create_pgd_attack
    
    heston_config = config['data']['heston']
    K = heston_config.get('K', 100.0)
    T = config['data']['T']
    n_steps = config['data']['n_steps']
    dt = T / n_steps
    
    model.eval()
    
    # Clean evaluation
    clean_pnls = []
    with torch.no_grad():
        for S, v, Z in test_loader:
            S, v, Z = S.to(device), v.to(device), Z.to(device)
            features = compute_features(S, v, K, T, dt)
            deltas, y = model(features, S)
            pnl = loss_fn.compute_pnl(deltas, S, Z, dt)
            clean_pnls.append(pnl.cpu())
    
    clean_pnls = torch.cat(clean_pnls).numpy()
    
    # FGSM attack
    fgsm = create_fgsm_attack(model, loss_fn, config)
    fgsm_pnls = []
    
    for S, v, Z in test_loader:
        S, v, Z = S.to(device), v.to(device), Z.to(device)
        features = compute_features(S, v, K, T, dt)
        features_adv, _ = fgsm.attack(features, S, Z, dt)
        
        with torch.no_grad():
            deltas, y = model(features_adv, S)
            pnl = loss_fn.compute_pnl(deltas, S, Z, dt)
            fgsm_pnls.append(pnl.cpu())
    
    fgsm_pnls = torch.cat(fgsm_pnls).numpy()
    
    # PGD attack
    pgd = create_pgd_attack(model, loss_fn, config)
    pgd_pnls = []
    
    for S, v, Z in test_loader:
        S, v, Z = S.to(device), v.to(device), Z.to(device)
        features = compute_features(S, v, K, T, dt)
        features_adv, _ = pgd.attack(features, S, Z, dt)
        
        with torch.no_grad():
            deltas, y = model(features_adv, S)
            pnl = loss_fn.compute_pnl(deltas, S, Z, dt)
            pgd_pnls.append(pnl.cpu())
    
    pgd_pnls = torch.cat(pgd_pnls).numpy()
    
    # Compute metrics
    clean_metrics = compute_all_metrics(clean_pnls)
    fgsm_metrics = compute_all_metrics(fgsm_pnls)
    pgd_metrics = compute_all_metrics(pgd_pnls)
    
    # Log results
    logger.log_adversarial_results(clean_metrics, fgsm_metrics, "FGSM")
    logger.log_adversarial_results(clean_metrics, pgd_metrics, "PGD")
    
    return {
        'clean': clean_metrics,
        'fgsm': fgsm_metrics,
        'pgd': pgd_metrics,
        'fgsm_gap': clean_metrics['pnl_mean'] - fgsm_metrics['pnl_mean'],
        'pgd_gap': clean_metrics['pnl_mean'] - pgd_metrics['pnl_mean']
    }


def main():
    """Main adversarial training function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with CLI args
    if args.mode is not None:
        if 'adversarial' not in config:
            config['adversarial'] = {}
        config['adversarial']['mode'] = args.mode
    
    if args.epsilon is not None:
        if 'adversarial' not in config:
            config['adversarial'] = {}
        config['adversarial']['pgd'] = config['adversarial'].get('pgd', {})
        config['adversarial']['pgd']['epsilon'] = args.epsilon
        config['adversarial']['fgsm'] = config['adversarial'].get('fgsm', {})
        config['adversarial']['fgsm']['epsilon'] = args.epsilon
    
    if args.device:
        config['device'] = args.device
    if args.seed:
        config['seed'] = args.seed
    
    # Ensure adversarial mode is set
    adv_mode = config.get('adversarial', {}).get('mode', 'pgd')
    config.setdefault('adversarial', {})['mode'] = adv_mode
    
    # Update experiment name
    base_name = config.get('experiment_name', 'deep_hedging')
    if args.pruned_model:
        config['experiment_name'] = base_name + f'_adv_{adv_mode}_pruned'
    else:
        config['experiment_name'] = base_name + f'_adv_{adv_mode}'
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Get device
    device = get_device(config)
    print(f"Device: {device}")
    print(f"Adversarial mode: {adv_mode}")
    
    # Experiment directory
    experiment_dir = get_experiment_dir(config)
    save_config_to_experiment(config)
    
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
    # MODEL
    # =========================================================================
    logger.info("Creating model...")
    model = create_model(config)
    loss_fn = create_loss_function(config)
    
    # Initialize pruning manager (will be set if loading pruned model)
    pruning_manager = None
    
    # Load pruned model (lottery ticket) if specified
    if args.pruned_model:
        logger.info(f"Loading pruned model (lottery ticket) from {args.pruned_model}")
        model, pruning_manager, ticket_info = load_pruned_model(model, args.pruned_model, device)
        logger.info(f"  Ticket sparsity: {ticket_info['sparsity']:.2%}")
        if ticket_info.get('metrics'):
            logger.info(f"  Original Sharpe: {ticket_info['metrics'].get('sharpe_ratio', 'N/A')}")
    
    # Optionally load pretrained baseline (non-pruned)
    elif args.load_baseline:
        logger.info(f"Loading pretrained model from {args.load_baseline}")
        model, _ = load_trained_model(model, args.load_baseline, device)
    
    model = model.to(device)
    logger.log_model_summary(model.summary())
    
    # =========================================================================
    # ADVERSARIAL TRAINING
    # =========================================================================
    logger.info("=" * 60)
    if pruning_manager:
        sparsity = pruning_manager.get_sparsity().get('total', 0)
        logger.info(f"ADVERSARIAL TRAINING (mode={adv_mode}, sparsity={sparsity:.1%})")
    else:
        logger.info(f"ADVERSARIAL TRAINING (mode={adv_mode})")
    logger.info("=" * 60)
    
    trainer = create_adversarial_trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
        device=device,
        experiment_dir=str(experiment_dir),
        pruning_manager=pruning_manager
    )
    
    # Resume if requested
    start_epoch = 0
    if args.resume:
        ckpt_path = check_existing_checkpoint(trainer.checkpoint_dir, 'latest')
        if ckpt_path:
            start_epoch = trainer.load_checkpoint('latest')
    
    # Train
    training_results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        start_epoch=start_epoch
    )
    
    # =========================================================================
    # ROBUSTNESS EVALUATION
    # =========================================================================
    logger.info("=" * 60)
    logger.info("ROBUSTNESS EVALUATION")
    logger.info("=" * 60)
    
    # Load best model
    trainer.load_checkpoint('best')
    
    # Verify pruning integrity after training (if pruned)
    if pruning_manager:
        if pruning_manager.verify_integrity():
            logger.info("Pruning integrity after adversarial training: PASS")
        else:
            logger.warning("Pruning integrity after adversarial training: FAIL")
        
        final_sparsity = pruning_manager.get_sparsity()
        logger.info(f"Final sparsity: {final_sparsity['total']:.2%}")
    
    robustness_results = evaluate_robustness(
        model, loss_fn, test_loader, config, device, logger
    )
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    logger.info("=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)
    
    comparison = {
        'Clean': {
            'mean_pnl': robustness_results['clean']['pnl_mean'],
            'cvar_05': robustness_results['clean']['cvar_05'],
            'sharpe': robustness_results['clean']['sharpe_ratio']
        },
        'FGSM Attack': {
            'mean_pnl': robustness_results['fgsm']['pnl_mean'],
            'cvar_05': robustness_results['fgsm']['cvar_05'],
            'sharpe': robustness_results['fgsm']['sharpe_ratio']
        },
        'PGD Attack': {
            'mean_pnl': robustness_results['pgd']['pnl_mean'],
            'cvar_05': robustness_results['pgd']['cvar_05'],
            'sharpe': robustness_results['pgd']['sharpe_ratio']
        }
    }
    
    logger.log_comparison_table(comparison, metric_keys=['mean_pnl', 'cvar_05', 'sharpe'])
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    final_results = {
        'training': training_results,
        'robustness': robustness_results,
        'config': config
    }
    
    # Add pruning info if applicable
    if pruning_manager:
        final_results['pruning'] = {
            'sparsity': pruning_manager.get_sparsity(),
            'integrity': pruning_manager.verify_integrity()
        }
    
    logger.save_results(final_results, 'final_results.json')
    logger.finalize()
    
    print(f"\nAdversarial training complete!")
    print(f"Results: {experiment_dir}")


if __name__ == '__main__':
    main()
