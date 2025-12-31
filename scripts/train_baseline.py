"""
Script to train baseline Deep Hedging model
"""

import sys
import torch
import numpy as np
import argparse
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_device
from src.models.deep_hedging import DeepHedgingNetwork
from src.models.trainer import Trainer
from src.data.preprocessor import create_dataloaders
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.baselines import delta_hedging_baseline


def main():
    parser = argparse.ArgumentParser(description='Train baseline Deep Hedging model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    # Set device
    device = get_device(config)
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data_dir = Path('data/processed')
    
    S_train = np.load(data_dir / 'S_train.npy')
    v_train = np.load(data_dir / 'v_train.npy')
    Z_train = np.load(data_dir / 'Z_train.npy')
    
    S_val = np.load(data_dir / 'S_val.npy')
    v_val = np.load(data_dir / 'v_val.npy')
    Z_val = np.load(data_dir / 'Z_val.npy')
    
    S_test = np.load(data_dir / 'S_test.npy')
    v_test = np.load(data_dir / 'v_test.npy')
    Z_test = np.load(data_dir / 'Z_test.npy')
    
    print(f"Data loaded: {S_train.shape[0]} train, {S_val.shape[0]} val, {S_test.shape[0]} test")
    
    # Create dataloaders
    batch_size = config['training']['batch_size'] or 256
    num_workers = config['compute']['num_parallel_workers']
    
    train_loader, val_loader, test_loader = create_dataloaders(
        S_train, v_train, Z_train,
        S_val, v_val, Z_val,
        S_test, v_test, Z_test,
        batch_size, num_workers
    )
    
    # Create model
    print("Creating model...")
    model = DeepHedgingNetwork(config['model'])
    print(f"Model created with {model.get_num_parameters()} parameters")
    
    # Save initial weights BEFORE training
    output_dir = Path('experiments/baseline')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    init_weights_path = output_dir / 'init_weights.pt'
    torch.save(model.state_dict(), init_weights_path)
    print(f"Initial weights saved to {init_weights_path}")
    
    # Create trainer (no mask for baseline)
    trainer = Trainer(model, config, device, mask=None)
    
    # Train
    print("Starting training...")
    K = config['data']['heston']['K']
    T = config['data']['T']
    dt = config['data']['dt']
    
    best_val_loss = trainer.fit(train_loader, val_loader, K, T, dt)
    
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Learned premium (y): {model.y.item():.6f}")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    
    trainer.load_checkpoint('experiments/baseline/best_model.pt')
    metrics = compute_all_metrics(model, test_loader, config, K, T, dt, device)
    
    print("\nTest metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {output_dir / 'metrics.json'}")
    
    # Compare with Delta Hedging baseline
    print("\n" + "="*60)
    print("Delta Hedging Baseline")
    print("="*60)
    
    r = config['data']['heston']['r']
    delta_metrics = delta_hedging_baseline(
        S_test, v_test, Z_test, K, T, r, dt,
        c_prop=config['data']['transaction_cost']['c_prop']
    )
    
    print("\nDelta Hedging metrics:")
    for key, value in delta_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Save delta metrics
    with open(output_dir / 'delta_hedging_metrics.json', 'w') as f:
        json.dump(delta_metrics, f, indent=2)
    
    # Comparison table
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"{'Metric':<30} {'Deep Hedging':<20} {'Delta Hedging':<20}")
    print("-"*70)
    for key in ['mean_pnl', 'cvar_005', 'hedging_error_rmse', 'total_trading_volume']:
        dh_val = metrics.get(key, 0.0)
        bs_val = delta_metrics.get(key, 0.0)
        print(f"{key:<30} {dh_val:<20.6f} {bs_val:<20.6f}")
    
    print("\nBaseline training complete")


if __name__ == '__main__':
    main()