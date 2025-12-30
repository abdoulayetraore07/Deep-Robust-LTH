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
    
    # ✅ CORRECTION: Save initial weights BEFORE training
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
    
    # Evaluate on test set
    print("Evaluating on test set...")
    from src.evaluation.metrics import compute_all_metrics
    
    trainer.load_checkpoint('experiments/baseline/best_model.pt')
    metrics = compute_all_metrics(model, test_loader, config, K, T, dt, device)
    
    print("Test metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {output_dir / 'metrics.json'}")
    print("Baseline training complete")


if __name__ == '__main__':
    main()