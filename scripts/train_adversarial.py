"""
Script to train adversarial models (FGSM or PGD)
"""

import sys
import torch
import numpy as np
import argparse
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_device
from src.models.deep_hedging import DeepHedgingNetwork
from src.attacks.adversarial_trainer import AdversarialTrainer
from src.data.preprocessor import create_dataloaders
from src.evaluation.metrics import evaluate_robustness


def main():
    parser = argparse.ArgumentParser(description='Train adversarial Deep Hedging model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--attack', type=str, choices=['fgsm', 'pgd'], required=True, help='Attack type')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
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
    
    batch_size = config['training']['batch_size'] or 256
    num_workers = config['compute']['num_parallel_workers']
    
    train_loader, val_loader, test_loader = create_dataloaders(
        S_train, v_train, Z_train,
        S_val, v_val, Z_val,
        S_test, v_test, Z_test,
        batch_size, num_workers
    )
    
    # Create model
    print(f"Creating model for {args.attack} adversarial training...")
    model = DeepHedgingNetwork(config['model'])
    
    # Passer mask=None explicitement
    trainer = AdversarialTrainer(model, config, attack_type=args.attack, device=device, mask=None)
    
    # Train
    print("Starting adversarial training...")
    K = config['data']['heston']['K']
    T = config['data']['T']
    dt = config['data']['dt']
    
    best_val_loss = trainer.fit(train_loader, val_loader, K, T, dt)
    
    print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
    print(f"Learned premium (y): {model.y.item():.6f}")
    
    # Save model
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), output_dir / 'best_model.pt')
    print(f"Model saved to {output_dir / 'best_model.pt'}")
    
    # Evaluate robustness
    print("Evaluating robustness...")
    results = evaluate_robustness(model, test_loader, config, K, T, dt, device)
    
    print("\nRobustness Results:")
    print(f"  Clean CVaR: {results['clean']['cvar_005']:.6f}")
    print(f"  FGSM CVaR: {results['fgsm']['cvar_005']:.6f}")
    print(f"  PGD-10 CVaR: {results['pgd10']['cvar_005']:.6f}")
    print(f"  PGD-20 CVaR: {results['pgd20']['cvar_005']:.6f}")
    
    # Save results
    with open(output_dir / 'robustness_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_dir / 'robustness_results.json'}")


if __name__ == '__main__':
    main()