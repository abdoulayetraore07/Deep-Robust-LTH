"""
Unit tests for pruning with masking
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import torch.nn as nn
from src.models.deep_hedging import DeepHedgingNetwork
from src.pruning.magnitude import magnitude_pruning, rewind_weights, get_sparsity
from src.models.trainer import Trainer


@pytest.fixture
def dummy_config():
    """Dummy configuration"""
    return {
        'model': {
            'input_dim': 8,
            'hidden_dims': [64, 64, 32],
            'dropout_rates': [0.1, 0.1, 0.05],
            'use_batch_norm': True
        },
        'training': {
            'optimizer_name': 'adam',
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'lr_scheduler': 'cosine',
            'epochs': 2,
            'batch_size': 32,
            'patience': 20,
            'cvar_alpha': 0.05
        },
        'data': {
            'transaction_cost': {
                'c_prop': 0.001
            }
        }
    }


def test_magnitude_pruning_sparsity(dummy_config):
    """Test that magnitude pruning achieves target sparsity"""
    model = DeepHedgingNetwork(dummy_config['model'])
    
    target_sparsity = 0.8
    mask = magnitude_pruning(model, sparsity=target_sparsity, exclude_output=False)
    
    # Apply mask
    for name, param in model.named_parameters():
        if name in mask:
            param.data *= mask[name]
    
    actual_sparsity = get_sparsity(model)
    
    # Should be close to target (within 5%)
    assert abs(actual_sparsity - target_sparsity) < 0.05


def test_mask_persistence_during_training(dummy_config, tmp_path):
    """Test that mask is maintained during training"""
    model = DeepHedgingNetwork(dummy_config['model'])
    
    # Save init weights
    init_weights_path = tmp_path / 'init.pt'
    torch.save(model.state_dict(), init_weights_path)
    
    # Prune
    mask = magnitude_pruning(model, sparsity=0.5)

    
    # Create dummy data with correct shape
    from torch.utils.data import DataLoader, TensorDataset
    
    n_paths = 50
    n_steps = 63
    
    S_dummy = torch.rand(n_paths, n_steps) * 50 + 75  
    v_dummy = torch.rand(n_paths, n_steps) * 0.05 + 0.01  
    Z_dummy = torch.rand(n_paths) * 10  
    
    dataset = TensorDataset(S_dummy, v_dummy, Z_dummy)
    train_loader = DataLoader(dataset, batch_size=10)
    val_loader = DataLoader(dataset, batch_size=10)
    
    # Rewind
    rewind_weights(model, str(init_weights_path), mask)
    
    # Train with mask
    trainer = Trainer(model, dummy_config, device='cpu', mask=mask)
    
    # Train one epoch
    trainer.train_epoch(train_loader, K=100, T=0.25, dt=0.004)
    
    # Check that pruned weights are still zero
    for name, param in model.named_parameters():
        if name in mask:
            pruned_indices = (mask[name] == 0)
            assert torch.all(param.data[pruned_indices] == 0), f"Pruned weights in {name} are not zero after training!"


def test_rewind_weights(dummy_config, tmp_path):
    """Test rewinding to initial weights"""
    model = DeepHedgingNetwork(dummy_config['model'])
    
    # Save init
    init_weights_path = tmp_path / 'init.pt'
    init_weights = {name: param.clone() for name, param in model.named_parameters()}
    torch.save(model.state_dict(), init_weights_path)
    
    # Modify weights
    for param in model.parameters():
        param.data += torch.randn_like(param.data) * 0.1
    
    # Prune
    mask = magnitude_pruning(model, sparsity=0.5)
    
    # Rewind
    rewind_weights(model, str(init_weights_path), mask)
    
    # Check that unpruned weights match init
    for name, param in model.named_parameters():
        if name in mask:
            unpruned_indices = (mask[name] == 1)
            init_param = init_weights[name]
            assert torch.allclose(param.data[unpruned_indices], init_param[unpruned_indices]), \
                f"Unpruned weights in {name} don't match initialization!"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])