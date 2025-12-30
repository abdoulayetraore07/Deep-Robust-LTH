"""
Unit tests for adversarial attacks
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import numpy as np
from src.models.deep_hedging import DeepHedgingNetwork
from src.attacks.fgsm import fgsm_attack
from src.attacks.pgd import pgd_attack


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
        'data': {
            'transaction_cost': {
                'c_prop': 0.001
            }
        },
        'training': {
            'cvar_alpha': 0.05
        }
    }


@pytest.fixture
def dummy_model(dummy_config):
    """Create dummy model"""
    model = DeepHedgingNetwork(dummy_config['model'])
    model.eval()
    return model


def dummy_features_fn(S, v):
    """Dummy feature computation"""
    batch_size, n_steps = S.shape
    features = torch.zeros(batch_size, n_steps, 8)
    features[:, :, 0] = torch.log(S / 100.0)
    features[:, :, 2] = torch.sqrt(v)
    return features


def test_fgsm_perturbation_bounded(dummy_model, dummy_config):
    """Test FGSM perturbations are within epsilon"""
    S = torch.rand(10, 10) * 50 + 75  # Price in [75, 125]
    v = torch.rand(10, 10) * 0.05 + 0.01  # Variance in [0.01, 0.06]
    Z = torch.rand(10) * 10
    
    epsilon_S = 0.02
    epsilon_v = 0.2
    
    S_adv, v_adv = fgsm_attack(
        dummy_model, S, v, Z, dummy_features_fn,
        dummy_config, epsilon_S, epsilon_v
    )
    
    # Check perturbations are bounded
    delta_S = torch.abs((S_adv - S) / S)
    delta_v = torch.abs((v_adv - v) / v)
    
    assert torch.all(delta_S <= epsilon_S + 1e-6)
    assert torch.all(delta_v <= epsilon_v + 1e-6)


def test_fgsm_positivity(dummy_model, dummy_config):
    """Test FGSM preserves positivity"""
    S = torch.rand(10, 10) * 50 + 75
    v = torch.rand(10, 10) * 0.05 + 0.01
    Z = torch.rand(10) * 10
    
    S_adv, v_adv = fgsm_attack(
        dummy_model, S, v, Z, dummy_features_fn,
        dummy_config, 0.02, 0.2
    )
    
    assert torch.all(S_adv > 0)
    assert torch.all(v_adv > 0)


def test_pgd_perturbation_bounded(dummy_model, dummy_config):
    """Test PGD perturbations are within epsilon"""
    S = torch.rand(10, 10) * 50 + 75
    v = torch.rand(10, 10) * 0.05 + 0.01
    Z = torch.rand(10) * 10
    
    epsilon_S = 0.05
    epsilon_v = 0.5
    
    S_adv, v_adv = pgd_attack(
        dummy_model, S, v, Z, dummy_features_fn,
        dummy_config, epsilon_S, epsilon_v,
        alpha_S=0.01, alpha_v=0.1, num_steps=5
    )
    
    delta_S = torch.abs((S_adv - S) / S)
    delta_v = torch.abs((v_adv - v) / v)
    
    assert torch.all(delta_S <= epsilon_S + 1e-5)
    assert torch.all(delta_v <= epsilon_v + 1e-5)


def test_pgd_positivity(dummy_model, dummy_config):
    """Test PGD preserves positivity"""
    S = torch.rand(10, 10) * 50 + 75
    v = torch.rand(10, 10) * 0.05 + 0.01
    Z = torch.rand(10) * 10
    
    S_adv, v_adv = pgd_attack(
        dummy_model, S, v, Z, dummy_features_fn,
        dummy_config, 0.05, 0.5,
        alpha_S=0.01, alpha_v=0.1, num_steps=10
    )
    
    assert torch.all(S_adv > 0)
    assert torch.all(v_adv > 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])