"""
Unit tests for Heston simulator
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from src.data.heston import HestonSimulator


@pytest.fixture
def heston_params():
    """Default Heston parameters"""
    return {
        'S_0': 100.0,
        'v_0': 0.0175,      
        'kappa': 1.5768,    
        'theta': 0.0398,    
        'xi': 0.5751,       
        'rho': -0.5711,     
        'mu': 0.05,
        'r': 0.02,
        'K': 100.0
    }


def test_heston_initialization(heston_params):
    """Test Heston simulator initialization"""
    sim = HestonSimulator(heston_params)
    
    assert sim.S_0 == 100.0
    assert sim.v_0 == 0.0175
    assert sim.kappa == 1.5768


def test_heston_simulation_shape(heston_params):
    """Test simulation output shapes"""
    sim = HestonSimulator(heston_params)
    
    n_paths = 1000
    T = 1.0
    n_steps = 100
    
    S, v = sim.simulate(n_paths, T, n_steps, seed=42)
    
    assert S.shape == (n_paths, n_steps + 1)
    assert v.shape == (n_paths, n_steps + 1)
    assert S[:, 0].mean() == pytest.approx(100.0, rel=1e-10)


def test_heston_positivity(heston_params):
    """Test that variance is always positive"""
    sim = HestonSimulator(heston_params)
    
    n_paths = 1000
    T = 1.0
    n_steps = 1000
    
    S, v = sim.simulate(n_paths, T, n_steps, seed=42)
    
    assert np.all(v > 0), "Variance must be positive"
    assert np.all(S > 0), "Price must be positive"


def test_heston_moments(heston_params):
    """Test variance moments match theory"""
    sim = HestonSimulator(heston_params)
    
    n_paths = 50000
    T = 1.0
    n_steps = 1000
    
    S, v = sim.simulate(n_paths, T, n_steps, seed=42)
    
    validation = sim.validate_moments(v, T, tolerance=0.01)
    
    assert validation['E_v_T_pass'], "Mean validation failed"
    assert validation['Var_v_T_pass'], "Variance validation failed"
    assert validation['positivity_pass'], "Positivity check failed"


def test_heston_reproducibility(heston_params):
    """Test that same seed gives same results"""
    sim = HestonSimulator(heston_params)
    
    S1, v1 = sim.simulate(100, 1.0, 10, seed=42)
    S2, v2 = sim.simulate(100, 1.0, 10, seed=42)
    
    np.testing.assert_array_equal(S1, S2)
    np.testing.assert_array_equal(v1, v2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])