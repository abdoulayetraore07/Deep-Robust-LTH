"""
Unit tests for feature engineering
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from src.data.preprocessor import compute_features


def test_features_shape():
    """Test feature computation output shape"""
    n_paths = 100
    n_steps = 10
    
    S = np.random.rand(n_paths, n_steps) * 100 + 50
    v = np.random.rand(n_paths, n_steps) * 0.1 + 0.01
    K = 100.0
    T = 1.0
    dt = 1.0 / n_steps
    
    features = compute_features(S, v, K, T, dt)
    
    assert features.shape == (n_paths, n_steps, 8)


def test_features_no_nan():
    """Test that features don't contain NaN"""
    n_paths = 100
    n_steps = 10
    
    S = np.random.rand(n_paths, n_steps) * 100 + 50
    v = np.random.rand(n_paths, n_steps) * 0.1 + 0.01
    K = 100.0
    T = 1.0
    dt = 1.0 / n_steps
    
    features = compute_features(S, v, K, T, dt)
    
    # Allow NaN only at t=0 for features that require previous timestep
    assert not np.any(np.isnan(features[:, 1:, :]))


def test_log_moneyness():
    """Test log-moneyness feature"""
    S = np.array([[100.0, 110.0], [90.0, 95.0]])
    v = np.ones_like(S) * 0.04
    K = 100.0
    T = 1.0
    dt = 0.5
    
    features = compute_features(S, v, K, T, dt)
    
    # Feature 0: log(S/K)
    expected_0 = np.log(100.0 / 100.0)
    expected_1 = np.log(110.0 / 100.0)
    
    assert features[0, 0, 0] == pytest.approx(expected_0)
    assert features[0, 1, 0] == pytest.approx(expected_1)


def test_time_to_maturity():
    """Test time to maturity feature"""
    S = np.ones((10, 5)) * 100
    v = np.ones_like(S) * 0.04
    K = 100.0
    T = 1.0
    dt = 0.25
    
    features = compute_features(S, v, K, T, dt)
    
    # Feature 4: (T - t*dt) / T
    assert features[0, 0, 4] == pytest.approx(1.0)  # t=0
    assert features[0, 1, 4] == pytest.approx(0.75)  # t=1
    assert features[0, 4, 4] == pytest.approx(0.0)  # t=4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])