"""
Feature engineering and data preprocessing

IMPORTANT: This module computes ONLY exogenous market features.
Recurrent features (delta_prev, trading_volume, pnl_cumulative) are 
computed during the forward pass in the model, NOT here.

Exogenous features (5):
    1. log(S_t / K)           - Log-moneyness
    2. (S_t - S_{t-1})/S_{t-1} - Return
    3. sqrt(v_t)              - Volatility
    4. v_t - v_{t-1}          - Variance change
    5. (T - t*dt) / T         - Normalized time to maturity
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Optional, Union

# Number of exogenous features
N_EXOGENOUS_FEATURES = 5


def compute_features(
    S: Union[np.ndarray, torch.Tensor],
    v: Union[np.ndarray, torch.Tensor],
    K: float,
    T: float,
    dt: float
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute exogenous market features from Heston paths.
    
    These are the 5 features that do NOT depend on the model's actions:
        1. log(S_t / K)           - Log-moneyness
        2. (S_t - S_{t-1})/S_{t-1} - Return (0 at t=0)
        3. sqrt(v_t)              - Volatility
        4. v_t - v_{t-1}          - Variance change (0 at t=0)
        5. (T - t*dt) / T         - Normalized time to maturity
    
    Note: Recurrent features (delta_prev, trading_volume, pnl_cumulative)
    are computed in the model's forward pass, NOT here.
    
    Args:
        S: Stock prices (n_paths, n_steps) - numpy array OR torch tensor
        v: Variances (n_paths, n_steps) - numpy array OR torch tensor
        K: Strike price
        T: Time to maturity
        dt: Time step
        
    Returns:
        features: (n_paths, n_steps, 5) - same type as input
    """
    is_torch = isinstance(S, torch.Tensor)
    
    if is_torch:
        return _compute_features_torch(S, v, K, T, dt)
    else:
        return _compute_features_numpy(S, v, K, T, dt)


def _compute_features_torch(
    S: torch.Tensor,
    v: torch.Tensor,
    K: float,
    T: float,
    dt: float
) -> torch.Tensor:
    """
    Compute features using PyTorch (GPU-compatible).
    
    Args:
        S: Stock prices (n_paths, n_steps)
        v: Variances (n_paths, n_steps)
        K: Strike price
        T: Time to maturity
        dt: Time step
        
    Returns:
        features: (n_paths, n_steps, 5)
    """
    device = S.device
    dtype = S.dtype
    n_paths, n_steps = S.shape
    
    features = torch.zeros(n_paths, n_steps, N_EXOGENOUS_FEATURES, 
                          device=device, dtype=dtype)
    
    # Feature 1: Log-moneyness
    features[:, :, 0] = torch.log(S / K)
    
    # Feature 2: Return (0 at t=0)
    features[:, 1:, 1] = (S[:, 1:] - S[:, :-1]) / S[:, :-1]
    
    # Feature 3: Volatility (sqrt of variance)
    features[:, :, 2] = torch.sqrt(torch.clamp(v, min=1e-8))
    
    # Feature 4: Variance change (0 at t=0)
    features[:, 1:, 3] = v[:, 1:] - v[:, :-1]
    
    # Feature 5: Normalized time to maturity
    time_steps = torch.arange(n_steps, device=device, dtype=dtype)
    features[:, :, 4] = (T - time_steps * dt) / T
    
    return features


def _compute_features_numpy(
    S: np.ndarray,
    v: np.ndarray,
    K: float,
    T: float,
    dt: float
) -> np.ndarray:
    """
    Compute features using NumPy.
    
    Args:
        S: Stock prices (n_paths, n_steps)
        v: Variances (n_paths, n_steps)
        K: Strike price
        T: Time to maturity
        dt: Time step
        
    Returns:
        features: (n_paths, n_steps, 5)
    """
    n_paths, n_steps = S.shape
    
    features = np.zeros((n_paths, n_steps, N_EXOGENOUS_FEATURES))
    
    # Feature 1: Log-moneyness
    features[:, :, 0] = np.log(S / K)
    
    # Feature 2: Return (0 at t=0)
    features[:, 1:, 1] = (S[:, 1:] - S[:, :-1]) / S[:, :-1]
    
    # Feature 3: Volatility (sqrt of variance)
    features[:, :, 2] = np.sqrt(np.maximum(v, 1e-8))
    
    # Feature 4: Variance change (0 at t=0)
    features[:, 1:, 3] = v[:, 1:] - v[:, :-1]
    
    # Feature 5: Normalized time to maturity
    for t in range(n_steps):
        features[:, t, 4] = (T - t * dt) / T
    
    return features


def create_dataloaders(
    S_train: np.ndarray,
    v_train: np.ndarray,
    Z_train: np.ndarray,
    S_val: np.ndarray,
    v_val: np.ndarray,
    Z_val: np.ndarray,
    S_test: np.ndarray,
    v_test: np.ndarray,
    Z_test: np.ndarray,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from numpy arrays.
    
    Note: We pass raw S, v, Z to the dataloader. Features are computed
    on-the-fly in the training loop (on GPU if available).
    
    Args:
        S_train, v_train, Z_train: Training data
        S_val, v_val, Z_val: Validation data
        S_test, v_test, Z_test: Test data
        batch_size: Batch size
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Convert to tensors
    S_train_t = torch.FloatTensor(S_train)
    v_train_t = torch.FloatTensor(v_train)
    Z_train_t = torch.FloatTensor(Z_train)
    
    S_val_t = torch.FloatTensor(S_val)
    v_val_t = torch.FloatTensor(v_val)
    Z_val_t = torch.FloatTensor(Z_val)
    
    S_test_t = torch.FloatTensor(S_test)
    v_test_t = torch.FloatTensor(v_test)
    Z_test_t = torch.FloatTensor(Z_test)
    
    # Create datasets
    train_dataset = TensorDataset(S_train_t, v_train_t, Z_train_t)
    val_dataset = TensorDataset(S_val_t, v_val_t, Z_val_t)
    test_dataset = TensorDataset(S_test_t, v_test_t, Z_test_t)
    
    # Create dataloaders
    # Note: num_workers=0 is often faster for small datasets on GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        features: (n_paths, n_steps, n_features)
        mean: Pre-computed mean (if None, compute from features)
        std: Pre-computed std (if None, compute from features)
        
    Returns:
        normalized_features, mean, std
    """
    if mean is None:
        # Compute mean over paths and time
        mean = features.reshape(-1, features.shape[-1]).mean(axis=0)
    
    if std is None:
        # Compute std over paths and time
        std = features.reshape(-1, features.shape[-1]).std(axis=0)
        std = np.maximum(std, 1e-8)  # Avoid division by zero
    
    normalized = (features - mean) / std
    
    return normalized, mean, std


def get_feature_names() -> list:
    """
    Get names of the exogenous features.
    
    Returns:
        List of feature names
    """
    return [
        'log_moneyness',    # log(S/K)
        'return',           # (S_t - S_{t-1}) / S_{t-1}
        'volatility',       # sqrt(v)
        'variance_change',  # v_t - v_{t-1}
        'time_to_maturity'  # (T - t*dt) / T
    ]


def describe_features(features: Union[np.ndarray, torch.Tensor]) -> dict:
    """
    Compute descriptive statistics for features.
    
    Args:
        features: (n_paths, n_steps, n_features)
        
    Returns:
        Dictionary with statistics per feature
    """
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    
    feature_names = get_feature_names()
    stats = {}
    
    for i, name in enumerate(feature_names):
        f = features[:, :, i].flatten()
        stats[name] = {
            'mean': float(np.mean(f)),
            'std': float(np.std(f)),
            'min': float(np.min(f)),
            'max': float(np.max(f)),
            'median': float(np.median(f))
        }
    
    return stats