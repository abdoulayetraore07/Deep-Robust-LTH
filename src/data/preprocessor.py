"""
Feature engineering and data preprocessing
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, Optional


def compute_features(
    S,  # Accepte np.ndarray OU torch.Tensor
    v,  # Accepte np.ndarray OU torch.Tensor
    K: float,
    T: float,
    dt: float,
    delta_prev=None,
    pnl_prev=None
):
    """
    Compute 8 features from Heston paths
    Détecte automatiquement numpy vs torch et compute sur GPU si possible
    
    Features:
    1. log(S_t / K) - log-moneyness
    2. (S_t - S_{t-1})/S_{t-1} - return
    3. sqrt(v_t) - volatility
    4. v_t - v_{t-1} - variance change
    5. (T - t*dt) / T - time to maturity
    6. delta_{t-1} - previous position
    7. |delta_t - delta_{t-1}| - trading volume
    8. PnL_{t-1} - cumulative P&L
    
    Args:
        S: Stock prices (n_paths, n_steps) - numpy array OR torch tensor
        v: Variances (n_paths, n_steps) - numpy array OR torch tensor
        K: Strike price
        T: Time to maturity
        dt: Time step
        delta_prev: Previous positions (n_paths, n_steps), optional
        pnl_prev: Cumulative P&L (n_paths, n_steps), optional
        
    Returns:
        features: (n_paths, n_steps, 8) - same type as input (numpy or torch)
    """
    # Détection automatique du type
    is_torch = isinstance(S, torch.Tensor)
    
    if is_torch:

        device = S.device
        n_paths, n_steps = S.shape
        features = torch.zeros(n_paths, n_steps, 8, device=device, dtype=S.dtype)
        
        # Feature 1: Log-moneyness
        features[:, :, 0] = torch.log(S / K)
        
        # Feature 2: Return
        features[:, 1:, 1] = (S[:, 1:] - S[:, :-1]) / S[:, :-1]
        
        # Feature 3: Volatility (sqrt of variance)
        features[:, :, 2] = torch.sqrt(v)
        
        # Feature 4: Variance change
        features[:, 1:, 3] = v[:, 1:] - v[:, :-1]
        
        # Feature 5: Time to maturity
        for t in range(n_steps):
            features[:, t, 4] = (T - t * dt) / T
        
        # Feature 6: Previous delta
        if delta_prev is not None:
            features[:, :, 5] = delta_prev
        
        # Feature 7: Trading volume (computed during training)
        # Will be filled during forward pass
        
        # Feature 8: Cumulative PnL
        if pnl_prev is not None:
            features[:, :, 7] = pnl_prev
            
    else:
       
        n_paths, n_steps = S.shape
        features = np.zeros((n_paths, n_steps, 8))
        
        for t in range(n_steps):
            # Feature 1: Log-moneyness
            features[:, t, 0] = np.log(S[:, t] / K)
            
            # Feature 2: Return
            if t > 0:
                features[:, t, 1] = (S[:, t] - S[:, t-1]) / S[:, t-1]
            
            # Feature 3: Volatility (sqrt of variance)
            features[:, t, 2] = np.sqrt(v[:, t])
            
            # Feature 4: Variance change
            if t > 0:
                features[:, t, 3] = v[:, t] - v[:, t-1]
            
            # Feature 5: Time to maturity
            features[:, t, 4] = (T - t * dt) / T
            
            # Feature 6: Previous delta
            if delta_prev is not None:
                features[:, t, 5] = delta_prev[:, t]
            
            # Feature 7: Trading volume (computed during training)
            # Will be filled during forward pass
            
            # Feature 8: Cumulative PnL
            if pnl_prev is not None:
                features[:, t, 7] = pnl_prev[:, t]
    
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
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders from numpy arrays
    
    Args:
        S_train, v_train, Z_train: Training data
        S_val, v_val, Z_val: Validation data
        S_test, v_test, Z_test: Test data
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features to zero mean and unit variance
    
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