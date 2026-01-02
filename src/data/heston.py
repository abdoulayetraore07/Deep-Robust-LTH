"""
Heston stochastic volatility model simulation using Inverse Gaussian scheme
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import hashlib
import json


class HestonSimulator:
    """
    Heston model simulator with Inverse Gaussian discretization scheme
    
    Under risk-neutral measure Q:
    dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW_S
    dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_v
    
    Corr(dW_S, dW_v) = rho * dt
    
    Note: For hedging, we simulate under Q (risk-neutral), so drift = r
    """
    
    def __init__(self, params: dict):
        """
        Initialize Heston simulator.
        
        Args:
            params: Dictionary with keys:
                - S_0: Initial stock price
                - v_0: Initial variance
                - kappa: Mean reversion speed
                - theta: Long-term variance
                - xi: Vol of vol
                - rho: Correlation
                - r: Risk-free rate (used for risk-neutral drift)
                - mu: Real-world drift (stored but not used in Q-simulation)
        """
        self.S_0 = params['S_0']
        self.v_0 = params['v_0']
        self.kappa = params['kappa']
        self.theta = params['theta']
        self.xi = params['xi']
        self.rho = params['rho']
        self.r = params['r']
        self.mu = params.get('mu', params['r'])  # Store but use r for simulation
        
        # Store full params for hashing
        self._params = params
        
    def simulate(
        self,
        n_paths: int,
        T: float,
        n_steps: int,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston paths using Inverse Gaussian scheme for variance.
        
        Args:
            n_paths: Number of Monte Carlo paths
            T: Time horizon in years
            n_steps: Number of time steps
            seed: Random seed for reproducibility
            
        Returns:
            S: Stock prices array (n_paths, n_steps + 1)
            v: Variance array (n_paths, n_steps + 1)
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = T / n_steps
        
        # Allocate arrays
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        
        # Initial conditions
        S[:, 0] = self.S_0
        v[:, 0] = self.v_0
        
        # Parameters for IG scheme
        a = self.kappa * self.theta
        c = self.xi
        
        # Pre-generate all random numbers
        Z_IG = np.random.randn(n_paths, n_steps)      # For IG generator
        U_IG = np.random.uniform(0, 1, (n_paths, n_steps))  # For IG generator
        Z_S = np.random.randn(n_paths, n_steps)       # For stock price
        
        # Simulate variance process using IG scheme
        for t in range(n_steps):
            v_t = v[:, t]
            
            # IG parameters
            alpha_t = v_t + a * dt
            mu_ig = alpha_t / (1.0 + self.kappa * dt)
            lambda_ig = (alpha_t ** 2) / (c ** 2 * dt)
            
            # Generate next variance using Inverse Gaussian
            v[:, t+1] = self._inverse_gaussian_generator(
                mu_ig, lambda_ig, Z_IG[:, t], U_IG[:, t]
            )
        
        # Generate correlated Brownian increments for stock price
        dW_S_all = np.zeros((n_paths, n_steps))
        for t in range(n_steps):
            dW_v_t = Z_IG[:, t] * np.sqrt(dt)
            dW_S_all[:, t] = (self.rho * dW_v_t + 
                             np.sqrt(1 - self.rho**2) * Z_S[:, t] * np.sqrt(dt))
        
        # Simulate stock price (under risk-neutral measure, drift = r)
        drift_term = (self.r - 0.5 * v[:, :-1]) * dt
        diffusion_term = np.sqrt(np.maximum(v[:, :-1], 0)) * dW_S_all
        log_returns = drift_term + diffusion_term
        S[:, 1:] = self.S_0 * np.exp(np.cumsum(log_returns, axis=1))
        
        return S, v
    
    @staticmethod
    def _inverse_gaussian_generator(
        mu: np.ndarray,
        lambda_p: np.ndarray,
        Z: np.ndarray,
        U: np.ndarray
    ) -> np.ndarray:
        """
        Generate Inverse Gaussian random variates.
        
        Args:
            mu: Mean parameter
            lambda_p: Shape parameter
            Z: Standard normal samples
            U: Uniform samples
            
        Returns:
            Inverse Gaussian samples
        """
        # Avoid division by zero
        mu = np.maximum(mu, 1e-10)
        lambda_p = np.maximum(lambda_p, 1e-10)
        
        Y = Z ** 2
        X = mu + (mu ** 2 * Y) / (2 * lambda_p) - (mu / (2 * lambda_p)) * np.sqrt(
            4 * mu * lambda_p * Y + mu ** 2 * Y ** 2
        )
        
        # Accept/reject step
        condition = U <= mu / (mu + X)
        V = np.where(condition, X, mu ** 2 / X)
        
        # Ensure positivity
        V = np.maximum(V, 1e-10)
        
        return V
    
    def compute_payoff(
        self,
        S: np.ndarray,
        K: float,
        option_type: str = "call"
    ) -> np.ndarray:
        """
        Compute option payoff at maturity.
        
        For a SHORT position (seller), we PAY the payoff.
        
        Args:
            S: Stock prices (n_paths, n_steps + 1)
            K: Strike price
            option_type: 'call' or 'put'
            
        Returns:
            Z: Payoff array (n_paths,) - positive values mean we pay
        """
        S_T = S[:, -1]  # Terminal stock price
        
        if option_type == "call":
            Z = np.maximum(S_T - K, 0)
        elif option_type == "put":
            Z = np.maximum(K - S_T, 0)
        else:
            raise ValueError(f"Unknown option type: {option_type}")
        
        return Z
    
    def validate_moments(
        self,
        v: np.ndarray,
        T: float,
        tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Validate simulated variance against theoretical moments.
        
        Args:
            v: Simulated variance paths (n_paths, n_steps + 1)
            T: Time horizon
            tolerance: Relative tolerance for validation
            
        Returns:
            Dictionary with validation results
        """
        v_T = v[:, -1]
        
        # Theoretical moments under Q
        exp_kT = np.exp(-self.kappa * T)
        
        E_v_T_theo = self.theta + (self.v_0 - self.theta) * exp_kT
        
        Var_v_T_theo = (
            self.v_0 * (self.xi ** 2 / self.kappa) * (exp_kT - np.exp(-2 * self.kappa * T))
            + self.theta * (self.xi ** 2 / (2 * self.kappa)) * (1 - exp_kT) ** 2
        )
        
        # Empirical moments
        E_v_T_emp = np.mean(v_T)
        Var_v_T_emp = np.var(v_T)
        
        # Validation
        mean_error = np.abs(E_v_T_theo - E_v_T_emp) / E_v_T_theo
        var_error = np.abs(Var_v_T_theo - Var_v_T_emp) / max(Var_v_T_theo, 1e-10)
        
        results = {
            'E_v_T_theoretical': E_v_T_theo,
            'E_v_T_empirical': E_v_T_emp,
            'E_v_T_relative_error': mean_error,
            'E_v_T_pass': mean_error < tolerance,
            'Var_v_T_theoretical': Var_v_T_theo,
            'Var_v_T_empirical': Var_v_T_emp,
            'Var_v_T_relative_error': var_error,
            'Var_v_T_pass': var_error < tolerance,
            'min_v': np.min(v),
            'positivity_pass': np.min(v) > 0,
            'all_pass': (mean_error < tolerance) and (var_error < tolerance) and (np.min(v) > 0)
        }
        
        return results
    
    def get_params_hash(self) -> str:
        """
        Get a hash of the simulator parameters for caching.
        
        Returns:
            8-character hex hash
        """
        params_str = json.dumps(self._params, sort_keys=True, default=str)
        return hashlib.sha256(params_str.encode()).hexdigest()[:8]


def generate_dataset(
    config: Dict[str, Any],
    dataset_type: str = "train",
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a complete dataset (S, v, Z) for training/validation/testing.
    
    Args:
        config: Configuration dictionary
        dataset_type: 'train', 'val', or 'test'
        save_path: Optional path to save the dataset
        
    Returns:
        S: Stock prices (n_paths, n_steps + 1)
        v: Variance (n_paths, n_steps + 1)
        Z: Payoffs (n_paths,)
    """
    # Get parameters
    heston_params = config['data']['heston']
    T = config['data']['T']
    n_steps = config['data']['n_steps']
    K = heston_params['K']
    
    # Get dataset size and seed
    if dataset_type == "train":
        n_paths = config['data']['n_train']
        seed = config['data']['seeds']['train']
    elif dataset_type == "val":
        n_paths = config['data']['n_val']
        seed = config['data']['seeds']['val']
    elif dataset_type == "test":
        n_paths = config['data']['n_test']
        seed = config['data']['seeds']['test']
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create simulator and generate data
    simulator = HestonSimulator(heston_params)
    S, v = simulator.simulate(n_paths, T, n_steps, seed=seed)
    Z = simulator.compute_payoff(S, K, option_type="call")
    
    # Validate moments
    validation = simulator.validate_moments(v, T)
    if not validation['all_pass']:
        print(f"[Heston] WARNING: Moment validation failed for {dataset_type} set")
        print(f"  E[v_T] error: {validation['E_v_T_relative_error']:.2%}")
        print(f"  Var[v_T] error: {validation['Var_v_T_relative_error']:.2%}")
        print(f"  Positivity: {validation['positivity_pass']}")
    
    # Save if path provided
    if save_path:
        save_dataset(S, v, Z, save_path)
        print(f"[Heston] Saved {dataset_type} dataset to {save_path}")
    
    return S, v, Z


def save_dataset(
    S: np.ndarray,
    v: np.ndarray,
    Z: np.ndarray,
    filepath: str
) -> None:
    """
    Save dataset to disk.
    
    Args:
        S: Stock prices
        v: Variance
        Z: Payoffs
        filepath: Path to save (without extension)
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        str(path),
        S=S,
        v=v,
        Z=Z
    )


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load dataset from disk.
    
    Args:
        filepath: Path to dataset file
        
    Returns:
        S, v, Z arrays
    """
    data = np.load(filepath)
    return data['S'], data['v'], data['Z']


def check_cached_dataset(
    config: Dict[str, Any],
    dataset_type: str,
    cache_dir: str = "data/processed"
) -> Optional[str]:
    """
    Check if a cached dataset exists for this configuration.
    
    Args:
        config: Configuration dictionary
        dataset_type: 'train', 'val', or 'test'
        cache_dir: Directory to check for cached data
        
    Returns:
        Path to cached file if exists, None otherwise
    """
    # Create hash from relevant config
    hash_config = {
        'heston': config['data']['heston'],
        'T': config['data']['T'],
        'n_steps': config['data']['n_steps'],
        f'n_{dataset_type}': config['data'].get(f'n_{dataset_type}'),
        'seed': config['data']['seeds'].get(dataset_type)
    }
    
    hash_str = json.dumps(hash_config, sort_keys=True, default=str)
    config_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:8]
    
    # Check if file exists
    cache_path = Path(cache_dir) / f"{dataset_type}_{config_hash}.npz"
    
    if cache_path.exists():
        return str(cache_path)
    
    return None


def get_or_generate_dataset(
    config: Dict[str, Any],
    dataset_type: str,
    cache_dir: str = "data/processed",
    force_regenerate: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get dataset from cache or generate if not exists.
    
    Args:
        config: Configuration dictionary
        dataset_type: 'train', 'val', or 'test'
        cache_dir: Directory for cached data
        force_regenerate: If True, regenerate even if cached
        
    Returns:
        S, v, Z arrays
    """
    if not force_regenerate:
        cached_path = check_cached_dataset(config, dataset_type, cache_dir)
        if cached_path:
            print(f"[Heston] Loading cached {dataset_type} dataset from {cached_path}")
            return load_dataset(cached_path)
    
    # Generate new dataset
    print(f"[Heston] Generating {dataset_type} dataset...")
    
    # Compute cache path
    hash_config = {
        'heston': config['data']['heston'],
        'T': config['data']['T'],
        'n_steps': config['data']['n_steps'],
        f'n_{dataset_type}': config['data'].get(f'n_{dataset_type}'),
        'seed': config['data']['seeds'].get(dataset_type)
    }
    hash_str = json.dumps(hash_config, sort_keys=True, default=str)
    config_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:8]
    cache_path = Path(cache_dir) / f"{dataset_type}_{config_hash}.npz"
    
    # Generate and save
    S, v, Z = generate_dataset(config, dataset_type, save_path=str(cache_path).replace('.npz', ''))
    
    return S, v, Z