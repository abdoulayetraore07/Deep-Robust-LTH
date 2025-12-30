"""
Heston stochastic volatility model simulation using Inverse Gaussian scheme
"""

import numpy as np
from typing import Tuple, Optional


class HestonSimulator:
    """
    Heston model simulator with Inverse Gaussian discretization scheme
    
    dS_t = mu * S_t * dt + sqrt(v_t) * S_t * dW_S
    dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_v
    
    Corr(dW_S, dW_v) = rho * dt
    """
    
    def __init__(self, params: dict):
        """
        Initialize Heston simulator
        
        Args:
            params: Dictionary with Heston parameters
                - S_0: Initial stock price
                - v_0: Initial variance
                - kappa: Mean reversion speed
                - theta: Long-term variance
                - xi: Volatility of volatility
                - rho: Correlation between S and v
                - mu: Drift
                - r: Risk-free rate
        """
        self.S_0 = params['S_0']
        self.v_0 = params['v_0']
        self.kappa = params['kappa']
        self.theta = params['theta']
        self.xi = params['xi']
        self.rho = params['rho']
        self.mu = params['mu']
        self.r = params['r']
        
    def simulate(
        self,
        n_paths: int,
        T: float,
        n_steps: int,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston paths using Inverse Gaussian scheme
        
        Args:
            n_paths: Number of sample paths
            T: Time horizon (in years)
            n_steps: Number of time steps
            seed: Random seed for reproducibility
            
        Returns:
            S: Stock price paths (n_paths, n_steps+1)
            v: Variance paths (n_paths, n_steps+1)
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
        b = -self.kappa
        c = self.xi
        
        # Generate random numbers (all at once for speed)
        Z_v = np.random.randn(n_paths, n_steps)  # For variance process
        U = np.random.uniform(0, 1, (n_paths, n_steps))  # For IG generator
        Z_S = np.random.randn(n_paths, n_steps)  # Independent noise for S
        
        # Simulate variance process using IG scheme
        for t in range(n_steps):
            v_t = v[:, t]
            
            # IG parameters
            alpha_t = v_t + a * dt
            mu_ig = alpha_t / (1.0 + self.kappa * dt)
            lambda_ig = (alpha_t ** 2) / (c ** 2 * dt)
            
            # Generate v_{t+1} from Inverse Gaussian
            v[:, t+1] = self._inverse_gaussian_generator(mu_ig, lambda_ig, Z_v[:, t], U[:, t])
        
        # Simulate stock price process
        for t in range(n_steps):
            v_t = v[:, t]
            
            # Correlated Brownian motion
            dW_v = Z_v[:, t] * np.sqrt(dt)
            dW_S = self.rho * dW_v + np.sqrt(1 - self.rho**2) * Z_S[:, t] * np.sqrt(dt)
            
            # Stock price update
            S[:, t+1] = S[:, t] * np.exp(
                (self.r - 0.5 * v_t) * dt + np.sqrt(np.maximum(v_t, 0)) * dW_S
            )
        
        return S, v
    
    @staticmethod
    def _inverse_gaussian_generator(
        mu: np.ndarray,
        lambda_p: np.ndarray,
        Z: np.ndarray,
        U: np.ndarray
    ) -> np.ndarray:
        """
        Generate samples from Inverse Gaussian distribution
        
        Michael, Schucany & Haas algorithm
        
        Args:
            mu: Mean parameter
            lambda_p: Shape parameter
            Z: Standard normal random variables
            U: Uniform random variables
            
        Returns:
            Samples from IG(mu, lambda_p)
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
    
    def validate_moments(
        self,
        v: np.ndarray,
        T: float,
        tolerance: float = 0.01
    ) -> dict:
        """
        Validate simulated variance against theoretical moments
        
        Args:
            v: Simulated variance paths (n_paths, n_steps+1)
            T: Time horizon
            tolerance: Tolerance for moment matching
            
        Returns:
            Dictionary with validation results
        """
        v_T = v[:, -1]
        
        # Theoretical moments
        E_v_T_theo = self.theta + (self.v_0 - self.theta) * np.exp(-self.kappa * T)
        
        Var_v_T_theo = (self.xi ** 2 / (2 * self.kappa)) * (self.v_0 - self.theta) * (
            np.exp(-self.kappa * T) - np.exp(-2 * self.kappa * T)
        ) + (self.xi ** 2 * self.theta / (2 * self.kappa)) * (1 - np.exp(-self.kappa * T)) ** 2
        
        # Empirical moments
        E_v_T_emp = np.mean(v_T)
        Var_v_T_emp = np.var(v_T)
        
        # Validation
        mean_error = np.abs(E_v_T_theo - E_v_T_emp)
        var_error = np.abs(Var_v_T_theo - Var_v_T_emp)
        
        mean_pass = mean_error < tolerance * E_v_T_theo
        var_pass = var_error < tolerance * Var_v_T_theo
        
        results = {
            'E_v_T_theoretical': E_v_T_theo,
            'E_v_T_empirical': E_v_T_emp,
            'E_v_T_error': mean_error,
            'E_v_T_pass': mean_pass,
            'Var_v_T_theoretical': Var_v_T_theo,
            'Var_v_T_empirical': Var_v_T_emp,
            'Var_v_T_error': var_error,
            'Var_v_T_pass': var_pass,
            'min_v': np.min(v),
            'positivity_pass': np.min(v) > 0,
        }
        
        return results