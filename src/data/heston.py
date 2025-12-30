"""
Heston stochastic volatility model simulation using Inverse Gaussian scheme
"""

import numpy as np
from typing import Tuple, Optional


class HestonSimulator:
    """
    Heston model simulator with Inverse Gaussian discretization scheme
    
    dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW_S   sous Q (risk-neutral measure)
    dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_v
    
    Corr(dW_S, dW_v) = rho * dt
    """
    
    def __init__(self, params: dict):
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
        
       
        Z_IG = np.random.randn(n_paths, n_steps)    # Pour générateur IG
        U_IG = np.random.uniform(0, 1, (n_paths, n_steps))  # Pour générateur IG
        Z_S = np.random.randn(n_paths, n_steps)     # Pour prix S
        
        # Simulate variance process using IG scheme
        for t in range(n_steps):
            v_t = v[:, t]
            
            # IG parameters
            alpha_t = v_t + a * dt
            mu_ig = alpha_t / (1.0 + self.kappa * dt)
            lambda_ig = (alpha_t ** 2) / (c ** 2 * dt)
            
            
            v[:, t+1] = self._inverse_gaussian_generator(mu_ig, lambda_ig, Z_IG[:, t], U_IG[:, t])
        
       
        # Brownien corrélé (on ne peut pas le précalculer car on a besoin de v)
        dW_S_all = np.zeros((n_paths, n_steps))
        for t in range(n_steps):
            # Composante corrélée
            dW_v_t = Z_IG[:, t] * np.sqrt(dt)  # Utilise Z_IG pour cohérence
            # Composante indépendante
            dW_S_all[:, t] = self.rho * dW_v_t + np.sqrt(1 - self.rho**2) * Z_S[:, t] * np.sqrt(dt)
        
        # Prix S
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
        v_T = v[:, -1]
        
        # Theoretical moments
        E_v_T_theo = self.theta + (self.v_0 - self.theta) * np.exp(-self.kappa * T)
        
        Var_v_T_theo = self.v_0 * (self.xi ** 2 / self.kappa) * (
            np.exp(-self.kappa * T) - np.exp(-2 * self.kappa * T)
        ) + self.theta * (self.xi ** 2 / (2 * self.kappa)) * (1 - np.exp(-self.kappa * T)) ** 2
        
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