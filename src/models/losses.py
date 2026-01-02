"""
Loss Functions for Deep Hedging

Implements the OCE (Optimized Certainty Equivalent) formulation from Buehler et al. (2019).

Key insight: The P&L is calculated WITHOUT the premium y.
The premium y only appears in the loss function, not in the P&L itself.

OCE Formulation:
    U(X) = sup_y { E[u(X + y)] - y }
    
    For CVaR with level Î±:
        u(x) = (1 + Î») * min(0, x)    where Î» = 1/Î± - 1
        
    Loss = -U(X) = y - E[u(PnL + y)]
         = y + (1/Î±) * E[max(-PnL - y, 0)]

P&L Calculation (for SHORT option position):
    PnL = Î£ Î´_{t-1} * (S_t - S_{t-1}) - Z - transaction_costs
    
    Where:
        - Î´_{t-1} * (S_t - S_{t-1}): hedging gains
        - Z: option payoff (positive for short position loss)
        - transaction_costs: proportional to |Î´_t - Î´_{t-1}|

=============================================================================
LOSS FUNCTION INTERFACE CONTRACT
=============================================================================

All loss functions must implement the following interface to be compatible
with the Trainer and AdversarialTrainer classes:

1. forward(deltas, S, Z, y, dt) -> Tuple[loss, info]
   - deltas: Hedging positions (batch, n_steps)
   - S: Stock prices (batch, n_steps)
   - Z: Option payoff at maturity (batch,)
   - y: Learned premium (scalar tensor)
   - dt: Time step size (float)
   - Returns: (loss scalar, info dict)

2. compute_pnl(deltas, S, Z, dt) -> pnl
   - Same inputs as forward (minus y)
   - Returns: P&L for each path (batch,)

3. info dict REQUIRED fields:
   - 'pnl_mean': Mean P&L across batch
   - 'pnl_std': Std of P&L across batch
   - 'premium_y': The premium parameter y

4. info dict OPTIONAL fields (recommended for consistency):
   - 'pnl_min': Minimum P&L
   - 'pnl_max': Maximum P&L
   - 'expected_shortfall': (OCELoss specific)
   - 'cvar': (OCELoss specific)

Note: The Trainer computes CVaR for monitoring using config['training']['cvar_alpha'],
NOT from the loss function. This allows consistent monitoring across all loss types.
=============================================================================
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class OCELoss(nn.Module):
    """
    Optimized Certainty Equivalent Loss for Deep Hedging.
    
    This implements the correct OCE formulation where:
    1. P&L is computed WITHOUT the premium y
    2. Loss = y + (1/Î±) * E[max(-PnL - y, 0)]
    
    The network learns both the hedging strategy AND the optimal premium.
    """
    
    def __init__(
        self, 
        alpha: float = 0.05,
        transaction_cost: float = 0.0,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize OCE Loss.
        
        Args:
            alpha: CVaR level (e.g., 0.05 for CVaR_5%)
            transaction_cost: Proportional transaction cost
            risk_free_rate: Risk-free rate for discounting
        """
        super(OCELoss, self).__init__()
        
        self.alpha = alpha
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        
        # Lambda for CVaR utility: Î» = 1/Î± - 1
        self.lam = 1.0 / alpha - 1.0
    
    def compute_pnl(
        self,
        deltas: torch.Tensor,
        S: torch.Tensor,
        Z: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """
        Compute P&L for SHORT option position (WITHOUT premium y).
        
        PnL = hedging_gains - option_payoff - transaction_costs
            = Î£ Î´_{t-1} * (S_t - S_{t-1}) - Z - TC
        
        Args:
            deltas: Hedging positions (batch, n_steps)
            S: Stock prices (batch, n_steps)
            Z: Option payoff at maturity (batch,) - POSITIVE for short position loss
            dt: Time step size
            
        Returns:
            pnl: P&L for each path (batch,) - WITHOUT premium y
        """
        batch_size, n_steps = deltas.shape
        device = deltas.device
        dtype = deltas.dtype
        
        # Stock price changes: dS_t = S_t - S_{t-1}
        dS = S[:, 1:] - S[:, :-1]  # (batch, n_steps - 1)
        
        # Hedging gains: Î£ Î´_{t-1} * dS_t
        # Use delta from previous step (delta_{t-1}) for gain at step t
        delta_prev = deltas[:, :-1]  # (batch, n_steps - 1)
        hedging_gains = (delta_prev * dS).sum(dim=1)  # (batch,)
        
        # Transaction costs: Î£ |Î´_t - Î´_{t-1}| * cost * S_t
        if self.transaction_cost > 0:
            # Initial trade: |Î´_0 - 0| = |Î´_0|
            initial_trade = torch.abs(deltas[:, 0]) * S[:, 0]
            
            # Subsequent trades: |Î´_t - Î´_{t-1}|
            delta_changes = torch.abs(deltas[:, 1:] - deltas[:, :-1])
            subsequent_trades = (delta_changes * S[:, 1:]).sum(dim=1)
            
            # Final unwind: |0 - Î´_{T-1}| = |Î´_{T-1}|
            final_trade = torch.abs(deltas[:, -1]) * S[:, -1]
            
            total_tc = self.transaction_cost * (initial_trade + subsequent_trades + final_trade)
        else:
            total_tc = torch.zeros(batch_size, device=device, dtype=dtype)
        
        # P&L = hedging_gains - payoff - transaction_costs
        # Z is POSITIVE when option is exercised (loss for short position)
        pnl = hedging_gains - Z - total_tc
        
        return pnl
    
    def cvar_utility(self, x: torch.Tensor) -> torch.Tensor:
        """
        CVaR utility function: u(x) = (1 + Î») * min(0, x)
        
        Args:
            x: Input values (batch,)
            
        Returns:
            u(x): Utility values (batch,)
        """
        return (1.0 + self.lam) * torch.clamp(x, max=0.0)
    
    def forward(
        self,
        deltas: torch.Tensor,
        S: torch.Tensor,
        Z: torch.Tensor,
        y: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute OCE Loss.
        
        Loss = y - E[u(PnL + y)]
             = y + (1/Î±) * E[max(-PnL - y, 0)]
        
        Args:
            deltas: Hedging positions (batch, n_steps)
            S: Stock prices (batch, n_steps)
            Z: Option payoff at maturity (batch,)
            y: Learned premium (scalar)
            dt: Time step size
            
        Returns:
            loss: Scalar loss value
            info: Dictionary with diagnostic values
        """
        # Compute P&L WITHOUT premium
        pnl = self.compute_pnl(deltas, S, Z, dt)
        
        # OCE Loss = y + (1/Î±) * E[max(-PnL - y, 0)]
        # Equivalent to: y - E[u(PnL + y)] where u(x) = (1+Î»)*min(0,x)
        shortfall = torch.clamp(-pnl - y, min=0.0)  # max(-PnL - y, 0)
        expected_shortfall = shortfall.mean()
        
        loss = y + (1.0 / self.alpha) * expected_shortfall
        
        # Compute diagnostic info
        with torch.no_grad():
            info = {
                'pnl_mean': pnl.mean(),
                'pnl_std': pnl.std(),
                'pnl_min': pnl.min(),
                'pnl_max': pnl.max(),
                'premium_y': y,
                'expected_shortfall': expected_shortfall,
                'cvar': -y.item(),  # CVaR â‰ˆ -y at optimum
            }
        
        return loss, info


class EntropicRiskLoss(nn.Module):
    """
    Entropic Risk Measure Loss (alternative to CVaR).
    
    u(x) = 1 - exp(-Î»x)
    
    Loss = (1/Î») * log(E[exp(-Î» * (PnL + y))]) - y
    """
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        transaction_cost: float = 0.0
    ):
        """
        Initialize Entropic Risk Loss.
        
        Args:
            risk_aversion: Risk aversion parameter Î»
            transaction_cost: Proportional transaction cost
        """
        super(EntropicRiskLoss, self).__init__()
        
        self.lam = risk_aversion
        self.transaction_cost = transaction_cost
    
    def compute_pnl(
        self,
        deltas: torch.Tensor,
        S: torch.Tensor,
        Z: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """Compute P&L (same as OCELoss)."""
        batch_size, n_steps = deltas.shape
        device = deltas.device
        dtype = deltas.dtype
        
        dS = S[:, 1:] - S[:, :-1]
        delta_prev = deltas[:, :-1]
        hedging_gains = (delta_prev * dS).sum(dim=1)
        
        if self.transaction_cost > 0:
            initial_trade = torch.abs(deltas[:, 0]) * S[:, 0]
            delta_changes = torch.abs(deltas[:, 1:] - deltas[:, :-1])
            subsequent_trades = (delta_changes * S[:, 1:]).sum(dim=1)
            final_trade = torch.abs(deltas[:, -1]) * S[:, -1]
            total_tc = self.transaction_cost * (initial_trade + subsequent_trades + final_trade)
        else:
            total_tc = torch.zeros(batch_size, device=device, dtype=dtype)
        
        pnl = hedging_gains - Z - total_tc
        return pnl
    
    def forward(
        self,
        deltas: torch.Tensor,
        S: torch.Tensor,
        Z: torch.Tensor,
        y: torch.Tensor,
        dt: float
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute Entropic Risk Loss.
        
        Loss = (1/Î») * log(E[exp(-Î» * (PnL + y))]) - y
        """
        pnl = self.compute_pnl(deltas, S, Z, dt)
        
        # Numerically stable log-sum-exp
        x = -self.lam * (pnl + y)
        max_x = x.max()
        loss = (1.0 / self.lam) * (max_x + torch.log(torch.exp(x - max_x).mean())) - y
        
        with torch.no_grad():
            info = {
                'pnl_mean': pnl.mean(),
                'pnl_std': pnl.std(),
                'pnl_min': pnl.min(),
                'pnl_max': pnl.max(),
                'premium_y': y,
            }
        
        return loss, info


def create_loss_function(config: Dict) -> nn.Module:
    """
    Factory function to create loss function.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Loss function module
    """
    training_config = config.get('training', {})
    loss_type = training_config.get('loss_type', 'cvar')
    
    if loss_type == 'cvar':
        alpha = training_config.get('cvar_alpha', 0.05)
        tc = config.get('data', {}).get('transaction_cost', {}).get('c_prop', 0.0)
        rf = config['data']['heston'].get('r', 0.0)
        
        loss_fn = OCELoss(
            alpha=alpha,
            transaction_cost=tc,
            risk_free_rate=rf
        )
        print(f"[Loss] Created OCELoss (CVaR Î±={alpha}, TC={tc})")
        
    elif loss_type == 'entropic':
        lam = training_config.get('risk_aversion', 1.0)
        tc = config.get('data', {}).get('transaction_cost', {}).get('c_prop', 0.0)
        
        loss_fn = EntropicRiskLoss(
            risk_aversion=lam,
            transaction_cost=tc
        )
        print(f"[Loss] Created EntropicRiskLoss (Î»={lam}, TC={tc})")
    
    return loss_fn

