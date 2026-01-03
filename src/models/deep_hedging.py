"""
Deep Hedging Network Architecture

This implements the Deep Hedging agent from Buehler et al. (2019).
The agent takes market features and previous position as input,
and outputs the new hedging position.

Key insight: The network must see delta_{t-1} to decide delta_t,
which requires an explicit temporal loop in the forward pass.

Input features (8 total):
    Exogenous (5): log_moneyness, return, volatility, variance_change, time_to_maturity
    Recurrent (3): delta_prev, pnl_cumulative, trading_volume
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


# Feature dimensions
N_EXOGENOUS_FEATURES = 5
N_RECURRENT_FEATURES = 3
N_TOTAL_FEATURES = N_EXOGENOUS_FEATURES + N_RECURRENT_FEATURES  # 8


class DeepHedgingNetwork(nn.Module):
    """
    Deep Hedging MLP Network with temporal loop.
    
    At each time step t, the network:
    1. Receives exogenous features (5) + recurrent features (3) = 8 inputs
    2. Outputs the new delta position
    
    Recurrent features (computed in forward loop):
        - delta_prev: previous hedging position
        - pnl_cumulative: cumulative P&L up to t-1
        - trading_volume: cumulative |delta_i - delta_{i-1}|
    
    Architecture: 8 -> hidden_dims -> 1
    
    The learnable premium y is a separate parameter optimized
    via the OCE (Optimized Certainty Equivalent) formulation.
    """
    
    def __init__(self, config: dict):
        """
        Initialize network.
        
        Args:
            config: Model configuration dictionary with keys:
                - input_dim: Number of exogenous features (default: 5)
                - hidden_dims: List of hidden layer sizes
                - dropout_rates: List of dropout rates per layer
                - use_batch_norm: Whether to use batch normalization
                - activation: Activation function name
        """
        super(DeepHedgingNetwork, self).__init__()
        
        # Store config
        self.exogenous_dim = config.get('input_dim', N_EXOGENOUS_FEATURES)
        self.recurrent_dim = config.get('recurrent_dim', N_RECURRENT_FEATURES)
        self.hidden_dims = config['hidden_dims']
        self.dropout_rates = config.get('dropout_rates', [0.0] * len(self.hidden_dims))
        self.use_batch_norm = config.get('use_batch_norm', False)
        self.config = config  # Store config for later use
        activation_name = config.get('activation', 'relu')
        
        # Total input dim = exogenous features + recurrent features
        self.input_dim = self.exogenous_dim + self.recurrent_dim
        
        # Get activation function
        self.activation_fn = self._get_activation(activation_name)
        
        # Build network layers
        self.layers = self._build_network()
        
        # Learnable premium (y) - scalar parameter for OCE formulation
        # Initialized to 0, will be learned during training
        self.y = nn.Parameter(torch.tensor(0.0))
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
            'gelu': nn.GELU()
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name.lower()]
    
    def _build_network(self) -> nn.Sequential:
        """Build the MLP layers."""
        layers = []
        
        prev_dim = self.input_dim
        for i, (hidden_dim, dropout_rate) in enumerate(zip(self.hidden_dims, self.dropout_rates)):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization (optional)
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(self._get_activation(self.config.get('activation', 'relu')))
            
            # Dropout (optional)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation - delta can be any value)
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        features: torch.Tensor,
        S: torch.Tensor,
        delta_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with explicit temporal loop.
        
        At each time step t:
        1. Build recurrent features from previous states
        2. Concatenate [exogenous_features_t (5), recurrent_features (3)]
        3. Pass through network to get delta_t
        4. Update recurrent states
        
        Args:
            features: Exogenous market features (batch, n_steps, exogenous_dim)
            S: Stock prices (batch, n_steps) - needed for P&L calculation
            delta_init: Initial delta position (batch,). Default: zeros
            
        Returns:
            deltas: Hedging positions (batch, n_steps)
            y: Learned premium (scalar)
        """
        batch_size, n_steps, n_features = features.shape
        device = features.device
        dtype = features.dtype
        
        # Validate input dimension
        assert n_features == self.exogenous_dim, \
            f"Expected {self.exogenous_dim} exogenous features, got {n_features}"
        
        # Initialize recurrent states
        if delta_init is None:
            delta_prev = torch.zeros(batch_size, device=device, dtype=dtype)
        else:
            delta_prev = delta_init
        
        pnl_cumulative = torch.zeros(batch_size, device=device, dtype=dtype)
        trading_volume = torch.zeros(batch_size, device=device, dtype=dtype)
        
        # Storage for all deltas
        deltas = torch.zeros(batch_size, n_steps, device=device, dtype=dtype)
        
        # Temporal loop
        for t in range(n_steps):
            # Get exogenous features at time t
            feat_t = features[:, t, :]  # (batch, exogenous_dim)
            
            # Build recurrent features tensor
            recurrent = torch.stack([
                delta_prev,        # Previous position
                pnl_cumulative,    # Cumulative P&L
                trading_volume     # Cumulative trading volume
            ], dim=1)  # (batch, 3)
            
            # Concatenate: [exogenous (5), recurrent (3)] = 8
            network_input = torch.cat([feat_t, recurrent], dim=1)  # (batch, input_dim)
            
            # Forward through network
            delta_t = self.layers(network_input).squeeze(-1)  # (batch,)
            
            # Store delta
            deltas[:, t] = delta_t
            
            # Update recurrent states for next step
            if t > 0:
                # P&L from this step: delta_{t-1} * (S_t - S_{t-1})
                dS = S[:, t] - S[:, t-1]
                pnl_step = delta_prev * dS
                pnl_cumulative = pnl_cumulative + pnl_step
            
            # Trading volume: |delta_t - delta_{t-1}|
            trade = torch.abs(delta_t - delta_prev)
            trading_volume = trading_volume + trade
            
            # Update delta_prev
            delta_prev = delta_t
        
        return deltas, self.y
    
    def forward_no_recurrent(
        self, 
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass WITHOUT recurrent features (for ablation study).
        
        WARNING: This version does NOT use delta_prev, pnl, or volume.
        The agent is "blind" to its previous actions and state.
        Use only for ablation studies to show importance of recurrent features.
        
        Args:
            features: Exogenous features (batch, n_steps, exogenous_dim)
            
        Returns:
            deltas: (batch, n_steps)
            y: scalar
        """
        batch_size, n_steps, n_features = features.shape
        device = features.device
        dtype = features.dtype
        
        # Add zeros for recurrent features (ablation: agent is "blind")
        zeros = torch.zeros(batch_size, n_steps, self.recurrent_dim, device=device, dtype=dtype)
        full_features = torch.cat([features, zeros], dim=-1)  # (batch, n_steps, input_dim)
        
        # Reshape for batch processing
        x = full_features.reshape(-1, self.input_dim)  # (batch * n_steps, input_dim)
        
        # Forward through network
        delta = self.layers(x)  # (batch * n_steps, 1)
        
        # Reshape back
        delta = delta.reshape(batch_size, n_steps)
        
        return delta, self.y
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_weights(self) -> int:
        """Get number of weight parameters (excluding biases and y)."""
        count = 0
        for name, p in self.named_parameters():
            if 'weight' in name and p.requires_grad:
                count += p.numel()
        return count
    
    def get_sparsity(self) -> float:
        """Compute current sparsity (fraction of zero weights)."""
        total = 0
        zeros = 0
        for name, p in self.named_parameters():
            if 'weight' in name:
                total += p.numel()
                zeros += (p.data == 0).sum().item()
        return zeros / total if total > 0 else 0.0
    
    def summary(self) -> str:
        """Get a summary string of the model architecture."""
        lines = [
            "DeepHedgingNetwork Summary",
            "=" * 50,
            f"Exogenous features: {self.exogenous_dim}",
            f"Recurrent features: {self.recurrent_dim}",
            f"Total input dim: {self.input_dim}",
            f"Hidden dims: {self.hidden_dims}",
            f"Use batch norm: {self.use_batch_norm}",
            f"Dropout rates: {self.dropout_rates}",
            f"Total parameters: {self.get_num_parameters():,}",
            f"Weight parameters: {self.get_num_weights():,}",
            f"Current sparsity: {self.get_sparsity():.2%}",
            f"Learned premium y: {self.y.item():.6f}",
            "=" * 50
        ]
        return "\n".join(lines)


def create_model(config: Dict) -> DeepHedgingNetwork:
    """
    Factory function to create a DeepHedgingNetwork.
    
    Args:
        config: Full configuration dictionary
        
    Returns:
        Initialized model
    """
    model_config = config['model']
    model = DeepHedgingNetwork(model_config)
    
    print(f"[Model] Created DeepHedgingNetwork")
    print(f"        Input: {model.exogenous_dim} exogenous + {model.recurrent_dim} recurrent = {model.input_dim} total")
    print(f"        Parameters: {model.get_num_parameters():,}")
    
    return model