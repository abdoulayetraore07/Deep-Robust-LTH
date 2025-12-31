"""
Deep Hedging MLP architecture
"""

import torch
import torch.nn as nn



class DeepHedgingNetwork(nn.Module):
    """
    MLP network for Deep Hedging
    
    Architecture: 8 -> 512 -> 512 -> 256 -> 1
    With BatchNorm and Dropout after each hidden layer
    """
    
    def __init__(self, config: dict):
        """
        Initialize network
        
        Args:
            config: Model configuration dictionary
        """
        super(DeepHedgingNetwork, self).__init__()
        
        self.input_dim = config['input_dim']
        self.hidden_dims = config['hidden_dims']
        self.dropout_rates = config['dropout_rates']
        self.use_batch_norm = config['use_batch_norm']
        
        # Build network
        layers = []
        
        prev_dim = self.input_dim
        for i, (hidden_dim, dropout_rate) in enumerate(zip(self.hidden_dims, self.dropout_rates)):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Learnable premium (y) - scalar parameter
        self.y = nn.Parameter(torch.tensor(0.0))
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> tuple:
        """
        Forward pass
        
        Args:
            features: (batch, n_steps, input_dim)
            
        Returns:
            delta: (batch, n_steps) - hedging positions
            y: scalar - learned premium
        """
        batch_size, n_steps, n_features = features.shape
        
        # Reshape: (batch * n_steps, n_features)
        x = features.reshape(-1, n_features)
        
        # Forward through network
        delta = self.network(x)  # (batch * n_steps, 1)
        
        # Reshape back: (batch, n_steps)
        delta = delta.reshape(batch_size, n_steps)
        
        return delta, self.y
    
    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)