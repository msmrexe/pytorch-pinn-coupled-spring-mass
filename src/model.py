"""
Defines the PINN model architecture.
"""

import torch
import torch.nn as nn
import logging

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) model.

    A simple feed-forward neural network that maps time (t) to the
    positions of N masses (x_1(t), ..., x_N(t)).
    """
    def __init__(self, num_masses: int, num_hidden_layers: int = 4, num_neurons: int = 64):
        """
        Initializes the PINN model.

        Args:
            num_masses (int): Number of masses (N), which is the output dimension.
            num_hidden_layers (int): Number of hidden layers.
            num_neurons (int): Number of neurons in each hidden layer.
        """
        super(PINN, self).__init__()
        
        layers = []

        # Input layer (from t to first hidden layer)
        layers.append(nn.Linear(1, num_neurons))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.Tanh())

        # Output layer (from last hidden to num_masses outputs)
        layers.append(nn.Linear(num_neurons, num_masses))

        self.model = nn.Sequential(*layers)
        
        logging.info(f"PINN model initialized with {num_hidden_layers} hidden layers and {num_neurons} neurons per layer.")
        logging.info(f"Input dim: 1 (time), Output dim: {num_masses} (positions)")

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            t (torch.Tensor): Input tensor of time points, shape [batch_size, 1].

        Returns:
            torch.Tensor: Predicted positions of the masses, shape [batch_size, num_masses].
        """
        return self.model(t)
