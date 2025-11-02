"""
Data generation module for the PINN spring-mass system.
"""

import numpy as np
import torch
import logging

def generate_training_data(N: int, T: float, x0: float, 
                             num_physics: int, num_ic: int, num_val: int, 
                             device: torch.device):
    """
    Generates training and validation data for the PINN.

    Args:
        N (int): Number of masses.
        T (float): Total time.
        x0 (float): Initial displacement of the first mass.
        num_physics (int): Number of collocation points for physics loss.
        num_ic (int): Number of points for initial condition loss (at t=0).
        num_val (int): Number of validation points.
        device (torch.device): The device to move tensors to (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: A tuple containing:
            - t_physics (torch.Tensor): Collocation points for physics loss.
            - t_ic (torch.Tensor): Points for initial condition loss (all t=0).
            - ic_data (dict): Dictionary with initial positions and velocities.
            - t_val (torch.Tensor): Validation time points.
    """
    logging.info(f"Generating training data: {num_physics} physics points, {num_ic} IC points, {num_val} validation points.")
    
    # 1. Physics collocation points (t > 0)
    # We sample uniformly from (0, T]
    t_physics = np.random.uniform(low=0.0, high=T, size=(num_physics, 1))
    t_physics_tensor = torch.tensor(t_physics, dtype=torch.float32, requires_grad=True).to(device)

    # 2. Initial condition points (t = 0)
    t_ic = np.zeros((num_ic, 1))
    t_ic_tensor = torch.tensor(t_ic, dtype=torch.float32, requires_grad=True).to(device)
    
    # Define initial conditions
    # Initial positions: -x0 for mass 1, 0 for all others
    ic_pos = np.zeros((num_ic, N))
    ic_pos[:, 0] = -x0
    
    # Initial velocities: 0 for all masses
    ic_vel = np.zeros((num_ic, N))
    
    ic_data = {
        'pos': torch.tensor(ic_pos, dtype=torch.float32).to(device),
        'vel': torch.tensor(ic_vel, dtype=torch.float32).to(device)
    }

    # 3. Validation points
    t_val = np.random.uniform(low=0.0, high=T, size=(num_val, 1))
    t_val_tensor = torch.tensor(t_val, dtype=torch.float32, requires_grad=True).to(device)

    logging.info("Data generation complete.")
    return t_physics_tensor, t_ic_tensor, ic_data, t_val_tensor
