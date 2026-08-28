"""
Utility functions for American Option PINN and Numerical Benchmarking.
Handles domain sampling, tensor conversions, and mesh generation.
"""

import numpy as np
import torch
from src.configs.default_config import OptionConfig, default_config

def generate_pinn_training_data(config: OptionConfig = default_config, device: str = "cpu"):
    """
    Generates interior, boundary, and initial/terminal collocation points for PINN training.
    """
    K = config.K
    T = config.T
    S_max = config.S_max
    
    # 1. Interior / Continuation Collocation Points (S, t)
    # Use stratified sampling to ensure dense coverage near strike K and free boundary
    N_c = config.num_collocation
    
    # 60% points concentrated in high-curvature zone [0.2*K, 1.8*K]
    N_dense = int(0.6 * N_c)
    N_wide = N_c - N_dense
    
    S_dense = np.random.uniform(0.2 * K, 1.8 * K, (N_dense, 1))
    t_dense = np.random.uniform(0.0, T, (N_dense, 1))
    
    S_wide = np.random.uniform(0.0, S_max, (N_wide, 1))
    t_wide = np.random.uniform(0.0, T, (N_wide, 1))
    
    S_int = np.vstack([S_dense, S_wide])
    t_int = np.vstack([t_dense, t_wide])
    
    # 2. Terminal Conditions at t = T (Payoff data)
    N_ic = config.num_initial
    S_ic = np.random.uniform(0.0, S_max, (N_ic, 1))
    t_ic = np.full((N_ic, 1), T)
    V_ic = np.maximum(K - S_ic, 0.0)
    
    # 3. Boundary Conditions
    N_bc = config.num_boundary
    # Left boundary S = 0 -> For American put V(0, t) = K (immediate exercise)
    t_bc_left = np.random.uniform(0.0, T, (N_bc // 2, 1))
    S_bc_left = np.zeros((N_bc // 2, 1))
    V_bc_left = np.full_like(t_bc_left, K)
    
    # Right boundary S = S_max -> V(S_max, t) = 0
    t_bc_right = np.random.uniform(0.0, T, (N_bc // 2, 1))
    S_bc_right = np.full((N_bc // 2, 1), S_max)
    V_bc_right = np.zeros((N_bc // 2, 1))
    
    S_bc = np.vstack([S_bc_left, S_bc_right])
    t_bc = np.vstack([t_bc_left, t_bc_right])
    V_bc = np.vstack([V_bc_left, V_bc_right])
    
    # Convert to PyTorch Tensors with gradient tracking for interior points
    tensors = {
        "S_int": torch.tensor(S_int, dtype=torch.float32, requires_grad=True, device=device),
        "t_int": torch.tensor(t_int, dtype=torch.float32, requires_grad=True, device=device),
        "S_ic": torch.tensor(S_ic, dtype=torch.float32, device=device),
        "t_ic": torch.tensor(t_ic, dtype=torch.float32, device=device),
        "V_ic": torch.tensor(V_ic, dtype=torch.float32, device=device),
        "S_bc": torch.tensor(S_bc, dtype=torch.float32, device=device),
        "t_bc": torch.tensor(t_bc, dtype=torch.float32, device=device),
        "V_bc": torch.tensor(V_bc, dtype=torch.float32, device=device),
    }
    
    return tensors
