"""
Default Configuration for 1D American Option Free-Boundary Problem.
Standard benchmark financial parameters and discretization settings.
"""

from dataclasses import dataclass

@dataclass
class OptionConfig:
    # Financial Market Parameters
    K: float = 100.0          # Strike Price (guaranteed selling threshold)
    T: float = 1.0            # Expiry / Maturity time in years
    r: float = 0.05           # Risk-free annual bank interest rate (5%)
    sigma: float = 0.20       # Annual volatility of underlying asset (20%)
    
    # Spatial Domain [0, S_max]
    S_min: float = 0.0        # Minimum stock price
    S_max: float = 300.0      # Maximum stock price (3x Strike)
    
    # Numerical Grid Settings for Finite Difference (PSOR)
    M_space: int = 300        # Number of spatial grid intervals (S)
    N_time: int = 300         # Number of time steps (t)
    omega_psor: float = 1.25  # Over-relaxation parameter for PSOR
    tol_psor: float = 1e-7    # Convergence tolerance for PSOR
    max_iter_psor: int = 1000 # Max iterations per time step
    
    # PINN Architecture & Training Settings
    hidden_layers: int = 4
    hidden_dim: int = 64
    activation: str = "tanh"  # smooth C^infty activation for clean derivatives
    num_collocation: int = 8000
    num_boundary: int = 1000
    num_initial: int = 1000
    epochs_adam: int = 2000
    lr_adam: float = 1e-3
    epochs_lbfgs: int = 300   # L-BFGS for fine optimization convergence
    
    # Loss Weights
    w_pde: float = 1.0
    w_early: float = 10.0     # Penalty weight for V >= h(S)
    w_bc: float = 2.0
    w_ic: float = 5.0

default_config = OptionConfig()
