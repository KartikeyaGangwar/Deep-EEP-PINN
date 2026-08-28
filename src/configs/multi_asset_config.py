"""
Multi-Asset Configuration for High-Dimensional American Basket Option Free-Boundary Problem.
Standard 5-Asset Benchmark with Cross-Asset Correlation Matrix.
"""

from dataclasses import dataclass, field
import numpy as np
import torch

@dataclass
class MultiAssetConfig:
    # Asset Dimensionality
    d: int = 5                        # Number of underlying correlated assets in basket
    
    # Financial Market Parameters
    K: float = 100.0                  # Strike Price
    T: float = 1.0                    # Maturity Time (Years)
    r: float = 0.05                   # Risk-free rate (5%)
    
    # Per-Asset Volatilities (Annualized)
    volatilities: list = field(default_factory=lambda: [0.20, 0.22, 0.18, 0.25, 0.20])
    
    # Basket Weights (Equal Weighted)
    weights: list = field(default_factory=lambda: [0.20, 0.20, 0.20, 0.20, 0.20])
    
    # Cross-Asset Correlation Matrix (Positive-Definite)
    # Default: Uniform pairwise correlation rho = 0.40
    rho: float = 0.40
    
    # Spatial Domain Boundaries
    S_min: float = 0.0
    S_max: float = 300.0
    
    # Monte Carlo (Longstaff-Schwartz) Benchmark Settings
    mc_paths: int = 100000            # 100,000 simulated correlated paths for ultimate statistical rigor
    mc_time_steps: int = 100          # Exercise monitoring intervals (dt = T/100)
    
    # PINN Training Settings
    hidden_layers: int = 4
    hidden_dim: int = 128             # Wider network for d=5 geometry
    num_collocation: int = 12000      # High-dimensional Latin Hypercube collocation points
    epochs_adam: int = 2500
    lr_adam: float = 1e-3
    epochs_lbfgs: int = 300
    
    def get_correlation_matrix(self) -> np.ndarray:
        corr = np.full((self.d, self.d), self.rho)
        np.fill_diagonal(corr, 1.0)
        return corr

    def get_covariance_matrix(self) -> np.ndarray:
        vols = np.array(self.volatilities)
        corr = self.get_correlation_matrix()
        return np.outer(vols, vols) * corr

default_multi_config = MultiAssetConfig()

# Institutional 10-Asset Benchmark Configuration (d=10, 45 Cross-Correlations, 100k LSM Paths)
config_10d = MultiAssetConfig(
    d=10,
    K=100.0,
    T=1.0,
    r=0.05,
    volatilities=[0.20, 0.22, 0.18, 0.25, 0.21, 0.19, 0.23, 0.24, 0.17, 0.20],
    weights=[0.10] * 10,
    rho=0.35,
    S_min=0.0,
    S_max=300.0,
    mc_paths=100000,          # 100,000 simulated correlated paths for ultimate statistical rigor
    mc_time_steps=100,        # 100 exercise monitoring intervals
    hidden_layers=4,
    hidden_dim=160,           # Scaled hidden dimension for 10D manifold
    num_collocation=8000,     # Dense 10D collocation sampling
    epochs_adam=1200,
    lr_adam=1e-3,
    epochs_lbfgs=50
)

# Institutional 30-Asset Benchmark Configuration (d=30, Dow Jones 30 Scale, 435 Cross-Correlations)
config_30d = MultiAssetConfig(
    d=30,
    K=100.0,
    T=1.0,
    r=0.05,
    volatilities=[0.18 + 0.08 * (i % 5) / 4.0 for i in range(30)],
    weights=[1.0 / 30.0] * 30,
    rho=0.30,
    S_min=0.0,
    S_max=300.0,
    mc_paths=100000,          # 100,000 simulated correlated paths
    mc_time_steps=100,
    hidden_layers=4,
    hidden_dim=192,
    num_collocation=10000,
    epochs_adam=1200,
    lr_adam=1e-3,
    epochs_lbfgs=50
)

# Institutional 50-Asset Benchmark Configuration (d=50, Nifty 50 / S&P Sector Scale, 1225 Cross-Correlations)
config_50d = MultiAssetConfig(
    d=50,
    K=100.0,
    T=1.0,
    r=0.05,
    volatilities=[0.15 + 0.12 * (i % 7) / 6.0 for i in range(50)],
    weights=[1.0 / 50.0] * 50,
    rho=0.25,
    S_min=0.0,
    S_max=300.0,
    mc_paths=100000,          # 100,000 simulated correlated paths
    mc_time_steps=100,
    hidden_layers=4,
    hidden_dim=192,
    num_collocation=6000,     # 6000 tri-modal collocation points for optimal memory
    epochs_adam=1200,
    lr_adam=1e-3,
    epochs_lbfgs=50
)
