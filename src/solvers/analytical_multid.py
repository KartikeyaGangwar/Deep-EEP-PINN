"""
Analytical Closed-Form Solutions for Multi-Asset European Basket Options.
Uses the Geometric Basket Decomposition property (product of lognormals is lognormal).
"""

import numpy as np
import torch
from scipy.stats import norm
from src.configs.multi_asset_config import MultiAssetConfig, default_multi_config

def compute_effective_basket_params(cfg: MultiAssetConfig):
    """
    Computes effective volatility sigma_G and dividend drift correction q_G
    for a d-dimensional geometric basket option.
    """
    w = np.array(cfg.weights)
    cov = cfg.get_covariance_matrix()
    
    # sigma_G^2 = w^T * Cov * w
    sigma_G_sq = float(w @ cov @ w)
    sigma_G = np.sqrt(max(sigma_G_sq, 1e-8))
    
    # q_G = 0.5 * sum(w_i * sigma_i^2) - 0.5 * sigma_G^2
    vols = np.array(cfg.volatilities)
    weighted_vol_sq = np.sum(w * (vols ** 2))
    q_G = float(0.5 * weighted_vol_sq - 0.5 * sigma_G_sq)
    
    return sigma_G, q_G

def multi_asset_geometric_european_put(S_matrix: np.ndarray, t_array: np.ndarray, cfg: MultiAssetConfig = default_multi_config) -> np.ndarray:
    """
    Evaluates exact closed-form European geometric basket put option price in NumPy.
    
    Parameters:
        S_matrix: (N, d) array of asset prices
        t_array: (N,) array of time points
    """
    S_matrix = np.asarray(S_matrix, dtype=float)
    t_array = np.asarray(t_array, dtype=float)
    w = np.array(cfg.weights)
    
    # Geometric mean: G(S) = prod(S_i^w_i) = exp(sum(w_i * ln(S_i)))
    log_S = np.log(np.maximum(S_matrix, 1e-12))
    G = np.exp(log_S @ w)
    
    sigma_G, q_G = compute_effective_basket_params(cfg)
    tau = np.maximum(cfg.T - t_array, 1e-12)
    
    d1 = (np.log(np.maximum(G, 1e-12) / cfg.K) + (cfg.r - q_G + 0.5 * sigma_G**2) * tau) / (sigma_G * np.sqrt(tau))
    d2 = d1 - sigma_G * np.sqrt(tau)
    
    put_price = cfg.K * np.exp(-cfg.r * tau) * norm.cdf(-d2) - G * np.exp(-q_G * tau) * norm.cdf(-d1)
    
    # Handle at maturity tau == 0
    payoff = np.maximum(cfg.K - G, 0.0)
    put_price = np.where(tau <= 1e-10, payoff, put_price)
    
    return np.maximum(put_price, 0.0)

def torch_multi_asset_geometric_put(S_tensor: torch.Tensor, t_tensor: torch.Tensor, cfg: MultiAssetConfig) -> torch.Tensor:
    """
    Differentiable PyTorch implementation of the exact Multi-Asset European Basket Put price.
    S_tensor: (N, d)
    t_tensor: (N, 1)
    """
    device = S_tensor.device
    w = torch.tensor(cfg.weights, dtype=torch.float32, device=device).unsqueeze(1) # (d, 1)
    sigma_G_val, q_G_val = compute_effective_basket_params(cfg)
    
    sigma_G = torch.tensor(sigma_G_val, dtype=torch.float32, device=device)
    q_G = torch.tensor(q_G_val, dtype=torch.float32, device=device)
    K = torch.tensor(cfg.K, dtype=torch.float32, device=device)
    r = torch.tensor(cfg.r, dtype=torch.float32, device=device)
    T = torch.tensor(cfg.T, dtype=torch.float32, device=device)
    
    # G(S) = exp(log(S) @ w)
    log_S = torch.log(torch.clamp(S_tensor, min=1e-8))
    G = torch.exp(log_S @ w) # (N, 1)
    
    tau = torch.clamp(T - t_tensor, min=1e-8)
    
    d1 = (torch.log(torch.clamp(G, min=1e-8) / K) + (r - q_G + 0.5 * sigma_G**2) * tau) / (sigma_G * torch.sqrt(tau))
    d2 = d1 - sigma_G * torch.sqrt(tau)
    
    cdf_neg_d1 = 0.5 * (1.0 + torch.special.erf(-d1 / np.sqrt(2.0)))
    cdf_neg_d2 = 0.5 * (1.0 + torch.special.erf(-d2 / np.sqrt(2.0)))
    
    V_euro = K * torch.exp(-r * tau) * cdf_neg_d2 - G * torch.exp(-q_G * tau) * cdf_neg_d1
    payoff = torch.clamp(K - G, min=0.0)
    V_euro = torch.where(tau <= 1e-7, payoff, V_euro)
    
    return torch.clamp(V_euro, min=0.0)

# Function alias for multidimensional geometric basket put
torch_multi_asset_european_basket_put = torch_multi_asset_geometric_put
