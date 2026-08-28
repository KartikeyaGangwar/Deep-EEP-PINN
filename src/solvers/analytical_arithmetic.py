"""
Analytical Closed-Form Moment-Matching Solver (Levy 1992 / Gentle 1993 / Milevsky-Posner 1998)
for Multi-Asset European Arithmetic Basket Options.

Matches the 1st and 2nd moments of the correlated arithmetic sum B(T) = sum(w_i * S_i(T))
to an effective lognormal distribution, yielding an exact terminal payoff (K - sum(w_i S_i))^+ at t=T.

Fix C2 Applied:
- Instantaneous variance limit sigma_inst^2 = (w^T Cov w) / B(t)^2 used when tau -> 0,
  preventing numerical cancellation/blowup near expiration.
"""

import numpy as np
import torch
from scipy.stats import norm
from src.configs.multi_asset_config import MultiAssetConfig, default_multi_config

def compute_arithmetic_basket_moments(S_matrix: np.ndarray, tau: np.ndarray, cfg: MultiAssetConfig):
    """
    Computes first moment M1 and second moment M2 for arithmetic basket B = sum(w_i S_i).
    S_matrix: (N, d)
    tau: (N,) or float
    """
    w = np.array(cfg.weights) # (d,)
    cov = cfg.get_covariance_matrix() # (d, d)
    r = cfg.r
    
    # B(t) = sum(w_i * S_i(t))
    B_t = S_matrix @ w # (N,)
    
    # M1 = E[B(T)] = B(t) * exp(r * tau)
    M1 = B_t * np.exp(r * tau) # (N,)
    
    # M2 = E[B(T)^2] = sum_{i,j} w_i w_j S_i S_j exp((2r + cov_ij)*tau)
    N = S_matrix.shape[0]
    M2 = np.zeros(N)
    instantaneous_cov_sum = np.zeros(N)
    for i in range(cfg.d):
        for j in range(cfg.d):
            w_ij = w[i] * w[j]
            cov_ij = cov[i, j]
            growth = np.exp((2.0 * r + cov_ij) * tau)
            term = w_ij * S_matrix[:, i] * S_matrix[:, j]
            M2 += term * growth
            instantaneous_cov_sum += term * cov_ij
            
    # Theoretical instantaneous variance limit as tau -> 0: sigma_inst^2 = sum(w_i w_j cov_ij S_i S_j) / B(t)^2
    sigma_inst_sq = instantaneous_cov_sum / np.maximum(B_t**2, 1e-10)
    
    # Effective variance: sigma_A^2 * tau = ln(M2 / M1^2)
    ratio = np.maximum(M2 / np.maximum(M1**2, 1e-12), 1.0 + 1e-10)
    
    tau_safe = np.maximum(tau, 1e-8)
    vol_A_sq_dynamic = np.log(ratio) / tau_safe
    
    # Use exact instantaneous limit when tau < 1e-4 to prevent numerical blowup
    vol_A_sq = np.where(tau < 1e-4, sigma_inst_sq, vol_A_sq_dynamic)
    vol_A = np.sqrt(np.maximum(vol_A_sq, 1e-6))
    
    return B_t, M1, vol_A

def multi_asset_arithmetic_european_put(S_matrix: np.ndarray, t_array: np.ndarray, cfg: MultiAssetConfig = default_multi_config) -> np.ndarray:
    """
    Evaluates European Arithmetic Basket Put Option via Moment Matching in NumPy.
    """
    S_matrix = np.asarray(S_matrix, dtype=float)
    t_array = np.asarray(t_array, dtype=float)
    tau = np.maximum(cfg.T - t_array, 0.0)
    
    B_t, M1, vol_A = compute_arithmetic_basket_moments(S_matrix, tau, cfg)
    
    tau_clamped = np.maximum(tau, 1e-8)
    d1 = (np.log(np.maximum(B_t, 1e-12) / cfg.K) + (cfg.r + 0.5 * vol_A**2) * tau_clamped) / (vol_A * np.sqrt(tau_clamped))
    d2 = d1 - vol_A * np.sqrt(tau_clamped)
    
    put_price = cfg.K * np.exp(-cfg.r * tau) * norm.cdf(-d2) - B_t * norm.cdf(-d1)
    
    # Terminal condition at tau == 0: exact arithmetic payoff max(K - sum(w_i S_i), 0)
    payoff = np.maximum(cfg.K - B_t, 0.0)
    put_price = np.where(tau <= 1e-6, payoff, put_price)
    
    return np.maximum(put_price, 0.0)

def torch_multi_asset_arithmetic_put(S_tensor: torch.Tensor, t_tensor: torch.Tensor, cfg: MultiAssetConfig) -> torch.Tensor:
    """
    Differentiable PyTorch implementation of the European Arithmetic Basket Put price.
    S_tensor: (N, d)
    t_tensor: (N, 1)
    """
    device = S_tensor.device
    w = torch.tensor(cfg.weights, dtype=torch.float32, device=device) # (d,)
    cov = torch.tensor(cfg.get_covariance_matrix(), dtype=torch.float32, device=device) # (d, d)
    K = torch.tensor(cfg.K, dtype=torch.float32, device=device)
    r = torch.tensor(cfg.r, dtype=torch.float32, device=device)
    T = torch.tensor(cfg.T, dtype=torch.float32, device=device)
    
    tau = torch.clamp(T - t_tensor, min=0.0) # (N, 1)
    
    # B(t) = sum(w_i S_i)
    B_t = torch.sum(S_tensor * w, dim=1, keepdim=True) # (N, 1)
    M1 = B_t * torch.exp(r * tau)
    
    # M2 = sum_{i,j} w_i w_j S_i S_j exp((2r + cov_ij)*tau)
    M2 = torch.zeros_like(B_t)
    instantaneous_cov_sum = torch.zeros_like(B_t)
    for i in range(cfg.d):
        for j in range(cfg.d):
            w_ij = w[i] * w[j]
            cov_ij = cov[i, j]
            growth = torch.exp((2.0 * r + cov_ij) * tau)
            term = w_ij * S_tensor[:, i:i+1] * S_tensor[:, j:j+1]
            M2 += term * growth
            instantaneous_cov_sum += term * cov_ij
            
    sigma_inst_sq = instantaneous_cov_sum / torch.clamp(B_t**2, min=1e-8)
    
    ratio = torch.clamp(M2 / torch.clamp(M1**2, min=1e-8), min=1.0 + 1e-7)
    tau_safe = torch.clamp(tau, min=1e-8)
    vol_A_sq_dynamic = torch.log(ratio) / tau_safe
    
    vol_A_sq = torch.where(tau < 1e-4, sigma_inst_sq, vol_A_sq_dynamic)
    vol_A = torch.sqrt(torch.clamp(vol_A_sq, min=1e-6))
    
    d1 = (torch.log(torch.clamp(B_t, min=1e-8) / K) + (r + 0.5 * vol_A**2) * tau_safe) / (vol_A * torch.sqrt(tau_safe))
    d2 = d1 - vol_A * torch.sqrt(tau_safe)
    
    cdf_neg_d1 = 0.5 * (1.0 + torch.special.erf(-d1 / np.sqrt(2.0)))
    cdf_neg_d2 = 0.5 * (1.0 + torch.special.erf(-d2 / np.sqrt(2.0)))
    
    V_euro = K * torch.exp(-r * tau) * cdf_neg_d2 - B_t * cdf_neg_d1
    payoff = torch.clamp(K - B_t, min=0.0)
    V_euro = torch.where(tau <= 1e-6, payoff, V_euro)
    
    return torch.clamp(V_euro, min=0.0)
