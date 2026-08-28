"""
Analytical Closed-Form Solutions for European Options (Black-Scholes-Merton 1973).
Includes both NumPy and PyTorch (vectorized, autograd-compatible) implementations.
"""

import numpy as np
import torch
from scipy.stats import norm

def black_scholes_european_put(S: np.ndarray, t: np.ndarray, K: float, T: float, r: float, sigma: float) -> np.ndarray:
    """
    Computes exact European Put option price using Black-Scholes formula in NumPy.
    """
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    tau = np.maximum(T - t, 1e-12)
    
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(np.maximum(S, 1e-12) / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        put_price = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    put_price = np.where(S <= 1e-12, K * np.exp(-r * tau), put_price)
    put_price = np.where(tau <= 1e-10, np.maximum(K - S, 0.0), put_price)
    return np.maximum(put_price, 0.0)

def analytical_black_scholes_put(S: np.ndarray, t: np.ndarray, K: float, T: float, r: float, sigma: float) -> np.ndarray:
    return black_scholes_european_put(S, t, K, T, r, sigma)

def analytical_black_scholes_delta(S: np.ndarray, t: np.ndarray, K: float, T: float, r: float, sigma: float) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    tau = np.maximum(T - t, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(np.maximum(S, 1e-12) / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        delta = -norm.cdf(-d1)
    return delta

def analytical_black_scholes_gamma(S: np.ndarray, t: np.ndarray, K: float, T: float, r: float, sigma: float) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    tau = np.maximum(T - t, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(np.maximum(S, 1e-12) / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        gamma = norm.pdf(d1) / (np.maximum(S, 1e-12) * sigma * np.sqrt(tau))
    return gamma

def torch_black_scholes_put(S: torch.Tensor, t: torch.Tensor, K: float, T: float, r: float, sigma: float) -> torch.Tensor:
    """
    Computes exact European Put price as a differentiable PyTorch Tensor.
    Uses torch.special.erf for exact normal CDF: N(x) = 0.5 * (1 + erf(x / sqrt(2)))
    """
    tau = torch.clamp(T - t, min=1e-8)
    S_safe = torch.clamp(S, min=1e-8)
    
    d1 = (torch.log(S_safe / K) + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))
    d2 = d1 - sigma * torch.sqrt(tau)
    
    cdf_neg_d1 = 0.5 * (1.0 + torch.special.erf(-d1 / np.sqrt(2.0)))
    cdf_neg_d2 = 0.5 * (1.0 + torch.special.erf(-d2 / np.sqrt(2.0)))
    
    V_euro = K * torch.exp(-r * tau) * cdf_neg_d2 - S_safe * cdf_neg_d1
    V_euro = torch.where(S <= 1e-8, K * torch.exp(-r * tau), V_euro)
    V_euro = torch.where(tau <= 1e-7, torch.clamp(K - S, min=0.0), V_euro)
    return torch.clamp(V_euro, min=0.0)

def torch_black_scholes_delta(S: torch.Tensor, t: torch.Tensor, K: float, T: float, r: float, sigma: float) -> torch.Tensor:
    tau = torch.clamp(T - t, min=1e-8)
    S_safe = torch.clamp(S, min=1e-8)
    d1 = (torch.log(S_safe / K) + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))
    cdf_neg_d1 = 0.5 * (1.0 + torch.special.erf(-d1 / np.sqrt(2.0)))
    return -cdf_neg_d1

def torch_black_scholes_gamma(S: torch.Tensor, t: torch.Tensor, K: float, T: float, r: float, sigma: float) -> torch.Tensor:
    tau = torch.clamp(T - t, min=1e-8)
    S_safe = torch.clamp(S, min=1e-8)
    d1 = (torch.log(S_safe / K) + (r + 0.5 * sigma**2) * tau) / (sigma * torch.sqrt(tau))
    phi_d1 = torch.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    return phi_d1 / (S_safe * sigma * torch.sqrt(tau))

def torch_european_greeks(S: torch.Tensor, t: torch.Tensor, K: float, T: float, r: float, sigma: float):
    Delta_euro = torch_black_scholes_delta(S, t, K, T, r, sigma)
    Gamma_euro = torch_black_scholes_gamma(S, t, K, T, r, sigma)
    return Delta_euro, Gamma_euro
