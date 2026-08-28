"""
Ground Truth Benchmark for Multi-Asset American Geometric Basket Options (d=5):
Longstaff-Schwartz Least Squares Monte Carlo (LSM 2001).

Upgraded with:
- Full 23-term quadratic polynomial basis including cross-terms (S_i * S_j) / K^2.
- Antithetic variates for robust variance reduction.
"""

import time
import numpy as np
from src.configs.multi_asset_config import MultiAssetConfig, default_multi_config

class LongstaffSchwartzBasketSolver:
    def __init__(self, config: MultiAssetConfig = default_multi_config):
        self.cfg = config
        self.d = config.d
        self.K = config.K
        self.T = config.T
        self.r = config.r
        self.vols = np.array(config.volatilities)
        self.weights = np.array(config.weights)
        self.corr_matrix = config.get_correlation_matrix()
        self.L = np.linalg.cholesky(self.corr_matrix)
        
        self.num_paths = config.mc_paths
        self.num_steps = config.mc_time_steps
        self.dt = self.T / self.num_steps
        self.df = np.exp(-self.r * self.dt)
        
    def simulate_paths(self, S0: np.ndarray = None) -> np.ndarray:
        if S0 is None:
            S0 = np.full(self.d, self.K)
            
        n_half = self.num_paths // 2
        paths = np.zeros((self.num_paths, self.num_steps + 1, self.d))
        paths[:, 0, :] = S0
        
        drift = (self.r - 0.5 * (self.vols ** 2)) * self.dt
        vol_sqrt_dt = self.vols * np.sqrt(self.dt)
        
        # Antithetic variate generation
        Z_half = np.random.normal(0.0, 1.0, size=(n_half, self.num_steps, self.d))
        Z_uncorr = np.vstack([Z_half, -Z_half])
        Z_corr = Z_uncorr @ self.L.T
        
        for k in range(self.num_steps):
            paths[:, k+1, :] = paths[:, k, :] * np.exp(drift + vol_sqrt_dt * Z_corr[:, k, :])
            
        return paths

    def compute_basket_payoff(self, S_t: np.ndarray) -> np.ndarray:
        """
        Computes geometric basket payoff: max(K - prod(S_i^w_i), 0).
        """
        log_S = np.log(np.maximum(S_t, 1e-12))
        G_t = np.exp(log_S @ self.weights)
        return np.maximum(self.K - G_t, 0.0)

    def build_quadratic_basis(self, S_itm: np.ndarray, G_itm: np.ndarray) -> np.ndarray:
        """
        Constructs full 23-term quadratic basis:
        {1, S_i/K, (S_i/K)^2, (S_i*S_j)/K^2 for i < j, G/K, (G/K)^2}.
        """
        N_pts = S_itm.shape[0]
        S_norm = S_itm / self.K
        G_norm = G_itm / self.K
        
        basis = [np.ones(N_pts)]
        
        # 1. Linear terms (d terms)
        for i in range(self.d):
            basis.append(S_norm[:, i])
            
        # 2. Squared terms (d terms)
        for i in range(self.d):
            basis.append(S_norm[:, i] ** 2)
            
        # 3. Cross terms (d*(d-1)/2 = 10 terms)
        for i in range(self.d):
            for j in range(i + 1, self.d):
                basis.append(S_norm[:, i] * S_norm[:, j])
                
        # 4. Geometric basket terms (2 terms)
        basis.append(G_norm)
        basis.append(G_norm ** 2)
        
        return np.column_stack(basis)

    def price(self, S0: np.ndarray = None):
        start_time = time.perf_counter()
        paths = self.simulate_paths(S0)
        
        # 1. Payoff at maturity T (step N)
        cash_flows = self.compute_basket_payoff(paths[:, -1, :])
        
        # 2. Backward Induction from N-1 down to 1
        for k in range(self.num_steps - 1, 0, -1):
            S_k = paths[:, k, :]
            intrinsic_val = self.compute_basket_payoff(S_k)
            
            itm_mask = intrinsic_val > 0.0
            
            if np.sum(itm_mask) > 30:
                Y = cash_flows[itm_mask] * self.df
                
                log_S_itm = np.log(np.maximum(S_k[itm_mask], 1e-12))
                G_itm = np.exp(log_S_itm @ self.weights)
                
                A = self.build_quadratic_basis(S_k[itm_mask], G_itm)
                
                beta, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
                continuation_val = A @ beta
                
                exercise_now = intrinsic_val[itm_mask] > continuation_val
                
                cash_flows = cash_flows * self.df
                itm_indices = np.where(itm_mask)[0]
                cash_flows[itm_indices[exercise_now]] = intrinsic_val[itm_indices[exercise_now]]
            else:
                cash_flows = cash_flows * self.df
                
        discounted_val = cash_flows * self.df
        price_estimate = np.mean(discounted_val)
        std_err = np.std(discounted_val) / np.sqrt(self.num_paths)
        exec_time = time.perf_counter() - start_time
        
        return price_estimate, std_err, exec_time

# Alias for backwards compatibility
LongstaffSchwartzSolver = LongstaffSchwartzBasketSolver
