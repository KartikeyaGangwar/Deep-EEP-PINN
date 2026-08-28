"""
Novel Method: Multi-Asset Early Exercise Premium PINN for American Geometric Basket Options (d=5):
V_American(S_1..S_d, t) = V_European_exact_basket(S_1..S_d, t) + e_theta(S_1..S_d, t)

Includes All Peer-Review Fixes:
- Robust high-dimensional Hessian autograd contraction
- Smooth Sigmoid continuation mask
- Full domain [0, S_max] collocation sampling with strike concentration
- Separate AdamW parameter groups for log_vars
- Detached obstacle targets for clean autograd graphs
"""

import time
import torch
import torch.nn as nn
import numpy as np
from src.configs.multi_asset_config import MultiAssetConfig, default_multi_config
from src.solvers.analytical_multid import torch_multi_asset_geometric_put

class MultiAssetEEPPINN(nn.Module):
    def __init__(self, config: MultiAssetConfig = default_multi_config):
        super().__init__()
        self.cfg = config
        self.d = config.d
        self.S_max = config.S_max
        self.T = config.T
        self.K = config.K
        self.r = config.r
        self.vols = config.volatilities
        self.weights = config.weights
        
        # Robust max premium scaling
        self.max_premium = max(self.K * (1.0 - np.exp(-self.r * self.T)), self.K * 0.05, 1.0)
        
        layers = []
        in_dim = self.d + 1 # (S_1, ..., S_d, t)
        hidden_dim = config.hidden_dim
        
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(config.hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        
        self.log_vars = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        self.register_buffer("weights_tensor", torch.tensor(config.weights, dtype=torch.float32).unsqueeze(1))

    def forward_premium(self, S_tensor: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        S_norm = S_tensor / self.S_max
        t_norm = t_tensor / self.T
        x = torch.cat([S_norm, t_norm], dim=1)
        raw = self.net(x)
        
        # Exact Hard Boundary Encodings (Basket-Invariant Geometric Coordinate)
        tau_factor = torch.clamp((self.T - t_tensor) / self.T, min=0.0)
        log_S = torch.log(torch.clamp(S_tensor, min=1e-8))
        G_val = torch.exp(torch.matmul(log_S, self.weights_tensor))
        spatial_factor = torch.clamp(1.0 - G_val / self.S_max, min=0.0)
        
        e = torch.nn.functional.softplus(raw) * tau_factor * spatial_factor * self.max_premium
        return e

    def forward(self, S_tensor: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        V_euro = torch_multi_asset_geometric_put(S_tensor, t_tensor, self.cfg)
        e = self.forward_premium(S_tensor, t_tensor)
        return V_euro + e

class MultiAssetEEPTrainer:
    def __init__(self, config: MultiAssetConfig = default_multi_config, device: str = None):
        self.cfg = config
        self.d = config.d
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MultiAssetEEPPINN(config).to(self.device)
        self.cov_matrix = torch.tensor(config.get_covariance_matrix(), dtype=torch.float32, device=self.device)
        self.weights = torch.tensor(config.weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        
    def generate_collocation_data(self):
        N = self.cfg.num_collocation
        N1 = N // 3
        N2 = N // 3
        N3 = N - N1 - N2
        
        # 1. Full hypercube uniform sampling [0, S_max] (Fix C6)
        S_uniform = torch.rand((N1, self.d), device=self.device) * self.cfg.S_max
        
        # 2. Correlated Moneyness Ray sampling covering full ITM to OTM [20, 250]
        base_spot = (torch.rand((N2, 1), device=self.device) * (self.cfg.S_max - 20.0) + 20.0)
        perturb = torch.exp(torch.randn((N2, self.d), device=self.device) * 0.20)
        S_moneyness = torch.clamp(base_spot * perturb, min=1.0, max=self.cfg.S_max)
        
        # 3. Strike-concentrated near [0.4 K, 1.6 K]
        S_concentrated = torch.rand((N3, self.d), device=self.device) * 1.2 * self.cfg.K + 0.4 * self.cfg.K
        
        S_base = torch.vstack([S_uniform, S_moneyness, S_concentrated])
        t_base = torch.rand((N, 1), device=self.device) * self.cfg.T
        
        S_base.requires_grad_(True)
        t_base.requires_grad_(True)
        return S_base, t_base

    def _compute_loss_chunk(self, S_int, t_int):
        e = self.model.forward_premium(S_int, t_int)
        
        grad_t = torch.autograd.grad(outputs=e, inputs=t_int, grad_outputs=torch.ones_like(e), create_graph=True, retain_graph=True)[0]
        grad_S = torch.autograd.grad(outputs=e, inputs=S_int, grad_outputs=torch.ones_like(e), create_graph=True, retain_graph=True)[0]
        
        # High-dimensional Hessian trace contraction
        diffusion = torch.zeros_like(e)
        for i in range(self.d):
            g_i = grad_S[:, i:i+1]
            H_i = torch.autograd.grad(outputs=g_i, inputs=S_int, grad_outputs=torch.ones_like(g_i), create_graph=True, retain_graph=True)[0]
            cov_row_i = self.cov_matrix[i, :]
            weighted_H = torch.sum(H_i * (cov_row_i * S_int), dim=1, keepdim=True)
            diffusion += 0.5 * S_int[:, i:i+1] * weighted_H
            
        convection = self.cfg.r * torch.sum(S_int * grad_S, dim=1, keepdim=True)
        pde_res = grad_t + diffusion + convection - self.cfg.r * e
        
        # Obstacle Constraint on Geometric Premium with detached target
        with torch.no_grad():
            V_euro = torch_multi_asset_geometric_put(S_int, t_int, self.cfg)
            log_S = torch.log(torch.clamp(S_int, min=1e-8))
            G_int = torch.exp(torch.matmul(log_S, self.weights))
            payoff = torch.clamp(self.cfg.K - G_int, min=0.0)
            h_e = torch.clamp(payoff - V_euro, min=0.0)
        
        # Smooth Sigmoid Continuation Mask (Fix C3)
        continuation_mask = torch.sigmoid(20.0 * (e - h_e))
        loss_pde = torch.mean((pde_res * continuation_mask)**2) + 0.05 * torch.mean(torch.clamp(pde_res, min=0.0)**2)
        loss_obstacle = torch.mean(torch.clamp(h_e - e, min=0.0)**2)
        
        clamped_log_vars = torch.clamp(self.model.log_vars, min=-4.0, max=4.0)
        precision = torch.exp(-clamped_log_vars)
        chunk_loss = (
            precision[0] * loss_pde + 0.5 * clamped_log_vars[0] +
            precision[1] * loss_obstacle + 0.5 * clamped_log_vars[1]
        )
        return chunk_loss

    def compute_loss(self, S_int, t_int, chunk_size: int = 3000):
        N = S_int.shape[0]
        if N <= chunk_size or self.d < 20:
            return self._compute_loss_chunk(S_int, t_int)
        
        # Chunked evaluation for ultra-high dimensions (d >= 20) to cap peak VRAM under 3.5 GB
        total_loss = 0.0
        num_chunks = (N + chunk_size - 1) // chunk_size
        for c in range(num_chunks):
            start_idx = c * chunk_size
            end_idx = min(start_idx + chunk_size, N)
            S_chunk = S_int[start_idx:end_idx]
            t_chunk = t_int[start_idx:end_idx]
            chunk_weight = (end_idx - start_idx) / float(N)
            total_loss = total_loss + chunk_weight * self._compute_loss_chunk(S_chunk, t_chunk)
        return total_loss

    def train(self, verbose: bool = True):
        start_time = time.perf_counter()
        S_int, t_int = self.generate_collocation_data()
        
        # Fix M5: Separate parameter groups (weight_decay=0.0 on log_vars)
        optimizer_adamw = torch.optim.AdamW([
            {"params": self.model.net.parameters(), "weight_decay": 1e-5},
            {"params": [self.model.log_vars], "weight_decay": 0.0}
        ], lr=self.cfg.lr_adam)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adamw, T_max=self.cfg.epochs_adam, eta_min=1e-5)
        
        if verbose:
            print(f">> [Multi-Asset EEP-PINN] Training d={self.d} Basket Model on {self.device.upper()}...")
            
        for epoch in range(1, self.cfg.epochs_adam + 1):
            optimizer_adamw.zero_grad()
            loss = self.compute_loss(S_int, t_int)
            loss.backward()
            optimizer_adamw.step()
            scheduler.step()
            
            if verbose and (epoch % 500 == 0 or epoch == self.cfg.epochs_adam):
                print(f"   [EEP-PINN d={self.d} Epoch {epoch:4d}] Loss: {loss.item():.4e}")
                
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if self.cfg.epochs_lbfgs > 0:
            optimizer_lbfgs = torch.optim.LBFGS(
                self.model.net.parameters(),
                max_iter=self.cfg.epochs_lbfgs,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                history_size=20,
                line_search_fn="strong_wolfe"
            )
            def closure():
                optimizer_lbfgs.zero_grad()
                loss = self.compute_loss(S_int, t_int)
                loss.backward()
                return loss
            optimizer_lbfgs.step(closure)
            
        train_time = time.perf_counter() - start_time
        if verbose:
            print(f"   [Multi-Asset EEP-PINN Completed in {train_time:.2f}s]")
        return train_time

    def price_spot(self, S_spot: np.ndarray, t_val: float = 0.0):
        self.model.eval()
        start_time = time.perf_counter()
        S_tensor = torch.tensor(S_spot[None, :], dtype=torch.float32, device=self.device)
        t_tensor = torch.tensor([[t_val]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            V_pred = self.model(S_tensor, t_tensor).item()
            e_pred = self.model.forward_premium(S_tensor, t_tensor).item()
        eval_time = time.perf_counter() - start_time
        return V_pred, e_pred, eval_time
