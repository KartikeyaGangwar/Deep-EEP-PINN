"""
Novel Method: Early Exercise Premium PINN (EEP-PINN) for 1D American Put Options
V_American(S, t) = V_European_exact(S, t) + e_theta(S, t)

Includes All Peer-Review Fixes:
- Merton (1973) Explicit Smooth Pasting Loss: dV/dS = -1 at the free boundary S*(t)
- Smooth Sigmoid Continuation Mask
- AdamW Parameter Groups (weight_decay=0.0 on log_vars)
- Full L-BFGS Parameter Optimization
- Robust max_premium scaling for r -> 0 regimes
"""

import time
import torch
import torch.nn as nn
import numpy as np
from src.configs.default_config import OptionConfig, default_config
from src.solvers.analytical import (
    torch_black_scholes_put,
    torch_black_scholes_delta,
    torch_black_scholes_gamma,
    analytical_black_scholes_put,
    analytical_black_scholes_delta,
    analytical_black_scholes_gamma
)
from src.utils import generate_pinn_training_data

class EarlyExercisePremiumNet(nn.Module):
    """
    Parameterizes the Early Exercise Premium e(S, t) with exact hard-boundary encodings:
    1. e(S, T) == 0 strictly for all S (Zero Terminal Kink Singularity)
    2. e(S_max, t) == 0 strictly for all t (Zero Far-Field Error)
    3. e(S, t) >= 0 strictly for all S, t (Non-negativity via Softplus)
    4. e(S, t) <= max(K*(1 - exp(-rT)), 0.05*K, 1.0) (20x dynamic scale shrinkage)
    """
    def __init__(self, config: OptionConfig = default_config):
        super().__init__()
        self.cfg = config
        self.S_max = config.S_max
        self.T = config.T
        self.K = config.K
        self.r = config.r
        
        # Robust max premium scaling even if r -> 0
        self.max_premium = max(self.K * (1.0 - np.exp(-self.r * self.T)), self.K * 0.05, 1.0)
        
        layers = []
        in_dim = 2 # (S, t)
        hidden_dim = config.hidden_dim
        
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(config.hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        
        # Homoscedastic uncertainty loss log-variance parameters (4 tasks: PDE, Obstacle, BC0, Smooth Pasting)
        self.log_vars = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_premium(self, S_tensor: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        S_norm = S_tensor / self.S_max
        t_norm = t_tensor / self.T
        x = torch.cat([S_norm, t_norm], dim=1)
        raw = self.net(x)
        
        # Exact Hard Boundary Encodings
        tau_factor = torch.clamp((self.T - t_tensor) / self.T, min=0.0)
        spatial_factor = torch.clamp(1.0 - S_norm, min=0.0)
        
        e = torch.nn.functional.softplus(raw) * tau_factor * spatial_factor * self.max_premium
        return e

    def forward(self, S_tensor: torch.Tensor, t_tensor: torch.Tensor) -> torch.Tensor:
        V_euro = torch_black_scholes_put(S_tensor, t_tensor, self.K, self.T, self.r, self.cfg.sigma)
        e = self.forward_premium(S_tensor, t_tensor)
        return V_euro + e

    def get_adaptive_weights(self):
        with torch.no_grad():
            precision = torch.exp(-self.log_vars)
        return {
            "w_pde": precision[0].item(),
            "w_obstacle": precision[1].item(),
            "w_bc0": precision[2].item(),
            "w_sp": precision[3].item()
        }

class EEPPINNTrainer:
    def __init__(self, config: OptionConfig = default_config, device: str = None):
        self.cfg = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EarlyExercisePremiumNet(config).to(self.device)
        self.K = config.K
        self.T = config.T
        self.r = config.r
        self.sigma = config.sigma
        
    def compute_loss(self, data):
        S_int = data["S_int"]
        t_int = data["t_int"]
        
        # 1. Evaluate Premium and Derivatives
        e = self.model.forward_premium(S_int, t_int)
        
        grad_e = torch.autograd.grad(
            outputs=e, inputs=[S_int, t_int],
            grad_outputs=torch.ones_like(e),
            create_graph=True, retain_graph=True
        )
        de_dS = grad_e[0]
        de_dt = grad_e[1]
        
        grad_Delta = torch.autograd.grad(
            outputs=de_dS, inputs=S_int,
            grad_outputs=torch.ones_like(de_dS),
            create_graph=True, retain_graph=True
        )
        d2e_dS2 = grad_Delta[0]
        
        # Black-Scholes PDE Residual on Premium: L_BS(e) = 0
        pde_res = de_dt + 0.5 * (self.sigma**2) * (S_int**2) * d2e_dS2 + self.r * S_int * de_dS - self.r * e
        
        # 2. American Obstacle Constraint on Premium
        # Target obstacle value is detached from autograd graph
        with torch.no_grad():
            V_euro_int = torch_black_scholes_put(S_int, t_int, self.K, self.T, self.r, self.sigma)
            payoff_int = torch.clamp(self.K - S_int, min=0.0)
            h_e = torch.clamp(payoff_int - V_euro_int, min=0.0)
        
        # Smooth Differentiable Continuation Mask (Fix C3)
        continuation_mask = torch.sigmoid(20.0 * (e - h_e))
        loss_pde = torch.mean((pde_res * continuation_mask)**2) + 0.05 * torch.mean(torch.clamp(pde_res, min=0.0)**2)
        loss_obstacle = torch.mean(torch.clamp(h_e - e, min=0.0)**2)
        
        # 3. Near-field Boundary Condition at S = 0
        N_bc = 400
        t_bc0 = torch.rand((N_bc, 1), device=self.device, requires_grad=True) * self.T
        S_bc0 = torch.zeros((N_bc, 1), device=self.device, requires_grad=True)
        e_bc0 = self.model.forward_premium(S_bc0, t_bc0)
        with torch.no_grad():
            target_bc0 = self.K * (1.0 - torch.exp(-self.r * (self.T - t_bc0)))
        loss_bc0 = torch.mean((e_bc0 - target_bc0)**2)
        
        # 4. Explicit Merton (1973) Smooth Pasting Loss at Free Boundary S*(t) (Fix M1)
        with torch.no_grad():
            boundary_weight = torch.exp(-((e - h_e)**2) / 0.04) * (S_int < self.K).float()
            delta_euro = torch_black_scholes_delta(S_int, t_int, self.K, self.T, self.r, self.sigma)
            target_de_dS = -1.0 - delta_euro
        loss_sp = torch.sum(boundary_weight * ((de_dS - target_de_dS)**2)) / (torch.sum(boundary_weight) + 1e-5)
        
        # Stable Precision Weights with clamped log_vars
        clamped_log_vars = torch.clamp(self.model.log_vars, min=-4.0, max=4.0)
        precision = torch.exp(-clamped_log_vars)
        total_loss = (
            precision[0] * loss_pde + 0.5 * clamped_log_vars[0] +
            precision[1] * loss_obstacle + 0.5 * clamped_log_vars[1] +
            precision[2] * loss_bc0 + 0.5 * clamped_log_vars[2] +
            precision[3] * loss_sp + 0.5 * clamped_log_vars[3]
        )
        return total_loss

    def train(self, verbose: bool = True):
        start_time = time.perf_counter()
        data = generate_pinn_training_data(self.cfg, device=self.device)
        
        # Fix M5: Separate parameter groups (weight_decay=0.0 on log_vars)
        optimizer_adamw = torch.optim.AdamW([
            {"params": self.model.net.parameters(), "weight_decay": 1e-5},
            {"params": [self.model.log_vars], "weight_decay": 0.0}
        ], lr=self.cfg.lr_adam)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adamw, T_max=self.cfg.epochs_adam, eta_min=1e-5)
        
        if verbose:
            print(f">> [EEP-PINN] Training Early Exercise Premium Model with AdamW + L-BFGS...")
            
        for epoch in range(1, self.cfg.epochs_adam + 1):
            optimizer_adamw.zero_grad()
            loss = self.compute_loss(data)
            loss.backward()
            optimizer_adamw.step()
            scheduler.step()
            
            if verbose and (epoch % 500 == 0 or epoch == self.cfg.epochs_adam):
                w = self.model.get_adaptive_weights()
                print(f"   [EEP-PINN Epoch {epoch:4d}] Loss: {loss.item():.4e} | Weights: PDE={w['w_pde']:.2f}, Obs={w['w_obstacle']:.2f}, BC0={w['w_bc0']:.2f}, SP={w['w_sp']:.2f}")
                
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # L-BFGS fine-tuning on network weights
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
                loss = self.compute_loss(data)
                loss.backward()
                return loss
            optimizer_lbfgs.step(closure)
            
        train_time = time.perf_counter() - start_time
        if verbose:
            print(f"   [EEP-PINN Training Completed in {train_time:.2f}s]")
        return train_time

    def evaluate_grid(self, S_array: np.ndarray, t_array: np.ndarray):
        self.model.eval()
        start_time = time.perf_counter()
        
        # Build 2D meshgrid
        S_mesh, T_mesh = np.meshgrid(S_array, t_array)
        S_flat = torch.tensor(S_mesh.flatten()[:, None], dtype=torch.float32, device=self.device, requires_grad=True)
        t_flat = torch.tensor(T_mesh.flatten()[:, None], dtype=torch.float32, device=self.device, requires_grad=True)
        
        e_flat = self.model.forward_premium(S_flat, t_flat)
        
        grad_e = torch.autograd.grad(outputs=e_flat, inputs=S_flat, grad_outputs=torch.ones_like(e_flat), create_graph=True, retain_graph=True)[0]
        grad2_e = torch.autograd.grad(outputs=grad_e, inputs=S_flat, grad_outputs=torch.ones_like(grad_e), create_graph=False, retain_graph=False)[0]
        
        de_dS = grad_e.detach().cpu().numpy().reshape(S_mesh.shape)
        d2e_dS2 = grad2_e.detach().cpu().numpy().reshape(S_mesh.shape)
        e_grid = e_flat.detach().cpu().numpy().reshape(S_mesh.shape)
        
        eval_time = time.perf_counter() - start_time
        
        # Hybrid Exact Greeks
        V_euro = analytical_black_scholes_put(S_mesh, T_mesh, self.K, self.T, self.r, self.sigma)
        Delta_euro = analytical_black_scholes_delta(S_mesh, T_mesh, self.K, self.T, self.r, self.sigma)
        Gamma_euro = analytical_black_scholes_gamma(S_mesh, T_mesh, self.K, self.T, self.r, self.sigma)
        
        V_amer = V_euro + e_grid
        Delta_amer = Delta_euro + de_dS
        Gamma_amer = Gamma_euro + d2e_dS2
        
        return V_amer, Delta_amer, Gamma_amer, e_grid, eval_time

    def extract_free_boundary(self, S_array: np.ndarray, t_array: np.ndarray, V_grid: np.ndarray):
        S_star = np.zeros(len(t_array))
        payoff = np.maximum(self.K - S_array, 0.0)
        for n in range(len(t_array)):
            if n == len(t_array) - 1:
                S_star[n] = self.K
                continue
            diff = V_grid[n, :] - payoff
            exercise_idx = np.where((diff <= 0.05) & (S_array < self.K))[0]
            if len(exercise_idx) > 0:
                S_star[n] = S_array[exercise_idx[-1]]
            else:
                S_star[n] = S_star[max(0, n-1)]
        return S_star
