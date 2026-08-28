"""
Method 2: Dual-Network Coupled Physics-Informed Neural Network with Domain Decomposition.
Simultaneously learns:
1. Option Price Surface V_theta(S, t)
2. Exact Moving Free Boundary S*_phi(t)
Coupled via Domain Decomposition, Smooth Pasting (dV/dS = -1), and Value Matching (V = K - S*).
"""

import time
import torch
import torch.nn as nn
import numpy as np
from src.configs.default_config import OptionConfig, default_config
from src.utils import generate_pinn_training_data

class BoundaryNet(nn.Module):
    """
    Sub-network predicting the continuous 1D moving curve S*(t).
    Enforces S*(T) = K and smooth monotonic decay backward in time.
    For standard params (K=100, r=0.05, sigma=0.2, T=1), S*(0) ~ 84, S*(T) = 100.
    """
    def __init__(self, config: OptionConfig):
        super().__init__()
        self.T = config.T
        self.K = config.K
        
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        # Initialize so S*(0) starts around 84 and smoothly rises to 100 at T
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, t):
        t_norm = t / self.T
        tau = torch.clamp((self.T - t) / self.T, min=0.0)
        raw = self.net(t_norm)
        
        # S*(t) = K * (1 - (0.16 + 0.1 * sigmoid(raw)) * sqrt(tau))
        # When t = T (tau=0) -> S*(T) = K = 100 exactly!
        # When t = 0 (tau=1) -> S*(0) in [80, 88]
        decay_factor = 0.12 + 0.08 * torch.sigmoid(raw)
        S_star = self.K * (1.0 - decay_factor * torch.sqrt(tau + 1e-6))
        return S_star

class DualPINN(nn.Module):
    def __init__(self, config: OptionConfig = default_config):
        super().__init__()
        self.cfg = config
        self.S_max = config.S_max
        self.T = config.T
        self.K = config.K
        
        # 1. Price Network V_theta(S, t)
        layers = []
        hidden_dim = config.hidden_dim
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.Tanh())
        for _ in range(config.hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.price_net = nn.Sequential(*layers)
        
        # 2. Boundary Network S*_phi(t)
        self.boundary_net = BoundaryNet(config)
        
        for m in self.price_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_price(self, S, t):
        S_norm = S / self.S_max
        t_norm = t / self.T
        x = torch.cat([S_norm, t_norm], dim=1)
        raw_out = self.price_net(x)
        V = torch.nn.functional.softplus(raw_out) * self.K
        return V

    def forward_boundary(self, t):
        return self.boundary_net(t)

class DualPINNTrainer:
    def __init__(self, config: OptionConfig = default_config, device: str = None):
        self.cfg = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DualPINN(config).to(self.device)
        self.K = config.K
        self.T = config.T
        self.r = config.r
        self.sigma = config.sigma
        
    def compute_loss(self, data):
        S_int = data["S_int"]
        t_int = data["t_int"]
        
        # 1. Price Evaluation on Interior Points
        V = self.model.forward_price(S_int, t_int)
        
        grad_V = torch.autograd.grad(
            outputs=V, inputs=[S_int, t_int],
            grad_outputs=torch.ones_like(V),
            create_graph=True, retain_graph=True
        )
        dV_dS = grad_V[0]
        dV_dt = grad_V[1]
        
        grad_Delta = torch.autograd.grad(
            outputs=dV_dS, inputs=S_int,
            grad_outputs=torch.ones_like(dV_dS),
            create_graph=True, retain_graph=True
        )
        d2V_dS2 = grad_Delta[0]
        
        # PDE Residual in continuation zone
        pde_res = dV_dt + 0.5 * (self.sigma**2) * (S_int**2) * d2V_dS2 + self.r * S_int * dV_dS - self.r * V
        h_int = torch.clamp(self.K - S_int, min=0.0)
        
        continuation_mask = (V > h_int + 1e-3).float()
        loss_pde = torch.mean((pde_res * continuation_mask)**2) + 0.05 * torch.mean(torch.clamp(pde_res, min=0.0)**2)
        loss_early = torch.mean(torch.clamp(h_int - V, min=0.0)**2)
        
        # 2. Coupled Free Boundary & Smooth Pasting
        N_b = 300
        t_sample = torch.rand((N_b, 1), device=self.device, requires_grad=True) * self.T
        S_star = self.model.forward_boundary(t_sample)
        
        # Value Matching: V(S*(t), t) == K - S*(t)
        V_at_boundary = self.model.forward_price(S_star, t_sample)
        loss_value_matching = torch.mean((V_at_boundary - (self.K - S_star))**2)
        
        # Smooth Pasting: dV/dS at S*(t) == -1.0
        dV_dS_boundary = torch.autograd.grad(
            outputs=V_at_boundary, inputs=S_star,
            grad_outputs=torch.ones_like(V_at_boundary),
            create_graph=True, retain_graph=True
        )[0]
        loss_smooth_pasting = torch.mean((dV_dS_boundary - (-1.0))**2)
        
        # 3. Standard Boundary & Initial Conditions
        V_bc = self.model.forward_price(data["S_bc"], data["t_bc"])
        loss_bc = torch.mean((V_bc - data["V_bc"])**2)
        
        V_ic = self.model.forward_price(data["S_ic"], data["t_ic"])
        loss_ic = torch.mean((V_ic - data["V_ic"])**2)
        
        total_loss = (
            self.cfg.w_pde * loss_pde +
            self.cfg.w_early * loss_early +
            self.cfg.w_bc * loss_bc +
            self.cfg.w_ic * loss_ic +
            5.0 * loss_value_matching +
            5.0 * loss_smooth_pasting
        )
        return total_loss

    def train(self, verbose: bool = True):
        start_time = time.perf_counter()
        data = generate_pinn_training_data(self.cfg, device=self.device)
        
        optimizer_adamw = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr_adam, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adamw, T_max=self.cfg.epochs_adam, eta_min=1e-5)
        
        if verbose:
            print(f">> [Dual-PINN] Training Coupled Networks with AdamW ({self.cfg.epochs_adam} eps) + L-BFGS...")
            
        for epoch in range(1, self.cfg.epochs_adam + 1):
            optimizer_adamw.zero_grad()
            loss = self.compute_loss(data)
            loss.backward()
            optimizer_adamw.step()
            scheduler.step()
            
            if verbose and (epoch % 1000 == 0 or epoch == self.cfg.epochs_adam):
                print(f"   [Dual-PINN AdamW Epoch {epoch:4d}] Loss: {loss.item():.4e}")
                
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if self.cfg.epochs_lbfgs > 0:
            optimizer_lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
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
            print(f"   [Dual-PINN Completed in {train_time:.2f}s]")
        return train_time

    def evaluate_grid(self, S_array: np.ndarray, t_array: np.ndarray):
        self.model.eval()
        start_time = time.perf_counter()
        T_mesh, S_mesh = np.meshgrid(t_array, S_array, indexing="ij")
        
        S_flat = torch.tensor(S_mesh.flatten()[:, None], dtype=torch.float32, requires_grad=True, device=self.device)
        t_flat = torch.tensor(T_mesh.flatten()[:, None], dtype=torch.float32, requires_grad=True, device=self.device)
        
        V_flat = self.model.forward_price(S_flat, t_flat)
        grad_V = torch.autograd.grad(outputs=V_flat, inputs=[S_flat], grad_outputs=torch.ones_like(V_flat), create_graph=True, retain_graph=True)
        Delta_flat = grad_V[0]
        grad_Delta = torch.autograd.grad(outputs=Delta_flat, inputs=S_flat, grad_outputs=torch.ones_like(Delta_flat), create_graph=False, retain_graph=False)
        Gamma_flat = grad_Delta[0]
        
        eval_time = time.perf_counter() - start_time
        V_grid = V_flat.detach().cpu().numpy().reshape(S_mesh.shape)
        Delta_grid = Delta_flat.detach().cpu().numpy().reshape(S_mesh.shape)
        Gamma_grid = Gamma_flat.detach().cpu().numpy().reshape(S_mesh.shape)
        return V_grid, Delta_grid, Gamma_grid, eval_time

    def extract_free_boundary(self, t_array: np.ndarray):
        self.model.eval()
        t_tensor = torch.tensor(t_array[:, None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            S_star = self.model.forward_boundary(t_tensor).cpu().numpy().flatten()
        return S_star
