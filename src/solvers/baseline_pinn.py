"""
Method 1: Standard Single-Network Penalty PINN (Baseline)
Trained using AdamW optimizer with Cosine Annealing + L-BFGS.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from src.configs.default_config import OptionConfig, default_config
from src.utils import generate_pinn_training_data

class BaselinePINN(nn.Module):
    def __init__(self, config: OptionConfig = default_config):
        super().__init__()
        self.cfg = config
        self.S_max = config.S_max
        self.T = config.T
        self.K = config.K
        
        layers = []
        in_dim = 2
        hidden_dim = config.hidden_dim
        
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(config.hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)
        
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, S, t):
        S_norm = S / self.S_max
        t_norm = t / self.T
        x = torch.cat([S_norm, t_norm], dim=1)
        raw_out = self.network(x)
        V = torch.nn.functional.softplus(raw_out) * self.K
        return V

class BaselineTrainer:
    def __init__(self, config: OptionConfig = default_config, device: str = None):
        self.cfg = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BaselinePINN(config).to(self.device)
        self.K = config.K
        self.T = config.T
        self.r = config.r
        self.sigma = config.sigma
        
    def compute_loss(self, data):
        S_int = data["S_int"]
        t_int = data["t_int"]
        
        V = self.model(S_int, t_int)
        
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
        
        pde_residual = dV_dt + 0.5 * (self.sigma**2) * (S_int**2) * d2V_dS2 + self.r * S_int * dV_dS - self.r * V
        
        h_int = torch.clamp(self.K - S_int, min=0.0)
        continuation_mask = (V > h_int + 1e-3).float()
        loss_pde = torch.mean((pde_residual * continuation_mask)**2) + 0.05 * torch.mean(torch.clamp(pde_residual, min=0.0)**2)
        
        violation = torch.clamp(h_int - V, min=0.0)
        loss_early = torch.mean(violation**2)
        
        V_bc_pred = self.model(data["S_bc"], data["t_bc"])
        loss_bc = torch.mean((V_bc_pred - data["V_bc"])**2)
        
        V_ic_pred = self.model(data["S_ic"], data["t_ic"])
        loss_ic = torch.mean((V_ic_pred - data["V_ic"])**2)
        
        total_loss = (
            self.cfg.w_pde * loss_pde +
            self.cfg.w_early * loss_early +
            self.cfg.w_bc * loss_bc +
            self.cfg.w_ic * loss_ic
        )
        return total_loss

    def train(self, verbose: bool = True):
        start_time = time.perf_counter()
        data = generate_pinn_training_data(self.cfg, device=self.device)
        
        # Optimizer: AdamW with weight decay
        optimizer_adamw = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr_adam, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adamw, T_max=self.cfg.epochs_adam, eta_min=1e-5)
        
        if verbose:
            print(f">> [Baseline PINN] Training with AdamW ({self.cfg.epochs_adam} eps) + L-BFGS ({self.cfg.epochs_lbfgs} iters)...")
            
        for epoch in range(1, self.cfg.epochs_adam + 1):
            optimizer_adamw.zero_grad()
            loss = self.compute_loss(data)
            loss.backward()
            optimizer_adamw.step()
            scheduler.step()
            
            if verbose and (epoch % 1000 == 0 or epoch == self.cfg.epochs_adam):
                print(f"   [Baseline AdamW Epoch {epoch:4d}] Loss: {loss.item():.4e}")
                
        # L-BFGS Fine-Tuning
        if self.cfg.epochs_lbfgs > 0:
            optimizer_lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                max_iter=self.cfg.epochs_lbfgs,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                history_size=50,
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
            print(f"   [Baseline PINN Completed in {train_time:.2f}s]")
        return train_time

    def evaluate_grid(self, S_array: np.ndarray, t_array: np.ndarray):
        self.model.eval()
        start_time = time.perf_counter()
        T_mesh, S_mesh = np.meshgrid(t_array, S_array, indexing="ij")
        
        S_flat = torch.tensor(S_mesh.flatten()[:, None], dtype=torch.float32, requires_grad=True, device=self.device)
        t_flat = torch.tensor(T_mesh.flatten()[:, None], dtype=torch.float32, requires_grad=True, device=self.device)
        
        V_flat = self.model(S_flat, t_flat)
        grad_V = torch.autograd.grad(outputs=V_flat, inputs=[S_flat], grad_outputs=torch.ones_like(V_flat), create_graph=True, retain_graph=True)
        Delta_flat = grad_V[0]
        grad_Delta = torch.autograd.grad(outputs=Delta_flat, inputs=S_flat, grad_outputs=torch.ones_like(Delta_flat), create_graph=False, retain_graph=False)
        Gamma_flat = grad_Delta[0]
        
        eval_time = time.perf_counter() - start_time
        V_grid = V_flat.detach().cpu().numpy().reshape(S_mesh.shape)
        Delta_grid = Delta_flat.detach().cpu().numpy().reshape(S_mesh.shape)
        Gamma_grid = Gamma_flat.detach().cpu().numpy().reshape(S_mesh.shape)
        return V_grid, Delta_grid, Gamma_grid, eval_time

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
