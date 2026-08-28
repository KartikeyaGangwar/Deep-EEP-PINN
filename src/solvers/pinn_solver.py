"""
Physics-Informed Neural Network (PINN) Solver for 1D American Put Options.
Solves the Parabolic Variational Inequality using Automatic Differentiation
and a differentiable penalty loss for early-exercise constraint enforcement.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from src.configs.default_config import OptionConfig, default_config
from src.utils import generate_pinn_training_data

class AmericanOptionPINN(nn.Module):
    def __init__(self, config: OptionConfig = default_config):
        super().__init__()
        self.cfg = config
        self.S_max = config.S_max
        self.T = config.T
        self.K = config.K
        
        # Network Architecture: Input (S, t) -> 2D
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
        
        # Xavier Initialization for stable initial training
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, S, t):
        """
        Forward pass with internal normalization.
        Inputs: S in [0, S_max], t in [0, T].
        Output: Option price V >= 0 (enforced via Softplus/ReLU scaling).
        """
        # Normalize inputs to [0, 1] range for optimal gradient flow
        S_norm = S / self.S_max
        t_norm = t / self.T
        x = torch.cat([S_norm, t_norm], dim=1)
        raw_out = self.network(x)
        
        # Scale output back to price scale [0, K]
        # Using softplus to naturally enforce V >= 0
        V = torch.nn.functional.softplus(raw_out) * self.K
        return V

class PINNTrainer:
    def __init__(self, config: OptionConfig = default_config, device: str = None):
        self.cfg = config
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = AmericanOptionPINN(config).to(self.device)
        self.K = config.K
        self.T = config.T
        self.r = config.r
        self.sigma = config.sigma
        
    def compute_loss(self, data):
        """
        Evaluates the composite physics loss:
        L = w_pde * L_pde + w_early * L_early + w_bc * L_bc + w_ic * L_ic
        """
        S_int = data["S_int"]
        t_int = data["t_int"]
        
        # 1. Forward Pass on Interior Points
        V = self.model(S_int, t_int)
        
        # 2. Automatic Differentiation for Greeks and Time Derivative
        grad_V = torch.autograd.grad(
            outputs=V, inputs=[S_int, t_int],
            grad_outputs=torch.ones_like(V),
            create_graph=True, retain_graph=True
        )
        dV_dS = grad_V[0] # Delta
        dV_dt = grad_V[1] # Theta
        
        grad_Delta = torch.autograd.grad(
            outputs=dV_dS, inputs=S_int,
            grad_outputs=torch.ones_like(dV_dS),
            create_graph=True, retain_graph=True
        )
        d2V_dS2 = grad_Delta[0] # Gamma
        
        # 3. Black-Scholes PDE Residual:
        # Res = dV/dt + 0.5 * sigma^2 * S^2 * d2V/dS2 + r * S * dV/dS - r * V
        pde_residual = dV_dt + 0.5 * (self.sigma**2) * (S_int**2) * d2V_dS2 + self.r * S_int * dV_dS - self.r * V
        
        # Payoff Obstacle at collocation points
        h_int = torch.clamp(self.K - S_int, min=0.0)
        
        # Continuation region indicator (where V > h)
        # In continuation region: PDE must hold (pde_residual == 0)
        # In stopping region: pde_residual <= 0 (since early exercise pays off)
        continuation_mask = (V > h_int + 1e-3).float()
        loss_pde = torch.mean((pde_residual * continuation_mask)**2) + 0.1 * torch.mean(torch.clamp(pde_residual, min=0.0)**2)
        
        # 4. Early-Exercise Penalty Loss: Enforce V >= h(S) everywhere
        violation = torch.clamp(h_int - V, min=0.0)
        loss_early = torch.mean(violation**2)
        
        # 5. Boundary Condition Loss (S=0 and S=S_max)
        V_bc_pred = self.model(data["S_bc"], data["t_bc"])
        loss_bc = torch.mean((V_bc_pred - data["V_bc"])**2)
        
        # 6. Terminal / Initial Condition Loss at t = T
        V_ic_pred = self.model(data["S_ic"], data["t_ic"])
        loss_ic = torch.mean((V_ic_pred - data["V_ic"])**2)
        
        total_loss = (
            self.cfg.w_pde * loss_pde +
            self.cfg.w_early * loss_early +
            self.cfg.w_bc * loss_bc +
            self.cfg.w_ic * loss_ic
        )
        
        loss_dict = {
            "total": total_loss,
            "pde": loss_pde.item(),
            "early": loss_early.item(),
            "bc": loss_bc.item(),
            "ic": loss_ic.item()
        }
        return total_loss, loss_dict

    def train(self, verbose: bool = True):
        """
        Trains the PINN using Adam optimizer followed by L-BFGS for high precision.
        """
        start_time = time.perf_counter()
        data = generate_pinn_training_data(self.cfg, device=self.device)
        
        # Phase A: Adam Optimization
        optimizer_adam = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr_adam)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=self.cfg.epochs_adam, eta_min=1e-5)
        
        if verbose:
            print(f">> Starting PINN Training on device: {self.device.upper()}")
            print(f">> Phase A: Adam Optimizer for {self.cfg.epochs_adam} epochs...")
            
        for epoch in range(1, self.cfg.epochs_adam + 1):
            optimizer_adam.zero_grad()
            loss, loss_dict = self.compute_loss(data)
            loss.backward()
            optimizer_adam.step()
            scheduler.step()
            
            if verbose and (epoch % 500 == 0 or epoch == self.cfg.epochs_adam):
                print(f"   [Adam Epoch {epoch:4d}/{self.cfg.epochs_adam}] "
                      f"Total Loss: {loss.item():.4e} | PDE: {loss_dict['pde']:.4e} | "
                      f"Early: {loss_dict['early']:.4e} | BC: {loss_dict['bc']:.4e} | IC: {loss_dict['ic']:.4e}")
                
        # Phase B: L-BFGS Fine-Tuning
        if self.cfg.epochs_lbfgs > 0:
            if verbose:
                print(f">> Phase B: L-BFGS Optimizer for {self.cfg.epochs_lbfgs} iterations...")
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
                loss, _ = self.compute_loss(data)
                loss.backward()
                return loss
                
            optimizer_lbfgs.step(closure)
            
        train_time = time.perf_counter() - start_time
        if verbose:
            final_loss, final_dict = self.compute_loss(data)
            print(f">> Training Completed in {train_time:.2f}s | Final Total Loss: {final_loss.item():.4e}")
            
        return train_time

    def evaluate_grid(self, S_array: np.ndarray, t_array: np.ndarray):
        """
        Evaluates the trained PINN across a 2D meshgrid (S, t).
        Returns:
            V_pred: 2D numpy array of shape (len(t_array), len(S_array))
            Delta_pred: 2D numpy array (dV/dS)
            Gamma_pred: 2D numpy array (d2V/dS2)
            eval_time: Elapsed time in seconds
        """
        self.model.eval()
        start_time = time.perf_counter()
        
        # Create meshgrid
        T_mesh, S_mesh = np.meshgrid(t_array, S_array, indexing="ij")
        
        S_flat = torch.tensor(S_mesh.flatten()[:, None], dtype=torch.float32, requires_grad=True, device=self.device)
        t_flat = torch.tensor(T_mesh.flatten()[:, None], dtype=torch.float32, requires_grad=True, device=self.device)
        
        V_flat = self.model(S_flat, t_flat)
        
        # Exact Greeks via Automatic Differentiation
        grad_V = torch.autograd.grad(
            outputs=V_flat, inputs=[S_flat],
            grad_outputs=torch.ones_like(V_flat),
            create_graph=True, retain_graph=True
        )
        Delta_flat = grad_V[0]
        
        grad_Delta = torch.autograd.grad(
            outputs=Delta_flat, inputs=S_flat,
            grad_outputs=torch.ones_like(Delta_flat),
            create_graph=False, retain_graph=False
        )
        Gamma_flat = grad_Delta[0]
        
        eval_time = time.perf_counter() - start_time
        
        V_grid = V_flat.detach().cpu().numpy().reshape(S_mesh.shape)
        Delta_grid = Delta_flat.detach().cpu().numpy().reshape(S_mesh.shape)
        Gamma_grid = Gamma_flat.detach().cpu().numpy().reshape(S_mesh.shape)
        
        return V_grid, Delta_grid, Gamma_grid, eval_time

    def extract_free_boundary(self, S_array: np.ndarray, t_array: np.ndarray, V_grid: np.ndarray):
        """
        Extracts optimal exercise boundary S*(t) from PINN predictions.
        """
        S_star = np.zeros(len(t_array))
        payoff = np.maximum(self.K - S_array, 0.0)
        
        for n, t_val in enumerate(t_array):
            if n == len(t_array) - 1: # t = T
                S_star[n] = self.K
                continue
            # Point where V(S,t) is on obstacle
            diff = V_grid[n, :] - payoff
            exercise_idx = np.where((diff <= 0.05) & (S_array < self.K))[0]
            if len(exercise_idx) > 0:
                S_star[n] = S_array[exercise_idx[-1]]
            else:
                S_star[n] = S_star[max(0, n-1)]
                
        return S_star
