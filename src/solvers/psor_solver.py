"""
Classical Finite Difference Solver for 1D American Put Options
Using Crank-Nicolson Time-Stepping and Projected Successive Over-Relaxation (PSOR).
This serves as the high-precision numerical ground-truth benchmark.
"""

import time
import numpy as np
from src.configs.default_config import OptionConfig, default_config

class PSORSolver:
    def __init__(self, config: OptionConfig = default_config):
        self.cfg = config
        self.K = config.K
        self.T = config.T
        self.r = config.r
        self.sigma = config.sigma
        self.S_max = config.S_max
        self.M = config.M_space
        self.N = config.N_time
        self.omega = config.omega_psor
        self.tol = config.tol_psor
        self.max_iter = config.max_iter_psor
        
        # Spatial and Temporal Discretization
        self.S = np.linspace(0, self.S_max, self.M + 1)
        self.dS = self.S_max / self.M
        self.dt = self.T / self.N
        self.t = np.linspace(0, self.T, self.N + 1)
        
        # Payoff Obstacle h(S) = max(K - S, 0)
        self.payoff = np.maximum(self.K - self.S, 0.0)
        
    def _build_tridiagonal_matrices(self):
        """Constructs Crank-Nicolson tridiagonal matrices A and B."""
        j = np.arange(1, self.M) # Interior points 1 to M-1
        
        # Spatial differential operator coefficients
        alpha = 0.25 * self.dt * (self.sigma**2 * (j**2) - self.r * j)
        beta = -0.5 * self.dt * (self.sigma**2 * (j**2) + self.r)
        gamma = 0.25 * self.dt * (self.sigma**2 * (j**2) + self.r * j)
        
        # Matrix A (Implicit: I - 0.5*dt*D)
        # Main diagonal of A
        A_diag = 1.0 - beta
        # Sub-diagonal of A (lower)
        A_sub = -alpha[1:] # corresponds to A[j, j-1]
        # Super-diagonal of A (upper)
        A_sup = -gamma[:-1] # corresponds to A[j, j+1]
        
        # Matrix B (Explicit: I + 0.5*dt*D)
        B_diag = 1.0 + beta
        B_sub = alpha[1:]
        B_sup = gamma[:-1]
        
        return alpha, beta, gamma, A_diag, A_sub, A_sup, B_diag, B_sub, B_sup
        
    def solve(self):
        """
        Solves the Parabolic Variational Inequality backward in time.
        Returns:
            V_grid: 2D array of option prices shape (N+1, M+1) where axis 0 is time t, axis 1 is S
            S_star: 1D array of free boundary values S*(t) at each time step
            solve_time: Elapsed time in seconds
        """
        start_time = time.perf_counter()
        
        alpha, beta, gamma, A_diag, A_sub, A_sup, B_diag, B_sub, B_sup = self._build_tridiagonal_matrices()
        
        # Initialize solution matrix V(t_n, S_j)
        # rows = time t (0 to T), cols = stock price S (0 to S_max)
        V_grid = np.zeros((self.N + 1, self.M + 1))
        
        # Terminal condition at maturity t = T (tau = 0)
        V_grid[-1, :] = self.payoff.copy()
        
        # Array to record free boundary S*(t) at each time step
        S_star = np.zeros(self.N + 1)
        S_star[-1] = self.K # At maturity S*(T) = K
        
        # Current solution vector for interior points
        V_current = self.payoff.copy()
        
        # Step backward in time from tau_0 to tau_N (t = T to t = 0)
        for n in range(self.N - 1, -1, -1):
            tau = (self.N - n) * self.dt # Time to maturity
            
            # Boundary conditions at time tau
            # For American put, immediate exercise at S=0 yields K cash: V_Amer(0, t) = K for all t
            BC_left = self.K                        # S = 0 (American boundary condition)
            BC_right = 0.0                          # S = S_max
            
            # Compute RHS vector = B * V_interior + boundary contributions
            # V_interior has indices 1 to M-1
            V_int = V_current[1:self.M]
            RHS = B_diag * V_int.copy()
            RHS[1:] += B_sub * V_int[:-1]
            RHS[:-1] += B_sup * V_int[1:]
            
            # Add explicit boundary term to first and last interior equations
            RHS[0] += alpha[0] * (V_current[0] + BC_left)
            RHS[-1] += gamma[-1] * (V_current[-1] + BC_right)
            
            # Solve A * V_new = RHS subject to V_new >= payoff using PSOR iteration
            V_new_int = V_int.copy()
            h_int = self.payoff[1:self.M]
            
            for it in range(self.max_iter):
                max_diff = 0.0
                for j in range(self.M - 1):
                    # Compute Gauss-Seidel sum
                    sum_val = 0.0
                    if j > 0:
                        sum_val += -alpha[j] * V_new_int[j - 1]
                    if j < self.M - 2:
                        sum_val += -gamma[j] * V_new_int[j + 1]
                        
                    # SOR update
                    V_sor = (RHS[j] - sum_val) / A_diag[j]
                    V_relaxed = (1.0 - self.omega) * V_new_int[j] + self.omega * V_sor
                    
                    # Project onto obstacle (The American early-exercise constraint)
                    V_projected = max(h_int[j], V_relaxed)
                    
                    diff = abs(V_projected - V_new_int[j])
                    if diff > max_diff:
                        max_diff = diff
                    V_new_int[j] = V_projected
                    
                if max_diff < self.tol:
                    break
                    
            # Assemble full solution vector at time t_n
            V_current[0] = BC_left
            V_current[1:self.M] = V_new_int
            V_current[-1] = BC_right
            V_grid[n, :] = V_current.copy()
            
            # Locate free boundary S*(t_n):
            # Highest S where V(S,t) is effectively on the obstacle (V == K - S)
            exercise_idx = np.where((V_current <= self.payoff + 1e-4) & (self.S < self.K))[0]
            if len(exercise_idx) > 0:
                S_star[n] = self.S[exercise_idx[-1]]
            else:
                S_star[n] = S_star[min(n+1, self.N)]
                
        solve_time = time.perf_counter() - start_time
        return V_grid, S_star, solve_time

    def compute_greeks(self, V_grid: np.ndarray):
        """Computes Delta and Gamma using central finite differences."""
        # Delta = dV/dS
        Delta = np.zeros_like(V_grid)
        Delta[:, 1:-1] = (V_grid[:, 2:] - V_grid[:, :-2]) / (2 * self.dS)
        Delta[:, 0] = -1.0 # At S=0
        Delta[:, -1] = 0.0 # As S -> S_max
        
        # Gamma = d2V/dS2
        Gamma = np.zeros_like(V_grid)
        Gamma[:, 1:-1] = (V_grid[:, 2:] - 2 * V_grid[:, 1:-1] + V_grid[:, :-2]) / (self.dS**2)
        
        return Delta, Gamma
