"""
Master Benchmark Pipeline: 1D American Option Free-Boundary Problem
Compares:
1. Classical Crank-Nicolson PSOR (Ground Truth Benchmark)
2. Method 1: Baseline Single-Network Penalty PINN (Standard Full-Surface Solver)
3. Method 2: Novel EEP-PINN (Early Exercise Premium Decomposition Solver)
"""

import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configs.default_config import default_config
from src.solvers.psor_solver import PSORSolver
from src.solvers.baseline_pinn import BaselineTrainer
from src.solvers.eep_pinn import EEPPINNTrainer
from src.evaluation.metrics import compute_error_metrics
from src.evaluation.visualizer import (
    plot_3d_surfaces,
    plot_free_boundary_comparison,
    plot_greeks_comparison,
    plot_error_heatmap,
    plot_benchmark_barchart,
    export_latex_table
)

def run_master_benchmark():
    print("=" * 80)
    print("  PHASE 1 RESEARCH BENCHMARK: 1D AMERICAN OPTION FREE-BOUNDARY PROBLEM")
    print("=" * 80)
    
    cfg = default_config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> Environment: PyTorch {torch.__version__} | Device: {device.upper()}")
    print(f">> Parameters : Strike K={cfg.K}, Expiry T={cfg.T}y, Rate r={cfg.r}, Vol sigma={cfg.sigma}")
    print(f">> Optimizer  : AdamW ({cfg.epochs_adam} eps) + L-BFGS ({cfg.epochs_lbfgs} iters)")
    
    # -------------------------------------------------------------
    # Method 0: Classical Crank-Nicolson PSOR (Ground Truth)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[1/3] Solving via Classical Crank-Nicolson PSOR (Ground Truth Benchmark)...")
    psor = PSORSolver(cfg)
    V_psor, S_star_psor, psor_time = psor.solve()
    Delta_psor, Gamma_psor = psor.compute_greeks(V_psor)
    print(f"      PSOR Completed in {psor_time*1000:.2f} ms")
    
    # -------------------------------------------------------------
    # Method 1: Standard Single-Network Penalty PINN (Baseline)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[2/3] Training Standard Single-Network Penalty PINN (Baseline)...")
    base_trainer = BaselineTrainer(cfg, device=device)
    base_time = base_trainer.train(verbose=False)
    V_base, Delta_base, Gamma_base, base_eval_time = base_trainer.evaluate_grid(psor.S, psor.t)
    S_star_base = base_trainer.extract_free_boundary(psor.S, psor.t, V_base)
    print(f"      Baseline PINN Completed in {base_time:.2f}s | Latency: {base_eval_time*1000:.2f} ms")
    
    # -------------------------------------------------------------
    # Method 2: Novel EEP-PINN (Early Exercise Premium Decomposition)
    # -------------------------------------------------------------
    print("\n" + "-" * 80)
    print("[3/3] Training Novel EEP-PINN (Early Exercise Premium Decomposition Solver)...")
    eep_trainer = EEPPINNTrainer(cfg, device=device)
    eep_time = eep_trainer.train(verbose=False)
    V_eep, Delta_eep, Gamma_eep, e_grid, eep_eval_time = eep_trainer.evaluate_grid(psor.S, psor.t)
    S_star_eep = eep_trainer.extract_free_boundary(psor.S, psor.t, V_eep)
    print(f"      Novel EEP-PINN Completed in {eep_time:.2f}s | Latency: {eep_eval_time*1000:.2f} ms")
    
    learned_weights = eep_trainer.model.get_adaptive_weights()
    print(f"      >> Learned Dynamic Precision: PDE={learned_weights['w_pde']:.2f}, "
          f"Obstacle={learned_weights['w_obstacle']:.2f}, BC0={learned_weights['w_bc0']:.2f}")
    
    # -------------------------------------------------------------
    # Compute Metrics and Formatted Reporting
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                    QUANTITATIVE BENCHMARK REPORT")
    print("=" * 80)
    
    def get_model_stats(V_pred, S_star_pred, eval_time, train_time):
        m = compute_error_metrics(V_psor, V_pred)
        b_rmse = np.sqrt(np.mean((S_star_pred - S_star_psor)**2))
        m["boundary_rmse"] = b_rmse
        m["eval_time"] = eval_time
        m["train_time"] = train_time
        return m
        
    benchmark_results = {
        "Baseline PINN (Standard)": get_model_stats(V_base, S_star_base, base_eval_time, base_time),
        "Novel EEP-PINN (Proposed)": get_model_stats(V_eep, S_star_eep, eep_eval_time, eep_time)
    }
    
    header = f"{'Model Architecture':<30} | {'Rel. L2 Error':<13} | {'Max Error':<11} | {'MAE':<10} | {'Boundary RMSE':<14} | {'Eval Latency':<12}"
    print(header)
    print("-" * len(header))
    for name, res in benchmark_results.items():
        print(f"{name:<30} | {res['rel_L2_error']*100:>11.2f}% | Rs. {res['L_inf_error']:>7.2f} | Rs. {res['MAE']:>6.2f} | Rs. {res['boundary_rmse']:>10.2f} | {res['eval_time']*1000:>9.2f} ms")
        
    print("=" * 80)
    
    # -------------------------------------------------------------
    # Generate Publication Figures and LaTeX Tables
    # -------------------------------------------------------------
    print("\n>> Generating High-Resolution Publication Figures & LaTeX Table...")
    plot_3d_surfaces(psor.S, psor.t, V_psor, V_eep, e_grid=e_grid, save_path="figures/phase1_1d_option_price_surface_3d.png")
    plot_free_boundary_comparison(psor.t, S_star_psor, S_star_base, S_star_eep, K=cfg.K,
                                  save_path="figures/phase1_1d_free_boundary_comparison.png")
    plot_greeks_comparison(psor.S, Delta_psor, Delta_eep, Gamma_psor, Gamma_eep,
                           t_indices=[0, int(cfg.N_time*0.5), int(cfg.N_time*0.85)],
                           t_grid=psor.t, save_path="figures/phase1_1d_greeks_comparison.png")
    export_latex_table(benchmark_results, save_path="figures/phase1_1d_benchmark_summary_table.tex")
    
    # -------------------------------------------------------------
    # Export Structured JSON and CSV
    # -------------------------------------------------------------
    from src.evaluation.results_exporter import append_to_master_json, save_phase1_csv
    save_phase1_csv(benchmark_results, csv_path="results/phase1_1d_results.csv")
    append_to_master_json("phase1_1d_american_option", {
        "models": {k: {m_k: float(m_v) if isinstance(m_v, (np.floating, float)) else m_v for m_k, m_v in v.items()} 
                   for k, v in benchmark_results.items()}
    })
    
    print("\n>> All Phase 1 Deliverables (Figures, LaTeX, CSV, JSON) Generated Successfully!")

if __name__ == "__main__":
    run_master_benchmark()
