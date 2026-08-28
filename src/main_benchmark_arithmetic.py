"""
Master Benchmark Pipeline: High-Dimensional American Arithmetic Basket Options (d=5).
Compares:
1. Analytical Closed-Form Moment-Matched European Arithmetic Basket Benchmark
2. Arithmetic Longstaff-Schwartz Least Squares Monte Carlo (100,000 paths) - Ground Truth
3. Novel Arithmetic Multi-Asset EEP-PINN (Mesh-Free Neural Solver)
"""

import sys
import os
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configs.multi_asset_config import default_multi_config
from src.solvers.analytical_arithmetic import multi_asset_arithmetic_european_put
from src.solvers.lsm_arithmetic import LongstaffSchwartzArithmeticSolver
from src.solvers.eep_pinn_arithmetic import ArithmeticEEPTrainer
from src.evaluation.visualizer_arithmetic import (
    plot_arithmetic_pricing_comparison,
    plot_arithmetic_premium_slice,
    export_arithmetic_latex_table
)

def run_arithmetic_master_benchmark():
    print("=" * 90)
    print("   AMERICAN ARITHMETIC BASKET OPTION BENCHMARK (d=5, 100k LSM paths)")
    print("=" * 90)
    
    cfg = default_multi_config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> Environment    : PyTorch {torch.__version__} | Device: {device.upper()}")
    print(f">> Basket Specs   : {cfg.d} Correlated Assets | Strike K={cfg.K}, Expiry T={cfg.T}y, Rate r={cfg.r}")
    print(f">> Asset Vols     : {cfg.volatilities}")
    print(f">> Cross-Corr rho : {cfg.rho}")
    print(f">> Payoff Type    : Arithmetic max(K - sum(w_i S_i), 0)")
    print(f">> Optimizer      : AdamW ({cfg.epochs_adam} eps) + L-BFGS ({cfg.epochs_lbfgs} iters)")
    
    # -------------------------------------------------------------
    # 1. Train Novel Arithmetic Multi-Asset EEP-PINN
    # -------------------------------------------------------------
    print("\n" + "-" * 90)
    print("[1/2] Training Novel Arithmetic Multi-Asset EEP-PINN on GPU...")
    pinn_trainer = ArithmeticEEPTrainer(cfg, device=device)
    pinn_train_time = pinn_trainer.train(verbose=True)
    
    # -------------------------------------------------------------
    # 2. Benchmark Across Moneyness Levels S_0 in [80, 90, 100, 110, 120]
    # -------------------------------------------------------------
    print("\n" + "-" * 90)
    print("[2/2] Evaluating Arithmetic Ground Truth (LSM 100,000 paths) vs. Novel Arithmetic EEP-PINN...")
    lsm_solver = LongstaffSchwartzArithmeticSolver(cfg)
    
    spot_levels = [80.0, 90.0, 100.0, 110.0, 120.0]
    results_summary = []
    
    euro_prices = []
    lsm_prices = []
    lsm_errors = []
    eep_prices = []
    
    total_lsm_time = 0.0
    total_eep_eval_time = 0.0
    
    print("\n" + "=" * 90)
    print(f"{'Spot S_0':<10} | {'European Arith MM':<18} | {'Arithmetic LSM (100k)':<25} | {'Arith EEP-PINN':<16} | {'Diff vs LSM':<12}")
    print("-" * 90)
    
    for s_val in spot_levels:
        S_vec = np.full(cfg.d, s_val)
        
        # 1. European Arithmetic Price (Moment Matching)
        V_euro = multi_asset_arithmetic_european_put(S_vec[None, :], np.array([0.0]), cfg)[0]
        
        # 2. Arithmetic LSM Monte Carlo Ground Truth
        V_lsm, se_lsm, lsm_time = lsm_solver.price(S_vec)
        total_lsm_time += lsm_time
        
        # 3. Novel Arithmetic EEP-PINN Price
        V_eep, e_prem, eep_time = pinn_trainer.price_spot(S_vec, t_val=0.0)
        total_eep_eval_time += eep_time
        
        diff = abs(V_eep - V_lsm)
        
        euro_prices.append(V_euro)
        lsm_prices.append(V_lsm)
        lsm_errors.append(se_lsm)
        eep_prices.append(V_eep)
        
        results_summary.append({
            "spot": s_val,
            "euro": V_euro,
            "lsm": V_lsm,
            "lsm_se": se_lsm,
            "eep": V_eep,
            "premium": e_prem,
            "abs_error": diff
        })
        
        print(f"Rs. {s_val:<6.1f} | Rs. {V_euro:>14.2f} | Rs. {V_lsm:>7.2f} +/- {se_lsm:<5.2f} ({lsm_time*1000:>5.1f}ms) | Rs. {V_eep:>10.2f} ({eep_time*1000:>4.2f}ms) | Rs. {diff:>8.2f}")
        
    print("=" * 90)
    
    # -------------------------------------------------------------
    # 3. Compute 2D Arithmetic Premium Slice (S_1, S_2) with S_3=S_4=S_5=100
    # -------------------------------------------------------------
    print("\n>> Computing 5D Arithmetic Early-Exercise Premium Manifold Slice...")
    n_slice = 60
    s_range = np.linspace(60.0, 140.0, n_slice)
    S1_grid, S2_grid = np.meshgrid(s_range, s_range)
    
    slice_points = np.zeros((n_slice * n_slice, cfg.d))
    slice_points[:, 0] = S1_grid.flatten()
    slice_points[:, 1] = S2_grid.flatten()
    slice_points[:, 2] = 100.0
    slice_points[:, 3] = 100.0
    slice_points[:, 4] = 100.0
    
    t_slice = torch.zeros((n_slice * n_slice, 1), dtype=torch.float32, device=device)
    S_slice_tensor = torch.tensor(slice_points, dtype=torch.float32, device=device)
    
    pinn_trainer.model.eval()
    with torch.no_grad():
        e_slice_flat = pinn_trainer.model.forward_premium(S_slice_tensor, t_slice).cpu().numpy()
        
    e_slice = e_slice_flat.reshape((n_slice, n_slice))
    
    # -------------------------------------------------------------
    # 4. Generate Visualizations and LaTeX Tables
    # -------------------------------------------------------------
    print("\n>> Generating High-Resolution Phase 2B Arithmetic Publication Figures & LaTeX Table...")
    plot_arithmetic_pricing_comparison(spot_levels, euro_prices, lsm_prices, lsm_errors, eep_prices,
                                       save_path="figures/phase2b_5d_arith_pricing_comparison.png")
    plot_arithmetic_premium_slice(S1_grid, S2_grid, e_slice,
                                  save_path="figures/phase2b_5d_arith_premium_surface_slice.png")
    export_arithmetic_latex_table(results_summary,
                                  save_path="figures/phase2b_5d_arith_benchmark_summary_table.tex")
    
    avg_lsm_latency = (total_lsm_time / len(spot_levels)) * 1000
    avg_eep_latency = (total_eep_eval_time / len(spot_levels)) * 1000
    speedup = avg_lsm_latency / max(avg_eep_latency, 1e-4)
    
    # -------------------------------------------------------------
    # 5. Export Structured JSON and CSV
    # -------------------------------------------------------------
    from src.evaluation.results_exporter import append_to_master_json, save_multid_csv
    summary_stats = {
        "avg_lsm_latency_ms": f"{avg_lsm_latency:.2f}",
        "avg_eep_latency_ms": f"{avg_eep_latency:.2f}",
        "speedup_factor": f"{speedup:.1f}x",
        "max_diff_vs_lsm_rs": f"{max(r['abs_error'] for r in results_summary):.2f}",
        "mean_absolute_diff_rs": f"{np.mean([r['abs_error'] for r in results_summary]):.2f}"
    }
    save_multid_csv(results_summary, csv_path="results/phase2b_5d_arithmetic_results.csv", summary_stats=summary_stats)
    append_to_master_json("phase2b_5d_arithmetic_basket", {
        "pricing_table": results_summary,
        "summary_statistics": summary_stats,
        "train_time_seconds": round(pinn_train_time, 2)
    })
    
    print(f">> All Phase 2B Deliverables (Figures, LaTeX, CSV, JSON) Generated Successfully!")

if __name__ == "__main__":
    run_arithmetic_master_benchmark()
