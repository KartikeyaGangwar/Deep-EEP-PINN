"""
Ultra-High-Dimensional Master Benchmark Pipeline: Arbitrary d-Asset American Basket Options (d=10, 30, 50).
Compares:
1. Exact Closed-Form European Basket Benchmark
2. Longstaff-Schwartz Least Squares Monte Carlo (100,000 correlated paths) - Ground Truth
3. Novel High-Dimensional EEP-PINN (Mesh-Free Neural Solver)
"""

import sys
import os
import argparse
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configs.multi_asset_config import config_10d, config_30d, config_50d, MultiAssetConfig
from src.solvers.analytical_multid import multi_asset_geometric_european_put
from src.solvers.lsm_monte_carlo import LongstaffSchwartzSolver
from src.solvers.eep_pinn_multid import MultiAssetEEPTrainer

def run_highdim_benchmark(dim=30):
    if dim == 10:
        cfg = config_10d
    elif dim == 30:
        cfg = config_30d
    elif dim == 50:
        cfg = config_50d
    else:
        cfg = MultiAssetConfig(
            d=dim,
            K=100.0,
            T=1.0,
            r=0.05,
            volatilities=[0.18 + 0.08 * (i % 5) / 4.0 for i in range(dim)],
            weights=[1.0 / dim] * dim,
            rho=0.30,
            mc_paths=100000,
            num_collocation=min(15000, 500 * dim),
            epochs_adam=1200,
            epochs_lbfgs=50
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 90)
    print(f"   ULTRA-HIGH-DIMENSIONAL AMERICAN BASKET BENCHMARK (d={cfg.d} ASSETS, 100k LSM PATHS)")
    print("=" * 90)
    print(f">> Environment    : PyTorch {torch.__version__} | Device: {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f">> GPU Device Name: {gpu_name} ({gpu_vram:.1f} GB VRAM)")
    print(f">> Basket Specs   : {cfg.d} Correlated Assets | Strike K={cfg.K}, Expiry T={cfg.T}y, Rate r={cfg.r}")
    print(f">> Cross-Corr rho : {cfg.rho} ({cfg.d * (cfg.d - 1) // 2} Pairwise Correlation Terms)")
    print(f">> Optimizer      : AdamW ({cfg.epochs_adam} eps) + L-BFGS ({cfg.epochs_lbfgs} iters)")
    
    # -------------------------------------------------------------
    # 1. Train Novel High-Dimensional EEP-PINN
    # -------------------------------------------------------------
    print("\n" + "-" * 90)
    print(f"[1/2] Training Novel Multi-Asset EEP-PINN for d={cfg.d} on GPU...")
    pinn_trainer = MultiAssetEEPTrainer(cfg, device=device)
    pinn_train_time = pinn_trainer.train(verbose=True)
    
    # -------------------------------------------------------------
    # 2. Benchmark Across Moneyness Levels S_0 in [80, 90, 100, 110, 120]
    # -------------------------------------------------------------
    print("\n" + "-" * 90)
    print(f"[2/2] Evaluating Multi-Asset Ground Truth (LSM 100,000 paths) vs. Novel EEP-PINN...")
    lsm_solver = LongstaffSchwartzSolver(cfg)
    
    spot_levels = [80.0, 90.0, 100.0, 110.0, 120.0]
    results_summary = []
    
    total_lsm_time = 0.0
    total_eep_eval_time = 0.0
    
    print("\n" + "=" * 90)
    print(f"{'Spot S_0':<10} | {'European Exact':<15} | {'LSM Ground Truth (100k)':<25} | {'Novel EEP-PINN':<16} | {'Diff vs LSM':<12}")
    print("-" * 90)
    
    for s_val in spot_levels:
        S_vec = np.full(cfg.d, s_val)
        
        # 1. Exact European Price
        V_euro = multi_asset_geometric_european_put(S_vec[None, :], np.array([0.0]), cfg)[0]
        
        # 2. LSM Monte Carlo Ground Truth
        V_lsm, se_lsm, lsm_time = lsm_solver.price(S_vec)
        total_lsm_time += lsm_time
        
        # 3. Novel EEP-PINN Price
        V_eep, e_prem, eep_time = pinn_trainer.price_spot(S_vec, t_val=0.0)
        total_eep_eval_time += eep_time
        
        diff = abs(V_eep - V_lsm)
        
        results_summary.append({
            "spot": s_val,
            "euro": V_euro,
            "lsm": V_lsm,
            "lsm_se": se_lsm,
            "eep": V_eep,
            "premium": e_prem,
            "abs_error": diff
        })
        
        print(f"Rs. {s_val:<6.1f} | Rs. {V_euro:>11.2f} | Rs. {V_lsm:>7.2f} +/- {se_lsm:<5.2f} ({lsm_time*1000:>5.1f}ms) | Rs. {V_eep:>10.2f} ({eep_time*1000:>4.2f}ms) | Rs. {diff:>8.2f}")
        
    print("=" * 90)
    avg_lsm_latency = (total_lsm_time / len(spot_levels)) * 1000
    avg_eep_latency = (total_eep_eval_time / len(spot_levels)) * 1000
    speedup = avg_lsm_latency / max(avg_eep_latency, 1e-4)
    
    # -------------------------------------------------------------
    # 3. Generate Publication Figures and LaTeX Tables
    # -------------------------------------------------------------
    from src.evaluation.visualizer_10d import plot_10d_pricing_comparison, export_10d_latex_table
    euro_prices = [r["euro"] for r in results_summary]
    lsm_prices = [r["lsm"] for r in results_summary]
    lsm_errors = [r["lsm_se"] for r in results_summary]
    eep_prices = [r["eep"] for r in results_summary]
    
    plot_10d_pricing_comparison(spot_levels, euro_prices, lsm_prices, lsm_errors, eep_prices,
                                save_path=f"figures/scalability_{cfg.d}d_pricing_comparison.png")
    export_10d_latex_table(results_summary,
                           save_path=f"figures/scalability_{cfg.d}d_benchmark_summary_table.tex")

    # -------------------------------------------------------------
    # 4. Export Structured JSON and CSV
    # -------------------------------------------------------------
    from src.evaluation.results_exporter import append_to_master_json, save_multid_csv
    summary_stats = {
        "avg_lsm_latency_ms": f"{avg_lsm_latency:.2f}",
        "avg_eep_latency_ms": f"{avg_eep_latency:.2f}",
        "speedup_factor": f"{speedup:.1f}x",
        "max_diff_vs_lsm_rs": f"{max(r['abs_error'] for r in results_summary):.2f}",
        "mean_absolute_diff_rs": f"{np.mean([r['abs_error'] for r in results_summary]):.2f}"
    }
    save_multid_csv(results_summary, csv_path=f"results/scalability_{cfg.d}d_results.csv", summary_stats=summary_stats)
    append_to_master_json(f"scalability_{cfg.d}d_basket", {
        "dimension": cfg.d,
        "pricing_table": results_summary,
        "summary_statistics": summary_stats,
        "train_time_seconds": round(pinn_train_time, 2)
    })
    
    print(f">> Deliverables for d={cfg.d} (Figures, LaTeX, CSV, JSON) Generated Successfully in 'figures/' and 'results/'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=30, help="Dimension of basket (e.g. 10, 30, 50)")
    args = parser.parse_args()
    run_highdim_benchmark(dim=args.d)
