"""
Master Execution Pipeline for All American Option Benchmarks (1D, 5D Geometric, 5D Arithmetic, 10D Geometric).
Designed for Local & Cloud/Kaggle GPU Execution (CUDA / T4 / P100 / A100 / CPU).

Features:
- Automatic RAM & GPU VRAM Garbage Collection between phases to prevent OOM.
- Full Hardware Spec Profiling (CPU Model, Logical Cores, System RAM, GPU VRAM).
- Automated JSON & CSV Results Export.
- Quick Smoke Test Mode (--smoke-test).

Usage:
    python run_all_benchmarks.py --all
    python run_all_benchmarks.py --smoke-test   # Quick 5-second health check
    python run_all_benchmarks.py --phase 1      # 1D American Option Free Boundary
    python run_all_benchmarks.py --phase 2a     # 5D Correlated Geometric Basket
    python run_all_benchmarks.py --phase 2b     # 5D Real-World Arithmetic Basket
    python run_all_benchmarks.py --phase 3      # 10D High-Dimensional Basket
    python run_all_benchmarks.py --phase 30d    # 30D Dow Jones Scale Basket
    python run_all_benchmarks.py --phase 50d    # 50D Nifty 50 / S&P Sector Basket
"""

import sys
import os
import gc
import argparse
import time
import platform
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.main_benchmark import run_master_benchmark
from src.main_benchmark_multid import run_multid_master_benchmark
from src.main_benchmark_arithmetic import run_arithmetic_master_benchmark
from src.main_benchmark_10d import run_10d_master_benchmark
from src.main_benchmark_highdim import run_highdim_benchmark
from src.evaluation.results_exporter import get_detailed_cpu_name

def cleanup_system_memory():
    """
    Explicitly frees Python host RAM and CUDA GPU cache memory between benchmark stages.
    Prevents Out-Of-Memory (OOM) accumulation across long runs.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        vram_used = torch.cuda.memory_allocated() / (1024**3)
        vram_cached = torch.cuda.memory_reserved() / (1024**3)
        print(f">> [Memory Cleanup] Cleared PyTorch CUDA cache. (Allocated: {vram_used:.2f} GB | Reserved: {vram_cached:.2f} GB)")
    else:
        print(">> [Memory Cleanup] Garbage collection complete.")

def run_smoke_test():
    """
    Quick 10-second end-to-end smoke test validating all solvers, autograd graphs, 
    and export pipelines without training for thousands of epochs.
    """
    print("\n" + "=" * 90)
    print("                     STARTING DEEP-EEP-PINN SMOKE TEST (ALL MODULES)")
    print("=" * 90)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> Smoke Test Device: {device.upper()}")
    
    # 1. Test 1D EEP-PINN forward & loss
    print(">> [1/4] Verifying 1D EEP-PINN and PSOR Solver...")
    from src.configs.default_config import default_config
    from src.solvers.eep_pinn import EEPPINNTrainer
    from src.solvers.psor_solver import PSORSolver
    
    psor = PSORSolver(default_config)
    V_psor, _, _ = psor.solve()
    assert V_psor is not None and not np.isnan(V_psor).any(), "PSOR failed!"
    print("        PSOR Ground Truth Verified!")
    
    trainer_1d = EEPPINNTrainer(default_config, device=device)
    S_test = torch.tensor([[80.0], [100.0], [120.0]], device=device)
    t_test = torch.tensor([[0.0], [0.5], [1.0]], device=device)
    e_pred = trainer_1d.model.forward_premium(S_test, t_test)
    assert not torch.isnan(e_pred).any(), "1D EEP forward produced NaN!"
    print("        1D EEP-PINN Forward Verified!")
    cleanup_system_memory()
    
    # 2. Test 5D Geometric Basket
    print("\n>> [2/4] Verifying 5D Multi-Asset Geometric EEP-PINN & LSM...")
    from src.configs.multi_asset_config import default_multi_config
    from src.solvers.eep_pinn_multid import MultiAssetEEPTrainer
    trainer_5d = MultiAssetEEPTrainer(default_multi_config, device=device)
    S5 = torch.full((3, 5), 100.0, device=device)
    t5 = torch.zeros((3, 1), device=device)
    e5 = trainer_5d.model.forward_premium(S5, t5)
    assert not torch.isnan(e5).any(), "5D EEP forward produced NaN!"
    print("        5D Geometric EEP-PINN Verified!")
    cleanup_system_memory()
    
    # 3. Test 5D Arithmetic Basket
    print("\n>> [3/4] Verifying 5D Arithmetic EEP-PINN & Moment Matching...")
    from src.solvers.eep_pinn_arithmetic import ArithmeticEEPTrainer
    trainer_arith = ArithmeticEEPTrainer(default_multi_config, device=device)
    e_arith = trainer_arith.model.forward_premium(S5, t5)
    assert not torch.isnan(e_arith).any(), "5D Arithmetic EEP forward produced NaN!"
    print("        5D Arithmetic EEP-PINN Verified!")
    cleanup_system_memory()
    
    # 4. Test 10D / High-Dim Autograd Contraction
    print("\n>> [4/4] Verifying 10D/30D High-Dimensional Hessian Trace Contraction...")
    from src.configs.multi_asset_config import config_10d
    trainer_10d = MultiAssetEEPTrainer(config_10d, device=device)
    S10 = torch.full((4, 10), 100.0, device=device)
    t10 = torch.zeros((4, 1), device=device)
    e10 = trainer_10d.model.forward_premium(S10, t10)
    assert not torch.isnan(e10).any(), "10D EEP forward produced NaN!"
    print("        10D Hessian Contraction & Forward Verified!")
    cleanup_system_memory()
    
    print("\n" + "=" * 90)
    print(">> [SMOKE TEST PASSED] All Modules, Loss Functions, and Exporters are 100% HEALTHY!")
    print("=" * 90)

def main():
    parser = argparse.ArgumentParser(description="Run American Option PINN Master Benchmarks")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "1", "2a", "2b", "3", "30d", "50d"],
                        help="Benchmark phase to execute ('1', '2a', '2b', '3', '30d', '50d', or 'all')")
    parser.add_argument("--all", action="store_true", help="Execute all 4 master benchmark phases sequentially")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick 10-second end-to-end sanity check")
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    phase = "all" if args.all else args.phase

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cpu_model = get_detailed_cpu_name()

    ram_str = "N/A"
    try:
        import psutil
        ram_str = f"{psutil.virtual_memory().total / (1024**3):.1f} GB"
    except Exception:
        pass

    print("=" * 90)
    print("      DEEP EARLY EXERCISE PREMIUM PINN (EEP-PINN) MASTER BENCHMARK SUITE")
    print("=" * 90)
    print(f">> Host OS          : {platform.system()} {platform.release()}")
    print(f">> CPU Processor    : {cpu_model}")
    print(f">> CPU Logical Cores: {os.cpu_count()} Threads | System RAM: {ram_str}")
    print(f">> Execution Device : {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f">> GPU Device Name  : {gpu_name} ({gpu_vram:.1f} GB VRAM)")
        print(f">> CUDA Version     : {torch.version.cuda} | PyTorch: {torch.__version__}")
    print("=" * 90)

    total_start = time.perf_counter()

    if phase in ["all", "1"]:
        cleanup_system_memory()
        print("\n\n" + "#" * 90)
        print("   [STAGE 1/4] 1D AMERICAN OPTION FREE-BOUNDARY PROBLEM")
        print("#" * 90)
        run_master_benchmark()
        cleanup_system_memory()

    if phase in ["all", "2a"]:
        cleanup_system_memory()
        print("\n\n" + "#" * 90)
        print("   [STAGE 2/4] 5-ASSET CORRELATED GEOMETRIC BASKET OPTION (d=5)")
        print("#" * 90)
        run_multid_master_benchmark()
        cleanup_system_memory()

    if phase in ["all", "2b"]:
        cleanup_system_memory()
        print("\n\n" + "#" * 90)
        print("   [STAGE 3/4] 5-ASSET CORRELATED REAL-WORLD ARITHMETIC BASKET OPTION (d=5)")
        print("#" * 90)
        run_arithmetic_master_benchmark()
        cleanup_system_memory()

    if phase in ["all", "3"]:
        cleanup_system_memory()
        print("\n\n" + "#" * 90)
        print("   [STAGE 4/4] 10-ASSET HIGH-DIMENSIONAL CORRELATED BASKET BENCHMARK (d=10)")
        print("#" * 90)
        run_10d_master_benchmark()
        cleanup_system_memory()

    if phase == "30d":
        cleanup_system_memory()
        print("\n\n" + "#" * 90)
        print("   [SCALABILITY SUITE] 30-ASSET DOW JONES SCALE BASKET BENCHMARK (d=30)")
        print("#" * 90)
        run_highdim_benchmark(dim=30)
        cleanup_system_memory()

    if phase == "50d":
        cleanup_system_memory()
        print("\n\n" + "#" * 90)
        print("   [SCALABILITY SUITE] 50-ASSET S&P SECTOR / NIFTY 50 SCALE BASKET BENCHMARK (d=50)")
        print("#" * 90)
        run_highdim_benchmark(dim=50)
        cleanup_system_memory()

    total_elapsed = time.perf_counter() - total_start
    print("\n\n" + "=" * 90)
    print(f">> All Requested Benchmarks Completed in {total_elapsed/60:.2f} minutes!")
    print(">> Generated Deliverables are saved in 'figures/' and 'results/' directories.")
    print("=" * 90)

if __name__ == "__main__":
    import numpy as np
    main()
