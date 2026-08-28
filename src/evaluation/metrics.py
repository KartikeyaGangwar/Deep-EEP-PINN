"""
Evaluation and Benchmarking Metrics for Scientific Computing.
Computes relative L2 error, Linf error, Mean Absolute Error (MAE),
Greek gradients accuracy, and inference speedup factors.
"""

import numpy as np

def compute_error_metrics(V_true: np.ndarray, V_pred: np.ndarray):
    """
    Computes standard scientific error metrics between ground truth and neural solver.
    """
    diff = V_pred - V_true
    
    # 1. Pointwise Maximum Error (L_infinity norm)
    linf_error = np.max(np.abs(diff))
    
    # 2. Relative L2 Error (Standard paper metric)
    l2_error_relative = np.linalg.norm(diff) / np.maximum(np.linalg.norm(V_true), 1e-10)
    
    # 3. Mean Absolute Error (MAE)
    mae = np.mean(np.abs(diff))
    
    # 4. Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean(diff**2))
    
    return {
        "rel_L2_error": l2_error_relative,
        "L_inf_error": linf_error,
        "MAE": mae,
        "RMSE": rmse
    }

def print_benchmark_table(metrics: dict, psors_time: float, pinn_train_time: float, pinn_eval_time: float):
    """
    Prints a formatted ASCII benchmark table for dissertation/paper reporting.
    """
    speedup = psors_time / max(pinn_eval_time, 1e-6)
    
    print("\n" + "=" * 70)
    print("        SCIENTIFIC BENCHMARK REPORT: PSOR vs. NEURAL PINN")
    print("=" * 70)
    print(f"  Relative L2 Error (||V_pinn - V_psor|| / ||V_psor||) : {metrics['rel_L2_error']:.4%}")
    print(f"  Maximum Absolute Error (L_infinity norm)           : Rs. {metrics['L_inf_error']:.4f}")
    print(f"  Mean Absolute Error (MAE)                          : Rs. {metrics['MAE']:.4f}")
    print(f"  Root Mean Squared Error (RMSE)                     : Rs. {metrics['RMSE']:.4f}")
    print("-" * 70)
    print(f"  Classical PSOR Solve Time                          : {psors_time*1000:.2f} ms")
    print(f"  PINN Total Training Time                           : {pinn_train_time:.2f} s")
    print(f"  PINN Instantaneous Evaluation Latency              : {pinn_eval_time*1000:.2f} ms")
    print(f"  Inference Speedup Factor                           : {speedup:.1f}x Faster")
    print("=" * 70 + "\n")
