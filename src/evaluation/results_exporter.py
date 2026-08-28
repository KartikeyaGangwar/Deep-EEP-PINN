"""
Scientific Results Exporter: Saves all Quantitative Benchmark Outputs into Structured JSON and CSV files.
Maintains institutional experiment logs with system hardware metadata for reproducibility.
"""

import os
import sys
import json
import csv
import time
import platform
import datetime
import torch

def get_detailed_cpu_name() -> str:
    """
    Retrieves the exact CPU model name across Windows and Linux.
    """
    if sys.platform == "win32":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return cpu_name.strip()
        except Exception:
            return platform.processor()
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()

def get_system_hardware_profile() -> dict:
    """
    Captures complete host hardware specifications for scientific reproducibility.
    """
    profile = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cpu_model": get_detailed_cpu_name(),
        "cpu_logical_threads": os.cpu_count(),
    }
    
    # System RAM
    try:
        import psutil
        profile["cpu_physical_cores"] = psutil.cpu_count(logical=False)
        profile["system_ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        profile["cpu_physical_cores"] = "N/A"
        profile["system_ram_gb"] = "N/A"
        
    # GPU Hardware
    profile["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        profile["gpu_device_name"] = torch.cuda.get_device_name(0)
        profile["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        profile["cuda_version"] = torch.version.cuda
    else:
        profile["gpu_device_name"] = "CPU (No CUDA GPU)"
        profile["gpu_vram_gb"] = 0.0
        profile["cuda_version"] = "N/A"
        
    return profile

def append_to_master_json(stage_id: str, stage_data: dict, json_path: str = "results/benchmark_results.json"):
    """
    Appends structured benchmark metrics to a master JSON file.
    """
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    master_data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                master_data = json.load(f)
        except Exception:
            master_data = {}
            
    if "runs" not in master_data:
        master_data["runs"] = []
        
    run_entry = {
        "stage_id": stage_id,
        "hardware_profile": get_system_hardware_profile(),
        "metrics": stage_data
    }
    master_data["runs"].append(run_entry)
    master_data["latest_run"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(master_data, f, indent=4)
        
    print(f">> [JSON Export] Saved structured metrics to: {json_path}")

def save_phase1_csv(benchmark_results: dict, csv_path: str = "results/phase1_1d_results.csv"):
    """
    Exports Phase 1 1D American Option benchmark metrics to CSV.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    headers = ["Model Architecture", "Rel L2 Error (%)", "Max Error (Rs)", "MAE (Rs)", "Boundary RMSE (Rs)", "Eval Latency (ms)", "Train Time (s)"]
    
    rows = []
    for model_name, res in benchmark_results.items():
        rows.append([
            model_name,
            f"{res['rel_L2_error'] * 100:.2f}",
            f"{res['L_inf_error']:.2f}",
            f"{res['MAE']:.2f}",
            f"{res['boundary_rmse']:.2f}",
            f"{res['eval_time'] * 1000:.2f}",
            f"{res['train_time']:.2f}"
        ])
        
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
    print(f">> [CSV Export] Saved 1D benchmark table to: {csv_path}")

def save_multid_csv(results_summary: list, csv_path: str, summary_stats: dict = None):
    """
    Exports Multi-Asset (5D / 10D / 30D / 50D) pricing comparison tables to CSV.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    headers = ["Spot Price S0", "European Price (Rs)", "LSM Ground Truth (Rs)", "LSM Std Error (Rs)", "Novel EEP-PINN (Rs)", "Premium e_theta (Rs)", "Absolute Error (Rs)"]
    
    rows = []
    for r in results_summary:
        rows.append([
            f"{r['spot']:.1f}",
            f"{r['euro']:.2f}",
            f"{r['lsm']:.2f}",
            f"{r['lsm_se']:.2f}",
            f"{r['eep']:.2f}",
            f"{r.get('premium', 0.0):.2f}",
            f"{r['abs_error']:.2f}"
        ])
        
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
        
        if summary_stats:
            writer.writerow([])
            writer.writerow(["--- Summary Statistics ---"])
            for k, v in summary_stats.items():
                writer.writerow([k, v])
                
    print(f">> [CSV Export] Saved multi-asset table to: {csv_path}")
