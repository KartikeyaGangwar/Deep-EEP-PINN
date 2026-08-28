"""
Publication-Quality Visualization for Phase 3: 10-Asset Correlated Basket Option (d=10, 100k LSM paths).
Features:
- IEEE/SIAM academic LaTeX serif styling (Computer Modern math font).
- Explicit Phase 3 labeling with 100k LSM Monte Carlo ground truth.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
    "legend.fontsize": 10
})

def plot_10d_pricing_comparison(spot_levels, euro_prices, lsm_prices, lsm_errors, eep_prices, save_path="figures/phase3_10d_geom_pricing_comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9.5, 5.5))
    
    spots = np.array(spot_levels)
    plt.plot(spots, euro_prices, "g--", linewidth=2.0, label=r"$\mathrm{European\; Basket\; (Exact\;} V_{\mathrm{Euro}}\mathrm{)}$")
    plt.errorbar(spots, lsm_prices, yerr=1.96*np.array(lsm_errors), fmt="ko", capsize=5, capthick=1.5,
                 markersize=6, label=r"$\mathrm{LSM\; Monte\; Carlo\; (100k\; paths} \pm 1.96\mathrm{SE)}$")
    plt.plot(spots, eep_prices, "b-^", linewidth=2.5, markersize=7, label=r"$\mathrm{Novel\; 10D\; EEP-PINN\;} (V_{\mathrm{Euro}} + e_\theta)$")
    
    # Shade early exercise premium
    plt.fill_between(spots, euro_prices, eep_prices, color="purple", alpha=0.12, label=r"$\mathrm{Early\; Exercise\; Premium\;} e(\mathbf{S}, t)$")
    
    plt.axvline(x=100.0, color="gray", linestyle=":", alpha=0.7, label=r"$\mathrm{Strike\;} K = 100\$")
    plt.title(r"$\mathbf{Phase\; 3:\; 10-Asset\; Correlated\; American\; Basket\; Option\;} (d = 10, \rho = 0.35)$", pad=12, fontsize=12)
    plt.xlabel(r"$\mathrm{Basket\; Initial\; Spot\; Price\;} S_{0, i} \;(\$)$")
    plt.ylabel(r"$\mathrm{Option\; Fair\; Value\;} V(\mathbf{S}_0, 0) \;(\$)$")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="upper right", framealpha=0.95, fontsize=10)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> Saved Phase 3 Pricing Figure to: {save_path}")

def plot_10d_premium_slice(S1_grid, S2_grid, e_slice, save_path="figures/phase3_10d_geom_premium_surface_slice.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8.5, 6))
    
    plt.contourf(S1_grid, S2_grid, e_slice, levels=40, cmap="plasma")
    cbar = plt.colorbar()
    cbar.set_label(r"$\mathrm{Early\; Exercise\; Premium\;} e_\theta(S_1, S_2, \dots, t=0) \;(\$)$", labelpad=10)
    
    plt.title(r"$\mathbf{Phase\; 3:\; 10D\; Premium\; Manifold\; (Slice\;} S_3=\dots=S_{10}=100, \, t=0\mathbf{)}$", pad=12, fontsize=12)
    plt.xlabel(r"$\mathrm{Asset\; 1\; Price\;} S_1 \;(\$)$")
    plt.ylabel(r"$\mathrm{Asset\; 2\; Price\;} S_2 \;(\$)$")
    plt.grid(True, alpha=0.2, linestyle="--")
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> Saved Phase 3 Premium Slice to: {save_path}")

def export_10d_latex_table(results_data: list, save_path="figures/phase3_10d_geom_benchmark_summary_table.tex"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    latex_str = r"""\begin{table}[h!]
\centering
\small
\caption{Phase 3 High-Dimensional Benchmark: 10-Asset Correlated American Basket Put Option ($d=10, K=100, T=1.0, r=0.05, \rho=0.35, N_{\text{paths}}=100,000$).}
\label{tab:10d_benchmarks}
\begin{tabular}{@{}cccccc@{}}
\toprule
\textbf{Spot $S_0$} & \textbf{European Exact} & \textbf{LSM Monte Carlo (100k)} & \textbf{Novel 10D EEP-PINN} & \textbf{Premium $e_\theta$} & \textbf{Error vs LSM} \\ \midrule
"""
    for r in results_data:
        latex_str += f"\\$ {r['spot']:.1f} & \\$ {r['euro']:.2f} & \\$ {r['lsm']:.2f} $\\pm$ {r['lsm_se']:.2f} & \\$ {r['eep']:.2f} & \\$ {r.get('premium', 0.0):.2f} & \\$ {r['abs_error']:.2f} \\\\\n"
        
    latex_str += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f">> Saved Phase 3 LaTeX Table to: {save_path}")
