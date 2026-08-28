"""
Publication-Quality Visualization for Scalability Suite: 30D (Dow Jones) and 50D (Nifty 50) Basket Options.
Features:
- IEEE/SIAM academic LaTeX serif styling (Computer Modern math font).
- Explicit Scalability labeling with 100k LSM Monte Carlo ground truth.
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

def plot_highdim_pricing_comparison(dim, spot_levels, euro_prices, lsm_prices, lsm_errors, eep_prices, save_path=None):
    if save_path is None:
        save_path = f"figures/scalability_{dim}d_pricing_comparison.png"
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9.5, 5.5))
    
    spots = np.array(spot_levels)
    corr_pairs = dim * (dim - 1) // 2
    market_name = "Dow Jones Industrial Average (DJIA 30)" if dim == 30 else ("Nifty 50 / Euro Stoxx 50" if dim == 50 else f"{dim}-Asset Basket")
    
    plt.plot(spots, euro_prices, "g--", linewidth=2.0, label=r"$\mathrm{European\; Basket\; (Exact\;} V_{\mathrm{Euro}}\mathrm{)}$")
    plt.errorbar(spots, lsm_prices, yerr=1.96*np.array(lsm_errors), fmt="ko", capsize=5, capthick=1.5,
                 markersize=6, label=r"$\mathrm{LSM\; Monte\; Carlo\; (100k\; paths} \pm 1.96\mathrm{SE)}$")
    plt.plot(spots, eep_prices, "b-^", linewidth=2.5, markersize=7, label=rf"$\mathrm{{Novel\; {dim}D\; EEP-PINN\;}} (V_{{\mathrm{{Euro}}}} + e_\theta)$")
    
    plt.fill_between(spots, euro_prices, eep_prices, color="purple", alpha=0.12, label=r"$\mathrm{Early\; Exercise\; Premium\;} e(\mathbf{S}, t)$")
    
    plt.axvline(x=100.0, color="gray", linestyle=":", alpha=0.7, label=r"$\mathrm{Strike\;} K = 100\$")
    plt.title(rf"$\mathbf{{Scalability\; Suite:\; {dim}-Asset\; {market_name}\;}} (d = {dim}, \, {corr_pairs}\; \mathrm{{Pairs}})$", pad=12, fontsize=12)
    plt.xlabel(r"$\mathrm{Basket\; Initial\; Spot\; Price\;} S_{0, i} \;(\$)$")
    plt.ylabel(r"$\mathrm{Option\; Fair\; Value\;} V(\mathbf{S}_0, 0) \;(\$)$")
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="upper right", framealpha=0.95, fontsize=10)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> Saved Scalability {dim}D Pricing Figure to: {save_path}")

def export_highdim_latex_table(dim, results_data: list, save_path=None):
    if save_path is None:
        save_path = f"figures/scalability_{dim}d_benchmark_summary_table.tex"
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    corr_pairs = dim * (dim - 1) // 2
    
    latex_str = "\\begin{table}[h!]\n\\centering\n\\small\n"
    latex_str += f"\\caption{{Scalability Benchmark: {dim}-Asset Correlated American Basket Put Option ($d={dim}, K=100, T=1.0, r=0.05, {corr_pairs}\\text{{ Correlation Pairs}}, N_{{\\text{{paths}}}}=100,000$).}}\n"
    latex_str += f"\\label{{tab:scalability_{dim}d_benchmarks}}\n"
    latex_str += "\\begin{tabular}{@{}cccccc@{}}\n\\toprule\n"
    latex_str += f"\\textbf{{Spot $S_0$}} & \\textbf{{European Exact}} & \\textbf{{LSM Monte Carlo (100k)}} & \\textbf{{Novel {dim}D EEP-PINN}} & \\textbf{{Premium $e_\\theta$}} & \\textbf{{Error vs LSM}} \\\\ \\midrule\n"
    
    for r in results_data:
        latex_str += f"\\$ {r['spot']:.1f} & \\$ {r['euro']:.2f} & \\$ {r['lsm']:.2f} $\\pm$ {r['lsm_se']:.2f} & \\$ {r['eep']:.2f} & \\$ {r.get('premium', 0.0):.2f} & \\$ {r['abs_error']:.2f} \\\\\n"
        
    latex_str += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f">> Saved Scalability {dim}D LaTeX Table to: {save_path}")
