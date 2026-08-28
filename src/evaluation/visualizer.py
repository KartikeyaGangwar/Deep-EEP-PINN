"""
Publication-Quality Visualization for Scientific Machine Learning Papers (Phase 1: 1D American Option).
Features:
- IEEE/SIAM academic LaTeX serif styling (Computer Modern math font).
- Explicit Phase 1 labeling and mathematical precision.
"""

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def plot_3d_surfaces(S_grid, t_grid, V_psor, V_eep, e_grid=None, save_path="figures/phase1_1d_option_price_surface_3d.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    T_mesh, S_mesh = np.meshgrid(t_grid, S_grid, indexing="ij")
    
    fig = plt.figure(figsize=(15.5, 5.5))
    
    # Subplot 1: PSOR Ground Truth Surface
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    surf1 = ax1.plot_surface(S_mesh, T_mesh, V_psor, cmap="viridis", alpha=0.9, edgecolor="none")
    ax1.set_title(r"$\mathbf{(a)\; Ground\; Truth:\; PSOR}$", pad=10, fontsize=12)
    ax1.set_xlabel(r"Stock Price $S$ (\$)")
    ax1.set_ylabel(r"Time $t$ (Years)")
    ax1.set_zlabel(r"$V(S,t)$ (\$)")
    ax1.view_init(elev=25, azim=-120)
    fig.colorbar(surf1, ax=ax1, shrink=0.45, aspect=10, label=r"Option Price $V(S,t)$ (\$)")
    
    # Subplot 2: Novel EEP-PINN American Price
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    surf2 = ax2.plot_surface(S_mesh, T_mesh, V_eep, cmap="viridis", alpha=0.9, edgecolor="none")
    ax2.set_title(r"$\mathbf{(b)\; Novel\; EEP-PINN:\; V_{\mathrm{Amer}}}$", pad=10, fontsize=12)
    ax2.set_xlabel(r"Stock Price $S$ (\$)")
    ax2.set_ylabel(r"Time $t$ (Years)")
    ax2.set_zlabel(r"$V(S,t)$ (\$)")
    ax2.view_init(elev=25, azim=-120)
    fig.colorbar(surf2, ax=ax2, shrink=0.45, aspect=10, label=r"Option Price $V(S,t)$ (\$)")
    
    # Subplot 3: Early Exercise Premium Surface e(S, t)
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    if e_grid is not None:
        surf3 = ax3.plot_surface(S_mesh, T_mesh, e_grid, cmap="plasma", alpha=0.9, edgecolor="none")
        ax3.set_title(r"$\mathbf{(c)\; Learned\; Premium:\; e_\theta(S,t)}$", pad=10, fontsize=12)
        ax3.set_xlabel(r"Stock Price $S$ (\$)")
        ax3.set_ylabel(r"Time $t$ (Years)")
        ax3.set_zlabel(r"$e(S,t)$ (\$)")
        ax3.view_init(elev=25, azim=-120)
        fig.colorbar(surf3, ax=ax3, shrink=0.45, aspect=10, label=r"Premium $e(S,t)$ (\$)")
    
    plt.suptitle(r"$\mathbf{Phase\; 1:\; 1D\; American\; Option\; Free-Boundary\; Price\; \&\; Premium\; Surfaces}$", fontsize=14, y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> Saved 3D Surface Figure to: {save_path}")

def plot_free_boundary_comparison(t_grid, S_star_psor, S_star_base, S_star_eep, K=100.0, save_path="figures/phase1_1d_free_boundary_comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(9.5, 5.5))
    
    plt.plot(t_grid, S_star_psor, "k-", linewidth=3.0, label=r"$\mathrm{Crank-Nicolson\; PSOR\; (Ground\; Truth)}$")
    plt.plot(t_grid, S_star_base, "r:", linewidth=2.2, label=r"$\mathrm{Baseline\; Penalty\; PINN}$")
    plt.plot(t_grid, S_star_eep, "b--", linewidth=2.5, label=r"$\mathrm{Novel\; EEP-PINN\; (Decomposition)}$")
    plt.axhline(y=K, color="gray", linestyle="--", alpha=0.7, label=r"$\mathrm{Strike\; Price\;} K = 100\$")
    
    plt.fill_between(t_grid, 0, S_star_psor, color="red", alpha=0.08, label=r"$\mathrm{Stopping\; Region\; (Early\; Exercise)}$")
    plt.fill_between(t_grid, S_star_psor, 150, color="blue", alpha=0.05, label=r"$\mathrm{Continuation\; Region\; (Hold)}$")
    
    plt.title(r"$\mathbf{Phase\; 1:\; Optimal\; Early\; Exercise\; Boundary\;} S^*(t) \;\mathbf{(Free\; Boundary\; Problem)}$", pad=12, fontsize=13)
    plt.xlabel(r"$\mathrm{Time\;} t \;\mathrm{(Years)}$")
    plt.ylabel(r"$\mathrm{Optimal\; Exercise\; Threshold\;} S^*(t) \;\mathrm{(\$)}$")
    plt.xlim([0.0, t_grid[-1]])
    plt.ylim([45, 115])
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(loc="lower left", framealpha=0.95, fontsize=10)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> Saved Free Boundary Figure to: {save_path}")

def plot_greeks_comparison(S_grid, Delta_psor, Delta_eep, Gamma_psor, Gamma_eep, t_indices=[0, 150, 250], t_grid=None, save_path="figures/phase1_1d_greeks_comparison.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["teal", "darkorange", "crimson"]
    
    for idx, c in zip(t_indices, colors):
        t_val = t_grid[idx] if t_grid is not None else idx
        axes[0].plot(S_grid, Delta_psor[idx, :], color=c, linestyle="-", label=f"PSOR ($t={t_val:.2f}$)")
        axes[0].plot(S_grid, Delta_eep[idx, :], color=c, linestyle="--", label=f"EEP-PINN ($t={t_val:.2f}$)")
        
        axes[1].plot(S_grid, Gamma_psor[idx, :], color=c, linestyle="-", label=f"PSOR ($t={t_val:.2f}$)")
        axes[1].plot(S_grid, Gamma_eep[idx, :], color=c, linestyle="--", label=f"EEP-PINN ($t={t_val:.2f}$)")
        
    axes[0].set_title(r"$\mathbf{Delta\;} (\Delta = \partial V / \partial S)$", fontsize=12)
    axes[0].set_xlabel(r"$\mathrm{Stock\; Price\;} S \;(\$)$")
    axes[0].set_ylabel(r"$\Delta$")
    axes[0].set_xlim([0, 200])
    axes[0].set_ylim([-1.05, 0.05])
    axes[0].grid(True, alpha=0.3, linestyle="--")
    axes[0].legend(fontsize=9)
    
    axes[1].set_title(r"$\mathbf{Gamma\;} (\Gamma = \partial^2 V / \partial S^2)$", fontsize=12)
    axes[1].set_xlabel(r"$\mathrm{Stock\; Price\;} S \;(\$)$")
    axes[1].set_ylabel(r"$\Gamma$")
    axes[1].set_xlim([40, 180])
    axes[1].set_ylim([-0.005, 0.08])
    axes[1].grid(True, alpha=0.3, linestyle="--")
    axes[1].legend(fontsize=9)
    
    plt.suptitle(r"$\mathbf{Phase\; 1:\; Financial\; Greeks\; Comparison\; (\mathrm{Autograd}\; \mathrm{vs.}\; \mathrm{Finite}\; \mathrm{Differences})}$", fontsize=13, y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> Saved Greeks Figure to: {save_path}")

def plot_error_heatmap(S_grid, t_grid, V_psor, V_eep, save_path="figures/phase1_1d_error_heatmap.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    abs_error = np.abs(V_eep - V_psor)
    
    plt.figure(figsize=(8.5, 6))
    plt.contourf(S_grid, t_grid, abs_error, levels=50, cmap="magma")
    cbar = plt.colorbar()
    cbar.set_label(r"$\mathrm{Absolute\; Error\;} |V_{\mathrm{EEP-PINN}} - V_{\mathrm{PSOR}}| \;(\$)$", labelpad=10)
    
    plt.title(r"$\mathbf{Phase\; 1:\; Pointwise\; Absolute\; Error\; Heatmap\; (EEP-PINN\; vs.\; PSOR)}$", pad=12, fontsize=12)
    plt.xlabel(r"$\mathrm{Stock\; Price\;} S \;(\$)$")
    plt.ylabel(r"$\mathrm{Time\;} t \;\mathrm{(Years)}$")
    plt.xlim([0, 250])
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> Saved Error Heatmap Figure to: {save_path}")

def plot_benchmark_barchart(benchmark_data: dict, save_path="figures/phase1_1d_error_barchart.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    models = list(benchmark_data.keys())
    l2_errors = [benchmark_data[m]["rel_L2_error"] * 100 for m in models]
    b_rmse = [benchmark_data[m]["boundary_rmse"] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    
    rects1 = ax1.bar(x - width/2, l2_errors, width, label=r"$\mathrm{Option\; Price\; Rel.\;} L_2 \;\mathrm{Error\; (\%)}$", color="teal", alpha=0.85)
    ax1.set_ylabel(r"$\mathrm{Relative\;} L_2 \;\mathrm{Error\; (\%)}$", color="teal", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="teal")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontweight="bold", fontsize=10)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, b_rmse, width, label=r"$\mathrm{Free\; Boundary\;} S^* \;\mathrm{RMSE\; (\$)}$", color="crimson", alpha=0.85)
    ax2.set_ylabel(r"$\mathrm{Boundary\; RMSE\; (\$)}$", color="crimson", fontweight="bold")
    ax2.tick_params(axis="y", labelcolor="crimson")
    
    plt.title(r"$\mathbf{Phase\; 1:\; Accuracy\; Comparison\; (Baseline\; PINN\; vs.\; Novel\; EEP-PINN)}$", pad=12, fontsize=12)
    ax1.grid(True, alpha=0.3, axis="y", linestyle="--")
    
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax2.annotate(f"${h:.2f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
                     
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> Saved Benchmark Bar Chart to: {save_path}")

def export_latex_table(benchmark_data: dict, save_path="figures/phase1_1d_benchmark_summary_table.tex"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    latex_str = r"""\begin{table}[h!]
\centering
\small
\caption{Phase 1 Free-Boundary Benchmark: 1D American Put Option ($K=100, T=1.0, r=0.05, \sigma=0.20$).}
\label{tab:pinn_benchmarks}
\begin{tabular}{@{}lccccc@{}}
\toprule
\textbf{Model Architecture} & \textbf{Rel. $L_2$ Error} & \textbf{Max Error ($L_\infty$)} & \textbf{MAE} & \textbf{Boundary RMSE} & \textbf{Latency} \\ \midrule
"""
    for model_name, res in benchmark_data.items():
        latex_str += f"{model_name} & {res['rel_L2_error']*100:.2f}\\% & \\$ {res['L_inf_error']:.2f} & \\$ {res['MAE']:.2f} & \\$ {res['boundary_rmse']:.2f} & {res['eval_time']*1000:.2f} ms \\\\\n"
        
    latex_str += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f">> Saved LaTeX Table to: {save_path}")
