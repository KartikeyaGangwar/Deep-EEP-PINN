# Deep Early Exercise Premium PINN (Deep-EEP-PINN)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle: GPU P100](https://img.shields.io/badge/Kaggle-Tesla%20P100%20Verified-brightgreen.svg)]()

> **Mesh-Free Deep Scientific Machine Learning Framework for High-Dimensional American Basket Option Free-Boundary Partial Differential Equations ($d=1, 5, 10, 30, 50$)**.

---

## 📌 Executive Summary

Valuing **American Options** requires solving a non-linear free-boundary partial differential equation (PDE) with an unknown early-exercise stopping region $S^*(t)$. In high dimensions ($d > 3$), traditional grid-based methods (Crank-Nicolson Finite Difference, PSOR) completely collapse due to Bellman's **Curse of Dimensionality** ($N^d$ exponential complexity).

**Deep-EEP-PINN** solves high-dimensional American option pricing up to **$d=50$ correlated assets (1,225 pairwise correlations)** by integrating:
1. **Kim (1990) / Carr-Jarrow-Myneni (1992) Early Exercise Premium (EEP) Decomposition**: $V_{\text{Amer}}(\mathbf{S}, t) = V_{\text{Euro}}^{\text{exact}}(\mathbf{S}, t) + e_\theta(\mathbf{S}, t)$.
2. **Hard-Constraint Boundary-Encoded Neural Ansatz**: Strictly enforces $e_\theta(\mathbf{S}, T) \equiv 0$, absorbing the $t=T$ payoff singularity.
3. **$\mathcal{O}(d)$ Directional Autograd Hessian Trace Contraction**: Evaluates multi-asset cross-diffusion terms in linear GPU time without $d \times d$ memory explosion.
4. **Chunked Loss Backpropagation**: Enables training on $d=50$ (Nifty 50 scale) with peak VRAM $< 3\text{ GB}$ (Zero OOM).

---

## 🧮 Core Mathematical Formulation

$$V_{\text{American}}(\mathbf{S}, t) = \mathbf{V_{\text{European}}^{\text{exact}}(\mathbf{S}, t)} + \mathbf{e_\theta(\mathbf{S}, t)}$$

where the early exercise premium is parameterized as:
$$e_\theta(\mathbf{S}, t) = \text{Softplus}\left(\mathcal{N}_\theta(\mathbf{S}, t)\right) \cdot \left(\frac{T-t}{T}\right) \cdot \prod_{i=1}^d \left(1 - \frac{S_i}{S_{\max}}\right) \cdot K\left(1 - e^{-rT}\right)$$

---

## 📊 Verified Empirical Benchmark Suite (Kaggle Tesla P100, 100k LSM Paths)

### 1. Phase 1: 1D American Option Free-Boundary Problem (90,000 Grid Nodes)
| Model Architecture | Rel. $L_2$ Error | Max Error ($L_\infty$) | MAE | Boundary RMSE | Latency | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Crank-Nicolson PSOR (Ground Truth)** | Benchmark | Benchmark | Benchmark | Benchmark | $1149.78\text{ ms}$ | $1.0\times$ |
| **Baseline Penalty PINN** | $3.86\%$ | ₹$7.63$ | ₹$0.97$ | ₹$5.17$ | $8.00\text{ ms}$ | $143.7\times$ |
| **Novel EEP-PINN (Proposed)** | **$0.37\%$** | **₹$0.43$ (43 paise)** | **₹$0.08$ (8 paise)** | **₹$2.93$** | **$11.84\text{ ms}$** | **$97.1\times$** |

---

### 2. Phase 2A: 5-Asset Correlated Geometric Basket Option ($d=5, \rho=0.40$, 100k LSM)
| Spot Price $S_{0,i}$ | European Exact | LSM Monte Carlo ($100\text{k}$) | Novel EEP-PINN | Difference vs LSM |
| :---: | :---: | :---: | :---: | :---: |
| **₹80.0 (Deep ITM)** | ₹16.70 | ₹$19.96 \pm 0.00$ | **₹19.80** | ₹0.16 |
| **₹90.0 (ITM)** | ₹9.13 | ₹$10.47 \pm 0.02$ | **₹10.36** | ₹0.11 |
| **₹100.0 (ATM)** | ₹4.16 | ₹$4.55 \pm 0.02$ | **₹4.49** | **₹0.06 (6 paise)** |
| **₹110.0 (OTM)** | ₹1.58 | ₹$1.69 \pm 0.01$ | **₹1.66** | **₹0.04 (4 paise)** |
| **₹120.0 (Deep OTM)** | ₹0.52 | ₹$0.55 \pm 0.01$ | **₹0.53** | **₹0.02 (2 paise)** |

* **Speedup:** **$2712.2\times$ faster** ($3.50\text{ ms}$ vs. $9484.37\text{ ms}$).

---

### 3. Phase 2B: Real-World 5-Asset Arithmetic Basket Option ($d=5, \rho=0.40$, 100k LSM)
| Spot Price $S_{0,i}$ | European Arith MM | Arithmetic LSM ($100\text{k}$) | Novel Arith EEP-PINN | Difference vs LSM |
| :---: | :---: | :---: | :---: | :---: |
| **₹80.0 (Deep ITM)** | ₹15.96 | ₹$19.95 \pm 0.00$ | **₹19.79** | ₹0.16 |
| **₹90.0 (ITM)** | ₹8.53 | ₹$10.28 \pm 0.02$ | **₹10.14** | ₹0.14 |
| **₹100.0 (ATM)** | ₹3.78 | ₹$4.26 \pm 0.02$ | **₹4.24** | **₹0.02 (2 paise)** |
| **₹110.0 (OTM)** | ₹1.40 | ₹$1.53 \pm 0.01$ | **₹1.52** | **₹0.02 (2 paise)** |
| **₹120.0 (Deep OTM)** | ₹0.44 | ₹$0.47 \pm 0.01$ | **₹0.47** | **₹0.00 (0 paise!)** |

* **Speedup:** **$1010.0\times$ faster** ($7.55\text{ ms}$ vs. $7623.61\text{ ms}$).

---

### 4. Phase 3: 10-Asset High-Dimensional Basket Option ($d=10, \rho=0.35$, 100k LSM)
| Spot Price $S_{0,i}$ | European Exact | LSM Monte Carlo ($100\text{k}$) | Novel 10D EEP-PINN | Difference vs LSM |
| :---: | :---: | :---: | :---: | :---: |
| **₹80.0 (Deep ITM)** | ₹16.61 | ₹$19.96 \pm 0.00$ | **₹19.85** | ₹0.11 |
| **₹90.0 (ITM)** | ₹8.71 | ₹$10.18 \pm 0.01$ | **₹10.31** | ₹0.13 |
| **₹100.0 (ATM)** | ₹3.61 | ₹$3.99 \pm 0.02$ | **₹4.15** | ₹0.16 |
| **₹110.0 (OTM)** | ₹1.18 | ₹$1.29 \pm 0.01$ | **₹1.43** | ₹0.15 |
| **₹120.0 (Deep OTM)** | ₹0.31 | ₹$0.33 \pm 0.00$ | **₹0.48** | ₹0.15 |

* **Speedup:** **$6882.6\times$ faster** ($3.47\text{ ms}$ vs. $23900.48\text{ ms}$).

---

### 5. Scalability Suite: 30-Asset Dow Jones Scale ($d=30, \rho=0.30$, 435 Pairs, 100k LSM)
* **Mean Difference vs 100k LSM:** **₹0.06 (6 paise)** across all 30 assets.
* **Latency:** **$3.26\text{ ms}$** vs **$204.7\text{ seconds}$** in LSM Monte Carlo $\implies$ **$62,842\times$ Speedup!**

---

### 6. Scalability Suite: 50-Asset Nifty 50 Scale ($d=50, \rho=0.25$, 1225 Pairs, 100k LSM)
* **Mean Difference vs 100k LSM:** **₹0.08 (8 paise)** across all 50 dimensions.
* **Latency:** **$3.94\text{ ms}$** vs **$696.3\text{ seconds}$ (11.6 mins!)** in LSM Monte Carlo $\implies$ **$176,760\times$ Speedup!**

---

## 🏃‍♂️ Execution & Reproducibility

```bash
# 1. Quick 10-Second Sanity Check
python run_all_benchmarks.py --smoke-test

# 2. Master Benchmark Suite (Phases 1, 2A, 2B, 3)
python run_all_benchmarks.py --all

# 3. Scalability Suite
python run_all_benchmarks.py --phase 30d    # 30D Dow Jones Scale
python run_all_benchmarks.py --phase 50d    # 50D Nifty 50 Scale

# 4. Regenerate All Publication Figures with LaTeX Serif Fonts
python src/evaluation/regenerate_all_figures.py
```

---

## 📚 Academic References & Citations

1. **Black, F., & Scholes, M. (1973).** *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy, 81(3), 637-654.
2. **Kim, I. J. (1990).** *The Analytic Valuation of American Options*. The Review of Financial Studies, 3(4), 547-572.
3. **Carr, P., Jarrow, R., & Myneni, R. (1992).** *Alternative Methods for Valuing American Options*. Finance and Stochastics, 2(1), 87-106.
4. **Longstaff, F. A., & Schwartz, E. S. (2001).** *Valuing American Options by Simulation: A Simple Least-Squares Approach*. RFS, 14(1), 113-147.
5. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** *Physics-informed neural networks*. JCP, 378, 686-707.
