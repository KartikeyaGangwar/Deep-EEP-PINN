# Deep Early Exercise Premium Physics-Informed Neural Network (Deep-EEP-PINN)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Kaggle: GPU P100](https://img.shields.io/badge/Hardware-Tesla%20P100%20Verified-brightgreen.svg)]()

A mesh-free scientific machine learning framework for high-dimensional American basket option free-boundary partial differential equations ($d=1, 5, 10, 30, 50$).

---

## Abstract

Valuing American-style options requires solving a nonlinear parabolic free-boundary obstacle problem with an a priori unknown early-exercise stopping surface $S^*(t)$. In high dimensions ($d > 3$), classical grid-based numerical methods (e.g., Crank-Nicolson Finite Difference, Projected Successive Over-Relaxation (PSOR)) suffer from Bellman's Curse of Dimensionality ($N^d$ exponential computational scaling).

**Deep-EEP-PINN** extends mesh-free scientific machine learning to high-dimensional American option pricing up to **$d=50$ correlated assets (1,225 pairwise correlations)** through four structural contributions:
* **Early Exercise Premium (EEP) Decomposition:** Embeds the analytical Kim (1990) / Carr-Jarrow-Myneni (1992) decomposition, splitting the total price into an exact European base and a non-negative early exercise premium: $V_{\text{Amer}}(\mathbf{S}, t) = V_{\text{Euro}}^{\text{exact}}(\mathbf{S}, t) + e_\theta(\mathbf{S}, t)$.
* **Hard-Constraint Boundary-Encoded Neural Ansatz:** Strictly enforces $e_\theta(\mathbf{S}, T) \equiv 0$, absorbing the $t=T$ payoff kink singularity into the exact European formula.
* **Directional Autograd Hessian Trace Contraction:** Evaluates multi-asset cross-diffusion terms in $\mathcal{O}(d)$ linear computational complexity without storing large $d \times d$ Hessian tensors.
* **Chunked Loss Backpropagation:** Limits peak GPU VRAM consumption to $< 3\text{ GB}$ during $d=50$ training on standard accelerators (e.g., NVIDIA Tesla P100).

---

## Mathematical Formulation

The American basket option fair value is formulated as:
$$V_{\text{American}}(\mathbf{S}, t) = V_{\text{European}}^{\text{exact}}(\mathbf{S}, t) + e_\theta(\mathbf{S}, t)$$

where the early exercise premium is parameterized through the boundary-encoded neural network:
$$e_\theta(\mathbf{S}, t) = \text{Softplus}\left(\mathcal{N}_\theta(\mathbf{S}, t)\right) \cdot \left(\frac{T-t}{T}\right) \cdot \prod_{i=1}^d \left(1 - \frac{S_i}{S_{\max}}\right) \cdot K\left(1 - e^{-rT}\right)$$

### Structural Guarantees:
* **Zero Expiration Singularity:** $e_\theta(\mathbf{S}, T) \equiv 0$ identically, eliminating non-differentiable Dirac delta derivatives at maturity.
* **Dynamic Range Compression:** The neural network fits a compact range $[0, 4.88]$ rather than $[0, 100]$, improving gradient stability.
* **Analytically Regularized Sensitivities (Greeks):** Sensitivities are evaluated via exact automatic differentiation: $\Delta = \nabla_{\mathbf{S}} V_{\text{Euro}}^{\text{exact}} + \nabla_{\mathbf{S}} e_\theta$.

---

## Empirical Benchmark Suite

All benchmarks evaluated on NVIDIA Tesla P100 (16GB VRAM) and validated against 100,000-path Longstaff-Schwartz Least Squares Monte Carlo (LSM) regressions and Crank-Nicolson PSOR.

### 1. Phase 1: 1D American Option Free-Boundary Problem (90,000 Space-Time Grid Points)
* **Parameters:** Strike $K=100.0$, Expiration $T=1.0\text{ year}$, Risk-free rate $r=0.05$, Volatility $\sigma=0.20$.

| Model Architecture | Rel. $L_2$ Error (%) | Max Error ($L_\infty$) | MAE | Boundary RMSE | Forward Latency | Speedup vs PSOR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Crank-Nicolson PSOR (Ground Truth)** | Benchmark | Benchmark | Benchmark | Benchmark | $1149.78\text{ ms}$ | $1.0\times$ |
| **Baseline Penalty PINN** | $3.86\%$ | ₹$7.63$ | ₹$0.97$ | ₹$5.17$ | $8.00\text{ ms}$ | $143.7\times$ |
| **Novel EEP-PINN (Proposed)** | **$0.37\%$** | **₹$0.43$** | **₹$0.08$** | **₹$2.93$** | **$11.84\text{ ms}$** | **$97.1\times$** |

---

### 2. Phase 2A: 5-Asset Correlated Geometric Basket Option ($d=5, \rho=0.40$, 100k-Path LSM)
* **Parameters:** $d=5$, Strike $K=100.0$, Expiration $T=1.0$, Correlation $\rho=0.40$, 100k simulated paths.

| Initial Spot $S_{0,i}$ | European Exact | LSM Monte Carlo ($100\text{k}$) | Novel EEP-PINN | Difference vs LSM |
| :---: | :---: | :---: | :---: | :---: |
| **₹80.0 (Deep ITM)** | ₹16.70 | ₹$19.96 \pm 0.00$ | **₹19.80** | ₹0.16 |
| **₹90.0 (ITM)** | ₹9.13 | ₹$10.47 \pm 0.02$ | **₹10.36** | ₹0.11 |
| **₹100.0 (ATM)** | ₹4.16 | ₹$4.55 \pm 0.02$ | **₹4.49** | **₹0.06** |
| **₹110.0 (OTM)** | ₹1.58 | ₹$1.69 \pm 0.01$ | **₹1.66** | **₹0.04** |
| **₹120.0 (Deep OTM)** | ₹0.52 | ₹$0.55 \pm 0.01$ | **₹0.53** | **₹0.02** |

* **Forward Latency:** $3.50\text{ ms}$ ($2712.2\times$ faster than 100k-path LSM).
* **Mean Absolute Difference:** ₹0.08 across all moneyness levels.

---

### 3. Phase 2B: Real-World 5-Asset Arithmetic Basket Option ($d=5, \rho=0.40$, 100k-Path LSM)
* **Parameters:** $d=5$, Strike $K=100.0$, Arithmetic Payoff $\max(K - \frac{1}{5}\sum S_i, 0)$, Milevsky-Posner moment-matched anchor.

| Initial Spot $S_{0,i}$ | European Arith MM | Arithmetic LSM ($100\text{k}$) | Novel Arith EEP-PINN | Difference vs LSM |
| :---: | :---: | :---: | :---: | :---: |
| **₹80.0 (Deep ITM)** | ₹15.96 | ₹$19.95 \pm 0.00$ | **₹19.79** | ₹0.16 |
| **₹90.0 (ITM)** | ₹8.53 | ₹$10.28 \pm 0.02$ | **₹10.14** | ₹0.14 |
| **₹100.0 (ATM)** | ₹3.78 | ₹$4.26 \pm 0.02$ | **₹4.24** | **₹0.02** |
| **₹110.0 (OTM)** | ₹1.40 | ₹$1.53 \pm 0.01$ | **₹1.52** | **₹0.02** |
| **₹120.0 (Deep OTM)** | ₹0.44 | ₹$0.47 \pm 0.01$ | **₹0.47** | **₹0.00** |

* **Forward Latency:** $7.55\text{ ms}$ ($1010.0\times$ faster than 100k-path LSM).
* **Mean Absolute Difference:** ₹0.07.

---

### 4. Phase 3: 10-Asset High-Dimensional Basket Option ($d=10, \rho=0.35$, 45 Correlation Pairs)
* **Parameters:** $d=10$, Strike $K=100.0$, 45 pairwise correlations, 100k-path 68-basis LSM ground truth.

| Initial Spot $S_{0,i}$ | European Exact | LSM Monte Carlo ($100\text{k}$) | Novel 10D EEP-PINN | Difference vs LSM |
| :---: | :---: | :---: | :---: | :---: |
| **₹80.0 (Deep ITM)** | ₹16.61 | ₹$19.96 \pm 0.00$ | **₹19.85** | ₹0.11 |
| **₹90.0 (ITM)** | ₹8.71 | ₹$10.18 \pm 0.01$ | **₹10.31** | ₹0.13 |
| **₹100.0 (ATM)** | ₹3.61 | ₹$3.99 \pm 0.02$ | **₹4.15** | ₹0.16 |
| **₹110.0 (OTM)** | ₹1.18 | ₹$1.29 \pm 0.01$ | **₹1.43** | ₹0.15 |
| **₹120.0 (Deep OTM)** | ₹0.31 | ₹$0.33 \pm 0.00$ | **₹0.48** | ₹0.15 |

* **Forward Latency:** $3.47\text{ ms}$ ($6882.6\times$ faster than 100k-path LSM).
* **Mean Absolute Difference:** ₹0.14.

---

### 5. Scalability Suite: 30-Asset Dow Jones Industrial Scale ($d=30, \rho=0.30$, 435 Correlation Pairs)
* **Parameters:** $d=30$, 435 correlation pairs, 100k-path 498-basis polynomial LSM.
* **Mean Difference vs LSM:** **₹0.06 (6 paise)** across all 30 assets.
* **Forward Latency:** **$3.26\text{ ms}$** vs **$204.7\text{ seconds}$** in LSM Monte Carlo (**$62,842\times$ Speedup**).

---

### 6. Scalability Suite: 50-Asset Nifty 50 Scale ($d=50, \rho=0.25$, 1,225 Correlation Pairs)
* **Parameters:** $d=50$, 1,225 correlation pairs, 100k-path 1328-basis polynomial LSM.
* **Mean Difference vs LSM:** **₹0.08 (8 paise)** across all 50 dimensions.
* **ATM Difference:** **₹0.02 (2 paise)**.
* **Forward Latency:** **$3.94\text{ ms}$** vs **$696.3\text{ seconds}$ ($11.6\text{ minutes}$)** in LSM Monte Carlo (**$176,760\times$ Speedup**).

---

## Directory Organization

```
Deep-EEP-PINN/
├── figures/                            # 18 Standardized Publication Figures & LaTeX Tables
├── results/                            # Structured JSON & CSV Benchmark Metrics
├── logs/                               # Verified Execution Logs (1D/5D/10D, 30D, 50D)
├── references/                         # Foundation Academic Papers
├── src/                                # Core Engine Source Code
│   ├── configs/                        # Hyperparameter configurations (1D, 5D, 10D, 30D, 50D)
│   ├── evaluation/                     # Metrics, Results Exporters, LaTeX Visualizers
│   ├── solvers/                        # PSOR, Baseline PINN, EEP-PINN, LSM Monte Carlo
│   └── utils.py
├── requirements.txt                    # Project Dependencies
├── LICENSE                             # MIT License
└── run_all_benchmarks.py               # Master CLI Execution Script
```

---

## Execution & Reproducibility

### Installation
```bash
git clone https://github.com/KartikeyaGangwar/Deep-EEP-PINN.git
cd Deep-EEP-PINN
pip install -r requirements.txt
```

### Running Benchmarks
```bash
# 1. Quick Verification Smoke Test
python run_all_benchmarks.py --smoke-test

# 2. Complete Benchmark Suite (Phases 1, 2A, 2B, 3)
python run_all_benchmarks.py --all

# 3. Scalability Suite
python run_all_benchmarks.py --phase 30d    # 30D Dow Jones Scale
python run_all_benchmarks.py --phase 50d    # 50D Nifty 50 Scale

# 4. Regenerate All Publication Figures with LaTeX Serif Fonts
python src/evaluation/regenerate_all_figures.py
```

---

## Academic References

1. **Black, F., & Scholes, M. (1973).** *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy, 81(3), 637-654.
2. **Kim, I. J. (1990).** *The Analytic Valuation of American Options*. The Review of Financial Studies, 3(4), 547-572.
3. **Carr, P., Jarrow, R., & Myneni, R. (1992).** *Alternative Methods for Valuing American Options*. Finance and Stochastics, 2(1), 87-106.
4. **Longstaff, F. A., & Schwartz, E. S. (2001).** *Valuing American Options by Simulation: A Simple Least-Squares Approach*. The Review of Financial Studies, 14(1), 113-147.
5. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686-707.
6. **Cryer, C. W. (1971).** *The Solution of a Quadratic Programming Problem using Systematic Overrelaxation*. SIAM Journal on Control, 9(3), 385-392.
