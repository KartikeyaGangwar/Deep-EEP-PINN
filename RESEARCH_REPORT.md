# Comprehensive Research Report: Scientific Machine Learning for Optimal Stopping & Free-Boundary PDEs

## An Analytical Early-Exercise Premium Physics-Informed Neural Network (EEP-PINN) for 1D, 5D Geometric, and 5D Real-World Arithmetic American Basket Options

**Author:** Final Year B.Sc. (Hons) Mathematics, University of Delhi  
**Specialization:** Differential Equations, Numerical Analysis, Computational Finance & Scientific Machine Learning  

---

## Table of Contents
1. [Executive Summary & Core Novelty](#1-executive-summary--core-novelty)
2. [Mathematical Foundations & PDE Derivations](#2-mathematical-foundations--pde-derivations)
   - [Asset Dynamics & Itô's Lemma](#asset-dynamics--itôs-lemma)
   - [Delta-Hedging & Multi-Asset Black-Scholes Operator](#delta-hedging--multi-asset-black-scholes-operator)
   - [American Option Obstacle Problem & LCP](#american-option-obstacle-problem--lcp)
   - [Free-Boundary Conditions & Smooth Pasting (Merton 1973)](#free-boundary-conditions--smooth-pasting-merton-1973)
3. [Method 0: Classical Crank-Nicolson PSOR (1D Ground Truth)](#3-method-0-classical-crank-nicolson-psor-1d-ground-truth)
4. [Method 1: Baseline Single-Network Penalty PINN & Failure Analysis](#4-method-1-baseline-single-network-penalty-pinn--failure-analysis)
   - [Root Cause 1: Terminal Payoff Kink Singularity](#root-cause-1-terminal-payoff-kink-singularity)
   - [Root Cause 2: Dirac-Delta Gamma Singularity](#root-cause-2-dirac-delta-gamma-singularity)
   - [Root Cause 3: Wasted Neural Capacity](#root-cause-3-wasted-neural-capacity)
5. [Method 2: Our Novel Early-Exercise Premium PINN (EEP-PINN)](#5-method-2-our-novel-early-exercise-premium-pinn-eep-pinn)
   - [Analytical Kim (1990) / CJM (1992) Decomposition](#analytical-kim-1990--cjm-1992-decomposition)
   - [Exact Boundary Hard-Constraint Ansatz](#exact-boundary-hard-constraint-ansatz)
   - [Exact Analytical Hybrid Greeks Formulation](#exact-analytical-hybrid-greeks-formulation)
6. [Phase 2A: High-Dimensional Geometric Basket Options ($d = 5$)](#6-phase-2a-high-dimensional-geometric-basket-options-d--5)
7. [Phase 2B: Real-World Arithmetic Basket Options ($d = 5$)](#7-phase-2b-real-world-arithmetic-basket-options-d--5)
   - [Why Arithmetic Baskets are Traded on Real-World Exchanges](#why-arithmetic-baskets-are-traded-on-real-world-exchanges)
   - [The Non-Lognormal Sum Problem](#the-non-lognormal-sum-problem)
   - [Gentle (1993) / Milevsky-Posner (1998) Moment-Matching Derivation](#gentle-1993--milevsky-posner-1998-moment-matching-derivation)
   - [Exact Analytical Anchor & Zero-Kink Theorem Preservation](#exact-analytical-anchor--zero-kink-theorem-preservation)
8. [Comprehensive Master Quantitative Benchmarks](#8-comprehensive-master-quantitative-benchmarks)
   - [Phase 1: 1D American Option Results](#phase-1-1d-american-option-results)
   - [Phase 2A: 5-Asset Geometric Basket Results](#phase-2a-5-asset-geometric-basket-results)
   - [Phase 2B: 5-Asset Real-World Arithmetic Basket Results](#phase-2b-5-asset-real-world-arithmetic-basket-results)
   - [Phase 3: 10-Asset High-Dimensional Basket Results](#phase-3-10-asset-high-dimensional-basket-results)
   - [Scalability Suite: 30D Dow Jones Results](#scalability-suite-30-asset-dow-jones-results)
   - [Scalability Suite: 50D Nifty 50 Results](#scalability-suite-50-asset-nifty-50-results)
9. [Academic References & Literature Citations](#9-academic-references--literature-citations)

---

## 1. Executive Summary & Core Novelty

American option pricing represents a classic **parabolic free-boundary obstacle problem** in applied mathematics. Unlike European options which possess exact closed-form formulas (Black-Scholes 1973), American options allow early exercise at any time $t \in [0, T]$, requiring the simultaneous computation of the option price manifold $V(\mathbf{S}, t)$ and the optimal moving exercise boundary $\mathbf{S}^*(t)$.

### Key Empirical Achievements:
* **1D Single Stock Free-Boundary:** Relative $L_2$ Error reduced to **$0.37\%$**, Max Error = **₹0.43**, MAE = **₹0.08 (8 paise)** ($97.1\times$ faster than Crank-Nicolson PSOR).
* **5D Correlated Geometric Basket ($d=5$):** Overcomes the Curse of Dimensionality ($300^5 = 2.43\text{ Trillion}$ grid nodes) with **$2712.2\times$ speedup** over 100,000-path Longstaff-Schwartz Monte Carlo.
* **5D Real-World Arithmetic Basket ($d=5$):** Solved exchange-traded arithmetic average payoff $\max(K - \sum w_i S_i, 0)$ with a Mean Absolute Difference of **₹0.07 (7 paise)** and **$1010.0\times$ speedup** over 100,000-path Arithmetic Monte Carlo.
* **10D High-Dimensional Basket ($d=10$):** Solved 45 pairwise correlation PDE space in $3.47\text{ ms}$ (**$6882.6\times$ speedup** over 100,000-path LSM).
* **30D Dow Jones Scale Basket ($d=30$):** Solved 435 correlation pairs in $3.26\text{ ms}$ (**$62,842.1\times$ speedup** over 100,000-path LSM).
* **50D Nifty 50 Scale Basket ($d=50$):** Solved 1,225 correlation pairs in $3.94\text{ ms}$ (**$176,760.1\times$ speedup** over 100,000-path LSM) with zero OOM.

---

## 2. Mathematical Foundations & PDE Derivations

### Asset Dynamics & Itô's Lemma
Under the risk-neutral probability measure $\mathbb{Q}$, asset prices follow correlated Geometric Brownian Motion:
$$dS_i(t) = r S_i(t) dt + \sigma_i S_i(t) dW_i(t), \quad \mathbb{E}[dW_i dW_j] = \rho_{ij} dt$$

For contract valuation $V(\mathbf{S}, t)$, Itô's Lemma states:
$$dV = \left( \frac{\partial V}{\partial t} + r \sum_{i=1}^d S_i \frac{\partial V}{\partial S_i} + \frac{1}{2}\sum_{i=1}^d \sum_{j=1}^d \rho_{ij} \sigma_i \sigma_j S_i S_j \frac{\partial^2 V}{\partial S_i \partial S_j} \right) dt + \sum_{i=1}^d \sigma_i S_i \frac{\partial V}{\partial S_i} dW_i(t)$$

Delta-hedging ($\Pi = V - \sum \Delta_i S_i$ with $\Delta_i = \frac{\partial V}{\partial S_i}$) eliminates stochastic diffusion risk, yielding the **Black-Scholes PDE Operator**:
$$\mathcal{L}_{\text{BS}}^{(d)} V \equiv \frac{\partial V}{\partial t} + \frac{1}{2}\sum_{i=1}^d \sum_{j=1}^d \rho_{ij}\sigma_i\sigma_j S_i S_j \frac{\partial^2 V}{\partial S_i \partial S_j} + r\sum_{i=1}^d S_i \frac{\partial V}{\partial S_i} - rV = 0$$

### American Option Obstacle Problem & Linear Complementarity Problem (LCP)
$$\min\left( -\mathcal{L}_{\text{BS}}^{(d)} V(\mathbf{S},t), \quad V(\mathbf{S},t) - h(\mathbf{S}) \right) = 0$$
Boundary conditions at the moving interface $\mathbf{S} = \mathbf{S}^*(t)$:
1. **Value Matching ($C^0$ Continuity):** $V(\mathbf{S}^*(t), t) = h(\mathbf{S}^*(t))$
2. **Smooth Pasting ($C^1$ Contact - Merton 1973):** $\nabla_\mathbf{S} V(\mathbf{S}^*(t), t) = \nabla_\mathbf{S} h(\mathbf{S}^*(t))$
3. **Terminal Free Boundary Anchor:** $\mathbf{S}^*(T) = \partial \{\mathbf{S} : h(\mathbf{S}) > 0\}$

---

## 3. Method 0: Classical Crank-Nicolson PSOR (1D Ground Truth)

Discretized on $M=300, N=300$ grid ($90,000$ points) with second-order Crank-Nicolson $\theta=0.5$:
$$\mathbf{A} \mathbf{V}^n = \mathbf{B} \mathbf{V}^{n+1} + \mathbf{b}^n, \quad \text{subject to } \mathbf{V}^n \ge \mathbf{h}$$
Solved via iterative **Projected Successive Over-Relaxation (PSOR)** ($\omega = 1.25$):
$$V_j^{(k+1)} = \max \left( h_j, \, V_j^{(k)} + \frac{\omega}{A_{jj}} \left( b_j - \sum_{m < j} A_{jm} V_m^{(k+1)} - \sum_{m \ge j} A_{jm} V_m^{(k)} \right) \right)$$

---

## 4. Method 1: Baseline Single-Network Penalty PINN & Failure Analysis

Parameterization: $V_\theta(S, t) = \text{Softplus}(\mathcal{N}_\theta) \cdot K$.  
Loss: $\mathcal{L}_{\text{total}} = w_{\text{pde}} \mathcal{L}_{\text{PDE}} + w_{\text{early}} \mathcal{L}_{\text{early}} + w_{\text{bc}} \mathcal{L}_{\text{BC}} + w_{\text{ic}} \mathcal{L}_{\text{IC}}$.

### Why it Failed (Max Error = ₹5.46):
1. **Payoff Kink Singularity:** Expiry corner $\max(K-S, 0)$ induces **Gibbs Spectral Leakage** with smooth activations.
2. **Dirac-Delta Gamma Singularity:** $\Gamma = \partial_{SS} V \to \delta(S-K)$ blows up autograd gradients near $(K, T)$.
3. **Wasted Capacity:** 95% of neurons re-learn known European curvature.

---

## 5. Method 2: Our Novel Early-Exercise Premium PINN (EEP-PINN)

### Kim (1990) / CJM (1992) Analytical Decomposition:
$$\mathbf{V_{\text{American}}(\mathbf{S}, t) = V_{\text{European}}^{\text{exact}}(\mathbf{S}, t) + e_\theta(\mathbf{S}, t)}$$

### Exact Boundary Hard-Constraint Ansatz:
$$e_\theta(\mathbf{S}, t) = \text{Softplus}\left(\mathcal{N}_\theta(\mathbf{S}, t)\right) \cdot \left(\frac{T-t}{T}\right) \cdot \prod_{i=1}^d \left(1 - \frac{S_i}{S_{\max}}\right) \cdot K\left(1 - e^{-rT}\right)$$

### Mathematical Guarantees:
* **$e(\mathbf{S}, T) \equiv 0$ strictly:** Expiry kink is 100% absorbed by $V_{\text{Euro}}^{\text{exact}}$. Zero Gibbs oscillations!
* **$20\times$ Target Scale Shrinkage:** Network fits $[0, 4.88]$ instead of $[0, 100]$.
* **Exact Analytical Hybrid Greeks:**
  $$\Delta_{\text{Amer}} = \Delta_{\text{Euro}}^{\text{exact}} + \partial_S e, \quad \Gamma_{\text{Amer}} = \Gamma_{\text{Euro}}^{\text{exact}} + \partial_{SS} e$$

---

## 6. Phase 2A: High-Dimensional Geometric Basket Options ($d = 5$)

For geometric basket $G(\mathbf{S}) = \prod S_i^{w_i}$ with $\sigma_G^2 = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$ and $q_G = \frac{1}{2}\sum w_i\sigma_i^2 - \frac{1}{2}\sigma_G^2$:
$$V_{\text{Euro}}^{\text{geom}}(\mathbf{S}, t) = K e^{-r(T-t)} \mathcal{N}(-d_2) - G(\mathbf{S}) e^{-q_G(T-t)} \mathcal{N}(-d_1)$$

---

## 7. Phase 2B: Real-World Arithmetic Basket Options ($d = 5$)

### The Real-World Market Reality:
In real finance (Wall Street, NSE, CBOE), **100% of traded basket options are Arithmetic Baskets**:
$$h_{\text{arith}}(\mathbf{S}) = \max\left( K - \sum_{i=1}^d w_i S_i, \, 0 \right)$$

### Gentle (1993) / Milevsky-Posner (1998) Moment-Matching Anchor:
We match the 1st and 2nd theoretical moments of $B(T) = \sum w_i S_i(T)$ under $\mathbb{Q}$:
1. **First Moment:** $M_1(t) = \sum_{i=1}^d w_i S_i(t) e^{r(T-t)} = e^{r(T-t)} B(t)$
2. **Second Moment:** $M_2(t) = \sum_{i=1}^d \sum_{j=1}^d w_i w_j S_i(t) S_j(t) \exp\left( (2r + \rho_{ij}\sigma_i\sigma_j)(T-t) \right)$
3. **Effective Volatility:** $\sigma_A^2(t) = \frac{1}{T-t} \ln\left( \frac{M_2(t)}{M_1(t)^2} \right)$
4. **European Formula:**
   $$V_{\text{Euro}}^{\text{arith, MM}}(\mathbf{S}, t) = K e^{-r(T-t)} \mathcal{N}(-d_2) - B(t) \mathcal{N}(-d_1)$$

### Preserving the Zero-Kink Theorem:
At $t \to T$, $V_{\text{Euro}}^{\text{arith, MM}}(\mathbf{S}, T) \equiv \max\left(K - \sum w_i S_i, 0\right)$.  
Therefore, $\mathbf{e_\theta(\mathbf{S}, T) \equiv 0}$ strictly by mathematical identity!

## 8. Comprehensive 4-Tier Master Quantitative Benchmarks (100k LSM Rigor on Tesla P100)

### A. Phase 1: 1D American Option Benchmark (90,000 Grid Points)
| Model Architecture | Rel. $L_2$ Error | Max Error ($L_\infty$) | MAE | Boundary RMSE | Eval Latency | Speedup vs PSOR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Crank-Nicolson PSOR (Ground Truth)** | Benchmark | Benchmark | Benchmark | Benchmark | $1149.78\text{ ms}$ | $1.0\times$ |
| **Baseline Single-Network PINN** | $3.86\%$ | ₹$7.63$ | ₹$0.97$ | ₹$5.17$ | $8.00\text{ ms}$ | $143.7\times$ |
| **Novel EEP-PINN (1D Proposed)** | **$0.37\%$** | **₹$0.43$ (43 paise)** | **₹$0.08$ (8 paise)** | **₹$2.93$** | **$11.84\text{ ms}$** | **$97.1\times$** |

* **Accuracy Gain:** $10.4\times$ reduction in Relative $L_2$ error ($0.37\%$ vs $3.86\%$) and $17.7\times$ reduction in Max Error ($0.43$ vs $7.63$).
* **Learned Dynamic Uncertainty Weights:** $w_{\text{pde}}=2.51, w_{\text{obstacle}}=2.73, w_{\text{bc0}}=2.53$.

---

### B. Phase 2A: 5-Asset Geometric Basket Option ($d=5, \rho=0.40$, 100k-Path 23-Basis LSM)
| Spot Price $S_{0,i}$ | European Exact | LSM Monte Carlo ($100\text{k}$) | Novel Geometric EEP-PINN | Difference vs LSM |
| :---: | :---: | :---: | :---: | :---: |
| **₹80.0 (Deep ITM)** | ₹16.70 | ₹$19.96 \pm 0.00$ | **₹19.80** | ₹0.16 |
| **₹90.0 (ITM)** | ₹9.13 | ₹$10.47 \pm 0.02$ | **₹10.36** | ₹0.11 |
| **₹100.0 (ATM)** | ₹4.16 | ₹$4.55 \pm 0.02$ | **₹4.49** | **₹0.06 (6 paise)** |
| **₹110.0 (OTM)** | ₹1.58 | ₹$1.69 \pm 0.01$ | **₹1.66** | **₹0.04 (4 paise)** |
| **₹120.0 (Deep OTM)** | ₹0.52 | ₹$0.55 \pm 0.01$ | **₹0.53** | **₹0.02 (2 paise)** |

* **Speedup:** **$2712.2\times$ faster** ($3.50\text{ ms}$ vs. $9484.37\text{ ms}$).
* **Mean Absolute Difference vs LSM:** **₹0.08 (8 paise)** across all moneyness regimes.

---

### C. Phase 2B: Real-World 5-Asset Arithmetic Basket Option ($d=5, \rho=0.40$, 100k-Path 23-Basis LSM)
| Spot Price $S_{0,i}$ | European Arith MM | Arithmetic LSM ($100\text{k}$) | Novel Arithmetic EEP-PINN | Difference vs LSM |
| :---: | :---: | :---: | :---: | :---: |
| **₹80.0 (Deep ITM)** | ₹15.96 | ₹$19.95 \pm 0.00$ | **₹19.79** | ₹0.16 |
| **₹90.0 (ITM)** | ₹8.53 | ₹$10.28 \pm 0.02$ | **₹10.14** | ₹0.14 |
| **₹100.0 (ATM)** | ₹3.78 | ₹$4.26 \pm 0.02$ | **₹4.24** | **₹0.02 (2 paise)** |
| **₹110.0 (OTM)** | ₹1.40 | ₹$1.53 \pm 0.01$ | **₹1.52** | **₹0.02 (2 paise)** |
| **₹120.0 (Deep OTM)** | ₹0.44 | ₹$0.47 \pm 0.01$ | **₹0.47** | **₹0.00 (0 paise!)** |

* **Mean Absolute Difference vs LSM:** **₹0.07 (7 paise!)**.
* **Speedup:** **$1010.0\times$ faster** ($7.55\text{ ms}$ vs. $7623.61\text{ ms}$).

---

### D. Phase 3: 10-Asset High-Dimensional Correlated Basket Option ($d=10, \rho=0.35$, 100k-Path 68-Basis LSM)
| Spot Price $S_{0,i}$ | European Exact | LSM Monte Carlo ($100\text{k}$) | Novel High-Dim EEP-PINN | Difference vs LSM |
| :---: | :---: | :---: | :---: | :---: |
| **₹80.0 (Deep ITM)** | ₹16.61 | ₹$19.96 \pm 0.00$ | **₹19.85** | ₹0.11 |
| **₹90.0 (ITM)** | ₹8.71 | ₹$10.18 \pm 0.01$ | **₹10.31** | ₹0.13 |
| **₹100.0 (ATM)** | ₹3.61 | ₹$3.99 \pm 0.02$ | **₹4.15** | ₹0.16 |
| **₹110.0 (OTM)** | ₹1.18 | ₹$1.29 \pm 0.01$ | **₹1.43** | ₹0.15 |
| **₹120.0 (Deep OTM)** | ₹0.31 | ₹$0.33 \pm 0.00$ | **₹0.48** | ₹0.15 |

* **Speedup:** **$6882.6\times$ faster** ($3.47\text{ ms}$ vs. $23900.48\text{ ms}$).
* **Mean Absolute Difference vs LSM:** **₹0.14 (14 paise)** in 10-dimensional space!
* **Curse of Dimensionality Solution:** Evaluates the 10D PDE (45 correlation pairs) in $3.47\text{ ms}$ where classical finite difference would require $5.9 \times 10^{24}$ grid nodes.

---

### E. Scalability Suite: 30-Asset Dow Jones Industrial Scale Basket Option ($d=30, \rho=0.30$, 435 Correlation Pairs, 100k-Path 498-Basis LSM)
| Spot Price $S_{0,i}$ | European Exact | LSM Monte Carlo ($100\text{k}$) | Novel 30D EEP-PINN | Difference vs LSM |
| :---: | :---: | :---: | :---: | :---: |
| **₹80.0 (Deep ITM)** | ₹16.76 | ₹$19.99 \pm 0.00$ | **₹19.96** | **₹0.03 (3 paise!)** |
| **₹90.0 (ITM)** | ₹8.63 | ₹$10.19 \pm 0.01$ | **₹10.09** | ₹0.10 |
| **₹100.0 (ATM)** | ₹3.39 | ₹$3.84 \pm 0.01$ | **₹3.76** | **₹0.08 (8 paise)** |
| **₹110.0 (OTM)** | ₹1.00 | ₹$1.15 \pm 0.01$ | **₹1.09** | **₹0.06 (6 paise)** |
| **₹120.0 (Deep OTM)** | ₹0.23 | ₹$0.29 \pm 0.00$ | **₹0.25** | **₹0.04 (4 paise)** |

* **Speedup:** **$62,842.1\times$ faster** ($3.26\text{ ms}$ vs. $204,703.02\text{ ms} \approx 3.4\text{ mins}$ per spot level).
* **Mean Absolute Difference vs LSM:** **₹0.06 (6 paise)** across all 30 assets!

---

### F. Scalability Suite: 50-Asset Nifty 50 / Euro Stoxx 50 Scale Basket Option ($d=50, \rho=0.25$, 1225 Correlation Pairs, 100k-Path 1328-Basis LSM)
| Spot Price $S_{0,i}$ | European Exact | LSM Monte Carlo ($100\text{k}$) | Novel 50D EEP-PINN | Difference vs LSM |
| :---: | :---: | :---: | :---: | :---: |
| **₹80.0 (Deep ITM)** | ₹16.60 | ₹$20.02 \pm 0.00$ | **₹19.80** | ₹0.22 |
| **₹90.0 (ITM)** | ₹8.11 | ₹$10.15 \pm 0.01$ | **₹10.17** | **₹0.02 (2 paise!)** |
| **₹100.0 (ATM)** | ₹2.74 | ₹$3.28 \pm 0.01$ | **₹3.30** | **₹0.02 (2 paise!)** |
| **₹110.0 (OTM)** | ₹0.62 | ₹$0.78 \pm 0.01$ | **₹0.70** | **₹0.08 (8 paise)** |
| **₹120.0 (Deep OTM)** | ₹0.10 | ₹$0.16 \pm 0.00$ | **₹0.11** | **₹0.06 (6 paise)** |

* **Speedup:** **$176,760.1\times$ faster** ($3.94\text{ ms}$ vs. $696,333.20\text{ ms} \approx 11.6\text{ mins}$ per spot level).
* **Mean Absolute Difference vs LSM:** **₹0.08 (8 paise)** across all 50 dimensions!
* **Memory & Stability:** Chunked loss evaluation ensures peak VRAM remains $< 3\text{ GB}$ on Tesla P100 with zero OOM.

---

## 9. Academic References & Literature Citations

1. **Black, F., & Scholes, M. (1973).** *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy, 81(3), 637-654.
2. **Merton, R. C. (1973).** *Theory of Rational Option Pricing*. The Bell Journal of Economics and Management Science, 4(1), 141-183.
3. **Kim, I. J. (1990).** *The Analytic Valuation of American Options*. The Review of Financial Studies, 3(4), 547-572.
4. **Carr, P., Jarrow, R., & Myneni, R. (1992).** *Alternative Methods for Valuing American Options*. Finance and Stochastics / Mathematical Finance, 2(1), 87-106.
5. **Gentle, D. (1993).** *Basket Options*. Risk, 6(12), 51-52.
6. **Milevsky, M. A., & Posner, S. E. (1998).** *Valuing Exotic Options by Approximating the Swapping Density: The Case of Arithmetic Asian and Basket Options*. Journal of Financial and Quantitative Analysis, 33(3), 409-425.
7. **Longstaff, F. A., & Schwartz, E. S. (2001).** *Valuing American Options by Simulation: A Simple Least-Squares Approach*. The Review of Financial Studies, 14(1), 113-147.
8. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations*. Journal of Computational Physics, 378, 686-707.
9. **Cuomo, S., Di Cola, V. S., Giampaolo, F., Rozza, G., Raissi, M., & Piccialli, F. (2022).** *Scientific Machine Learning through Physics-Informed Neural Networks: Where we are and What’s next*. Journal of Scientific Computing, 92(3), 88.
10. **Cryer, C. W. (1971).** *The Solution of a Quadratic Programming Problem using Systematic Overrelaxation*. SIAM Journal on Control, 9(3), 385-392.
