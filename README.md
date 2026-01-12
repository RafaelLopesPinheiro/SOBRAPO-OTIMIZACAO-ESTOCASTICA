# 📊 SOBRAPO School 2025 - Stochastic Optimization

<div align="center">

![Julia](https://img.shields.io/badge/Julia-1.x-9558B2?style=for-the-badge&logo=julia&logoColor=white)
![JuMP](https://img.shields.io/badge/JuMP-Optimization-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

**Complete solutions to stochastic optimization problems from SOBRAPO School 2025**

[🔗 Original Course Materials](https://github.com/log-ufpb/sobrapo_school_2025) · [📄 Full Report (LaTeX)]() · [🎓 SOBRAPO](https://www.sobrapo.org. br/)

</div>

---

## 📚 Table of Contents

- [Overview](#-overview)
- [Problems Solved](#-problems-solved)
- [Key Results](#-key-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Methodologies](#-methodologies)
- [Visualizations](#-visualizations)
- [Technical Report](#-technical-report)
- [References](#-references)
- [Author](#-author)
- [License](#-license)

---

## 🎯 Overview

This repository contains **complete implementations and solutions** to exercises from the **SOBRAPO School 2025** on Stochastic Optimization, held at Universidade Federal da Paraíba (UFPB). All solutions are implemented in **Julia** using the **JuMP.  jl** optimization framework.

### What is Stochastic Optimization? 

Stochastic optimization deals with decision-making under uncertainty, where some problem parameters are random variables. Applications include:

- 📦 **Inventory Management** - Newsvendor problem
- 🌾 **Agricultural Planning** - Farmer's problem  
- 🛢️ **Capacity Expansion** - Petroleum extraction
- 🎲 **Sequential Decisions** - Markov Decision Processes (MDPs)
- 📈 **Risk Management** - CVaR and coherent risk measures

---

## 🧩 Problems Solved

<details>
<summary><b>📖 Classical Problems</b></summary>

### 1. Newsvendor Problem (Jornaleiro)
- **File:** `codigo-jornaleiro-3cenarios.jl`
- **Method:** Two-stage stochastic programming
- **Status:** ✅ Solved
- **Key Result:** Optimal order quantity balances shortage vs excess costs

### 2. Farmer's Problem (Fazendeiro)
- **File:** `fazendeiro.jl`
- **Method:** Linear programming with uncertainty
- **Status:** ✅ Solved  
- **Key Result:** Monoculture (sugar beet) maximizes profit at **$262,000**

### 3. Petroleum Capacity Expansion
- **File:** `lista_1.jl`
- **Method:** Sample Average Approximation (SAA) + Benders Decomposition
- **Status:** ✅ Solved
- **Key Result:** Converges with N ≥ 500 scenarios (CV = 0.65%)

</details>

<details>
<summary><b>🎲 Markov Decision Processes</b></summary>

### 4. MDP with Value Iteration (Question 1)
- **File:** `lista_2_1.jl`
- **Method:** Value iteration algorithm
- **Status:** ✅ Solved
- **Discount Factors:** γ ∈ {0.1, 0.9}
- **Convergence:** 6-18 iterations

### 5. MDP with Absorbing States (Question 2)
- **File:** `lista_2_2.jl`
- **Method:** Value iteration with γ = 1
- **Status:** ✅ Solved
- **Key Result:** Optimal policy maximizes reaching highest-reward terminal state

### 6. Inventory Management as MDP (Question 3)
- **File:** `lista_2_3.jl`
- **Method:** Backward induction + Monte Carlo simulation
- **Status:** ✅ Solved
- **Policy:** (s, S) policy with s=5, S=20

</details>

<details>
<summary><b>📐 Analytical Solutions & Advanced Topics</b></summary>

### 7. Analytical Derivation (Questions 4-5)
- **File:** `lista_2_q4-5.jl`
- **Method:** Lagrange multipliers (KKT) + SDDP verification
- **Status:** ✅ Verified
- **Closed-form:** x* = M/N, f* = M²/N

### 8. Dual MDP Formulation (Question 6)
- **File:** `lista_2_q6.jl`
- **Method:** Linear programming (occupancy measure)
- **Status:** ✅ Solved

### 9. Forest Planning (Question 7)
- **File:** `lista_2_q7.jl`
- **Method:** Nonlinear optimization (Ipopt)
- **Status:** ✅ Solved (single & multi-species)

### 10. CVaR Coherence (Question 8)
- **File:** `lista_2_q8.jl`
- **Method:** Analytical proof + numerical verification
- **Status:** ✅ Proven coherent

### 11. Combined Risk Measure (Question 9)
- **File:** `lista_2_q9.jl`
- **Result:** ρ[X] = λE[X] + (1-λ)CVaR is coherent ⟺ **0 ≤ λ ≤ 1**

### 12. Entropic Risk Measure (Question 10)
- **File:** `lista_2_q10.jl`
- **Result:** ❌ **NOT coherent** (violates positive homogeneity)

</details>

<details>
<summary><b>📚 Chapter Exercises</b></summary>

### Chapter 3: Mixture Problem
- **File:** `cap3.jl`
- **Verified:** 4 ≤ v*(x₁*, x₂*) ≤ 7 ✅

### Chapter 4: min vs 𝔼 Non-commutativity
- **File:** `cap4.jl`
- **Proven:** 𝔼[min f] ≠ min 𝔼[f] ✅

### Chapter 5: L-Shaped Method
- **File:** `cap5.jl`
- **Status:** ✅ Converged in 3 iterations
- **Optimal:** x* = 10, f* = 0

### Chapter 6: Stochastic Decomposition
- **File:** `cap6.jl`
- **Status:** ⚠️ Problem infeasible (identified correctly)

</details>

---

## 🏆 Key Results

### 📊 Sample Average Approximation (SAA) Convergence

| Scenarios (N) | Cost (R$) | Expansion (T) | Time (s) | Gap |
|---------------|-----------|---------------|----------|-----|
| 10 | 58,729. 50 | 354. 47 | 0.0015 | High |
| 100 | 56,613.57 | 354.47 | 0.0280 | Medium |
| **500** | **55,640.61** | **330.94** | **0.6260** | **0.65%** ✅ |
| 1000 | 55,110.14 | 327.19 | 2.6350 | 0.65% ✅ |

**Complexity:** O(n^1.66) (subquadratic scaling)

### 🔀 Benders Decomposition Performance

| Method | Objective | Iterations | Gap | Time |
|--------|-----------|------------|-----|------|
| **Multi-Cut** | 53,973.11 | **4** | 0.020% | ⚡ Fast |
| Single-Cut | 53,955.72 | 7 | 0.047% | Slower |
| SAA (reference) | 53,973.11 | - | - | - |

**Multi-cut is 1.75× faster** in iterations!  

### 🎯 MDP Value Iteration

**Impact of Discount Factor:**

| State | V*(γ=0.1) | V*(γ=0.9) | Change | Policy |
|-------|-----------|-----------|--------|--------|
| S₀ | 0.0000 | 0.0000 | - | a₁ |
| S₁ | 1.0207 | **1.2787** | **+25%** | a₀ |
| S₂ | 0.0319 | **0.5395** | **+1590%** | a₀ |

Higher discount = more value in future rewards! 

---

## 🚀 Installation

### Prerequisites

- Julia 1.6 or higher
- JuMP. jl
- Optimization solvers:  HiGHS, GLPK, Ipopt

### Setup

```bash
# Clone the repository
git clone https://github.com/RafaelLopesPinheiro/SOBRAPO-OTIMIZACAO-ESTOCASTICA.git
cd SOBRAPO-OTIMIZACAO-ESTOCASTICA

# Start Julia REPL
julia

# Install required packages
using Pkg
Pkg.add(["JuMP", "HiGHS", "GLPK", "Ipopt", "Distributions", 
         "Random", "Statistics", "Plots", "DataFrames", "SDDP"])
```

---

## 💻 Usage

### Running Individual Problems

```julia
# Farmer's problem
include("fazendeiro.jl")

# Petroleum SAA
include("lista_1.jl")

# MDP with value iteration
include("lista_2_1.jl")

# L-Shaped method
include("cap5.jl")
```

### Example: Solving the Newsvendor Problem

```julia
using JuMP, HiGHS

# Parameters
c = 0.30  # Cost
p = 1.00  # Price
s = 0.10  # Salvage value

# Scenarios
demand = [40, 50, 60]
probs = [0.3, 0.5, 0.2]

# Model
model = Model(HiGHS.Optimizer)
@variable(model, Q >= 0)  # Order quantity
@variable(model, y1[1:3] >= 0)  # Sold
@variable(model, y2[1:3] >= 0)  # Unsold

# Constraints
for (i, d) in enumerate(demand)
    @constraint(model, y1[i] + y2[i] == Q)
    @constraint(model, y1[i] <= d)
end

# Objective
@objective(model, Max, 
    -c*Q + sum(probs[i]*(p*y1[i] + s*y2[i]) for i in 1:3))

optimize!(model)

println("Optimal order:  ", value(Q))
println("Expected profit: ", objective_value(model))
```

---

## 📁 Project Structure

```
SOBRAPO-OTIMIZACAO-ESTOCASTICA/
├── 📄 README.md                    # This file
├── 📄 main.tex                     # Full LaTeX report
│
├── 📊 Classical Problems
│   ├── fazendeiro.jl               # Farmer's problem
│   ├── codigo-jornaleiro-3cenarios.jl  # Newsvendor
│   └── lista_1.jl                  # Petroleum SAA
│
├── 🎲 MDP & Advanced Topics
│   ├── lista_2_1.jl                # MDP value iteration
│   ├── lista_2_2.jl                # MDP absorbing states
│   ├── lista_2_3.jl                # Inventory MDP
│   ├── lista_2_q4-5.jl             # Analytical solution
│   ├── lista_2_q6.jl               # Dual MDP
│   ├── lista_2_q7.jl               # Forest planning
│   ├── lista_2_q8.jl               # CVaR coherence
│   ├── lista_2_q9.jl               # Combined risk
│   └── lista_2_q10.jl              # Entropic risk
│
├── 📚 Chapter Exercises
│   ├── cap3.jl                     # Mixture problem
│   ├── cap4.jl                     # min vs E
│   ├── cap5.jl                     # L-Shaped method
│   └── cap6.jl                     # Stochastic decomposition
│
└── 📊 Results & Figures
    ├── saa_convergence.png
    ├── benders_comparison.png
    ├── bootstrap_analysis.png
    └── problem_cap5_lshaped.png
```

---

## 🔬 Methodologies

### 1. Sample Average Approximation (SAA)

Approximates stochastic programs using sample averaging:

```
E[Q(x,ω)] ≈ (1/N) Σᵢ Q(x, ωᵢ)
```

**Advantages:** Simple, parallelizable  
**Convergence:** Requires N ≥ 500 for < 1% variance

### 2. Benders Decomposition (L-Shaped)

Decomposes two-stage problems into: 
- **Master problem:** First-stage decisions
- **Subproblems:** Second-stage recourse (one per scenario)

**Generates optimality cuts:**
```
θ ≥ Q(xₖ) + π'(h - Tx)(x - xₖ)
```

### 3. Value Iteration (MDPs)

Iteratively solves Bellman equation:
```
Vₖ₊₁(s) = max_a { Σₛ' P(s'|s,a)[R(s,a,s') + γVₖ(s')] }
```

**Convergence:** O(log(1/ε)) iterations

### 4. Stochastic Dual Dynamic Programming (SDDP)

For multistage problems (used in Q4-5 verification)

---

## 📈 Visualizations

<div align="center">

### SAA Convergence
![SAA Convergence](docs/saa_convergence_example.png)

### Benders L-Shaped Method  
![L-Shaped](docs/lshaped_example.png)

### Bootstrap Stability Analysis
![Bootstrap](docs/bootstrap_example.png)

</div>

---

## 📄 Technical Report

A comprehensive **LaTeX report** (`main.tex`) documents all solutions with:

- ✅ Mathematical formulations
- ✅ Analytical derivations  
- ✅ Numerical results & tables
- ✅ Algorithm convergence analysis
- ✅ Comparative performance studies
- ✅ Theoretical proofs (CVaR coherence, etc.)

**Compile:**
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use Overleaf for online compilation.

---

## 📖 References

### Course Materials

- SOBRAPO School 2025 - [Course Repository](https://github.com/log-ufpb/sobrapo_school_2025)
- III Bienal da Sociedade Brasileira de Matemática - [Stochastic Optimization Text](https://www.im-uff.mat.br/puc-rio/disciplinas/2006.  1/soe/arquivos/iii-bienal-sbm-texto. pdf)

### Key Textbooks

1. **Birge, J. R., & Louveaux, F. (2011)**  
   *Introduction to Stochastic Programming* (2nd ed.). Springer.

2. **Shapiro, A., Dentcheva, D., & Ruszczyński, A. (2021)**  
   *Lectures on Stochastic Programming:  Modeling and Theory* (3rd ed.). SIAM.

3. **Kall, P., & Wallace, S. W. (1994)**  
   *Stochastic Programming*. Wiley.

### Software

- **JuMP.jl** - Dunning, I., Huchette, J., & Lubin, M. (2017). *SIAM Review*, 59(2), 295-320.
- **Julia** - Bezanson, J., et al. (2017). *SIAM Review*, 59(1), 65-98. 

---

## 👤 Author

**Rafael Lopes Pinheiro**

- 🎓 Student at SOBRAPO School 2025
- 💻 GitHub: [@RafaelLopesPinheiro](https://github.com/RafaelLopesPinheiro)
- 📧 Email: [Your Email]

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **SOBRAPO** (Sociedade Brasileira de Pesquisa Operacional)
- **LOG-UFPB** (Laboratório de Otimização e Gestão - UFPB)
- **Course Instructors** at SOBRAPO School 2025
- **Open Source Community** - JuMP.jl, Julia contributors

---

<div align="center">

### ⭐ If this repository helped you, please consider giving it a star! 

**Made with ❤️ using Julia & JuMP**

[⬆ Back to Top](#-sobrapo-school-2025---stochastic-optimization)

</div>
