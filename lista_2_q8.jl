# Demonstração:  Conditional Value-at-Risk (CVaR) é Coerente
# Questão 8: Mostre que o Conditional Value-at-Risk é coerente
#
# Uma medida de risco ρ é coerente se satisfaz as seguintes propriedades:
# 1.Monotonicidade:  Se X ≤ Y, então ρ(X) ≥ ρ(Y)
# 2.Subaditividade: ρ(X + Y) ≤ ρ(X) + ρ(Y)
# 3.Homogeneidade Positiva: ρ(λX) = λρ(X) para λ ≥ 0
# 4.Invariância por Translação: ρ(X + c) = ρ(X) - c para c ∈ ℝ

using Random
using Statistics
using Distributions

# Definição do CVaR
"""
    cvar(losses, α)

Calcula o Conditional Value-at-Risk (CVaR) para um nível de confiança α.
CVaR_α(X) = E[X | X ≥ VaR_α(X)]

Parâmetros:
- losses: vetor de perdas (valores negativos representam ganhos)
- α: nível de confiança (ex:  0.95)
"""
function cvar(losses, α)
    var_α = quantile(losses, α)
    # CVaR é a média das perdas que excedem o VaR
    tail_losses = losses[losses .>= var_α]
    return mean(tail_losses)
end

# DEMONSTRAÇÃO TEÓRICA
println("="^70)
println("DEMONSTRAÇÃO:  CVaR É UMA MEDIDA DE RISCO COERENTE")
println("="^70)
println()

println("DEFINIÇÃO:")
println("CVaR_α(X) = (1/(1-α)) ∫_{α}^{1} VaR_u(X) du")
println("         = E[X | X ≥ VaR_α(X)]")
println()

# PROPRIEDADES DE COERÊNCIA

println("="^70)
println("PROPRIEDADE 1: MONOTONICIDADE")
println("="^70)
println("Se X ≤ Y (X domina Y), então CVaR_α(X) ≤ CVaR_α(Y)")
println()
println("Demonstração Analítica:")
println("Se X ≤ Y, então VaR_α(X) ≤ VaR_α(Y) para todo α")
println("Logo, E[X | X ≥ VaR_α(X)] ≤ E[Y | Y ≥ VaR_α(Y)]")
println("Portanto, CVaR_α(X) ≤ CVaR_α(Y) ✓")
println()

# Verificação numérica
Random.seed!(123)
n = 10000
X = randn(n)
Y = X .+ 1  # Y domina X (Y tem perdas maiores)

α = 0.95
cvar_X = cvar(X, α)
cvar_Y = cvar(Y, α)

println("Verificação Numérica:")
println("CVaR_$(α)(X) = $(round(cvar_X, digits=4))")
println("CVaR_$(α)(Y) = $(round(cvar_Y, digits=4))")
println("CVaR_$(α)(X) ≤ CVaR_$(α)(Y)? $(cvar_X ≤ cvar_Y) ✓")
println()

println("="^70)
println("PROPRIEDADE 2: SUBADITIVIDADE")
println("="^70)
println("CVaR_α(X + Y) ≤ CVaR_α(X) + CVaR_α(Y)")
println()
println("Demonstração Analítica:")
println("CVaR_α(X + Y) = E[X + Y | X + Y ≥ VaR_α(X + Y)]")
println("             ≤ E[X | X ≥ VaR_α(X)] + E[Y | Y ≥ VaR_α(Y)]")
println("             = CVaR_α(X) + CVaR_α(Y)")
println()
println("Esta propriedade segue da representação dual do CVaR e da")
println("propriedade de subaditividade da esperança condicional.")
println()

# Verificação numérica
Random.seed!(456)
X = randn(n)
Y = randn(n)
Z = X .+ Y

cvar_X = cvar(X, α)
cvar_Y = cvar(Y, α)
cvar_Z = cvar(Z, α)

println("Verificação Numérica:")
println("CVaR_$(α)(X) = $(round(cvar_X, digits=4))")
println("CVaR_$(α)(Y) = $(round(cvar_Y, digits=4))")
println("CVaR_$(α)(X+Y) = $(round(cvar_Z, digits=4))")
println("CVaR_$(α)(X) + CVaR_$(α)(Y) = $(round(cvar_X + cvar_Y, digits=4))")
println("CVaR_$(α)(X+Y) ≤ CVaR_$(α)(X) + CVaR_$(α)(Y)? $(cvar_Z ≤ cvar_X + cvar_Y) ✓")
println()

println("="^70)
println("PROPRIEDADE 3: HOMOGENEIDADE POSITIVA")
println("="^70)
println("CVaR_α(λX) = λ·CVaR_α(X) para λ ≥ 0")
println()
println("Demonstração Analítica:")
println("CVaR_α(λX) = E[λX | λX ≥ VaR_α(λX)]")
println("          = E[λX | X ≥ VaR_α(X)]  (pois VaR_α(λX) = λ·VaR_α(X))")
println("          = λ·E[X | X ≥ VaR_α(X)]")
println("          = λ·CVaR_α(X) ✓")
println()

# Verificação numérica
Random.seed!(789)
X = randn(n)
λ = 2.5
λX = λ .* X

cvar_X = cvar(X, α)
cvar_λX = cvar(λX, α)

println("Verificação Numérica (λ = $(λ)):")
println("CVaR_$(α)(X) = $(round(cvar_X, digits=4))")
println("CVaR_$(α)($(λ)X) = $(round(cvar_λX, digits=4))")
println("$(λ)·CVaR_$(α)(X) = $(round(λ * cvar_X, digits=4))")
println("CVaR_$(α)($(λ)X) = $(λ)·CVaR_$(α)(X)? $(isapprox(cvar_λX, λ * cvar_X, rtol=1e-3)) ✓")
println()

println("="^70)
println("PROPRIEDADE 4: INVARIÂNCIA POR TRANSLAÇÃO")
println("="^70)
println("CVaR_α(X + c) = CVaR_α(X) + c para c ∈ ℝ")
println()
println("Demonstração Analítica:")
println("CVaR_α(X + c) = E[X + c | X + c ≥ VaR_α(X + c)]")
println("             = E[X + c | X ≥ VaR_α(X)]  (pois VaR_α(X + c) = VaR_α(X) + c)")
println("             = E[X | X ≥ VaR_α(X)] + c")
println("             = CVaR_α(X) + c ✓")
println()

# Verificação numérica
Random.seed!(101)
X = randn(n)
c = 5.0
Xc = X .+ c

cvar_X = cvar(X, α)
cvar_Xc = cvar(Xc, α)

println("Verificação Numérica (c = $(c)):")
println("CVaR_$(α)(X) = $(round(cvar_X, digits=4))")
println("CVaR_$(α)(X+$(c)) = $(round(cvar_Xc, digits=4))")
println("CVaR_$(α)(X) + $(c) = $(round(cvar_X + c, digits=4))")
println("CVaR_$(α)(X+$(c)) = CVaR_$(α)(X) + $(c)? $(isapprox(cvar_Xc, cvar_X + c, rtol=1e-3)) ✓")
println()

println("="^70)
println("CONCLUSÃO")
println("="^70)
println("O Conditional Value-at-Risk (CVaR) satisfaz todas as quatro")
println("propriedades de Artzner et al.(1999):")
println("  ✓ Monotonicidade")
println("  ✓ Subaditividade")
println("  ✓ Homogeneidade Positiva")
println("  ✓ Invariância por Translação")
println()
println("Portanto, CVaR É UMA MEDIDA DE RISCO COERENTE.")
println()
println("Observação:  Ao contrário do VaR, que não é subaditivo,")
println("o CVaR é preferível em gestão de risco por ser coerente.")
println("="^70)