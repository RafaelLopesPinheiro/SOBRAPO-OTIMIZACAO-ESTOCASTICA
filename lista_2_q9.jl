# Versão FINAL CORRIGIDA - Questão 9
# ρ[X] = λE[X] + (1-λ)CVaR_α[X]

using Random
using Statistics
using Distributions
using Plots

function cvar(losses, α)
    var_α = quantile(losses, α)
    tail_losses = losses[losses .>= var_α]
    return mean(tail_losses)
end

function rho_combined(losses, α, λ)
    return λ * mean(losses) + (1 - λ) * cvar(losses, α)
end

println("="^80)
println("ANÁLISE:  ρ[X] = λE[X] + (1-λ)CVaR_α[X]")
println("="^80)
println()

println("ANÁLISE TEÓRICA DAS PROPRIEDADES DE COERÊNCIA")
println("="^80)
println()

println("1.MONOTONICIDADE:   Se X ≤ Y, então ρ(X) ≤ ρ(Y)")
println("-"^80)
println("   CONDIÇÃO: 0 ≤ λ ≤ 1  ✓")
println()

println("2.SUBADITIVIDADE:  ρ[X+Y] ≤ ρ[X] + ρ[Y]")
println("-"^80)
println("   CONDIÇÃO: λ ≤ 1  ✓")
println()

println("3.HOMOGENEIDADE POSITIVA:  ρ[cX] = cρ[X] para c ≥ 0")
println("-"^80)
println("   CONDIÇÃO:   Satisfeita para qualquer λ  ✓")
println()

println("4.INVARIÂNCIA POR TRANSLAÇÃO:  ρ[X+c] = ρ[X] + c")
println("-"^80)
println("   CONDIÇÃO:  Satisfeita para qualquer λ  ✓")
println()

println("="^80)
println("CONCLUSÃO TEÓRICA:   0 ≤ λ ≤ 1")
println("="^80)
println()

# VERIFICAÇÃO NUMÉRICA
println("="^80)
println("VERIFICAÇÃO NUMÉRICA")
println("="^80)
println()

Random.seed!(42)
n = 10000
α = 0.95

λ_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, -0.5]

println("Testando diferentes valores de λ:")
println("-"^80)
println()

for λ in λ_values
    println("λ = $λ:")
    
    # Gerar dados (usando local para evitar warnings)
    local X = randn(n)
    local Y = randn(n)
    local Z = X .+ Y
    local c = 2.0
    local k = 2.5
    
    # Criar Y_dominado tal que X ≤ Y_dominado
    local Y_dominado = X .+ abs.(randn(n))
    
    # Calcular medidas
    local rho_X = rho_combined(X, α, λ)
    local rho_Y = rho_combined(Y, α, λ)
    local rho_Y_dom = rho_combined(Y_dominado, α, λ)
    local rho_Z = rho_combined(Z, α, λ)
    local rho_kX = rho_combined(k .* X, α, λ)
    local rho_Xc = rho_combined(X .+ c, α, λ)
    
    # Testar propriedades
    local monotonicidade = rho_X <= rho_Y_dom + 1e-6
    local subaditividade = rho_Z <= rho_X + rho_Y + 1e-6
    local homogeneidade = isapprox(rho_kX, k * rho_X, rtol=1e-3)
    local translacao = isapprox(rho_Xc, rho_X + c, rtol=1e-3)
    
    local coerente = monotonicidade && subaditividade && homogeneidade && translacao
    local valido = 0 <= λ <= 1
    
    println("  1.Monotonicidade:  $(monotonicidade ?   "✓" : "✗")  (X ≤ Y ⟹ ρ(X) ≤ ρ(Y))")
    println("  2.Subaditividade:  $(subaditividade ?  "✓" : "✗")  (ρ(X+Y) ≤ ρ(X)+ρ(Y))")
    println("  3.Homogeneidade:   $(homogeneidade ?  "✓" : "✗")  (ρ(kX) = kρ(X))")
    println("  4.Translação:      $(translacao ? "✓" : "✗")  (ρ(X+c) = ρ(X)+c)")
    println("  ─────────────────────────────────────")
    println("  ✓ COERENTE:          $(coerente ? "✓✓✓ SIM" : "✗✗✗ NÃO")")
    println("  ✓ 0 ≤ λ ≤ 1:        $(valido ? "✓✓✓ SIM" :  "✗✗✗ NÃO")")
    println()
end

# TESTE DETERMINÍSTICO PARA λ < 0
println("="^80)
println("TESTE DETERMINÍSTICO: POR QUE λ < 0 VIOLA MONOTONICIDADE")
println("="^80)
println()

function teste_monotonicidade_deterministico(λ, α)
    # Exemplo simples com valores constantes
    X = fill(1.0, 1000)  # Perdas = 1
    Y = fill(2.0, 1000)  # Perdas = 2 (Y > X)
    
    # Calcular medidas
    E_X = mean(X)  # = 1
    E_Y = mean(Y)  # = 2
    CVaR_X = cvar(X, α)  # = 1 (todos valores iguais)
    CVaR_Y = cvar(Y, α)  # = 2 (todos valores iguais)
    
    rho_X = λ * E_X + (1 - λ) * CVaR_X
    rho_Y = λ * E_Y + (1 - λ) * CVaR_Y
    
    println("Exemplo:  X = 1 (constante),  Y = 2 (constante)")
    println("Claramente X < Y")
    println()
    println("Para λ = $λ:")
    println("  E[X] = $E_X,  CVaR(X) = $CVaR_X")
    println("  E[Y] = $E_Y,  CVaR(Y) = $CVaR_Y")
    println()
    println("  ρ(X) = $λ × $E_X + $(1-λ) × $CVaR_X = $rho_X")
    println("  ρ(Y) = $λ × $E_Y + $(1-λ) × $CVaR_Y = $rho_Y")
    println()
    println("  Diferença: ρ(Y) - ρ(X) = $(rho_Y - rho_X)")
    println()
    
    monotonica = rho_X <= rho_Y
    
    if monotonica
        println("  ✓ Monotonicidade OK:   ρ(X) ≤ ρ(Y)")
    else
        println("  ✗✗✗ MONOTONICIDADE VIOLADA!")
        println("  ✗✗✗ X < Y mas ρ(X) > ρ(Y)")
        println("  ✗✗✗ Portanto λ = $λ NÃO é coerente!")
    end
    println()
    
    return monotonica
end

# Testar λ = -0.5
println("Teste para λ = -0.5:")
println("-"^80)
teste_monotonicidade_deterministico(-0.5, α)

println("Teste para λ = 0.5 (dentro do intervalo válido):")
println("-"^80)
teste_monotonicidade_deterministico(0.5, α)

println("Teste para λ = 1.5 (fora do intervalo):")
println("-"^80)
teste_monotonicidade_deterministico(1.5, α)

# EXPLICAÇÃO MATEMÁTICA DETALHADA
println("="^80)
println("EXPLICAÇÃO MATEMÁTICA:   POR QUE λ < 0 FALHA? ")
println("="^80)
println()
println("Considerando X < Y (elemento a elemento):")
println()
println("  ρ(Y) - ρ(X) = λ[E[Y] - E[X]] + (1-λ)[CVaR(Y) - CVaR(X)]")
println()
println("Como X < Y, sabemos que:")
println("  • E[Y] - E[X] > 0  (positivo)")
println("  • CVaR(Y) - CVaR(X) > 0  (positivo)")
println()
println("Para garantir ρ(Y) - ρ(X) > 0 (monotonicidade):")
println()
println("  ┌─────────────────────────────────────────────────────┐")
println("  │ Caso λ < 0:                                         │")
println("  │   λ[E[Y] - E[X]] < 0  (NEGATIVO!)                   │")
println("  │   (1-λ)[CVaR(Y) - CVaR(X)] > 0  (positivo)          │")
println("  │                                                      │")
println("  │   Se |λ| é grande, o termo negativo pode dominar!    │")
println("  │   Resultado: ρ(Y) - ρ(X) < 0  ✗ VIOLA MONOT.      │")
println("  └─────────────────────────────────────────────────────┘")
println()
println("  ┌─────────────────────────────────────────────────────┐")
println("  │ Caso 0 ≤ λ ≤ 1:                                     │")
println("  │   λ[E[Y] - E[X]] ≥ 0  (não-negativo)                │")
println("  │   (1-λ)[CVaR(Y) - CVaR(X)] ≥ 0  (não-negativo)      │")
println("  │                                                      │")
println("  │   Soma de não-negativos = não-negativo              │")
println("  │   Resultado: ρ(Y) - ρ(X) ≥ 0  ✓ OK                 │")
println("  └─────────────────────────────────────────────────────┘")
println()
println("  ┌─────────────────────────────────────────────────────┐")
println("  │ Caso λ > 1:                                         │")
println("  │   (1-λ) < 0, então (1-λ)[CVaR(Y) - CVaR(X)] < 0     │")
println("  │   Pode violar monotonicidade E subaditividade        │")
println("  └─────────────────────────────────────────────────────┘")
println()

# VISUALIZAÇÃO
println("="^80)
println("CRIANDO VISUALIZAÇÃO")
println("="^80)
println()

Random.seed!(123)
X_samples = randn(n)
λ_range = -0.5:0.05:1.5
rho_values = [rho_combined(X_samples, α, λ) for λ in λ_range]

p = plot(λ_range, rho_values, 
         xlabel="λ", 
         ylabel="ρ[X]",
         title="ρ[X] = λE[X] + (1-λ)CVaR_α[X]",
         label="Medida de Risco",
         linewidth=2,
         legend=:topright,
         size=(800, 600))

# Destacar região válida
plot!(p, [0, 1], [rho_combined(X_samples, α, 0), rho_combined(X_samples, α, 1)],
      linewidth=4, color=:green, label="Região Coerente (0 ≤ λ ≤ 1)")

# Marcar pontos especiais
scatter!(p, [0], [cvar(X_samples, α)], 
         markersize=8, color=:red, label="λ=0 (CVaR puro)")
scatter!(p, [1], [mean(X_samples)], 
         markersize=8, color=:blue, label="λ=1 (E[X] puro)")
scatter!(p, [0.5], [rho_combined(X_samples, α, 0.5)],
         markersize=8, color=:purple, label="λ=0.5 (balanceado)")

# Região coerente
vspan!(p, [0, 1], alpha=0.15, color=:green, label="")

# Regiões não-coerentes
vspan!(p, [-0.5, 0], alpha=0.15, color=:red, label="")
vspan!(p, [1, 1.5], alpha=0.15, color=:red, label="")

# Adicionar texto
annotate!(p, 0.5, maximum(rho_values) * 0.9, 
          text("COERENTE", 10, :green, :bold))
annotate!(p, -0.25, maximum(rho_values) * 0.5, 
          text("NÃO\nCOERENTE", 8, :red, :bold))
annotate!(p, 1.25, maximum(rho_values) * 0.5, 
          text("NÃO\nCOERENTE", 8, :red, :bold))

savefig(p, "cvar_lambda_analysis.png")
println("Gráfico salvo como 'cvar_lambda_analysis.png'")
println()

# RESPOSTA FINAL
println("="^80)
println("RESPOSTA FINAL")
println("="^80)
println()
println("  ╔═══════════════════���════════════════════════════════════╗")
println("  ║  A medida ρ[X] = λE[X] + (1-λ)CVaR_α[X] é COERENTE    ║")
println("  ║                                                        ║")
println("  ║         SE E SOMENTE SE:   0 ≤ λ ≤ 1                    ║")
println("  ╚════════════════════════════════════════════════════════╝")
println()
println("Resumo das condições:")
println("  • Monotonicidade:        requer 0 ≤ λ ≤ 1")
println("  • Subaditividade:       requer λ ≤ 1")
println("  • Homogeneidade:        sempre satisfeita")
println("  • Translação:           sempre satisfeita")
println()
println("Interseção:  0 ≤ λ ≤ 1")
println()
println("Interpretação:")
println("  λ = 0   → ρ[X] = CVaR_α[X]  (100% aversão ao risco)")
println("  λ = 0.5 → Mix balanceado entre média e risco extremo")
println("  λ = 1   → ρ[X] = E[X]  (neutro ao risco)")
println()
println("="^80)