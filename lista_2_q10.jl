# Questão 10:  Mostre que a medida de risco entrópica NÃO é coerente
# VERSÃO CORRIGIDA

using Random
using Statistics
using Distributions
using Plots

# Função da medida de risco entrópica
function entropic_risk(Z, γ)
    return (1/γ) * log(mean(exp.(γ * Z)))
end

# Função geradora de momentos analítica para Normal
function entropic_risk_normal_analytical(μ, σ, γ)
    return μ + (γ * σ^2) / 2
end

println("="^80)
println("QUESTÃO 10:  Medida de Risco Entrópica NÃO é Coerente")
println("="^80)
println()

println("DEFINIÇÃO:")
println("-"^80)
println("ENT_γ[Z] = (1/γ) log(𝔼[e^(γZ)])  onde γ > 0")
println()
println("Esta é baseada na função geradora de momentos e está relacionada")
println("à entropia relativa (divergência de Kullback-Leibler).")
println()

# ANÁLISE DAS PROPRIEDADES
println("="^80)
println("ANÁLISE DAS PROPRIEDADES DE COERÊNCIA")
println("="^80)
println()

# PROPRIEDADE 1: MONOTONICIDADE
println("PROPRIEDADE 1:  MONOTONICIDADE")
println("-"^80)
println("Se X ≤ Y, então ENT_γ[X] ≤ ENT_γ[Y]?  ")
println()
println("Demonstração:")
println("  Se X ≤ Y e γ > 0, então γX ≤ γY")
println("  Logo e^(γX) ≤ e^(γY)  (exp é crescente)")
println("  Portanto 𝔼[e^(γX)] ≤ 𝔼[e^(γY)]")
println("  Como log é crescente:    log(𝔼[e^(γX)]) ≤ log(𝔼[e^(γY)])")
println("  Dividindo por γ > 0:  ENT_γ[X] ≤ ENT_γ[Y]")
println()
println("  ✓ MONOTONICIDADE:    SATISFEITA")
println()

# Verificação numérica
Random.seed!(123)
n = 10000
X = randn(n)
Y = X .+ abs.(randn(n))  # Y ≥ X
γ = 0.5

ent_X = entropic_risk(X, γ)
ent_Y = entropic_risk(Y, γ)

println("Verificação Numérica (γ = $γ):")
println("  ENT_γ[X] = $(round(ent_X, digits=4))")
println("  ENT_γ[Y] = $(round(ent_Y, digits=4))  (Y ≥ X)")
println("  ENT_γ[X] ≤ ENT_γ[Y]? $(ent_X <= ent_Y) ✓")
println()

# PROPRIEDADE 2: SUBADITIVIDADE
println("PROPRIEDADE 2:  SUBADITIVIDADE")
println("-"^80)
println("ENT_γ[X+Y] ≤ ENT_γ[X] + ENT_γ[Y]? ")
println()
println("Demonstração:")
println("  ENT_γ[X+Y] = (1/γ) log(𝔼[e^(γ(X+Y))])")
println("            = (1/γ) log(𝔼[e^(γX) · e^(γY)])")
println()
println("  Se X e Y são INDEPENDENTES:")
println("    𝔼[e^(γX) · e^(γY)] = 𝔼[e^(γX)] · 𝔼[e^(γY)]")
println("    ENT_γ[X+Y] = (1/γ) log(𝔼[e^(γX)] · 𝔼[e^(γY)])")
println("              = (1/γ)[log(𝔼[e^(γX)]) + log(𝔼[e^(γY)])]")
println("              = ENT_γ[X] + ENT_γ[Y]")
println()
println("  ⚠ Para variáveis INDEPENDENTES:  IGUALDADE (satisfaz subaditividade)")
println("  ⚠ Para variáveis DEPENDENTES: pode VIOLAR subaditividade!")
println()

# Verificação numérica com independentes
Random.seed!(456)
X_ind = randn(n)
Y_ind = randn(n)  # Independente de X
Z_ind = X_ind .+ Y_ind

ent_X_ind = entropic_risk(X_ind, γ)
ent_Y_ind = entropic_risk(Y_ind, γ)
ent_Z_ind = entropic_risk(Z_ind, γ)

println("Teste 1: Variáveis INDEPENDENTES (γ = $γ):")
println("  ENT_γ[X] = $(round(ent_X_ind, digits=4))")
println("  ENT_γ[Y] = $(round(ent_Y_ind, digits=4))")
println("  ENT_γ[X+Y] = $(round(ent_Z_ind, digits=4))")
println("  ENT_γ[X] + ENT_γ[Y] = $(round(ent_X_ind + ent_Y_ind, digits=4))")

subaditiva_ind = ent_Z_ind <= ent_X_ind + ent_Y_ind + 1e-4
println("  ENT_γ[X+Y] ≤ ENT_γ[X] + ENT_γ[Y]? $(subaditiva_ind) $(subaditiva_ind ?  "✓" : "✗")")
println()

# Verificação com variáveis DEPENDENTES (podem violar!)
Random.seed!(789)
X_dep = randn(n)
Y_dep = X_dep .+ 0.3 * randn(n)  # Correlacionadas! 
Z_dep = X_dep .+ Y_dep

ent_X_dep = entropic_risk(X_dep, γ)
ent_Y_dep = entropic_risk(Y_dep, γ)
ent_Z_dep = entropic_risk(Z_dep, γ)

println("Teste 2: Variáveis DEPENDENTES (correlacionadas, γ = $γ):")
println("  ENT_γ[X] = $(round(ent_X_dep, digits=4))")
println("  ENT_γ[Y] = $(round(ent_Y_dep, digits=4))")
println("  ENT_γ[X+Y] = $(round(ent_Z_dep, digits=4))")
println("  ENT_γ[X] + ENT_γ[Y] = $(round(ent_X_dep + ent_Y_dep, digits=4))")

subaditiva_dep = ent_Z_dep <= ent_X_dep + ent_Y_dep + 1e-4
println("  ENT_γ[X+Y] ≤ ENT_γ[X] + ENT_γ[Y]? $(subaditiva_dep) $(subaditiva_dep ? "✓" : "✗✗✗ VIOLADA!")")
println()

if ! subaditiva_dep
    println("  ✗✗✗ SUBADITIVIDADE:   PODE SER VIOLADA (com dependência)")
else
    println("  ⚠ SUBADITIVIDADE:   Satisfeita neste exemplo, mas não sempre")
end
println()

# PROPRIEDADE 3: HOMOGENEIDADE POSITIVA
println("PROPRIEDADE 3:  HOMOGENEIDADE POSITIVA")
println("-"^80)
println("ENT_γ[λX] = λ · ENT_γ[X]  para λ ≥ 0?  ")
println()
println("Demonstração:")
println("  ENT_γ[λX] = (1/γ) log(𝔼[e^(γλX)])")
println()
println("  Para que seja homogênea, precisamos:")
println("  (1/γ) log(𝔼[e^(γλX)]) = λ · (1/γ) log(𝔼[e^(γX)])")
println("  log(𝔼[e^(γλX)]) = λ log(𝔼[e^(γX)])")
println("  𝔼[e^(γλX)] = (𝔼[e^(γX)])^λ")
println()
println("  Mas isso NÃO é verdade em geral!")
println()
println("  EXEMPLO:   X ~ N(0, σ^2)")
println("    𝔼[e^(γX)] = e^(γ^2 σ^2 / 2)  (função geradora de momentos)")
println("    𝔼[e^(γλX)] = e^((γλ)^2 σ^2 / 2) = e^(γ^2 λ^2 σ^2 / 2)")
println()
println("    (𝔼[e^(γX)])^λ = (e^(γ^2 σ^2 / 2))^λ = e^(γ^2 λ σ^2 / 2)")
println()
println("    Comparando:")
println("    e^(γ^2 λ^2 σ^2 / 2) ≟ e^(γ^2 λ σ^2 / 2)")
println("    γ^2 λ^2 σ^2 / 2 ≟ γ^2 λ σ^2 / 2")
println("    λ^2 ≟ λ")
println()
println("    Isso só é verdade se λ = 0 ou λ = 1!")
println()
println("  ✗✗✗ HOMOGENEIDADE POSITIVA:   VIOLADA!")
println()

# Verificação numérica detalhada
println("Verificação Numérica Detalhada:")
println("-"^80)
println()

# Usar distribuição normal para ter cálculo analítico
μ = 0.0
σ = 1.0
γ_test = 0.5
λ_values = [0.5, 1.0, 2.0, 3.0]

println("Distribuição:   X ~ N($μ, $(σ)^2)")  # CORRIGIDO AQUI
println("γ = $γ_test")
println()

Random.seed!(789)
X_normal = randn(n) * σ .+ μ

# Valor analítico de ENT_γ[X]
ent_X_analytical = entropic_risk_normal_analytical(μ, σ, γ_test)
ent_X_numerical = entropic_risk(X_normal, γ_test)

println("ENT_γ[X] (analítico) = $(round(ent_X_analytical, digits=4))")
println("ENT_γ[X] (numérico)  = $(round(ent_X_numerical, digits=4))")
println()

for λ in λ_values
    # Calcular ENT_γ[λX]
    λX = λ .* X_normal
    ent_λX_numerical = entropic_risk(λX, γ_test)
    ent_λX_analytical = entropic_risk_normal_analytical(λ * μ, λ * σ, γ_test)
    
    # Calcular λ · ENT_γ[X]
    λ_times_ent_X = λ * ent_X_analytical
    
    # Comparar
    println("λ = $λ:")
    println("  ENT_γ[$(λ)X] (numérico)  = $(round(ent_λX_numerical, digits=4))")
    println("  ENT_γ[$(λ)X] (analítico) = $(round(ent_λX_analytical, digits=4))")
    println("  $(λ) × ENT_γ[X]          = $(round(λ_times_ent_X, digits=4))")
    println("  Diferença = $(round(abs(ent_λX_analytical - λ_times_ent_X), digits=4))")
    
    homogenea = isapprox(ent_λX_analytical, λ_times_ent_X, rtol=1e-3)
    println("  ENT_γ[$(λ)X] = $(λ) × ENT_γ[X]?    $(homogenea ?   "✓" : "✗ VIOLADA")")
    println()
end

# PROPRIEDADE 4: INVARIÂNCIA POR TRANSLAÇÃO
println("PROPRIEDADE 4:  INVARIÂNCIA POR TRANSLAÇÃO")
println("-"^80)
println("ENT_γ[X+c] = ENT_γ[X] + c?  ")
println()
println("Demonstração:")
println("  ENT_γ[X+c] = (1/γ) log(𝔼[e^(γ(X+c))])")
println("            = (1/γ) log(𝔼[e^(γX) · e^(γc)])")
println("            = (1/γ) log(e^(γc) · 𝔼[e^(γX)])  (e^(γc) é constante)")
println("            = (1/γ) [γc + log(𝔼[e^(γX)])]")
println("            = c + (1/γ) log(𝔼[e^(γX)])")
println("            = c + ENT_γ[X]")
println()
println("  ✓ INVARIÂNCIA POR TRANSLAÇÃO:  SATISFEITA")
println()

# Verificação numérica
Random.seed!(101)
X = randn(n)
c = 5.0

ent_X = entropic_risk(X, γ)
ent_Xc = entropic_risk(X .+ c, γ)

println("Verificação Numérica (γ = $γ, c = $c):")
println("  ENT_γ[X] = $(round(ent_X, digits=4))")
println("  ENT_γ[X+$c] = $(round(ent_Xc, digits=4))")
println("  ENT_γ[X] + $c = $(round(ent_X + c, digits=4))")
println("  ENT_γ[X+c] = ENT_γ[X] + c?   $(isapprox(ent_Xc, ent_X + c, rtol=1e-3)) ✓")
println()

# RESUMO
println("="^80)
println("RESUMO:    PROPRIEDADES DA MEDIDA ENTRÓPICA")
println("="^80)
println()
println("  1.Monotonicidade:             ✓ SATISFEITA")
println("  2.Subaditividade:           ⚠ PODE SER VIOLADA (dependendo das variáveis)")
println("  3.Homogeneidade Positiva:   ✗ VIOLADA")
println("  4.Invariância por Translação: ✓ SATISFEITA")
println()
println("  ╔═══════════════════════════════════════════════════════╗")
println("  ║  CONCLUSÃO:  A medida de risco entrópica NÃO É       ║")
println("  ║              COERENTE pois viola HOMOGENEIDADE!        ║")
println("  ║              (e pode violar SUBADITIVIDADE também)    ║")
println("  ╚═══════════════════════════════════════════════════════╝")
println()

# VISUALIZAÇÃO
println("="^80)
println("VISUALIZAÇÃO:    Violação da Homogeneidade")
println("="^80)
println()

# Criar gráfico mostrando a violação
λ_range = 0.0:0.1:3.0
μ_plot = 0.0
σ_plot = 1.0
γ_plot = 0.5

# ENT_γ[λX] (verdadeiro)
ent_λX = [entropic_risk_normal_analytical(λ * μ_plot, λ * σ_plot, γ_plot) for λ in λ_range]

# λ · ENT_γ[X] (se fosse homogênea)
ent_X_base = entropic_risk_normal_analytical(μ_plot, σ_plot, γ_plot)
λ_times_ent_X = [λ * ent_X_base for λ in λ_range]

p = plot(λ_range, ent_λX, 
         label="ENT_γ[λX] (real)",
         xlabel="λ",
         ylabel="Valor da Medida de Risco",
         title="Violação da Homogeneidade Positiva\nX ~ N(0,1), γ=$γ_plot",
         linewidth=3,
         color=:red,
         legend=:topleft,
         size=(800, 600))

plot!(p, λ_range, λ_times_ent_X,
      label="λ × ENT_γ[X] (se fosse homogênea)",
      linewidth=3,
      linestyle=:dash,
      color=:blue)

# Marcar ponto onde coincidem (λ=1)
scatter!(p, [1.0], [ent_X_base],
         markersize=8,
         color=:green,
         label="λ=1 (coincidem)")

# Adicionar área de diferença
plot!(p, λ_range, ent_λX,
      fillrange=λ_times_ent_X,
      fillalpha=0.2,
      fillcolor=:orange,
      label="Diferença (violação)")

savefig(p, "entropic_risk_nonhomogeneous.png")
println("Gráfico salvo como 'entropic_risk_nonhomogeneous.png'")
println()

# EXPLICAÇÃO ADICIONAL
println("="^80)
println("POR QUE ISSO IMPORTA?")
println("="^80)
println()
println("A violação da homogeneidade positiva significa que:")
println()
println("  Se você DOBRA sua posição (λ=2), o risco entrópico NÃO dobra!")
println()
println("Exemplo com X ~ N(0, 1) e γ = 0.5:")
Random.seed!(999)
X_example = randn(10000)
γ_ex = 0.5

ent_1X = entropic_risk(X_example, γ_ex)
ent_2X = entropic_risk(2 .* X_example, γ_ex)

println("  ENT_γ[X]  = $(round(ent_1X, digits=4))")
println("  ENT_γ[2X] = $(round(ent_2X, digits=4))")
println("  2 × ENT_γ[X] = $(round(2 * ent_1X, digits=4))")
println()
println("  ENT_γ[2X] > 2 × ENT_γ[X]  (risco cresce MAIS que linearmente! )")
println()
println("Isso reflete aversão ao risco crescente com o tamanho da posição,")
println("mas viola o axioma de coerência de Artzner et al.(1999).")
println()

println("="^80)
println("RESPOSTA FINAL")
println("="^80)
println()
println("A medida de risco entrópica ENT_γ[Z] = (1/γ) log(𝔼[e^(γZ)])")
println("NÃO é coerente porque:")
println()
println("  ✗  Viola HOMOGENEIDADE POSITIVA:")
println("     ENT_γ[λX] ≠ λ · ENT_γ[X] para λ > 0 (exceto λ = 1)")
println()
println("  ⚠  Pode violar SUBADITIVIDADE:")
println("     Para variáveis dependentes:  ENT_γ[X+Y] pode ser > ENT_γ[X] + ENT_γ[Y]")
println()
println("Especificamente, para X ~ N(μ, σ^2):")
println("  ENT_γ[λX] = λμ + (γλ^2 σ^2)/2")
println("  λ·ENT_γ[X] = λμ + (γλσ^2)/2")
println()
println("  λ^2 ≠ λ  (exceto λ ∈ {0, 1})")
println()
println("Apesar de não ser coerente, a medida entrópica é amplamente")
println("usada em finanças e teoria da decisão por suas propriedades")
println("de convexidade e tratamento de caudas pesadas.")
println("="^80)