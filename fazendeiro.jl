# Problema do Fazendeiro - Otimização Linear
# Exercícios de Programação Linear

using JuMP
using HiGHS  # ou outro solver como GLPK, Cbc, etc. 

println("=== PROBLEMA DO FAZENDEIRO ===\n")

# Dados do problema base
area_total = 500  # hectares
trigo_demanda = 200  # toneladas
milho_demanda = 240  # toneladas

# Rendimentos (T/ha)
rend_trigo = 2.5
rend_milho = 3.0
rend_cana = 20.0

# Preços de venda ($/T)
preco_venda_trigo = 170
preco_venda_milho = 150
preco_venda_cana = 36

# Preços de compra ($/T)
preco_compra_trigo = 238
preco_compra_milho = 210

println("Dados do problema:")
println("Área total disponível: $area_total ha")
println("Demanda mínima - Trigo: $trigo_demanda T, Milho: $milho_demanda T")
println()

# ====================
# EXERCÍCIO [01] - Problema base com variação de preços
# ====================
println("\n[01] Problema base do fazendeiro")

model01 = Model(HiGHS.Optimizer)
set_silent(model01)

# Variáveis de decisão
@variable(model01, x1 >= 0)  # hectares de trigo
@variable(model01, x2 >= 0)  # hectares de milho
@variable(model01, x3 >= 0)  # hectares de cana-de-açúcar
@variable(model01, w1 >= 0)  # toneladas de trigo compradas
@variable(model01, w2 >= 0)  # toneladas de milho compradas
@variable(model01, y1 >= 0)  # toneladas de trigo vendidas
@variable(model01, y2 >= 0)  # toneladas de milho vendidas
@variable(model01, y3 >= 0)  # toneladas de cana vendidas

# Restrições
@constraint(model01, x1 + x2 + x3 <= area_total)  # área total
@constraint(model01, rend_trigo * x1 + w1 - y1 >= trigo_demanda)  # demanda trigo
@constraint(model01, rend_milho * x2 + w2 - y2 >= milho_demanda)  # demanda milho
@constraint(model01, y3 <= rend_cana * x3)  # produção de cana

# Função objetivo:  maximizar lucro
@objective(model01, Max, 
    preco_venda_trigo * y1 + preco_venda_milho * y2 + preco_venda_cana * y3 -
    preco_compra_trigo * w1 - preco_compra_milho * w2
)

optimize!(model01)

println("Status: ", termination_status(model01))
println("Lucro máximo: \$", objective_value(model01))
println("Área de trigo: ", value(x1), " ha")
println("Área de milho: ", value(x2), " ha")
println("Área de cana-de-açúcar: ", value(x3), " ha")
println("Trigo comprado: ", value(w1), " T")
println("Milho comprado: ", value(w2), " T")
println("Trigo vendido: ", value(y1), " T")
println("Milho vendido: ", value(y2), " T")
println("Cana vendida:  ", value(y3), " T")

# Agora com variação de preços (±10%)
println("\n--- Análise com variação de preços ±10% ---")

# Preços 10% acima da média
precos_altos = [preco_venda_trigo * 1.1, preco_venda_milho * 1.1]
precos_baixos = [preco_venda_trigo * 0.9, preco_venda_milho * 0.9]

# Você pode adicionar mais cenários aqui

# ====================
# EXERCÍCIO [02] - Propriedade dividida em lotes
# ====================
println("\n\n[02] Propriedade dividida em 4 lotes")

lotes = [185, 145, 105, 65]  # hectares
println("Tamanhos dos lotes:  ", lotes, " ha")

model02 = Model(HiGHS.Optimizer)
set_silent(model02)

# Variáveis:  que cultura plantar em cada lote (apenas um tipo por lote)
@variable(model02, x_lote[1:4, 1:3], Bin)  # lote i, cultura j (1=trigo, 2=milho, 3=cana)
@variable(model02, w1_02 >= 0)
@variable(model02, w2_02 >= 0)
@variable(model02, y1_02 >= 0)
@variable(model02, y2_02 >= 0)
@variable(model02, y3_02 >= 0)

# Cada lote recebe apenas uma cultura
for i in 1:4
    @constraint(model02, sum(x_lote[i, j] for j in 1:3) == 1)
end

# Produção total por cultura
prod_trigo = sum(lotes[i] * rend_trigo * x_lote[i, 1] for i in 1:4)
prod_milho = sum(lotes[i] * rend_milho * x_lote[i, 2] for i in 1:4)
prod_cana = sum(lotes[i] * rend_cana * x_lote[i, 3] for i in 1:4)

# Restrições de demanda
@constraint(model02, prod_trigo + w1_02 - y1_02 >= trigo_demanda)
@constraint(model02, prod_milho + w2_02 - y2_02 >= milho_demanda)
@constraint(model02, y3_02 <= prod_cana)

# Objetivo
@objective(model02, Max,
    preco_venda_trigo * y1_02 + preco_venda_milho * y2_02 + preco_venda_cana * y3_02 -
    preco_compra_trigo * w1_02 - preco_compra_milho * w2_02
)

optimize!(model02)

println("Status: ", termination_status(model02))
println("Lucro máximo: \$", objective_value(model02))
for i in 1:4
    for j in 1:3
        if value(x_lote[i, j]) > 0.5
            cultura = ["Trigo", "Milho", "Cana-de-açúcar"][j]
            println("Lote $i ($(lotes[i]) ha): $cultura")
        end
    end
end

# ====================
# EXERCÍCIO [03] - Restrição de centenas de toneladas
# ====================
println("\n\n[03] Compra/venda apenas em centenas de toneladas")

model03 = Model(HiGHS.Optimizer)
set_silent(model03)

# Variáveis contínuas para área
@variable(model03, x1_03 >= 0)
@variable(model03, x2_03 >= 0)
@variable(model03, x3_03 >= 0)

# Variáveis inteiras para compra/venda (em centenas de toneladas)
@variable(model03, w1_cent >= 0, Int)  # centenas de T
@variable(model03, w2_cent >= 0, Int)
@variable(model03, y1_cent >= 0, Int)
@variable(model03, y2_cent >= 0, Int)
@variable(model03, y3_cent >= 0, Int)

# Conversão:  centenas -> toneladas
w1_03 = 100 * w1_cent
w2_03 = 100 * w2_cent
y1_03 = 100 * y1_cent
y2_03 = 100 * y2_cent
y3_03 = 100 * y3_cent

# Restrições
@constraint(model03, x1_03 + x2_03 + x3_03 <= area_total)
@constraint(model03, rend_trigo * x1_03 + w1_03 - y1_03 >= trigo_demanda)
@constraint(model03, rend_milho * x2_03 + w2_03 - y2_03 >= milho_demanda)
@constraint(model03, y3_03 <= rend_cana * x3_03)

# Objetivo
@objective(model03, Max,
    preco_venda_trigo * y1_03 + preco_venda_milho * y2_03 + preco_venda_cana * y3_03 -
    preco_compra_trigo * w1_03 - preco_compra_milho * w2_03
)

optimize!(model03)

println("Status: ", termination_status(model03))
println("Lucro máximo: \$", objective_value(model03))
println("Área de trigo: ", value(x1_03), " ha")
println("Área de milho: ", value(x2_03), " ha")
println("Área de cana:  ", value(x3_03), " ha")
println("Trigo comprado: ", value(w1_03), " T (", value(w1_cent), " centenas)")
println("Milho comprado: ", value(w2_03), " T (", value(w2_cent), " centenas)")
println("Trigo vendido: ", value(y1_03), " T (", value(y1_cent), " centenas)")
println("Milho vendido: ", value(y2_03), " T (", value(y2_cent), " centenas)")
println("Cana vendida: ", value(y3_03), " T (", value(y3_cent), " centenas)")

println("\n=== FIM ===")