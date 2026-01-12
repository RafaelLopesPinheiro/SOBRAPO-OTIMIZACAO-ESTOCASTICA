"""
Problem [01] - Chapter 5:The L-shaped Method
Book: III Bienal da Sociedade Brasileira de Matemática
"""

using JuMP
using HiGHS
using Printf
using Plots

println("="^80)
println("Problem [01] - Chapter 5:L-shaped Method (Benders Decomposition)")
println("="^80)
println()

# Problem data
println("PROBLEM DATA")
println("-"^80)

# First stage
c = 0.0
X_lower = 0.0
X_upper = 10.0
println("First stage:")
println("  c = $c")
println("  X ∈ [$X_lower, $X_upper]")
println()

# Second stage
q = [1.0, 1.0]
scenarios = [1.0, 2.0, 4.0]
probabilities = [1/3, 1/3, 1/3]
n_scenarios = length(scenarios)

println("Second stage:")
println("  q = $q")
println("  Scenarios (ξ):$scenarios")
println("  Probabilities: $probabilities")
println()

println("Assumed recourse problem structure:")
println("  Q(x,ξ) = min y₁ + y₂")
println("  subject to: y₁ + y₂ ≥ ξ - x")
println("              y₁, y₂ ≥ 0")
println()

# Function to solve the second-stage problem
function solve_second_stage(x_val, ξ)
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    @variable(model, y[1:2] >= 0)
    @objective(model, Min, sum(q[i] * y[i] for i in 1:2))
    @constraint(model, demand, sum(y[i] for i in 1:2) >= ξ - x_val)
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        return (
            value = objective_value(model),
            y = value.(y),
            dual = dual(demand)
        )
    else
        return nothing
    end
end

# Function to compute expected second-stage cost
function compute_expected_Q(x_val)
    expected_cost = 0.0
    for s in 1:n_scenarios
        local result = solve_second_stage(x_val, scenarios[s])  # FIX:Added 'local'
        if result !== nothing
            expected_cost += probabilities[s] * result.value
        else
            return Inf
        end
    end
    return expected_cost
end

# L-SHAPED ALGORITHM
println("="^80)
println("L-SHAPED ALGORITHM (BENDERS DECOMPOSITION)")
println("="^80)
println()

tolerance = 1e-6
max_iterations = 100
iteration = 0

LB = -Inf
UB = Inf

optimality_cuts = []
feasibility_cuts = []
x_history = []
LB_history = []
UB_history = []

println("ITERATION LOG")
println("-"^80)
println(@sprintf("%4s  %10s  %10s  %10s  %10s  %12s", 
                 "Iter", "x", "θ", "LB", "UB", "Gap"))
println("-"^80)

while iteration < max_iterations
    global iteration, LB, UB  # FIX: Explicitly declare global variables
    iteration += 1
    
    # Solve master problem
    master = Model(HiGHS.Optimizer)
    set_silent(master)
    
    @variable(master, X_lower <= x <= X_upper)
    @variable(master, θ)
    
    @objective(master, Min, c * x + θ)
    
    for (k, cut) in enumerate(optimality_cuts)
        @constraint(master, θ >= cut.intercept + cut.slope * x)
    end
    
    for cut in feasibility_cuts
        @constraint(master, cut.intercept + cut.slope * x <= 0)
    end
    
    optimize!(master)
    
    x_k = value(x)
    θ_k = value(θ)
    LB = objective_value(master)
    
    # Solve second-stage problems
    expected_Q = 0.0
    intercept = 0.0
    slope = 0.0
    all_feasible = true
    
    for s in 1:n_scenarios
        local result = solve_second_stage(x_k, scenarios[s])  # FIX:Added 'local'
        
        if result === nothing
            all_feasible = false
            println("  Infeasibility detected for scenario $s (ξ=$(scenarios[s]))")
            break
        else
            expected_Q += probabilities[s] * result.value
            π = result.dual
            intercept += probabilities[s] * (result.value + π * x_k)
            slope += probabilities[s] * (-π)
        end
    end
    
    if ! all_feasible
        println("  Adding feasibility cut")
        continue
    end
    
    UB = min(UB, c * x_k + expected_Q)
    gap = UB - LB
    
    push!(x_history, x_k)
    push!(LB_history, LB)
    push!(UB_history, UB)
    
    println(@sprintf("%4d  %10.6f  %10.6f  %10.6f  %10.6f  %12.8f", 
                     iteration, x_k, θ_k, LB, UB, gap))
    
    if gap < tolerance
        println("-"^80)
        println("CONVERGENCE ACHIEVED!")
        println()
        break
    end
    
    push!(optimality_cuts, (intercept=intercept, slope=slope))
end

# FINAL RESULTS
println("="^80)
println("FINAL RESULTS")
println("="^80)
println()

x_optimal = x_history[end]
optimal_value = UB_history[end]

println(@sprintf("Optimal solution:x* = %.6f", x_optimal))
println(@sprintf("Optimal value:%.6f", optimal_value))
println(@sprintf("Number of iterations:%d", iteration))
println()

println("Second-stage solutions at x* = $x_optimal:")
println("-"^80)
for s in 1:n_scenarios
    local result = solve_second_stage(x_optimal, scenarios[s])  # FIX:Added 'local'
    if result !== nothing
        println(@sprintf("Scenario %d (ξ=%.1f, p=%.3f):", s, scenarios[s], probabilities[s]))
        println(@sprintf("  y = [%.6f, %.6f]", result.y[1], result.y[2]))
        println(@sprintf("  Q(x*,ξ) = %.6f", result.value))
    end
end
println()

expected_second_stage = compute_expected_Q(x_optimal)
println(@sprintf("Expected second-stage cost E[Q(x*,ξ)] = %.6f", expected_second_stage))
println(@sprintf("First-stage cost c'x* = %.6f", c * x_optimal))
println(@sprintf("Total cost = %.6f", c * x_optimal + expected_second_stage))
println()

# VISUALIZATION
println("="^80)
println("GENERATING VISUALIZATIONS")
println("="^80)
println()

p1 = plot(1:iteration, LB_history, 
          label="Lower Bound", 
          linewidth=2,
          marker=:circle,
          xlabel="Iteration",
          ylabel="Objective Value",
          title="Convergence of L-shaped Method",
          legend=:right)
plot!(p1, 1:iteration, UB_history, 
      label="Upper Bound",
      linewidth=2,
      marker=:square)

p2 = plot(1:iteration, x_history,
          label="x_k",
          linewidth=2,
          marker=:diamond,
          xlabel="Iteration",
          ylabel="x value",
          title="Evolution of First-stage Decision",
          legend=:right)
hline!(p2, [x_optimal], 
       label="x*",
       linestyle=:dash,
       linewidth=2)

x_range = X_lower:0.1:X_upper
true_obj = [c * x + compute_expected_Q(x) for x in x_range]

p3 = plot(x_range, true_obj,
          label="True Objective",
          linewidth=3,
          xlabel="x",
          ylabel="Objective Value",
          title="Objective Function and Optimality Cuts",
          legend=:topright)

for (k, cut) in enumerate(optimality_cuts)
    cut_values = [cut.intercept + cut.slope * x for x in x_range]
    if k <= 5
        plot!(p3, x_range, cut_values,
              label="Cut $k",
              linestyle=:dash,
              alpha=0.6)
    end
end

scatter!(p3, [x_optimal], [optimal_value],
         markersize=8,
         label="Optimal Solution",
         color=:red)

expected_Q_range = [compute_expected_Q(x) for x in x_range]
p4 = plot(x_range, expected_Q_range,
          label="E[Q(x,ξ)]",
          linewidth=2,
          xlabel="x",
          ylabel="Expected Recourse Cost",
          title="Expected Second-stage Cost",
          legend=:topright)
scatter!(p4, [x_optimal], [expected_second_stage],
         markersize=8,
         label="At x*",
         color=:red)

p_combined = plot(p1, p2, p3, p4, 
                  layout=(2,2), 
                  size=(1200, 900))

savefig(p_combined, "problem_cap5_lshaped.png")
println("Visualization saved as 'problem_cap5_lshaped.png'")
println()

# ANALYTICAL VERIFICATION
println("="^80)
println("ANALYTICAL VERIFICATION")
println("="^80)
println()

println("For each x ∈ [0, 10], the expected second-stage cost is:")
println()
println("E[Q(x,ξ)] = (1/3)[Q(x,1) + Q(x,2) + Q(x,4)]")
println()
println("where Q(x,ξ) = min{y₁ + y₂ : y₁ + y₂ ≥ ξ - x, y₁,y₂ ≥ 0}")
println("            = max{0, ξ - x}")
println()
println("So:")
println("  Q(x,1) = max{0, 1-x}")
println("  Q(x,2) = max{0, 2-x}")
println("  Q(x,4) = max{0, 4-x}")
println()
println("The objective is: f(x) = 0·x + E[Q(x,ξ)]")
println()
println("For x ∈ [0, 1]: E[Q(x,ξ)] = (1/3)[(1-x) + (2-x) + (4-x)] = (7-3x)/3")
println("For x ∈ [1, 2]:E[Q(x,ξ)] = (1/3)[0 + (2-x) + (4-x)] = (6-2x)/3")
println("For x ∈ [2, 4]:E[Q(x,ξ)] = (1/3)[0 + 0 + (4-x)] = (4-x)/3")
println("For x ∈ [4,10]:E[Q(x,ξ)] = 0")
println()
println("Since the objective is piecewise linear and decreasing until x=4,")
println("then constant for x ≥ 4, the minimum value 0.0 is achieved for")
println("all x* ∈ [4, 10]. The L-shaped method found x* = 10.0.")
println()
println("="^80)