"""
Problem [01] - Chapter 6:Sampling Methods
Book:III Bienal da Sociedade Brasileira de Matemática
"""

using JuMP
using HiGHS
using Distributions
using Random
using LinearAlgebra
using Printf
using Plots

println("="^80)
println("Problem [01] - Chapter 6:Stochastic Decomposition")
println("="^80)
println()

Random.seed!(1234)

# PROBLEM DATA
println("PROBLEM DATA")
println("-"^80)

c = -0.75
X_lower = 0.0
X_upper = 5.0

println("First stage:")
println("  c = $c")
println("  X = [$X_lower, $X_upper]")
println()

q = [-1.0, 3.0, 1.0, 1.0]
n_y = 4

T = [10.0, 5.0]

W = [-1.0  1.0  -1.0   1.0;
      1.0  1.0   1.0  -1.0]

println("Second stage:")
println("  q = $q")
println("  T = $T")
println("  W = ")
display(W)
println()

println("Uncertain parameters:")
println("  ω₁ ~ Uniform[-1, 0]")
println("  ω₂ = 1 + ω₁")
println("  h(ω) = [-1, ω₂]ᵀ = [-1, 1+ω₁]ᵀ")
println()

dist_omega1 = Uniform(-1.0, 0.0)

function generate_h(ω1)
    ω2 = 1.0 + ω1
    return [-1.0, ω2]
end

function solve_second_stage(x_val, ω1; return_dual=false)
    h = generate_h(ω1)
    
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    @variable(model, y[1:n_y] >= 0)
    @objective(model, Min, sum(q[i] * y[i] for i in 1:n_y))
    
    Tx = T * x_val
    rhs = h - Tx
    
    @constraint(model, recourse[i=1:2], 
                sum(W[i,j] * y[j] for j in 1:n_y) == rhs[i])
    
    optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL
        if return_dual
            return (
                value = objective_value(model),
                y = value.(y),
                dual = dual.(recourse)
            )
        else
            return objective_value(model)
        end
    else
        if return_dual
            println("    Infeasible at x=$x_val, ω1=$ω1")
            println("    rhs = $rhs")
        end
        return nothing
    end
end

# STOCHASTIC DECOMPOSITION ALGORITHM
println("="^80)
println("STOCHASTIC DECOMPOSITION ALGORITHM")
println("="^80)
println()

max_iterations = 3
M = 10

k = 0
x_history = []
cuts = []
theta_history = []
obj_history = []

println("Algorithm parameters:")
println("  Maximum iterations:$max_iterations")
println("  Samples per iteration (M):$M")
println()

println("ITERATION LOG")
println("-"^80)
println(@sprintf("%4s  %10s  %10s  %10s", "Iter", "x", "θ", "Objective"))
println("-"^80)

while k < max_iterations
    global k  # FIX: Explicitly declare as global
    k += 1
    
    # Solve master problem
    master = Model(HiGHS.Optimizer)
    set_silent(master)
    
    @variable(master, X_lower <= x <= X_upper)
    @variable(master, θ)
    
    @objective(master, Min, c * x + θ)
    
    for cut in cuts
        @constraint(master, θ >= cut.intercept + cut.slope * x)
    end
    
    if k == 1
        @constraint(master, θ >= -100)
    end
    
    optimize!(master)
    
    x_k = value(x)
    θ_k = value(θ)
    obj_k = objective_value(master)
    
    push!(x_history, x_k)
    push!(theta_history, θ_k)
    push!(obj_history, obj_k)
    
    println(@sprintf("%4d  %10.6f  %10.6f  %10.6f", k, x_k, θ_k, obj_k))
    
    # Generate samples
    sample_values = Float64[]
    intercepts = Float64[]
    slopes = Float64[]
    
    for j in 1:M
        ω1_sample = rand(dist_omega1)
        
        local result = solve_second_stage(x_k, ω1_sample, return_dual=true)  # FIX:Added 'local'
        
        if result !== nothing
            push!(sample_values, result.value)
            
            π = result.dual
            slope_j = dot(π, T)
            intercept_j = result.value - slope_j * x_k
            
            push!(intercepts, intercept_j)
            push!(slopes, slope_j)
        end
    end
    
    if isempty(intercepts)
        println("  WARNING:No valid samples obtained at x=$x_k")
        println("  Trying to diagnose infeasibility...")
        local test_result = solve_second_stage(x_k, -0.5, return_dual=true)  # FIX:Added 'local'
        continue
    end
    
    avg_intercept = mean(intercepts)
    avg_slope = mean(slopes)
    
    push!(cuts, (intercept=avg_intercept, slope=avg_slope))
    
    avg_Q = mean(sample_values)
    
    # println("  Sample average Q(x_k):$(round(avg_Q, digits=6))")
    # println("  Cut added:θ ≥ $(round(avg_intercept, digits=4)) + $(round(avg_slope, digits=4)) * x")
    println("  Valid samples:$(length(sample_values))/$M")
    println()
end

println("-"^80)
println()

# FINAL RESULTS
println("="^80)
println("FINAL RESULTS (After $max_iterations iterations)")
println("="^80)
println()

if ! isempty(x_history)
    x_final = x_history[end]
    θ_final = theta_history[end]
    obj_final = obj_history[end]
    
    println(@sprintf("Final x: %.6f", x_final))
    println(@sprintf("Final θ:%.6f", θ_final))
    println(@sprintf("Final objective:%.6f", obj_final))
    println()
    
    # VERIFICATION
    println("="^80)
    println("VERIFICATION WITH MONTE CARLO (N=10000 samples)")
    println("="^80)
    println()
    
    N_verify = 10000
    
    Q_samples_final = Float64[]
    for i in 1:N_verify
        ω1 = rand(dist_omega1)
        local Q_val = solve_second_stage(x_final, ω1)  # FIX:Added 'local'
        if Q_val !== nothing
            push!(Q_samples_final, Q_val)
        end
    end
    
    if !isempty(Q_samples_final)
        E_Q_final = mean(Q_samples_final)
        total_cost_final = c * x_final + E_Q_final
        
        println(@sprintf("At x = %.6f:", x_final))
        println(@sprintf("  First-stage cost:c*x = %.6f", c * x_final))
        println(@sprintf("  Expected second-stage cost:E[Q(x)] = %.6f", E_Q_final))
        println(@sprintf("  Total expected cost:%.6f", total_cost_final))
        println(@sprintf("  Feasible samples:%d/%d", length(Q_samples_final), N_verify))
        println()
    else
        println("WARNING:All samples infeasible at final solution!")
        println()
    end
    
    println("Evaluating objective function on grid:")
    println("-"^80)
    x_grid = 0.0:0.5:5.0
    
    for x_test in x_grid
        Q_samples = Float64[]
        for i in 1:1000
            ω1 = rand(dist_omega1)
            local Q_val = solve_second_stage(x_test, ω1)  # FIX:Added 'local'
            if Q_val !== nothing
                push!(Q_samples, Q_val)
            end
        end
        if !isempty(Q_samples)
            E_Q = mean(Q_samples)
            total = c * x_test + E_Q
            println(@sprintf("  x = %.1f:E[Q(x)] = %8.4f, Total = %8.4f (%d/1000 feasible)", 
                             x_test, E_Q, total, length(Q_samples)))
        else
            println(@sprintf("  x = %.1f:All samples infeasible", x_test))
        end
    end
    println()
end

println("="^80)
println("PROBLEM STRUCTURE ANALYSIS")
println("="^80)
println()
println("The recourse problem is:")
println("  min  -y₁ + 3y₂ + y₃ + y₄")
println("  s.t. -y₁ + y₂ - y₃ + y₄ = -1 - 10x")
println("       y₁ + y₂ + y₃ - y₄ = ω₂ - 5x")
println("       y₁, y₂, y₃, y₄ ≥ 0")
println()
println("where ω₂ = 1 + ω₁ ∈ [0, 1]")
println()
println("For feasibility, we need the system Wy = h - Tx to have")
println("a non-negative solution y ≥ 0.")
println("="^80)