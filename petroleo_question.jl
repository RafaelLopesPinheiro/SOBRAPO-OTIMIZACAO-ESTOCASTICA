"""
FIXED VERSION - Better parameter calibration and numerical stability
"""

using JuMP
using GLPK
using Distributions
using Random
using Statistics
using Plots
using Printf
using DataFrames

# ============================================================================
# IMPROVED DATA GENERATION
# ============================================================================

struct PetroleumData
    n_sites::Int
    expansion_cost::Vector{Float64}
    extraction_cost::Vector{Float64}
    current_capacity::Vector{Float64}
    max_capacity::Vector{Float64}
    purchase_price_dist::Distribution
    demand_dist::Distribution
end

"""
Generate problem with better parameter calibration
"""
function generate_problem_data(n_sites::Int=10; seed:: Int=42)
    Random.seed!(seed)
    
    # Key insight: Make expansion + extraction MORE ATTRACTIVE than purchasing
    
    # Expansion costs:  50-100 R$/T capacity
    expansion_cost = 50 .+ 50 * rand(n_sites)
    
    # Extraction costs: 15-30 R$/T (lower than before)
    extraction_cost = 15 .+ 15 * rand(n_sites)
    
    # Current capacities
    current_capacity = 80 .+ 60 * rand(n_sites)
    
    # Maximum capacities (2x to 3x current)
    max_capacity = current_capacity .* (2 .+ rand(n_sites))
    
    # CRITICAL: Purchase price should be HIGH enough to make expansion attractive
    # Mean = 120, but with high uncertainty
    # Expected total cost of expansion+extraction ≈ 50-100 + 15-30 = 65-130
    # So purchase price around 120 makes expansion competitive
    purchase_price_dist = truncated(Normal(120, 40), 60, 300)
    
    # Demand:  higher than current capacity to force decision
    mean_demand = sum(current_capacity) * 1.3
    demand_dist = truncated(Normal(mean_demand, mean_demand * 0.15), 
                           sum(current_capacity) * 0.8, 
                           sum(current_capacity) * 2.0)
    
    return PetroleumData(
        n_sites,
        expansion_cost,
        extraction_cost,
        current_capacity,
        max_capacity,
        purchase_price_dist,
        demand_dist
    )
end

# ============================================================================
# IMPROVED SAA SOLVER WITH BETTER NUMERICS
# ============================================================================

function solve_saa(data::PetroleumData, n_scenarios::Int; 
                   solver=GLPK.Optimizer, verbose:: Bool=false, seed::Union{Int,Nothing}=nothing)
    
    # Use provided seed or hash-based seed
    if seed !== nothing
        Random.seed!(seed)
    else
        Random.seed!(42 + n_scenarios)  # Deterministic but different per n
    end
    
    # Generate scenarios with truncation to avoid extreme values
    purchase_prices = rand(data.purchase_price_dist, n_scenarios)
    demands = rand(data.demand_dist, n_scenarios)
    
    # Create model
    model = Model(solver)
    ! verbose && set_silent(model)
    
    # First-stage variables
    @variable(model, 0 <= x[i=1:data.n_sites] <= 
              data.max_capacity[i] - data.current_capacity[i])
    
    # Second-stage variables
    @variable(model, y[i=1:data.n_sites, s=1:n_scenarios] >= 0)
    @variable(model, z[s=1:n_scenarios] >= 0)
    
    # Objective
    @objective(model, Min,
        sum(data.expansion_cost[i] * x[i] for i in 1:data.n_sites) +
        (1/n_scenarios) * sum(
            sum(data.extraction_cost[i] * y[i,s] for i in 1:data.n_sites) +
            purchase_prices[s] * z[s]
            for s in 1:n_scenarios
        )
    )
    
    # Constraints
    for s in 1:n_scenarios
        for i in 1:data.n_sites
            @constraint(model, y[i,s] <= data.current_capacity[i] + x[i])
        end
        @constraint(model, 
            sum(y[i,s] for i in 1:data.n_sites) + z[s] >= demands[s]
        )
    end
    
    # Solve
    optimize!(model)
    
    # Extract production and purchase decisions
    y_vals = value.(y)
    z_vals = value.(z)
    
    return (
        model = model,
        expansion = value.(x),
        production = y_vals,
        purchase = z_vals,
        objective = objective_value(model),
        solve_time = solve_time(model),
        scenarios = (prices=purchase_prices, demands=demands)
    )
end

# ============================================================================
# IMPROVED CONVERGENCE ANALYSIS
# ============================================================================

function convergence_analysis(data::PetroleumData; 
                             scenario_counts=[10, 25, 50, 100, 250, 500, 1000, 2000])
    
    println("=" ^ 70)
    println("CONVERGENCE ANALYSIS")
    println("=" ^ 70)
    
    results = []
    
    for n in scenario_counts
        @printf("Solving with %4d scenarios...", n)
        flush(stdout)
        
        sol = solve_saa(data, n)
        
        # Calculate average purchase amount
        avg_purchase = mean(sol.purchase)
        
        push!(results, (
            n_scenarios = n,
            objective = sol.objective,
            total_expansion = sum(sol.expansion),
            avg_purchase = avg_purchase,
            expansion_by_site = copy(sol.expansion)
        ))
        
        @printf("Obj = %10.2f, Expansion = %8.2f, Avg Purchase = %8.2f\n", 
               sol.objective, sum(sol.expansion), avg_purchase)
    end
    
    # Enhanced plotting
    objectives = [r.objective for r in results]
    expansions = [r.total_expansion for r in results]
    purchases = [r.avg_purchase for r in results]
    
    p1 = plot(scenario_counts, objectives, 
              marker=:circle, linewidth=2, markersize=6,
              xlabel="Number of Scenarios", 
              ylabel="Optimal Objective Value (R\$)",
              title="Objective Value Convergence",
              legend=false, grid=true, minorgrid=true)
    
    p2 = plot(scenario_counts, expansions,
              marker=:square, linewidth=2, markersize=6, color=:red,
              xlabel="Number of Scenarios",
              ylabel="Total Expansion Volume (T)",
              title="Optimal Expansion Convergence",
              legend=false, grid=true, minorgrid=true)
    
    p3 = plot(scenario_counts, purchases,
              marker=:diamond, linewidth=2, markersize=6, color=:green,
              xlabel="Number of Scenarios",
              ylabel="Average Purchase (T)",
              title="Average Purchase Convergence",
              legend=false, grid=true, minorgrid=true)
    
    p = plot(p1, p2, p3, layout=(1,3), size=(1400, 400))
    savefig(p, "convergence_analysis.png")
    
    # Convergence statistics
    if length(objectives) >= 3
        last_3_obj = objectives[end-2: end]
        obj_std = std(last_3_obj)
        obj_cv = obj_std / mean(last_3_obj) * 100
        
        println("\nConvergence Statistics (last 3 points):")
        @printf("  Objective CV: %.2f%%\n", obj_cv)
        @printf("  Objective range: [%.2f, %.2f]\n", minimum(last_3_obj), maximum(last_3_obj))
        
        if obj_cv < 5.0
            println("  ✓ GOOD CONVERGENCE (CV < 5%)")
        elseif obj_cv < 10.0
            println("  ⚠ MODERATE CONVERGENCE (5% ≤ CV < 10%)")
        else
            println("  ❌ POOR CONVERGENCE (CV ≥ 10%) - Consider more scenarios")
        end
    end
    
    println("\n✓ Convergence analysis complete.Plot saved.")
    return results, p
end

# ============================================================================
# IMPROVED COMPUTATIONAL TIME ANALYSIS
# ============================================================================

function computational_time_analysis(data::PetroleumData;
                                    scenario_counts=[10, 25, 50, 100, 250, 500, 1000])
    
    println("\n" * "=" ^ 70)
    println("COMPUTATIONAL TIME ANALYSIS")
    println("=" ^ 70)
    
    times = []
    
    for n in scenario_counts
        @printf("Timing with %4d scenarios...", n)
        flush(stdout)
        
        # Run multiple times for small instances to get measurable time
        n_runs = n < 100 ? 10 : 1
        
        time_start = time()
        for _ in 1:n_runs
            sol = solve_saa(data, n)
        end
        elapsed = (time() - time_start) / n_runs
        
        push!(times, elapsed)
        
        @printf("Time = %.6f seconds\n", elapsed)
    end
    
    # Filter out zero times and fit
    valid_indices = findall(times .> 1e-6)
    
    if length(valid_indices) >= 3
        valid_n = scenario_counts[valid_indices]
        valid_t = times[valid_indices]
        
        log_n = log.(valid_n)
        log_t = log.(valid_t)
        
        n_points = length(log_n)
        b = (n_points * sum(log_n .* log_t) - sum(log_n) * sum(log_t)) /
            (n_points * sum(log_n.^2) - sum(log_n)^2)
        a = exp((sum(log_t) - b * sum(log_n)) / n_points)
        
        # Plot
        p = plot(scenario_counts, times,
                 marker=:circle, linewidth=2, markersize=6, label="Actual",
                 xlabel="Number of Scenarios",
                 ylabel="Computational Time (seconds)",
                 title="Computational Complexity Analysis",
                 yscale=:log10, xscale=:log10, grid=true, minorgrid=true)
        
        fitted_times = a .* scenario_counts.^b
        plot!(p, scenario_counts, fitted_times,
              linewidth=2, linestyle=:dash, 
              label=@sprintf("Fitted:  %.2e × n^%.2f", a, b))
        
        savefig(p, "computational_time.png")
        
        println("\n✓ Time complexity: O(n^$(round(b, digits=2)))")
        @printf("✓ Fitted model: T(n) = %.2e × n^%.2f seconds\n", a, b)
    else
        println("\n⚠ Warning: Too few measurable times for fitting")
        
        p = plot(scenario_counts, times,
                 marker=:circle, linewidth=2, markersize=6,
                 xlabel="Number of Scenarios",
                 ylabel="Computational Time (seconds)",
                 title="Computational Time",
                 grid=true)
        savefig(p, "computational_time.png")
    end
    
    println("✓ Computational time analysis complete.Plot saved.")
    
    return times, p
end

# ============================================================================
# FIXED BENDERS DECOMPOSITION
# ============================================================================

function solve_benders_singlecut(data:: PetroleumData, n_scenarios::Int;
                                 max_iter::Int=100, tol::Float64=1e-4,
                                 verbose::Bool=false, seed::Int=42)
    
    Random.seed!(seed)
    purchase_prices = rand(data.purchase_price_dist, n_scenarios)
    demands = rand(data.demand_dist, n_scenarios)
    
    master = Model(GLPK.Optimizer)
    set_silent(master)
    
    @variable(master, 0 <= x[i=1:data.n_sites] <= 
              data.max_capacity[i] - data.current_capacity[i])
    @variable(master, θ)  # Remove lower bound initially
    
    @objective(master, Min,
        sum(data.expansion_cost[i] * x[i] for i in 1:data.n_sites) + θ
    )
    
    # CRITICAL FIX: Initialize with a trivial lower bound on θ
    # This prevents unbounded solution in first iteration
    @constraint(master, θ >= 0)  # θ represents expected recourse cost, must be non-negative
    
    lb = -Inf
    ub = Inf
    iter = 0
    convergence_history = Float64[]
    lb_history = Float64[]
    ub_history = Float64[]
    
    while iter < max_iter
        iter += 1
        
        # Solve master problem
        optimize!(master)
        
        if termination_status(master) != MOI.OPTIMAL
            @warn "Master problem not optimal at iteration $iter"
            break
        end
        
        x_val = value.(x)
        θ_val = value(θ)
        lb = objective_value(master)
        
        # Solve all subproblems
        avg_recourse = 0.0
        cut_coef = zeros(data.n_sites)
        cut_rhs = 0.0
        
        for s in 1:n_scenarios
            sub = Model(GLPK.Optimizer)
            set_silent(sub)
            
            @variable(sub, y[i=1:data.n_sites] >= 0)
            @variable(sub, z >= 0)
            
            @objective(sub, Min,
                sum(data.extraction_cost[i] * y[i] for i in 1:data.n_sites) +
                purchase_prices[s] * z
            )
            
            # Capacity constraints
            @constraint(sub, cap[i=1:data.n_sites],
                y[i] <= data.current_capacity[i] + x_val[i]
            )
            
            # Demand constraint
            @constraint(sub, dem,
                sum(y[i] for i in 1:data.n_sites) + z >= demands[s]
            )
            
            optimize!(sub)
            
            if termination_status(sub) != MOI.OPTIMAL
                @warn "Subproblem $s not optimal at iteration $iter"
                continue
            end
            
            sub_obj = objective_value(sub)
            avg_recourse += sub_obj / n_scenarios
            
            # Get dual values (shadow prices)
            duals_cap = dual.(cap)
            
            # Build Benders cut:  θ >= E[Q(x,ξ)]
            # Cut:  θ >= (Q_s - π_s^T x_val) + π_s^T x
            # Accumulate for single cut
            cut_coef .+= duals_cap ./ n_scenarios
            cut_rhs += (sub_obj - sum(duals_cap[i] * x_val[i] for i in 1:data.n_sites)) / n_scenarios
        end
        
        # Calculate upper bound (best feasible solution found)
        first_stage_cost = sum(data.expansion_cost[i] * x_val[i] for i in 1:data.n_sites)
        current_ub = first_stage_cost + avg_recourse
        ub = min(ub, current_ub)
        
        # Store history
        gap = ub - lb
        push!(convergence_history, gap)
        push!(lb_history, lb)
        push!(ub_history, ub)
        
        # Check convergence
        relative_gap = abs(gap) / (abs(ub) + 1e-10)
        
        if verbose && (iter <= 5 || iter % 5 == 0)
            @printf("Iter %3d: LB = %10.2f, UB = %10.2f, Gap = %8.2f (%.3f%%)\n", 
                   iter, lb, ub, gap, 100*relative_gap)
        end
        
        if relative_gap < tol
            if verbose
                println("✓ Converged!")
            end
            break
        end
        
        # Add Benders cut (single cut - average over all scenarios)
        @constraint(master, 
            θ >= cut_rhs + sum(cut_coef[i] * x[i] for i in 1:data.n_sites)
        )
    end
    
    # Final solve
    optimize!(master)
    
    return (
        expansion = value.(x),
        theta = value(θ),
        objective = objective_value(master),
        iterations = iter,
        convergence = convergence_history,
        lb_history = lb_history,
        ub_history = ub_history,
        gap = length(convergence_history) > 0 ? convergence_history[end] :  Inf
    )
end

function solve_benders_multicut(data::PetroleumData, n_scenarios::Int;
                                max_iter::Int=100, tol::Float64=1e-4,
                                verbose:: Bool=false, seed::Int=42)
    
    Random.seed!(seed)
    purchase_prices = rand(data.purchase_price_dist, n_scenarios)
    demands = rand(data.demand_dist, n_scenarios)
    
    master = Model(GLPK.Optimizer)
    set_silent(master)
    
    @variable(master, 0 <= x[i=1:data.n_sites] <= 
              data.max_capacity[i] - data.current_capacity[i])
    @variable(master, θ[s=1:n_scenarios])  # One θ per scenario
    
    @objective(master, Min,
        sum(data.expansion_cost[i] * x[i] for i in 1:data.n_sites) + 
        (1/n_scenarios) * sum(θ[s] for s in 1:n_scenarios)
    )
    
    # CRITICAL FIX: Initialize bounds for each θ_s
    for s in 1:n_scenarios
        @constraint(master, θ[s] >= 0)
    end
    
    lb = -Inf
    ub = Inf
    iter = 0
    convergence_history = Float64[]
    lb_history = Float64[]
    ub_history = Float64[]
    
    while iter < max_iter
        iter += 1
        
        # Solve master
        optimize!(master)
        
        if termination_status(master) != MOI.OPTIMAL
            @warn "Master problem not optimal at iteration $iter"
            break
        end
        
        x_val = value.(x)
        θ_vals = value.(θ)
        lb = objective_value(master)
        
        # Solve subproblems and add one cut per scenario
        avg_recourse = 0.0
        
        for s in 1:n_scenarios
            sub = Model(GLPK.Optimizer)
            set_silent(sub)
            
            @variable(sub, y[i=1:data.n_sites] >= 0)
            @variable(sub, z >= 0)
            
            @objective(sub, Min,
                sum(data.extraction_cost[i] * y[i] for i in 1:data.n_sites) +
                purchase_prices[s] * z
            )
            
            @constraint(sub, cap[i=1:data.n_sites],
                y[i] <= data.current_capacity[i] + x_val[i]
            )
            
            @constraint(sub, dem,
                sum(y[i] for i in 1:data.n_sites) + z >= demands[s]
            )
            
            optimize!(sub)
            
            if termination_status(sub) != MOI.OPTIMAL
                @warn "Subproblem $s not optimal at iteration $iter"
                continue
            end
            
            sub_obj = objective_value(sub)
            avg_recourse += sub_obj / n_scenarios
            
            # Get duals
            duals_cap = dual.(cap)
            
            # Build cut for this specific scenario
            cut_rhs = sub_obj - sum(duals_cap[i] * x_val[i] for i in 1:data.n_sites)
            
            # Add cut:  θ_s >= (Q_s - π_s^T x_val) + π_s^T x
            @constraint(master, 
                θ[s] >= cut_rhs + sum(duals_cap[i] * x[i] for i in 1:data.n_sites)
            )
        end
        
        # Calculate upper bound
        first_stage_cost = sum(data.expansion_cost[i] * x_val[i] for i in 1:data.n_sites)
        current_ub = first_stage_cost + avg_recourse
        ub = min(ub, current_ub)
        
        # Store history
        gap = ub - lb
        push!(convergence_history, gap)
        push!(lb_history, lb)
        push!(ub_history, ub)
        
        # Check convergence
        relative_gap = abs(gap) / (abs(ub) + 1e-10)
        
        if verbose && (iter <= 5 || iter % 5 == 0)
            @printf("Iter %3d: LB = %10.2f, UB = %10.2f, Gap = %8.2f (%.3f%%)\n", 
                   iter, lb, ub, gap, 100*relative_gap)
        end
        
        if relative_gap < tol
            if verbose
                println("✓ Converged!")
            end
            break
        end
    end
    
    # Final solve
    optimize!(master)
    
    return (
        expansion = value.(x),
        theta = value.(θ),
        objective = objective_value(master),
        iterations = iter,
        convergence = convergence_history,
        lb_history = lb_history,
        ub_history = ub_history,
        gap = length(convergence_history) > 0 ? convergence_history[end] : Inf
    )
end


function benders_comparison(data::PetroleumData, n_scenarios::Int=100)
    
    println("\n" * "=" ^ 70)
    println("BENDERS DECOMPOSITION COMPARISON")
    println("=" ^ 70)
    
    # Fixed: Use string concatenation, not multiplication
    println("\n" * "-" ^ 35 * " Single-Cut Benders " * "-" ^ 16)
    sol_single = solve_benders_singlecut(data, n_scenarios, verbose=true, tol=1e-3)
    @printf("\nFinal Results:\n")
    @printf("  Objective:  %.2f R\$\n", sol_single.objective)
    @printf("  Iterations: %d\n", sol_single.iterations)
    @printf("  Final Gap: %.4f R\$ (%.3f%%)\n", sol_single.gap, 
            100*sol_single.gap/(abs(sol_single.objective)+1e-10))
    @printf("  Total Expansion: %.2f T\n", sum(sol_single.expansion))
    
    println("\n" * "-" ^ 35 * " Multi-Cut Benders " * "-" ^ 17)
    sol_multi = solve_benders_multicut(data, n_scenarios, verbose=true, tol=1e-3)
    @printf("\nFinal Results:\n")
    @printf("  Objective: %.2f R\$\n", sol_multi.objective)
    @printf("  Iterations: %d\n", sol_multi.iterations)
    @printf("  Final Gap: %.4f R\$ (%.3f%%)\n", sol_multi.gap,
            100*sol_multi.gap/(abs(sol_multi.objective)+1e-10))
    @printf("  Total Expansion: %.2f T\n", sum(sol_multi.expansion))
    
    # Validate against SAA
    println("\n" * "-" ^ 70)
    println("Validation against Sample Average Approximation:")
    saa_sol = solve_saa(data, n_scenarios, seed=42)
    @printf("  SAA Objective: %.2f R\$\n", saa_sol.objective)
    @printf("  SAA Total Expansion: %.2f T\n", sum(saa_sol.expansion))
    
    obj_diff_single = abs(sol_single.objective - saa_sol.objective)
    obj_diff_multi = abs(sol_multi.objective - saa_sol.objective)
    
    @printf("\n  Single-cut vs SAA difference: %.2f R\$ (%.3f%%)\n", 
            obj_diff_single, 100*obj_diff_single/abs(saa_sol.objective))
    @printf("  Multi-cut vs SAA difference: %.2f R\$ (%.3f%%)\n",
            obj_diff_multi, 100*obj_diff_multi/abs(saa_sol.objective))
    
    if obj_diff_single < 100 && obj_diff_multi < 100
        println("  ✓ Both Benders methods agree with SAA")
    else
        println("  ⚠ Warning: Large difference between Benders and SAA")
    end
    
    # Enhanced plotting with error handling
    if length(sol_single.convergence) > 0 && length(sol_multi.convergence) > 0
        # Plot 1: Convergence (gap vs iteration)
        p1 = plot(1:length(sol_single.convergence), 
                  max.(sol_single.convergence, 1e-6),  # Avoid log(0)
                  marker=:circle, linewidth=2, markersize=5, label="Single-Cut",
                  xlabel="Iteration", ylabel="Optimality Gap (R\$)",
                  title="Convergence:  Gap vs Iteration",
                  yscale=:log10, grid=true, minorgrid=true, legend=:topright)
        
        plot!(p1, 1:length(sol_multi.convergence), 
              max.(sol_multi.convergence, 1e-6),  # Avoid log(0)
              marker=:square, linewidth=2, markersize=5, label="Multi-Cut")
        
        # Plot 2: Bounds evolution
        p2 = plot(xlabel="Iteration", ylabel="Objective Bounds (R\$)",
                  title="Lower and Upper Bounds Evolution",
                  grid=true, minorgrid=true, legend=:right)
        
        if length(sol_single.lb_history) > 0
            plot!(p2, 1:length(sol_single.lb_history), sol_single.lb_history,
                  linewidth=2, label="Single-Cut LB", color=:blue)
            plot!(p2, 1:length(sol_single.ub_history), sol_single.ub_history,
                  linewidth=2, linestyle=:dash, label="Single-Cut UB", color=:blue)
        end
        
        if length(sol_multi.lb_history) > 0
            plot!(p2, 1:length(sol_multi.lb_history), sol_multi.lb_history,
                  linewidth=2, label="Multi-Cut LB", color=:red)
            plot!(p2, 1:length(sol_multi.ub_history), sol_multi.ub_history,
                  linewidth=2, linestyle=:dash, label="Multi-Cut UB", color=:red)
        end
        
        # Add SAA reference line
        hline!(p2, [saa_sol.objective], linewidth=2, linestyle=:dot, 
               label="SAA Solution", color=:green)
        
        p = plot(p1, p2, layout=(1,2), size=(1200, 400))
        savefig(p, "benders_comparison.png")
        
        println("\n✓ Benders comparison complete.Plot saved.")
        
        if sol_multi.iterations > 0 && sol_single.iterations > 0
            speedup = sol_single.iterations / sol_multi.iterations
            @printf("✓ Multi-cut is %.2fx faster in iterations\n", speedup)
        end
    else
        println("\n⚠ Warning: No convergence data to plot")
    end
    
    return sol_single, sol_multi, nothing
end


# ============================================================================
# BOOTSTRAP SAMPLING ANALYSIS
# ============================================================================

"""
Bootstrap analysis (item e) - Validation of SAA solution stability
"""
function bootstrap_analysis(data:: PetroleumData; 
                           n_scenarios::Int=200, n_bootstrap::Int=200)
    
    println("\n" * "=" ^ 70)
    println("BOOTSTRAP SAMPLING ANALYSIS")
    println("=" ^ 70)
    
    @printf("Running %d bootstrap samples with %d scenarios each...\n", 
           n_bootstrap, n_scenarios)
    
    bootstrap_expansions = []
    bootstrap_objectives = []
    
    # Progress tracking
    progress_marks = [25, 50, 75, 100, 125, 150, 175, 200]
    
    for b in 1:n_bootstrap
        if b in progress_marks
            @printf("Bootstrap iteration %d/%d\n", b, n_bootstrap)
            flush(stdout)
        end
        
        # Each bootstrap gets different random scenarios
        sol = solve_saa(data, n_scenarios, seed=1000+b)
        push!(bootstrap_expansions, sol.expansion)
        push!(bootstrap_objectives, sol.objective)
    end
    
    # Calculate statistics
    expansion_matrix = hcat(bootstrap_expansions...)
    mean_expansion = vec(mean(expansion_matrix, dims=2))
    std_expansion = vec(std(expansion_matrix, dims=2))
    median_expansion = vec(median(expansion_matrix, dims=2))
    
    # Reference solution (from convergence analysis)
    reference_sol = solve_saa(data, n_scenarios, seed=42)
    
    # Calculate confidence intervals
    obj_mean = mean(bootstrap_objectives)
    obj_std = std(bootstrap_objectives)
    obj_ci_lower = quantile(bootstrap_objectives, 0.025)
    obj_ci_upper = quantile(bootstrap_objectives, 0.975)
    
    # Create visualizations
    sites = 1:data.n_sites
    
    # Plot 1: Expansion by site with confidence intervals
    p1 = plot(sites, mean_expansion, 
              ribbon=1.96*std_expansion, fillalpha=0.3,
              marker=:circle, linewidth=2, markersize=6, 
              label="Bootstrap Mean ± 95% CI",
              xlabel="Site Index", ylabel="Optimal Expansion (T)",
              title="Expansion by Site:  Bootstrap Analysis",
              grid=true, minorgrid=true, legend=:topright)
    
    plot!(p1, sites, reference_sol.expansion,
          marker=:square, linewidth=2, markersize=6, linestyle=:dash,
          label="Reference Solution (N=$n_scenarios)", color=:red)
    
    # Plot 2: Histogram of objective values
    p2 = histogram(bootstrap_objectives, bins=30, normalize=:probability,
                   xlabel="Objective Value (R\$)", ylabel="Probability",
                   title="Distribution of Objective Values",
                   label="Bootstrap Distribution", alpha=0.7, 
                   grid=true, color=:skyblue)
    
    vline!(p2, [obj_mean], linewidth=3, 
           label="Mean = $(round(obj_mean, digits=2))", 
           linestyle=:dash, color=:blue)
    
    vline!(p2, [reference_sol.objective], linewidth=3,
           label="Reference = $(round(reference_sol.objective, digits=2))",
           linestyle=:dot, color=:red)
    
    # Plot 3: Total expansion distribution
    total_expansions = [sum(bootstrap_expansions[i]) for i in 1:n_bootstrap]
    
    p3 = histogram(total_expansions, bins=30, normalize=:probability,
                   xlabel="Total Expansion (T)", ylabel="Probability",
                   title="Distribution of Total Expansion",
                   label="Bootstrap Distribution", alpha=0.7,
                   grid=true, color=:lightgreen)
    
    vline!(p3, [mean(total_expansions)], linewidth=3,
           label="Mean = $(round(mean(total_expansions), digits=2))",
           linestyle=:dash, color=:green)
    
    vline!(p3, [sum(reference_sol.expansion)], linewidth=3,
           label="Reference = $(round(sum(reference_sol.expansion), digits=2))",
           linestyle=:dot, color=:red)
    
    # Combined plot
    p = plot(p1, p2, p3, layout=(1,3), size=(1600, 400))
    savefig(p, "bootstrap_analysis.png")
    
    # Print detailed statistics
    println("\n" * "-" ^ 70)
    println("Bootstrap Statistics Summary")
    println("-" ^ 70)
    
    println("\nObjective Value:")
    @printf("  Mean:              %.2f R\$\n", obj_mean)
    @printf("  Std Dev:          %.2f R\$\n", obj_std)
    @printf("  95%% CI:           [%.2f, %.2f] R\$\n", obj_ci_lower, obj_ci_upper)
    @printf("  Coefficient of Variation: %.2f%%\n", 100*obj_std/obj_mean)
    
    println("\nTotal Expansion:")
    @printf("  Mean:             %.2f T\n", mean(total_expansions))
    @printf("  Std Dev:          %.2f T\n", std(total_expansions))
    @printf("  95%% CI:           [%.2f, %.2f] T\n", 
           quantile(total_expansions, 0.025), quantile(total_expansions, 0.975))
    
    println("\nReference Solution (N=$n_scenarios):")
    @printf("  Objective:         %.2f R\$\n", reference_sol.objective)
    @printf("  Total Expansion:  %.2f T\n", sum(reference_sol.expansion))
    
    # Statistical tests
    println("\n" * "-" ^ 70)
    println("Statistical Validation")
    println("-" ^ 70)
    
    # Check if reference is within confidence interval
    ref_in_ci = obj_ci_lower <= reference_sol.objective <= obj_ci_upper
    
    if ref_in_ci
        println("  ✓ Reference solution is within 95% confidence interval")
        println("    This indicates good solution stability")
    else
        println("  ⚠ Reference solution is outside 95% confidence interval")
        println("    This may indicate high variance in scenarios")
    end
    
    # Calculate z-score
    z_score = abs(reference_sol.objective - obj_mean) / (obj_std + 1e-10)
    @printf("\n  Z-score: %.2f\n", z_score)
    
    if z_score < 1.96
        println("  ✓ Reference within 95% confidence (|z| < 1.96)")
    elseif z_score < 2.58
        println("  ⚠ Reference within 99% confidence (|z| < 2.58)")
    else
        println("  ❌ Reference is a statistical outlier (|z| ≥ 2.58)")
    end
    
    # Coefficient of variation check
    cv = 100 * obj_std / obj_mean
    
    println("\n  Solution Stability Assessment:")
    if cv < 1.0
        println("  ✓ EXCELLENT stability (CV < 1%)")
    elseif cv < 3.0
        println("  ✓ GOOD stability (1% ≤ CV < 3%)")
    elseif cv < 5.0
        println("  ⚠ MODERATE stability (3% ≤ CV < 5%)")
    else
        println("  ❌ POOR stability (CV ≥ 5%) - Consider more scenarios")
    end
    
    # Comparison with convergence analysis
    println("\n" * "-" ^ 70)
    println("Comparison:  Bootstrap vs Convergence Analysis")
    println("-" ^ 70)
    
    # Run a single convergence test at same scenario count
    conv_sol = solve_saa(data, n_scenarios, seed=42)
    
    @printf("Bootstrap mean objective:       %.2f R\$\n", obj_mean)
    @printf("Convergence analysis solution: %.2f R\$\n", conv_sol.objective)
    @printf("Absolute difference:           %.2f R\$ (%.3f%%)\n",
           abs(obj_mean - conv_sol.objective),
           100*abs(obj_mean - conv_sol.objective)/conv_sol.objective)
    
    if abs(obj_mean - conv_sol.objective) / conv_sol.objective < 0.01
        println("\n✓ Excellent agreement between bootstrap and convergence analysis (<1% difference)")
    elseif abs(obj_mean - conv_sol.objective) / conv_sol.objective < 0.05
        println("\n✓ Good agreement between bootstrap and convergence analysis (<5% difference)")
    else
        println("\n⚠ Significant difference between bootstrap and convergence analysis (≥5%)")
    end
    
    println("\n" * "=" ^ 70)
    println("✓ Bootstrap analysis complete. Plot saved.")
    println("=" ^ 70)
    
    return (
        mean_expansion = mean_expansion,
        std_expansion = std_expansion,
        objectives = bootstrap_objectives,
        obj_mean = obj_mean,
        obj_std = obj_std,
        obj_ci = (obj_ci_lower, obj_ci_upper),
        reference = reference_sol,
        plot = p
    )
end




# ============================================================================
# MAIN EXECUTION
# ============================================================================

function main()
    println("\n" * "=" ^ 70)
    println("PETROLEUM OPTIMIZATION - STOCHASTIC PROGRAMMING SOLUTION")
    println("Senior Data Scientist Implementation - FIXED VERSION")
    println("=" ^ 70)
    
    data = generate_problem_data(10)
    
    println("\nProblem Instance:")
    @printf("Number of sites: %d\n", data.n_sites)
    @printf("Total current capacity: %.2f T\n", sum(data.current_capacity))
    @printf("Total maximum capacity: %.2f T\n", sum(data.max_capacity))
    @printf("Mean demand: %.2f T\n", mean(data.demand_dist))
    @printf("Mean purchase price: %.2f R\$/T\n", mean(data.purchase_price_dist))
    @printf("\nExpansion cost range: [%.2f, %.2f] R\$/T\n", 
           minimum(data.expansion_cost), maximum(data.expansion_cost))
    @printf("Extraction cost range: [%.2f, %.2f] R\$/T\n",
           minimum(data.extraction_cost), maximum(data.extraction_cost))
    @printf("Expected marginal cost (exp+extr): [%.2f, %.2f] R\$/T\n",
           minimum(data.expansion_cost .+ data.extraction_cost),
           maximum(data.expansion_cost .+ data.extraction_cost))
    
    println("\n💡 Economic Logic:")
    println("   If purchase price > (expansion + extraction), expand capacity")
    println("   If purchase price < (expansion + extraction), buy from market")
    println("   Optimal solution balances these costs under uncertainty")
    
    # Run all analyses
    convergence_analysis(data)
    computational_time_analysis(data)
    benders_comparison(data, 100)
    bootstrap_analysis(data, n_scenarios=200, n_bootstrap=200)
    
    println("\n" * "=" ^ 70)
    println("ALL ANALYSES COMPLETE!")
    println("=" ^ 70)
end

main()