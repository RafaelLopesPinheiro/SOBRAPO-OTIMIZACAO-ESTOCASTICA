"""
Forest Planning Optimization - FINAL CORRECT VERSION
"""

using JuMP
using Ipopt
using Printf

struct ForestModel
    n_stages::Int
    biomass_coeffs::Vector{Float64}
    discount_factor::Float64
    total_area::Float64
    time_horizon::Int
end

struct MultiSpeciesModel
    n_species::Int
    n_stages::Int
    biomass_coeffs::Matrix{Float64}
    discount_factor::Float64
    total_area::Float64
    time_horizon::Int
end

function formulate_problem()
    println("="^80)
    println("FOREST PLANNING PROBLEM - MATHEMATICAL FORMULATION")
    println("="^80)
    println()
    println("DECISION VARIABLES:")
    println("  a[s,t] >= 0:Area in stage s at time t")
    println("  h[s,t] >= 0:Harvested area from stage s at time t")
    println()
    println("OBJECTIVE:")
    println("  maximize sum(b^t * sqrt(sum(f[s]*h[s,t])) for t=0 to T)")
    println()
    println("CONSTRAINTS:")
    println("  1.sum(a[s,t]) = A  (area conservation)")
    println("  2.a[1,t+1] = sum(h[s,t])  (replanting)")
    println("  3.a[s+1,t+1] = a[s,t] - h[s,t], s<n  (aging)")
    println("  4.h[n,t] = a[n,t]  (mandatory mature harvest)")
    println("  5.0 <= h[s,t] <= a[s,t]  (cannot overharvest)")
    println("="^80)
    println()
end

function solve_forest_problem(model::ForestModel; verbose=true)
    n = model.n_stages
    T = model.time_horizon
    b = model.discount_factor
    f = model.biomass_coeffs
    A = model.total_area
    
    a0 = fill(A/n, n)
    
    opt = Model(Ipopt.Optimizer)
    set_optimizer_attribute(opt, "print_level", verbose ?  5 : 0)
    set_optimizer_attribute(opt, "max_iter", 500)
    set_optimizer_attribute(opt, "tol", 1e-7)
    
    @variable(opt, a[s=1:n, t=0:T] >= 0)
    @variable(opt, h[s=1:n, t=0:T] >= 0)
    
    for s in 1:n
        JuMP.fix(a[s,0], a0[s]; force=true)
    end
    
    @constraint(opt, [t=0:T], sum(a[s,t] for s in 1:n) == A)
    @constraint(opt, [s=1:n, t=0:T], h[s,t] <= a[s,t])
    
    for t in 0:T-1
        @constraint(opt, a[1,t+1] == sum(h[s,t] for s in 1:n))
        for s in 1:n-1
            @constraint(opt, a[s+1,t+1] == a[s,t] - h[s,t])
        end
    end
    
    @constraint(opt, [t=0:T], h[n,t] == a[n,t])
    
    @NLobjective(opt, Max,
        sum(b^t * sqrt(sum(f[s]*h[s,t] for s in 1:n) + 1e-10) for t in 0:T))
    
    optimize!(opt)
    
    status = JuMP.termination_status(opt)
    
    if verbose
        println("\n" * "="^80)
        println("RESULTS - SINGLE SPECIES")
        println("="^80)
        println("Status: $status")
        
        if status in [:LOCALLY_SOLVED, :OPTIMAL]
            obj = JuMP.objective_value(opt)
            println("Objective: ", round(obj, digits=4))
            println()
            
            for t in [0, 1, 2, T-1, T]
                println("t=$t:")
                areas = [JuMP.value(a[s,t]) for s in 1:n]
                harvests = [JuMP.value(h[s,t]) for s in 1:n]
                biomass = sum(f[s]*harvests[s] for s in 1:n)
                
                println("  Areas:    ", round.(areas, digits=4))
                println("  Harvest:  ", round.(harvests, digits=4))
                println("  Biomass: ", round(biomass, digits=4))
                println("  Utility:  ", round(sqrt(biomass+1e-10), digits=4))
                println()
            end
        else
            println("WARNING: Solver did not converge!")
        end
        println("="^80)
    end
    
    return opt
end

function solve_multispecies(model::MultiSpeciesModel; verbose=true)
    K = model.n_species
    n = model.n_stages
    T = model.time_horizon
    b = model.discount_factor
    f = model.biomass_coeffs
    A = model.total_area
    
    a0 = fill(A/(K*n), K, n)
    
    opt = Model(Ipopt.Optimizer)
    set_optimizer_attribute(opt, "print_level", verbose ? 5 : 0)
    set_optimizer_attribute(opt, "max_iter", 500)
    
    @variable(opt, a[k=1:K, s=1:n, t=0:T] >= 0)
    @variable(opt, h[k=1:K, s=1:n, t=0:T] >= 0)
    
    for k in 1:K, s in 1:n
        JuMP.fix(a[k,s,0], a0[k,s]; force=true)
    end
    
    @constraint(opt, [t=0:T], 
        sum(a[k,s,t] for k in 1:K, s in 1:n) == A)
    
    @constraint(opt, [k=1:K, s=1:n, t=0:T], h[k,s,t] <= a[k,s,t])
    
    for k in 1:K
        for t in 0:T-1
            @constraint(opt, a[k,1,t+1] == sum(h[k,s,t] for s in 1:n))
            for s in 1:n-1
                @constraint(opt, a[k,s+1,t+1] == a[k,s,t] - h[k,s,t])
            end
        end
        @constraint(opt, [t=0:T], h[k,n,t] == a[k,n,t])
    end
    
    @NLobjective(opt, Max,
        sum(b^t * sqrt(sum(f[k,s]*h[k,s,t] for k in 1:K, s in 1:n) + 1e-10) 
            for t in 0:T))
    
    optimize!(opt)
    
    if verbose
        println("\n" * "="^80)
        println("RESULTS - MULTI-SPECIES")
        println("="^80)
        println("Status:", JuMP.termination_status(opt))
        
        if JuMP.termination_status(opt) in [:LOCALLY_SOLVED, :OPTIMAL]
            println("Objective:", round(JuMP.objective_value(opt), digits=4))
            println()
            
            println("Biomass coefficients:")
            for k in 1:K
                println("  Species $k: ", f[k,:])
            end
            println()
            
            for t in [0, 1, 2, T-1, T]
                println("t=$t:")
                for k in 1:K
                    areas = [JuMP.value(a[k,s,t]) for s in 1:n]
                    harvests = [JuMP.value(h[k,s,t]) for s in 1:n]
                    println("  Species $k:A=", round.(areas,digits=4), 
                            " H=", round.(harvests,digits=4))
                end
                biomass = sum(f[k,s]*JuMP.value(h[k,s,t]) for k in 1:K, s in 1:n)
                println("  Total biomass:", round(biomass, digits=4))
                println()
            end
        end
        println("="^80)
    end
    
    return opt
end

function main()
    println("\n" * "="^80)
    println("FOREST PLANNING OPTIMIZATION - COMPLETE SOLUTION")
    println("="^80)
    println()
    
    println("PART (a):MATHEMATICAL FORMULATION\n")
    formulate_problem()
    
    println("PART (b):SINGLE SPECIES\n")
    model1 = ForestModel(
        4,
        [1.0, 2.5, 4.0, 5.5],
        0.95,
        1.0,
        10
    )
    solve_forest_problem(model1, verbose=true)
    
    println("\nPART (c):MULTI-SPECIES (K=3)\n")
    model2 = MultiSpeciesModel(
        3,
        4,
        [1.0 2.0 3.0 4.0;
         0.8 2.2 4.0 5.5;
         0.5 1.8 4.5 7.0],
        0.95,
        1.0,
        10
    )
    solve_multispecies(model2, verbose=true)
    
    println("\n" * "="^80)
    println("ALL PARTS COMPLETE")
    println("="^80)
end

main()