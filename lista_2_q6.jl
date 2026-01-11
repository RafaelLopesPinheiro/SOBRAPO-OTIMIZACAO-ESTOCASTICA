using JuMP
using GLPK

function solve_mdp_dual_discounted_fixed(X, U, C, q; gamma=0.95, initial_dist=nothing)
    model = Model(GLPK. Optimizer)
    
    # Default:  uniform initial distribution
    if initial_dist === nothing
        initial_dist = Dict(x => 1.0/length(X) for x in X)
    end
    
    # Decision variables
    @variable(model, z[x in X, u in U(x)] >= 0)
    
    # Objective: maximize ∑∑ C(x,u) * z_x^u (for max formulation)
    # For cost minimization, typically we minimize, so let's use Min: 
    @objective(model, Min, 
        sum(C(x, u) * z[x, u] for x in X for u in U(x))
    )
    
    # Flow balance constraints (corrected):
    # ∑_u z_j^u = (1-γ)α_j + γ ∑_{i,u} P(j|i,u) z_i^u
    # Rearranging: ∑_u z_j^u - γ ∑_{i,u} q_{ij}^u z_i^u = (1-γ)α_j
    @constraint(model, flow[j in X],
        sum(z[j, u] for u in U(j)) - 
        gamma * sum(q(i, j, u) * z[i, u] for i in X for u in U(i)) == 
        (1 - gamma) * get(initial_dist, j, 0.0)
    )
    
    # Solve
    JuMP.optimize!(model)
    
    return (
        model = model,
        status = JuMP.termination_status(model),
        objective_value = JuMP. objective_value(model),
        z_optimal = JuMP. value.(z),
        dual_variables = JuMP.dual.(flow)
    )
end

# Test with the same example
function test_discounted()
    X = 1:3
    
    U_map = Dict(1 => [1, 2], 2 => [1], 3 => [1, 2])
    U(x) = U_map[x]
    
    C_map = Dict(
        (1, 1) => 1.0, (1, 2) => 2.0,
        (2, 1) => 1.5,
        (3, 1) => 0.5, (3, 2) => 1.0
    )
    C(x, u) = C_map[(x, u)]
    
    q_map = Dict(
        (1, 1, 1) => 0.7, (1, 2, 1) => 0.2, (1, 3, 1) => 0.1,
        (1, 1, 2) => 0.1, (1, 2, 2) => 0.6, (1, 3, 2) => 0.3,
        (2, 1, 1) => 0.3, (2, 2, 1) => 0.5, (2, 3, 1) => 0.2,
        (3, 1, 1) => 0.2, (3, 2, 1) => 0.3, (3, 3, 1) => 0.5,
        (3, 1, 2) => 0.4, (3, 2, 2) => 0.4, (3, 3, 2) => 0.2
    )
    q(x, j, u) = get(q_map, (x, j, u), 0.0)
    
    println("="^60)
    println("DISCOUNTED FORMULATION (γ = 0.95) - FIXED")
    println("="^60)
    
    # Try with uniform initial distribution
    result = solve_mdp_dual_discounted_fixed(X, U, C, q, gamma=0.95)
    
    println("Solution Status: ", result.status)
    println("Optimal Objective Value: ", result.objective_value)
    println("\nOptimal z values:")
    for x in X
        for u in U(x)
            println("  z[$x, $u] = ", result.z_optimal[x, u])
        end
    end
    println("\nDual variables (value function):")
    for x in X
        println("  v*[$x] = ", result.dual_variables[x])
    end
    
    return result
end

test_discounted()