# MDP with Absorbing States Solver
# Problem: Find optimal policy with discount factor α = 1

using LinearAlgebra
using Printf

# Define the MDP structure for absorbing states
struct AbsorbingMDP
    n_states::Int
    n_actions::Int
    transitions::Dict{Tuple{Int,Symbol}, Vector{Tuple{Int,Float64}}}  # (state, action) -> [(next_state, prob)]
    rewards::Dict{Int, Float64}  # Terminal state rewards
    absorbing_states::Set{Int}
    γ::Float64
end

# Create MDP from Figure 2
function create_absorbing_mdp()
    n_states = 7
    γ = 1.0  # Discount factor α = 1
    
    # Define absorbing (terminal) states with their rewards
    absorbing_states = Set([5, 6, 7])
    rewards = Dict(
        5 => 10.0,
        6 => 100.0,
        7 => 1000.0
    )
    
    # Define transitions:  (state, action) => [(next_state, probability), ...]
    transitions = Dict{Tuple{Int,Symbol}, Vector{Tuple{Int,Float64}}}()
    
    # State 1
    transitions[(1, :Rest)] = [(1, 1.0)]
    transitions[(1, :Work)] = [(2, 0.5), (3, 0.5)]
    
    # State 2
    transitions[(2, :Rest)] = [(2, 0.5), (5, 0.5)]
    transitions[(2, :Work)] = [(3, 0.3), (4, 0.4), (6, 0.3)]
    
    # State 3
    transitions[(3, :Rest)] = [(1, 0.5), (4, 0.5)]
    transitions[(3, :Work)] = [(2, 0.6), (5, 0.4)]
    
    # State 4
    transitions[(4, :Rest)] = [(6, 0.6), (7, 0.4)]
    transitions[(4, :Work)] = [(5, 0.1), (7, 0.9)]
    
    # Absorbing states (stay with probability 1)
    transitions[(5, :Rest)] = [(5, 1.0)]
    transitions[(5, :Work)] = [(5, 1.0)]
    
    transitions[(6, :Rest)] = [(6, 1.0)]
    transitions[(6, :Work)] = [(6, 1.0)]
    
    transitions[(7, :Rest)] = [(7, 1.0)]
    transitions[(7, :Work)] = [(7, 1.0)]
    
    return AbsorbingMDP(n_states, 2, transitions, rewards, absorbing_states, γ)
end

# Value Iteration for absorbing states
function value_iteration_absorbing(mdp::AbsorbingMDP; θ:: Float64=1e-8, max_iter::Int=10000)
    V = zeros(Float64, mdp.n_states)
    V_new = zeros(Float64, mdp.n_states)
    π = Dict{Int, Symbol}()
    
    # Initialize values for absorbing states
    for s in mdp.absorbing_states
        V[s] = mdp.rewards[s]
        V_new[s] = mdp.rewards[s]
    end
    
    actions = [:Rest, :Work]
    iteration = 0
    
    for iter in 1:max_iter
        iteration = iter
        δ = 0.0
        
        for s in 1:mdp.n_states
            # Skip absorbing states
            if s in mdp.absorbing_states
                continue
            end
            
            v_old = V[s]
            
            # Compute Q-values for all actions
            q_values = Dict{Symbol, Float64}()
            
            for action in actions
                if ! haskey(mdp.transitions, (s, action))
                    continue
                end
                
                q_val = 0.0
                for (next_state, prob) in mdp.transitions[(s, action)]
                    # Reward is only obtained when reaching absorbing state
                    immediate_reward = (next_state in mdp.absorbing_states) ? mdp.rewards[next_state] : 0.0
                    q_val += prob * (immediate_reward + mdp.γ * V[next_state])
                end
                q_values[action] = q_val
            end
            
            # Update value function with max Q-value
            if !isempty(q_values)
                best_action = argmax(q_values)
                V_new[s] = q_values[best_action]
                π[s] = best_action
                
                δ = max(δ, abs(v_old - V_new[s]))
            end
        end
        
        V .= V_new
        
        # Check convergence
        if δ < θ
            println("Converged after $iteration iterations (δ = $δ)")
            break
        end
        
        if iter == max_iter
            println("Warning: Maximum iterations reached. δ = $δ")
        end
    end
    
    return V, π, iteration
end

# Linear system solver (alternative exact method for α=1)
function solve_linear_system(mdp::AbsorbingMDP)
    println("\n" * "="^70)
    println("SOLVING USING LINEAR SYSTEM (Exact Method for γ=1)")
    println("="^70)
    
    # Separate transient and absorbing states
    transient_states = [s for s in 1:mdp.n_states if !(s in mdp.absorbing_states)]
    n_transient = length(transient_states)
    
    println("Transient states: $transient_states")
    println("Absorbing states: $(sort(collect(mdp.absorbing_states)))")
    
    # For each transient state, we need to compute optimal policy first
    # This is more complex, so we'll use value iteration results
    # But show that we could set up the linear system
    
    println("\nNote: With γ=1, the system may not have a unique solution")
    println("unless all paths lead to absorbing states (which is the case here).")
    println("Value iteration handles this naturally.")
end

# Print MDP structure
function print_mdp_structure(mdp::AbsorbingMDP)
    println("\n" * "="^70)
    println("MDP STRUCTURE - TRANSITIONS AND REWARDS")
    println("="^70)
    
    actions = [:Rest, :Work]
    
    for s in 1:mdp.n_states
        println("\n--- State $s ---")
        
        if s in mdp.absorbing_states
            println("  [ABSORBING STATE] Reward = $(mdp.rewards[s])")
            continue
        end
        
        for action in actions
            if haskey(mdp.transitions, (s, action))
                println("  Action:  $action")
                for (next_state, prob) in mdp.transitions[(s, action)]
                    reward_str = (next_state in mdp.absorbing_states) ? 
                                 " (Reward:  $(mdp.rewards[next_state]))" : ""
                    @printf("    → State %d:  P = %.1f%s\n", next_state, prob, reward_str)
                end
            end
        end
    end
end

# Compute expected reward for reaching absorbing states
function compute_absorption_probabilities(mdp::AbsorbingMDP, π::Dict{Int, Symbol})
    println("\n" * "="^70)
    println("ABSORPTION ANALYSIS")
    println("="^70)
    
    # For each non-absorbing state, compute probability of reaching each absorbing state
    # following the optimal policy
    
    transient_states = [s for s in 1:mdp.n_states if !(s in mdp.absorbing_states)]
    absorbing_list = sort(collect(mdp.absorbing_states))
    
    n_trans = length(transient_states)
    n_abs = length(absorbing_list)
    
    # Create mapping
    trans_idx = Dict(s => i for (i, s) in enumerate(transient_states))
    abs_idx = Dict(s => i for (i, s) in enumerate(absorbing_list))
    
    # Transition matrix from transient to transient (following policy)
    Q = zeros(n_trans, n_trans)
    # Transition matrix from transient to absorbing (following policy)
    R_mat = zeros(n_trans, n_abs)
    
    for (i, s) in enumerate(transient_states)
        if haskey(π, s)
            action = π[s]
            for (next_state, prob) in mdp.transitions[(s, action)]
                if next_state in mdp.absorbing_states
                    j = abs_idx[next_state]
                    R_mat[i, j] = prob
                else
                    j = trans_idx[next_state]
                    Q[i, j] = prob
                end
            end
        end
    end
    
    println("\nTransition matrix Q (transient → transient, following π*):")
    display(Q)
    
    println("\n\nTransition matrix R (transient → absorbing, following π*):")
    display(R_mat)
    
    # Fundamental matrix N = (I - Q)^(-1)
    # B = N * R gives absorption probabilities
    I = Matrix{Float64}(LinearAlgebra.I, n_trans, n_trans)
    
    if rank(I - Q) == n_trans
        N = inv(I - Q)
        B = N * R_mat
        
        println("\n\nAbsorption probabilities (starting state → absorbing state):")
        println("-" * "^"^70)
        @printf("%-15s", "Start \\ End")
        for abs_state in absorbing_list
            @printf("%15s", "State $abs_state")
        end
        println()
        println("-"^70)
        
        for (i, s) in enumerate(transient_states)
            @printf("%-15s", "State $s")
            for j in 1:n_abs
                @printf("%15.4f", B[i, j])
            end
            println()
        end
        
        # Expected reward calculation
        println("\n\nExpected rewards from each transient state:")
        println("-"^70)
        for (i, s) in enumerate(transient_states)
            exp_reward = sum(B[i, j] * mdp.rewards[absorbing_list[j]] for j in 1:n_abs)
            @printf("State %d: Expected reward = %.4f\n", s, exp_reward)
        end
    else
        println("\n\nWarning: (I-Q) is singular, cannot compute absorption probabilities")
    end
end

# Print results
function print_results(mdp::AbsorbingMDP, V::Vector{Float64}, π::Dict{Int, Symbol})
    println("\n" * "="^70)
    println("OPTIMAL VALUE FUNCTION AND POLICY (γ = $(mdp.γ))")
    println("="^70)
    
    for s in 1:mdp.n_states
        if s in mdp.absorbing_states
            @printf("State %d: V*(S%d) = %10.4f  [ABSORBING - Reward:  %.0f]\n", 
                    s, s, V[s], mdp.rewards[s])
        else
            action_str = haskey(π, s) ? string(π[s]) : "N/A"
            @printf("State %d: V*(S%d) = %10.4f  →  π*(S%d) = %s\n", 
                    s, s, V[s], s, action_str)
        end
    end
    
    println("\n" * "="^70)
    println("OPTIMAL POLICY INTERPRETATION")
    println("="^70)
    println("The optimal policy maximizes expected cumulative reward until")
    println("reaching an absorbing state.\n")
    
    for s in sort(collect(keys(π)))
        println("  • State $s: Choose action '$(π[s])'")
    end
end

# Main execution
function solve_absorbing_mdp_problem()
    println("="^70)
    println("MDP WITH ABSORBING STATES SOLVER")
    println("Problem 2:  Optimal Policy with γ = 1")
    println("="^70)
    
    # Create MDP
    mdp = create_absorbing_mdp()
    
    # Print structure
    print_mdp_structure(mdp)
    
    # Solve using value iteration
    println("\n" * "="^70)
    println("SOLVING USING VALUE ITERATION")
    println("="^70)
    V, π, iterations = value_iteration_absorbing(mdp)
    
    # Print results
    print_results(mdp, V, π)
    
    # Absorption analysis
    compute_absorption_probabilities(mdp, π)
    
    # Discussion of γ=1 case
    println("\n" * "="^70)
    println("ANALYSIS:  DISCOUNT FACTOR γ = 1")
    println("="^70)
    println("With γ = 1 (no discounting), the agent values all future rewards")
    println("equally. In an absorbing state MDP, this means:")
    println("  • The optimal policy maximizes the expected terminal reward")
    println("  • All paths eventually lead to an absorbing state")
    println("  • The value of a state = expected reward when eventually absorbed")
    println("\nThe optimal strategy is to maximize the probability of reaching")
    println("the highest-reward absorbing state (State 7:  reward = 1000).")
    println("="^70)
end

# Run the solver
solve_absorbing_mdp_problem()
