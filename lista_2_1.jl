# MDP Solver using Value Iteration
# Problem: MDP with 3 states and 2 actions per state

using LinearAlgebra
using Printf

# Define the MDP structure
struct MDP
    n_states::Int
    n_actions::Int
    transitions::Array{Float64, 3}  # [state, action, next_state]
    rewards::Array{Float64, 3}      # [state, action, next_state]
    γ::Float64                       # discount factor
end

# Extract MDP parameters from the figure
function create_mdp_from_figure(γ::Float64)
    n_states = 3  # S₀, S₁, S₂
    n_actions = 2  # a₀, a₁
    
    # Initialize transition probability matrix P[s, a, s']
    P = zeros(Float64, n_states, n_actions, n_states)
    
    # Initialize reward matrix R[s, a, s']
    R = zeros(Float64, n_states, n_actions, n_states)
    
    # State S₀ (index 1)
    # Action a₀:  stays in S₀ with prob 0.5, goes to S₂ with prob 0.5
    P[1, 1, 1] = 0.5
    P[1, 1, 3] = 0.5
    R[1, 1, 1] = 0.0
    R[1, 1, 3] = -1.0
    
    # Action a₁: goes to S₀ with prob 1.0
    P[1, 2, 1] = 1.0
    R[1, 2, 1] = 0.0
    
    # State S₁ (index 2)
    # Action a₀: goes to S₀ with prob 0.70, to S₁ with prob 0.20, to S₂ with prob 0.10
    P[2, 1, 1] = 0.70
    P[2, 1, 2] = 0.20
    P[2, 1, 3] = 0.10
    R[2, 1, 1] = 0.0
    R[2, 1, 2] = +5.0
    R[2, 1, 3] = 0.0
    
    # Action a₁: goes to S₁ with prob 0.95, to S₂ with prob 0.05
    P[2, 2, 2] = 0.95
    P[2, 2, 3] = 0.05
    R[2, 2, 2] = 0.0
    R[2, 2, 3] = 0.0
    
    # State S₂ (index 3)
    # Action a₀: goes to S₂ with prob 0.40, to S₁ with prob 0.30, to S₀ with prob 0.30
    P[3, 1, 3] = 0.40
    P[3, 1, 2] = 0.30
    P[3, 1, 1] = 0.30
    R[3, 1, 3] = 0.0
    R[3, 1, 2] = 0.0
    R[3, 1, 1] = 0.0
    
    # Action a₁: goes to S₀ with prob 0.60, to S₂ with prob 0.40
    P[3, 2, 1] = 0.60
    P[3, 2, 3] = 0.40
    R[3, 2, 1] = 0.0
    R[3, 2, 3] = 0.0
    
    return MDP(n_states, n_actions, P, R, γ)
end

# Compute expected reward for state-action pair
function expected_reward(mdp::MDP, s:: Int, a::Int)
    exp_r = 0.0
    for s_next in 1:mdp.n_states
        exp_r += mdp.transitions[s, a, s_next] * mdp.rewards[s, a, s_next]
    end
    return exp_r
end

# Value Iteration algorithm
function value_iteration(mdp::MDP; θ::Float64=1e-6, max_iter::Int=1000)
    V = zeros(Float64, mdp.n_states)
    V_new = zeros(Float64, mdp.n_states)
    π = zeros(Int, mdp.n_states)
    
    iteration = 0
    
    for iter in 1:max_iter
        iteration = iter
        δ = 0.0
        
        for s in 1:mdp.n_states
            v_old = V[s]
            
            # Compute Q-values for all actions
            q_values = zeros(Float64, mdp.n_actions)
            for a in 1:mdp.n_actions
                # Q(s,a) = E[R(s,a,s')] + γ * Σ P(s'|s,a) * V(s')
                exp_r = expected_reward(mdp, s, a)
                exp_future = 0.0
                for s_next in 1:mdp.n_states
                    exp_future += mdp.transitions[s, a, s_next] * V[s_next]
                end
                q_values[a] = exp_r + mdp.γ * exp_future
            end
            
            # Update value function with max Q-value
            V_new[s] = maximum(q_values)
            π[s] = argmax(q_values)
            
            δ = max(δ, abs(v_old - V_new[s]))
        end
        
        V .= V_new
        
        # Check convergence
        if δ < θ
            println("Converged after $iteration iterations (δ = $δ)")
            break
        end
    end
    
    return V, π, iteration
end

# Print transition probability table
function print_transition_probabilities(mdp::MDP)
    println("\n" * "="^70)
    println("TRANSITION PROBABILITY FUNCTION P(s'|s,a)")
    println("="^70)
    
    for s in 1:mdp.n_states
        for a in 1:mdp.n_actions
            println("\nFrom S$(s-1), Action a$(a-1):")
            for s_next in 1:mdp.n_states
                prob = mdp.transitions[s, a, s_next]
                if prob > 0
                    reward = mdp.rewards[s, a, s_next]
                    @printf("  → S%d:  P = %.2f, R = %+.1f\n", s_next-1, prob, reward)
                end
            end
        end
    end
end

# Print results
function print_results(mdp::MDP, V:: Vector{Float64}, π::Vector{Int}, γ::Float64)
    println("\n" * "="^70)
    println("OPTIMAL VALUE FUNCTION AND POLICY (γ = $γ)")
    println("="^70)
    
    for s in 1:mdp.n_states
        @printf("State S%d: V*(S%d) = %8.4f  →  π*(S%d) = a%d\n", 
                s-1, s-1, V[s], s-1, π[s]-1)
    end
    
    println("\n" * "="^70)
    println("INTERPRETATION")
    println("="^70)
    println("The optimal policy π* tells us which action to take in each state")
    println("to maximize the expected cumulative discounted reward.")
    println("\nOptimal actions:")
    for s in 1:mdp.n_states
        println("  • In state S$(s-1): take action a$(π[s]-1)")
    end
end

# Main execution
function solve_mdp_problem()
    println("="^70)
    println("MDP SOLVER - Value Iteration")
    println("="^70)
    
    # Solve for both discount factors
    for γ in [0.1, 0.9]
        println("\n\n")
        println("*"^70)
        println("SOLVING FOR DISCOUNT FACTOR γ = $γ")
        println("*"^70)
        
        # Create MDP
        mdp = create_mdp_from_figure(γ)
        
        # Print transition probabilities (only once)
        if γ == 0.1
            print_transition_probabilities(mdp)
        end
        
        # Solve using value iteration
        V, π, iterations = value_iteration(mdp)
        
        # Print results
        print_results(mdp, V, π, γ)
    end
    
    println("\n" * "="^70)
    println("ANALYSIS")
    println("="^70)
    println("• γ = 0.1 (low discount): The agent is myopic, prioritizing")
    println("  immediate rewards over future rewards.")
    println("\n• γ = 0.9 (high discount): The agent is farsighted, valuing")
    println("  future rewards almost as much as immediate rewards.")
    println("="^70)
end

# Run the solver
solve_mdp_problem()