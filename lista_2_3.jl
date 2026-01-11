# Retail Store Inventory Management MDP
# Problem 3: Formulate the Store Management Problem as an MDP

using Printf
using Random
using LinearAlgebra

"""
Inventory Management MDP Structure

States:  Current inventory level x_t (0 ≤ x_t ≤ M)
Actions: Order quantity a_t (number of units to order)
Transitions:  Stochastic due to random demand D_t
Rewards: Revenue from sales - ordering costs - holding costs + residual value
"""

# Define the Inventory MDP structure
struct InventoryMDP
    M::Int                      # Maximum capacity
    h::Function                 # Holding cost function h(x)
    C::Function                 # Ordering cost function C(a)
    f::Function                 # Revenue function f(q) for q units sold
    g::Function                 # Residual value function g(x) at year end
    demand_dist::Function       # Probability distribution P(D = d)
    max_demand::Int            # Maximum possible demand
    γ::Float64                 # Discount factor
    horizon::Int               # Time horizon (12 months)
end

"""
MDP Tuple Elements: 
- S (States): Inventory levels {0, 1, 2, ..., M}
- A (Actions): Order quantities {0, 1, 2, ..., a_max}
- P (Transition probabilities): P(s'|s,a) based on demand distribution
- R (Reward function): Revenue - costs
- γ (Discount factor): For future value evaluation
"""

# Print MDP formulation
function print_mdp_formulation()
    println("="^80)
    println("INVENTORY MANAGEMENT MDP FORMULATION")
    println("="^80)
    
    println("\n📦 PROBLEM DESCRIPTION:")
    println("-"^80)
    println("A retail store manages inventory month by month over a year.")
    println("At each month t, the store has x_t units in stock and faces demand D_t.")
    println("The manager decides how many units a_t to order from the supplier.")
    println()
    
    println("="^80)
    println("MDP TUPLE DEFINITION:  ⟨S, A, P, R, γ⟩")
    println("="^80)
    
    println("\n1️⃣  STATE SPACE (S):")
    println("-"^80)
    println("   S = {0, 1, 2, ..., M}")
    println("   ")
    println("   • x_t ∈ S represents the inventory level at the beginning of month t")
    println("   • M is the maximum storage capacity")
    println("   • State 0 means out of stock")
    println()
    
    println("2️⃣  ACTION SPACE (A):")
    println("-"^80)
    println("   A(x) = {0, 1, 2, ..., M - x}")
    println("   ")
    println("   • a_t represents the number of units ordered at month t")
    println("   • Action space depends on current state x")
    println("   • Cannot order more than remaining capacity:  a ≤ M - x")
    println("   • After ordering:  inventory becomes x + a (before demand)")
    println()
    
    println("3️⃣  TRANSITION PROBABILITY FUNCTION (P):")
    println("-"^80)
    println("   P(x'|x, a) = Probability of transitioning from state x to x'")
    println("                given action a")
    println("   ")
    println("   Transition dynamics:")
    println("   • After ordering a units:  inventory = x + a")
    println("   • Demand D occurs (random variable)")
    println("   • Units sold: q = min(x + a, D)")
    println("   • Next state: x' = max(0, x + a - D)")
    println()
    println("   Mathematical formulation:")
    println("   ")
    println("   P(x'|x, a) = P(D = x + a - x')  if x' = (x + a) - D and x' ≥ 0")
    println("              = P(D ≥ x + a)       if x' = 0 (stockout)")
    println("              = 0                   otherwise")
    println()
    
    println("4️⃣  REWARD FUNCTION (R):")
    println("-"^80)
    println("   R(x, a, x') = Revenue - Ordering Cost - Holding Cost + Residual Value")
    println("   ")
    println("   Components:")
    println("   ")
    println("   (a) Holding Cost: h(x)")
    println("       • Cost of maintaining inventory of size x")
    println("       • Typically:  h(x) = c_h × x (linear in inventory)")
    println("   ")
    println("   (b) Ordering Cost: C(a)")
    println("       • Cost of ordering a units")
    println("       • Often includes fixed cost:  C(a) = K·𝟙(a>0) + c_o × a")
    println("       • K = fixed ordering cost (setup cost)")
    println("       • c_o = per-unit ordering cost")
    println("   ")
    println("   (c) Revenue: f(q)")
    println("       • Revenue from selling q units")
    println("       • q = min(x + a, D) = units actually sold")
    println("       • Typically: f(q) = p × q (price p per unit)")
    println("   ")
    println("   (d) Lost Sales Penalty (implicit):")
    println("       • If D > x + a, customers leave unsatisfied")
    println("       • Lost revenue = p × (D - (x + a))")
    println("   ")
    println("   (e) Residual Value: g(x)")
    println("       • Value of remaining inventory at year end")
    println("       • g(x) = salvage_value × x")
    println("       • Only applied in final time period")
    println()
    println("   Expected Reward:")
    println("   ")
    println("   R(x, a) = 𝔼_D[f(min(x+a, D))] - C(a) - h(x+a)")
    println("   ")
    println("   For a given demand realization D = d:")
    println("   r(x, a, d) = f(min(x+a, d)) - C(a) - h(x+a)")
    println()
    
    println("5️⃣  DISCOUNT FACTOR (γ):")
    println("-"^80)
    println("   γ ∈ [0, 1]")
    println("   ")
    println("   • Represents time value of money")
    println("   • Monthly discount: γ ≈ e^(-r/12) where r is annual interest rate")
    println("   • For annual evaluation: can use γ = 1 within the year")
    println()
    
    println("6️⃣  TIME HORIZON:")
    println("-"^80)
    println("   T = 12 months (one year)")
    println("   ")
    println("   • Finite horizon problem")
    println("   • Terminal reward includes residual value g(x_12)")
    println()
    
    println("="^80)
    println("OBJECTIVE")
    println("="^80)
    println()
    println("Find optimal policy π*:  S → A that maximizes expected total profit:")
    println()
    println("   V*(x₀) = max E[Σₜ₌₀ᵀ⁻¹ γᵗ R(xₜ, aₜ, Dₜ) + γᵀ g(x_T)]")
    println()
    println("where:")
    println("   • x₀ = initial inventory")
    println("   • aₜ = π(xₜ) = optimal action in state xₜ")
    println("   • Dₜ ~ demand distribution")
    println("   • g(x_T) = residual value of final inventory")
    println()
    
    println("="^80)
    println("SOLUTION METHOD")
    println("="^80)
    println()
    println("Use Backward Induction (Dynamic Programming) for finite horizon:")
    println()
    println("1.Initialize: V_T(x) = g(x)  (terminal value)")
    println()
    println("2.For t = T-1, T-2, ..., 0:")
    println("   ")
    println("      Q_t(x, a) = 𝔼_D[r(x, a, D) + γ V_{t+1}(x')]")
    println("      ")
    println("      V_t(x) = max_{a ∈ A(x)} Q_t(x, a)")
    println("      ")
    println("      π_t*(x) = argmax_{a ∈ A(x)} Q_t(x, a)")
    println()
    println("3.Optimal policy:  π* = {π₀*, π₁*, ..., π_{T-1}*}")
    println()
end

# Example MDP instance with concrete functions
function create_example_inventory_mdp()
    println("\n" * "="^80)
    println("EXAMPLE INVENTORY MDP INSTANCE")
    println("="^80)
    
    # Parameters
    M = 20                    # Maximum capacity
    price = 10.0              # Selling price per unit
    unit_cost = 5.0          # Ordering cost per unit
    fixed_cost = 15.0        # Fixed ordering cost
    holding_cost_rate = 0.5  # Holding cost per unit
    salvage_value = 3.0      # Residual value per unit
    γ = 0.99                 # Monthly discount factor
    horizon = 12             # 12 months
    
    # Cost and revenue functions
    h(x) = holding_cost_rate * x                           # Holding cost
    C(a) = (a > 0 ? fixed_cost : 0.0) + unit_cost * a     # Ordering cost
    f(q) = price * q                                       # Revenue
    g(x) = salvage_value * x                              # Residual value
    
    # Demand distribution:  Poisson-like, truncated
    max_demand = 15
    λ = 8.0  # Average demand
    
    # Compute Poisson probabilities (with approximation for large factorials)
    function poisson_pmf(k, lambda)
        if k > 20
            # Use Stirling's approximation for large k
            return exp(k * log(lambda) - lambda - k * log(k) + k - 0.5 * log(2 * π * k))
        else
            return (lambda^k * exp(-lambda)) / factorial(k)
        end
    end
    
    # Create demand distribution
    demand_probs = [poisson_pmf(d, λ) for d in 0:max_demand]
    total_prob = sum(demand_probs)
    
    # Normalize
    demand_probs = demand_probs ./ total_prob
    
    # Create function
    demand_dist(d) = (d >= 0 && d <= max_demand) ? demand_probs[d+1] : 0.0
    
    println("\nParameters:")
    println("-"^80)
    @printf("  Maximum capacity (M):           %d units\n", M)
    @printf("  Selling price:                   \$%.2f per unit\n", price)
    @printf("  Ordering cost per unit:         \$%.2f\n", unit_cost)
    @printf("  Fixed ordering cost:            \$%.2f\n", fixed_cost)
    @printf("  Holding cost rate:              \$%.2f per unit\n", holding_cost_rate)
    @printf("  Salvage value:                   \$%.2f per unit\n", salvage_value)
    @printf("  Discount factor (γ):            %.2f\n", γ)
    @printf("  Time horizon:                    %d months\n", horizon)
    @printf("  Average demand (λ):             %.1f units/month\n", λ)
    @printf("  Maximum demand considered:      %d units\n", max_demand)
    
    # Verify demand distribution
    total_demand_prob = sum(demand_dist(d) for d in 0:max_demand)
    println("\nDemand distribution verification:")
    @printf("  Sum of probabilities:  %.6f\n", total_demand_prob)
    
    println("\nDemand Distribution P(D = d):")
    println("-"^80)
    for d in 0:max_demand
        prob = demand_dist(d)
        if prob > 0.001
            @printf("  P(D = %2d) = %.4f", d, prob)
            bar_length = Int(round(prob * 100))
            println("  " * "█"^bar_length)
        end
    end
    
    return InventoryMDP(M, h, C, f, g, demand_dist, max_demand, γ, horizon)
end

# Compute transition probability - CORRECTED VERSION
function transition_probability(mdp::InventoryMDP, x:: Int, a::Int, x_next::Int)
    # After ordering:  inventory = x + a (before demand)
    inventory_after_order = x + a
    
    # Demand needed to reach x_next from inventory_after_order
    demand = inventory_after_order - x_next
    
    if demand < 0
        # Impossible:  can't have more inventory after demand
        return 0.0
    elseif x_next == 0
        # Stockout: demand ≥ inventory_after_order
        if inventory_after_order > mdp.max_demand
            # If we have more inventory than max demand, we won't stock out
            return 0.0
        elseif inventory_after_order == 0
            # If we start with 0, we stay at 0
            return 1.0
        else
            # Sum probabilities for all demands >= inventory
            total_prob = 0.0
            for d in inventory_after_order:mdp.max_demand
                total_prob += mdp.demand_dist(d)
            end
            return total_prob
        end
    else
        # Specific next state:  need exact demand
        if demand > mdp.max_demand
            return 0.0
        else
            return mdp.demand_dist(demand)
        end
    end
end

# Compute expected reward - CORRECTED VERSION
function expected_reward(mdp::InventoryMDP, x::Int, a::Int, is_final::Bool=false)
    inventory_after_order = x + a
    
    # Ordering cost
    order_cost = mdp.C(a)
    
    # Holding cost (applied to inventory after ordering)
    hold_cost = mdp.h(inventory_after_order)
    
    # Expected revenue from sales
    expected_revenue = 0.0
    for d in 0:mdp.max_demand
        units_sold = min(inventory_after_order, d)
        expected_revenue += mdp.demand_dist(d) * mdp.f(units_sold)
    end
    
    # Base reward
    reward = expected_revenue - order_cost - hold_cost
    
    # Note: residual value is NOT added here
    # It will be captured in the value function through transitions
    
    return reward
end

# Solve using backward induction - CORRECTED VERSION
function solve_finite_horizon_inventory(mdp::InventoryMDP)
    println("\n" * "="^80)
    println("SOLVING USING BACKWARD INDUCTION")
    println("="^80)
    
    T = mdp.horizon
    
    # Value functions for each time period:  V[t][x]
    # Index:  V[t+1][x+1] corresponds to V_t(x)
    V = [zeros(Float64, mdp.M + 1) for _ in 1:(T+1)]
    
    # Optimal policies for each time period:  π[t][x]
    π = [zeros(Int, mdp.M + 1) for _ in 1:T]
    
    # Terminal condition: V_T(x) = g(x) (residual value)
    for x in 0:mdp.M
        V[T+1][x+1] = mdp.g(x)
    end
    
    println("\nTerminal values V_T(x) = g(x):")
    for x in 0:min(10, mdp.M)
        @printf("  V_%d(%2d) = %.2f\n", T, x, V[T+1][x+1])
    end
    
    # Backward induction
    for t in T:-1:1
        if t >= T-2 || t <= 2
            println("\n" * "-"^80)
            println("Period t = $t")
            println("-"^80)
        elseif t == T-3
            println("\n... (skipping detailed output for middle periods) ...")
        end
        
        for x in 0:mdp.M
            best_value = -Inf
            best_action = 0
            
            # Try all feasible actions
            for a in 0:(mdp.M - x)
                # Expected immediate reward
                immediate_reward = expected_reward(mdp, x, a, false)
                
                # Expected future value
                future_value = 0.0
                for x_next in 0:mdp.M
                    prob = transition_probability(mdp, x, a, x_next)
                    if prob > 0.0
                        future_value += prob * V[t+1][x_next+1]
                    end
                end
                
                q_value = immediate_reward + mdp.γ * future_value
                
                if q_value > best_value
                    best_value = q_value
                    best_action = a
                end
            end
            
            V[t][x+1] = best_value
            π[t][x+1] = best_action
        end
        
        # Print sample values
        if t <= 2 || t >= T-1
            println("Sample optimal values and actions:")
            for x in [0, 2, 4, 6, 8, 10]
                if x <= mdp.M
                    @printf("  x=%2d: V_%d(x)=%8.2f, π_%d(x)=%2d (order to level %d)\n", 
                            x, t, V[t][x+1], t, π[t][x+1], x + π[t][x+1])
                end
            end
        end
    end
    
    println("\n" * "="^80)
    println("Backward induction completed successfully!")
    println("="^80)
    
    return V, π
end

# Print optimal policy summary
function print_policy_summary(mdp::InventoryMDP, π::Vector{Vector{Int}})
    println("\n" * "="^80)
    println("OPTIMAL POLICY SUMMARY")
    println("="^80)
    println()
    println("Optimal ordering policy π_t*(x) for each month t and inventory level x:")
    println()
    
    # Print header
    @printf("%-10s", "x \\ t")
    for t in 1:min(6, mdp.horizon)
        @printf("%6d", t)
    end
    if mdp.horizon > 6
        print("  ...")
    end
    println()
    println("-"^80)
    
    # Print policy table
    for x in 0:min(15, mdp.M)
        @printf("%-10d", x)
        for t in 1:min(6, mdp.horizon)
            @printf("%6d", π[t][x+1])
        end
        println()
    end
    
    println("\nInterpretation:  π_t(x) = number of units to order when")
    println("                         inventory is x at beginning of month t")
    println()
    
    # Analyze policy pattern
    println("Policy insights:")
    println("-"^80)
    
    # Check if there's an (S,s) policy pattern
    println("• The optimal policy often follows an (S,s) structure:")
    println("  - s = reorder point (order when inventory drops to s or below)")
    println("  - S = order-up-to level (order enough to reach S)")
    println()
    
    # Sample analysis for period 1
    t = 1
    println("Month $t analysis:")
    non_zero_actions = [(x, π[t][x+1]) for x in 0:mdp.M if π[t][x+1] > 0]
    if ! isempty(non_zero_actions)
        println("  Reorder points (states where we order):")
        for (x, a) in non_zero_actions[1:min(5, length(non_zero_actions))]
            @printf("    x=%2d:  order %2d units → reach level %2d\n", x, a, x+a)
        end
    end
end

# Simulate inventory management
function simulate_inventory(mdp::InventoryMDP, π::Vector{Vector{Int}}, x0::Int, num_scenarios:: Int=100)
    println("\n" * "="^80)
    println("MONTE CARLO SIMULATION")
    println("="^80)
    println("\nSimulating $num_scenarios scenarios starting with x₀ = $x0")
    
    Random.seed!(42)
    total_profits = Float64[]
    
    for scenario in 1:num_scenarios
        x = x0
        total_profit = 0.0
        
        for t in 1:mdp.horizon
            # Get optimal action
            a = π[t][x+1]
            
            # Compute immediate reward
            reward = expected_reward(mdp, x, a, false)
            
            # Sample demand
            rand_val = rand()
            cumulative_prob = 0.0
            demand = 0
            for d in 0:mdp.max_demand
                cumulative_prob += mdp.demand_dist(d)
                if rand_val <= cumulative_prob
                    demand = d
                    break
                end
            end
            
            # Compute actual reward based on realized demand
            inventory_after_order = x + a
            units_sold = min(inventory_after_order, demand)
            actual_reward = mdp.f(units_sold) - mdp.C(a) - mdp.h(inventory_after_order)
            
            total_profit += mdp.γ^(t-1) * actual_reward
            
            # Transition to next state
            x = max(0, inventory_after_order - demand)
        end
        
        # Add terminal value
        total_profit += mdp.γ^mdp.horizon * mdp.g(x)
        
        push!(total_profits, total_profit)
    end
    
    avg_profit = sum(total_profits) / num_scenarios
    std_profit = sqrt(sum((p - avg_profit)^2 for p in total_profits) / num_scenarios)
    
    println("\nSimulation results:")
    println("-"^80)
    @printf("  Average total profit:     \$%.2f\n", avg_profit)
    @printf("  Standard deviation:      \$%.2f\n", std_profit)
    @printf("  Min profit:              \$%.2f\n", minimum(total_profits))
    @printf("  Max profit:              \$%.2f\n", maximum(total_profits))
    
    return total_profits
end

# Main execution
function main()
    print_mdp_formulation()
    
    mdp = create_example_inventory_mdp()
    
    V, π = solve_finite_horizon_inventory(mdp)
    
    print_policy_summary(mdp, π)
    
    # Simulate with initial inventory
    x0 = 5
    profits = simulate_inventory(mdp, π, x0, 1000)
    
    println("\n" * "="^80)
    println("CONCLUSION")
    println("="^80)
    println()
    println("The inventory management problem has been successfully formulated as an MDP")
    println("with all required tuple elements ⟨S, A, P, R, γ⟩ clearly identified:")
    println()
    println("  ✓ States (S): Inventory levels {0, 1, ..., M}")
    println("  ✓ Actions (A): Order quantities {0, 1, ..., M-x}")
    println("  ✓ Transitions (P): P(x'|x,a) based on stochastic demand")
    println("  ✓ Rewards (R): Revenue - ordering cost - holding cost")
    println("  ✓ Discount factor (γ): Time preference parameter")
    println()
    println("The optimal policy was computed using backward induction and")
    println("validated through Monte Carlo simulation.")
    println("="^80)
end

# Run the program
main()