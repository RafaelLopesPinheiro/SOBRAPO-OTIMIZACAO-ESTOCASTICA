"""
Problem [05] - Chapter 4: Optimization Under Uncertainty
Book: III Bienal da Sociedade Brasileira de Matemática

Problem Statement:
Consider f(x,ω) = (x-ω)², where x ∈ ℝ and ω is a random variable
with uniform distribution on the interval [0,1].Show that for this
situation, "min{·}" and "𝔼[·]" do not commute: 

𝔼[min_{x∈ℝ} f(x,ω)] ≠ min_{x∈ℝ} 𝔼[f(x,ω)]
"""

using Distributions
using Optim
using Plots
using Printf

# Define the function f(x,ω)
f(x, ω) = (x - ω)^2

println("="^70)
println("Problem [05] - Optimization Under Uncertainty")
println("="^70)
println()

# Part 1: Calculate 𝔼[min_{x∈ℝ} f(x,ω)]
println("PART 1: Computing 𝔼[min_{x∈ℝ} f(x,ω)]")
println("-"^70)

# For each fixed ω, find min_x f(x,ω)
# Since f(x,ω) = (x-ω)², the minimum occurs at x* = ω with f(ω,ω) = 0
println("For fixed ω, minimizing f(x,ω) = (x-ω)² over x:")
println("  ∂f/∂x = 2(x-ω) = 0  =>  x* = ω")
println("  Therefore:  min_{x∈ℝ} f(x,ω) = f(ω,ω) = 0")
println()

# The expected value of this minimum
min_then_expectation = 0.0
println("Taking the expectation:")
println("  𝔼[min_{x∈ℝ} f(x,ω)] = 𝔼[0] = 0")
println()
println("Result: 𝔼[min_{x∈ℝ} f(x,ω)] = $min_then_expectation")
println()

# Part 2: Calculate min_{x∈ℝ} 𝔼[f(x,ω)]
println("PART 2: Computing min_{x∈ℝ} 𝔼[f(x,ω)]")
println("-"^70)

# First compute 𝔼[f(x,ω)] for a given x
# ω ~ U[0,1], so 𝔼[f(x,ω)] = ∫₀¹ (x-ω)² dω
println("Computing 𝔼[f(x,ω)] for fixed x:")
println("  𝔼[f(x,ω)] = ∫₀¹ (x-ω)² dω")
println()

# Analytical computation of the integral
function expected_f(x)
    # ∫₀¹ (x-ω)² dω = ∫₀¹ (x² - 2xω + ω²) dω
    # = [x²ω - xω² + ω³/3]₀¹
    # = x² - x + 1/3
    return x^2 - x + 1/3
end

println("Analytical solution:")
println("  ∫₀¹ (x-ω)² dω = ∫₀¹ (x² - 2xω + ω²) dω")
println("               = [x²ω - xω² + ω³/3]₀¹")
println("               = x² - x + 1/3")
println()

# Now minimize 𝔼[f(x,ω)] over x
println("Minimizing 𝔼[f(x,ω)] = x² - x + 1/3 over x:")
println("  d/dx (x² - x + 1/3) = 2x - 1 = 0")
println("  => x* = 1/2")
println()

x_optimal = 1/2
expectation_then_min = expected_f(x_optimal)

println("Evaluating at x* = 1/2:")
println("  𝔼[f(1/2,ω)] = (1/2)² - 1/2 + 1/3")
println("              = 1/4 - 1/2 + 1/3")
println("              = 3/12 - 6/12 + 4/12")
println("              = 1/12")
println()
println("Result: min_{x∈ℝ} 𝔼[f(x,ω)] = $expectation_then_min")
println()

# Comparison
println("="^70)
println("CONCLUSION")
println("="^70)
println()
println(@sprintf("𝔼[min_{x∈ℝ} f(x,ω)] = %.6f", min_then_expectation))
println(@sprintf("min_{x∈ℝ} 𝔼[f(x,ω)] = %.6f", expectation_then_min))
println()
println("Since $min_then_expectation ≠ $expectation_then_min, we have shown that")
println("the operators 'min' and '𝔼' do NOT commute for this problem.")
println()

# Numerical verification using Monte Carlo simulation
println("="^70)
println("NUMERICAL VERIFICATION (Monte Carlo)")
println("="^70)
println()

n_samples = 100000
dist = Uniform(0, 1)

# Part 1 (numerical): 𝔼[min_x f(x,ω)]
# For each sample ω, min_x f(x,ω) = 0
samples_min_then_exp = zeros(n_samples)
for i in 1:n_samples
    ω = rand(dist)
    # min_x (x-ω)² = 0 at x = ω
    samples_min_then_exp[i] = 0.0
end
numerical_min_then_exp = mean(samples_min_then_exp)

# Part 2 (numerical): min_x 𝔼[f(x,ω)]
# Use optimization to find min_x of expected_f(x)
result = optimize(expected_f, -10.0, 10.0)
x_opt_numerical = Optim.minimizer(result)
numerical_exp_then_min = Optim.minimum(result)

println(@sprintf("Numerical 𝔼[min_{x∈ℝ} f(x,ω)] = %.6f", numerical_min_then_exp))
println(@sprintf("Numerical min_{x∈ℝ} 𝔼[f(x,ω)] = %.6f (at x = %.6f)", 
                 numerical_exp_then_min, x_opt_numerical))
println()

# Visualization
println("Generating visualization...")
println()

# Plot 1: Expected value function 𝔼[f(x,ω)] = x² - x + 1/3
x_range = -0.5:0.01:1.5
exp_values = expected_f.(x_range)

p1 = plot(x_range, exp_values, 
          linewidth=2, 
          label="𝔼[f(x,ω)] = x² - x + 1/3",
          xlabel="x", 
          ylabel="Expected Value",
          title="Expected Value of f(x,ω)",
          legend=:top)
scatter!([x_optimal], [expectation_then_min], 
         markersize=8, 
         label="min at x=1/2",
         color=:red)
hline!([min_then_expectation], 
       linestyle=:dash, 
       linewidth=2,
       label="𝔼[min f(x,ω)] = 0",
       color=:green)

# Plot 2: Sample paths of f(x,ω) for different ω values
ω_samples = [0.1, 0.3, 0.5, 0.7, 0.9]
p2 = plot(xlabel="x", 
          ylabel="f(x,ω)",
          title="Sample paths of f(x,ω) = (x-ω)² for different ω",
          legend=:top)
for ω in ω_samples
    plot!(p2, x_range, (x_range .- ω).^2, 
          label="ω = $ω",
          linewidth=1.5,
          alpha=0.7)
end

# Combine plots
p = plot(p1, p2, layout=(2,1), size=(800, 800))
savefig(p, "problem_05_visualization.png")
println("Visualization saved as 'problem_05_visualization.png'")
println()

println("="^70)
println("INTERPRETATION")
println("="^70)
println()
println("This problem illustrates a fundamental principle in stochastic optimization:")
println()
println("• When we minimize FIRST (for each ω) and then take expectation:")
println("  We adaptively choose x = ω for each scenario, achieving f = 0 always.")
println("  This represents a 'wait-and-see' or 'recourse' approach.")
println()
println("• When we take expectation FIRST and then minimize:")
println("  We choose a single x = 1/2 that works 'on average' across all scenarios.")
println("  This represents a 'here-and-now' approach.")
println()
println("The wait-and-see solution (0) is always at least as good as the")
println("here-and-now solution (1/12), which is why 𝔼[min f] ≤ min 𝔼[f].")
println()
println("This relates to Chapter 4's topic of 'Recourse Models' where the")
println("ability to make decisions after uncertainty is revealed (recourse)")
println("provides value compared to making all decisions upfront.")
println("="^70)