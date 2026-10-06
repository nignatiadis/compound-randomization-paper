# %% Setup: open this file in VS Code and execute from here.
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
using Random, Statistics, Distributions, CompoundRandomization
import Empirikos

# %% Parameters
n = 10_000
K = 5
π₁ = 0.005                 # Expected number of nonnulls: nπ₁ = 50.
α = 0.1
λ = 40.0                  # μ | σ² ~ N(0, λσ²/K).
ν₀ = 10.0
s₀² = 1.0
well_specified_prior = Empirikos.InverseScaledChiSquare(s₀², ν₀)
two_point_prior = DiscreteNonParametric([1.0, 10.0], [0.99, 0.01])
nreps = 1000                # Set to 10 for a quick check.
settings = (("inverse_chisquare", 930307329), ("two_point", 930516787))

# %% Gaussian data generation
# Each row of X contains K independent N(μᵢ, σᵢ²) observations.
function simulate(prior, rng; n=n, K=K, π₁=π₁, λ=λ)
    nonnull = rand(rng, Bernoulli(π₁), n)
    n₁ = count(nonnull)
    variance_prior = prior == "inverse_chisquare" ? well_specified_prior : two_point_prior
    σ² = rand(rng, variance_prior, n)
    μ = nonnull .* sqrt.(λ .* σ² ./ K) .* randn(rng, n)
    X = μ .+ sqrt.(σ²) .* randn(rng, n, K)
    Z = sqrt(K) .* vec(mean(X; dims=2))
    S² = vec(var(X; dims=2))
    samples = ReplicatedSample.(eachrow(X))
    (; X, samples, nonnull, σ², μ, Z, S², n₁)
end

# %% The four procedures (oracle is used only for the well-specified prior).
oracle = LocalFDROracle(π1=π₁, λ=λ, ν0=ν₀, s0²=s₀², α=α)
ttest = Empirikos.SimultaneousTTest(α=α)
limma = Empirikos.EmpiricalPartiallyBayesTTest(
    prior=Empirikos.Limma(), α=α, solver=nothing)
rotation = MultipleRandomizationTest(
    group=OrthogonalRotations(),
    statistic=ModeratedTScore(Empirikos.QuantileLimma(
        quantile_p=(0.25, 0.75), scale_p=0.5)),
    procedure=CompoundBH(α=α, compute_pvalues=false))
procedures = ("Oracle B-statistic" => oracle, "t-test + BH" => ttest,
    "Limma + BH" => limma, "Compound rotations" => rotation)

# %% Inspect one dataset: choose settings[1] or settings[2].
prior, seed = settings[2]
rng = MersenneTwister(seed)
data = simulate(prior, rng)
(; X, samples, nonnull, σ², μ, Z, S², n₁) = data

# %% Run each procedure individually and inspect its result.
ttest_result = fit(ttest, samples)
limma_result = fit(limma, samples)
rotation_result = fit(rotation, samples)
oracle_result = prior == "inverse_chisquare" ? fit(oracle, samples) : nothing

# %% Monte Carlo loop: FDR = mean(FDP), power = mean(true discoveries / n₁).
function run_simulation(prior, seed, nreps)
    active_methods = prior == "inverse_chisquare" ? procedures : procedures[2:end]
    fdp = zeros(nreps, length(active_methods))
    power = similar(fdp)
    rng = MersenneTwister(seed)
    for rep in 1:nreps
        data = simulate(prior, rng)
        for (j, (_, method)) in enumerate(active_methods)
            rejected = fit(method, data.samples).rj_idx
            R = count(rejected)
            T = count(rejected .& data.nonnull)
            fdp[rep,j] = (R-T)/max(R, 1)
            power[rep,j] = T/max(data.n₁, 1)
        end
    end
    [(; prior, method=label, FDR=100mean(fdp[:,j]), Power=100mean(power[:,j]))
        for (j, (label, _)) in enumerate(active_methods)]
end

results = reduce(vcat, [run_simulation(prior, seed, nreps) for (prior, seed) in settings])
