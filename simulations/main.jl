module MainSimulation

using CompoundRandomization
using Distributions
import Empirikos
using Random
using Future: randjump

const NREPS = 500
const TASK_GROUPS = (
    k3_5 = (K=(3, 5), reps=250),
    k7_9 = (K=(7, 9), reps=100),
    k11 = (K=(11,), reps=50),
    k13 = (K=(13,), reps=25),
)
const MASTER_SEED = 1
const RNG_JUMP = big(10)^20
const REPLICATE_RNGS = let
    rngs = [MersenneTwister(MASTER_SEED)]
    for rep in 2:NREPS
        push!(rngs, randjump(last(rngs), RNG_JUMP))
    end
    rngs
end
const QUANTILES = (0.25, 0.75)
const METHOD_NAMES = (
    "ttest", "usual_limma", "oracle_limma", "oracle_lfdr",
    "compound_rotation_quantile", "compound_signflip_quantile",
    "separate_signflip_quantile", "seqstep_quantile",
    "compound_rotation_quantile_corrected", "compound_signflip_quantile_corrected",
    "ddr_rotation_quantile", "ddr_signflip_quantile", "sens_gaussian", "sens_general",
)

"""
Main-text model (1)-(2). Supply a univariate noise distribution with mean zero
and variance one; scaling by sigma_i then gives conditional variance sigma_i^2.
"""
Base.@kwdef struct Scenario{D<:UnivariateDistribution}
    id::Int
    noise::D
    K::Int
    π1::Float64
    n::Int = 5000
    α::Float64 = 0.1
    λ::Float64 = 10.0
    ν0::Float64 = 10.0
    s0²::Float64 = 1.0
end

const SETTINGS = let
    noises = (Normal(), Uniform(-sqrt(3.0), sqrt(3.0)), Laplace(0.0, inv(sqrt(2.0))))
    # Panels A, C, D: vary K at fixed sparsity for each noise distribution.
    varying_K = [(noise, K, 0.025) for noise in noises for K in (3, 5, 7, 9, 11, 13)]
    # Panel B: vary sparsity at K=5. Its π1=0.025 point is already in panel A.
    varying_π1 = [(Normal(), 5, π1) for π1 in (0.005, 0.015, 0.035, 0.045)]
    settings = vcat(varying_K, varying_π1)
    [Scenario(; id, noise, K, π1) for (id, (noise, K, π1)) in enumerate(settings)]
end

"""
Map a group's Slurm array index to one scenario and a batch of repetitions.
Task indices start at one separately in each of k3_5, k7_9, k11, and k13.
Repetition numbers, and thus RNG streams, do not depend on the batching.
"""
function task_replicates(group::Symbol, task; nreps=NREPS)
    haskey(TASK_GROUPS, group) || throw(ArgumentError("unknown task group $group"))
    nreps > 0 || throw(ArgumentError("replicate count must be positive"))
    config = TASK_GROUPS[group]
    settings = filter(s -> s.K in config.K, SETTINGS)
    nbatches = cld(nreps, config.reps)
    1 <= task <= length(settings) * nbatches || throw(ArgumentError("invalid task number for $group"))
    scenario = settings[div(task - 1, nbatches) + 1]
    first_rep = mod(task - 1, nbatches) * config.reps + 1
    scenario, first_rep:min(first_rep + config.reps - 1, nreps)
end

"""Generate X once in Julia. Rows are hypotheses; columns are replicates."""
function simulate(rng, s::Scenario)
    X = Matrix{Float64}(undef, s.n, s.K)
    σ², μ = zeros(s.n), zeros(s.n)
    nonnull = falses(s.n)
    for i in 1:s.n
        σ²[i] = s.ν0 * s.s0² / rand(rng, Chisq(s.ν0))
        nonnull[i] = rand(rng) < s.π1
        nonnull[i] && (μ[i] = randn(rng) * sqrt(s.λ * σ²[i] / s.K))
        X[i, :] .= μ[i] .+ sqrt(σ²[i]) .* rand(rng, s.noise, s.K)
    end
    (; X, σ², μ, nonnull)
end

"""
Run the main-figure methods and compound comparisons on the same samples.
No method generates data.
Standalone Limma uses sample variances (K-1 df). Randomization methods share
the quantile prior fitted to orbit variances (K df), invariant under all three
groups used here. Reuse the sign-flip reference for compound and separate BH.
Load RCall before calling with include_sens=true.
"""
function run_methods(samples, s::Scenario; split_seed, include_sens=true)
    rejections = Dict{String,BitVector}()
    rejections["ttest"] = fit(Empirikos.SimultaneousTTest(α=s.α), samples).rj_idx
    rejections["usual_limma"] = fit(Empirikos.EmpiricalPartiallyBayesTTest(
        prior=Empirikos.Limma(), α=s.α, solver=nothing), samples).rj_idx
    prior = Empirikos.InverseScaledChiSquare(s.s0², s.ν0)
    rejections["oracle_limma"] = fit(Empirikos.EmpiricalPartiallyBayesTTest(
        prior=prior, α=s.α, solver=nothing), samples).rj_idx
    rejections["oracle_lfdr"] = fit(LocalFDROracle(
        π1=s.π1, λ=s.λ, ν0=s.ν0, s0²=s.s0², α=s.α), samples).rj_idx

    estimator = ModeratedTScore(Empirikos.QuantileLimma(quantile_p=QUANTILES, scale_p=0.5))
    score = fit_statistic(OrthogonalRotations(), estimator, samples)
    rotations = fit_reference(OrthogonalRotations(), score, samples)
    compound_rotation = fit(CompoundBH(α=s.α), rotations)
    rejections["compound_rotation_quantile"] = compound_rotation.rj_idx
    rejections["compound_rotation_quantile_corrected"] = compound_rotation.adjp .<= s.α / 1.93
    rejections["ddr_rotation_quantile"] = fit(DDR(α=s.α, τ=s.α / 10), rotations).rj_idx
    flips = fit_reference(SignFlips(), score, samples)
    compound_signflip = fit(CompoundBH(α=s.α), flips)
    rejections["compound_signflip_quantile"] = compound_signflip.rj_idx
    rejections["compound_signflip_quantile_corrected"] = compound_signflip.adjp .<= s.α / 1.93
    rejections["ddr_signflip_quantile"] = fit(DDR(α=s.α, τ=s.α / 10), flips).rj_idx
    rejections["separate_signflip_quantile"] = fit(SeparateBH(α=s.α), flips).rj_idx
    rejections["seqstep_quantile"] = fit(MultipleRandomizationTest(
        group=InvolutionGroup(), statistic=score, procedure=SeqStepPlus(α=s.α)), samples).rj_idx

    if include_sens
        for variant in (:gaussian, :general)
            rejections["sens_$variant"] = fit(SENS(; variant, seed=split_seed, α=s.α), samples).rj_idx
        end
    end
    rejections
end

"""FDP = V/max(R,1); power = true discoveries/max(number of alternatives,1)."""
function metrics(rejected, nonnull)
    discoveries = count(rejected)
    false_discoveries = count(rejected .& .!nonnull)
    true_discoveries = count(rejected .& nonnull)
    (; discoveries, false_discoveries, true_discoveries,
        fdp=false_discoveries / max(discoveries, 1),
        power=true_discoveries / max(count(nonnull), 1))
end

function run_replicate(s::Scenario, rep; include_sens=true)
    rng = copy(REPLICATE_RNGS[rep])
    split_seed = rand(rng, 0:Int(typemax(Int32)))
    data = simulate(rng, s)
    samples = ReplicatedSample.(eachrow(data.X))
    rejections = run_methods(samples, s; split_seed, include_sens)
    [(; scenario=s.id, rep, master_seed=MASTER_SEED, rng_jump=string(RNG_JUMP), split_seed, method,
        m=s.n, K=s.K, noise=lowercase(string(nameof(typeof(s.noise)))), pi1=s.π1, alpha=s.α,
        lambda=s.λ, nu0=s.ν0, s0sq=s.s0²,
        quantile_low=QUANTILES[1], quantile_high=QUANTILES[2], scale_quantile=0.5,
        bh_level=endswith(method, "_corrected") ? s.α / 1.93 : s.α,
        tau=startswith(method, "ddr_") ? s.α / 10 : missing,
        nnonnull=count(data.nonnull), metrics(rejections[method], data.nonnull)...)
        for method in METHOD_NAMES if haskey(rejections, method)]
end

end
