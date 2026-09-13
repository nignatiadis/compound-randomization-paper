function fit(method::Empirikos.SimultaneousTTest, data::AbstractVector{<:ReplicatedSample})
    fit(method, Empirikos.NormalChiSquareSample.(checked_samples(data)))
end

function fit(method::Empirikos.EmpiricalPartiallyBayesTTest, data::AbstractVector{<:ReplicatedSample})
    fit(method, Empirikos.NormalChiSquareSample.(checked_samples(data)))
end

"""
    LocalFDROracle(; π1, λ, ν0, s0², α=0.1)

Oracle LFDR_OR(alpha) from equations (5)-(6), under the Gaussian model (1)-(2).
The supplied parameters are known, not estimated from the observations.

`fit(method, samples)` rejects when `lfdr(Z_i) <= c_alpha`, using the largest cutoff with

    P(null and lfdr(Z) <= c_alpha) / P(lfdr(Z) <= c_alpha) <= alpha.

The cutoff is calibrated under the population model, not by averaging the
observed local FDRs. When `alpha >= pi0`, c_alpha is one (reject everyone).
If no finite t threshold attains the target, c_alpha is zero (reject no one).
Returns `lfdr`, `lfdr_cutoff`, the equivalent `t_cutoff`, and rejection indicators.
"""
Base.@kwdef struct LocalFDROracle
    π1::Float64
    λ::Float64
    ν0::Float64
    s0²::Float64
    α::Float64 = 0.1
end

function fit(method::LocalFDROracle, data::AbstractVector{<:ReplicatedSample})
    (; π1, λ, ν0, s0², α) = method
    check_level(α)
    check_level(π1)
    all(x -> isfinite(x) && x > 0, (λ, ν0, s0²)) ||
        throw(ArgumentError("oracle lambda, prior df and scale must be positive and finite"))
    s = checked_samples(data)
    π0 = 1 - π1
    ν = nobs(first(s)) - 1

    # Distributions of the oracle moderated t-statistic under (1)-(2).
    null = TDist(ν + ν0)
    alternative = sqrt(1 + λ) * null
    marginal = MixtureModel([null, alternative], [π0, π1])

    # Equations (5)-(6); symmetry cancels the factor of two in both tails.
    lfdr(t) = exp(log(π0) + logpdf(null, t) - logpdf(marginal, t))
    logFdr(t) = log(π0) + logccdf(null, t) - logccdf(marginal, t)
    log_prior_odds = log(π1) - log(π0)

    # Fdr decreases from pi0 at t=0 to this limit as t tends to infinity.
    minimum_fdr = inv(1 + exp(log_prior_odds + (ν + ν0) / 2 * log1p(λ)))
    if α >= π0
        t_cutoff, cα = 0.0, 1.0
    elseif α <= minimum_fdr
        t_cutoff, cα = Inf, 0.0
    else
        upper = 1.0
        while logFdr(upper) > log(α)
            upper *= 2
            isfinite(upper) || error("could not bracket the oracle cutoff")
        end
        t_cutoff = Roots.find_zero(t -> logFdr(t) - log(α), (0.0, upper), Roots.Bisection())
        cα = lfdr(t_cutoff)
    end

    score = ModeratedTScore(Empirikos.InverseScaledChiSquare(s0², ν0))
    local_fdr = lfdr.(score.(s))
    rejected = local_fdr .<= cα
    (; method, lfdr = local_fdr, t_cutoff, lfdr_cutoff = cα,
        rj_idx = rejected, total_rejections = count(rejected))
end

"""Original R SENS implementation. Load RCall and supply the R script and split seed."""
Base.@kwdef struct SENS
    variant::Symbol = :gaussian
    script::String
    seed::Int
    α::Float64 = 0.1
end
