"""
    ReplicatedSample(z)

Replicate measurements for one hypothesis, with cached mean, corrected sample
variance, and orbit variance `sum(abs2, z)/length(z)`.
Copies `z`; treat the stored replicates `Z` as read-only.
"""
struct ReplicatedSample{V<:AbstractVector{<:Real},M<:Real,S<:Real} <: Empirikos.EBayesSample{V}
    Z::V
    μ̂::M
    σ̂²::S
    τ̂²::S

    function ReplicatedSample(z::AbstractVector{<:Real})
        Z = copy(z)
        μ̂ = mean(Z)
        σ̂² = var(Z; mean=μ̂, corrected=true)
        τ̂² = sum(abs2, Z) / length(Z)
        new{typeof(Z),typeof(μ̂),typeof(σ̂²)}(Z, μ̂, σ̂², τ̂²)
    end
end

nobs(s::ReplicatedSample) = length(s.Z)
Statistics.mean(s::ReplicatedSample) = s.μ̂
Statistics.var(s::ReplicatedSample) = s.σ̂²

function checked_samples(s::AbstractVector{<:ReplicatedSample})
    isempty(s) && throw(ArgumentError("at least one hypothesis is required"))
    K = nobs(first(s))
    K >= 2 || throw(ArgumentError("at least two replicates are required"))
    all(x -> nobs(x) == K, s) || throw(DimensionMismatch("replicate counts differ"))
    all(x -> all(isfinite, x.Z), s) || throw(ArgumentError("replicates must be finite"))
    s
end

"""Ordinary sufficient statistics: sample variance with K-1 degrees of freedom."""
Empirikos.NormalChiSquareSample(x::ReplicatedSample) =
    Empirikos.NormalChiSquareSample(sqrt(nobs(x)) * mean(x), var(x), nobs(x) - 1)


struct AbsMean end
(::AbsMean)(s::ReplicatedSample) = abs(mean(s))

"""
    ModeratedTScore(prior)

Absolute moderated t-statistic, using a fitted variance prior.
A point-mass prior gives the absolute z-score.

To estimate the prior, call `fit(ModeratedTScore(estimator), samples)` first.
This uses the orbit variances tau_i^2 = sum(z_i.^2)/K with K degrees of freedom.
"""
struct ModeratedTScore{P}
    prior::P
end

sign_symmetric(::Any) = false
sign_symmetric(::AbsMean) = true
sign_symmetric(::ModeratedTScore) = true

function fit(score::ModeratedTScore{<:Empirikos.LimmaMethod}, data::AbstractVector{<:ReplicatedSample})
    s = checked_samples(data)
    tau = Empirikos.ScaledChiSquareSample.(getproperty.(s, :τ̂²), nobs.(s))
    ModeratedTScore(Empirikos.fit_prior(score.prior, tau))
end

function (score::ModeratedTScore{<:Empirikos.InverseScaledChiSquare})(x::ReplicatedSample)
    K = nobs(x)
    mu = mean(x)
    iszero(mu) && return 0.0
    variance = Empirikos.ScaledChiSquareSample(var(x), K - 1)
    post = Empirikos.posterior(variance, score.prior)
    sqrt(K) * abs(mu) / sqrt(post.σ²)
end

(score::ModeratedTScore{<:Dirac})(x::ReplicatedSample) =
    sqrt(nobs(x)) * abs(mean(x)) / sqrt(score.prior.value)
