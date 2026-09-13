"""
    AbstractRandomizationSample{T} <: Empirikos.EBayesSample{T}

Data for one hypothesis. Concrete types specify the testing problem and retain
the observations or summaries needed by their supported randomization groups.

Scores are callables `score(sample)`; no particular statistic, model, or summary
representation is required by this abstract type. Learned scores implement
`fit_statistic(group, estimator, samples)` using information invariant under that group.

New sample types should validate their inputs on construction, or specialize
`checked_samples` for checks involving the collection (e.g. a shared design).
"""
abstract type AbstractRandomizationSample{T} <: Empirikos.EBayesSample{T} end

"""
    ReplicatedSample(z)

Replicate measurements for one hypothesis, with cached mean, corrected sample
variance, and orbit variance `sum(abs2, z)/length(z)`.
Copies `z`; treat the stored replicates `Z` as read-only.
"""
struct ReplicatedSample{V<:AbstractVector{<:Real},M<:Real,S<:Real} <: AbstractRandomizationSample{V}
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

function checked_samples(s::AbstractVector{<:AbstractRandomizationSample})
    isempty(s) && throw(ArgumentError("at least one hypothesis is required"))
    s
end

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


"""
    TwoSample(a, b)

Two groups of observations for one hypothesis, testing equality of their means
(Section 7.1). Both groups must be nonempty, with K = length(a)+length(b) > 2.
Copies the observations into `Z`, with group A first and group B second.
The group sizes are stored in `K_A` and `K_B`.
Treat `Z` as read-only. Cached summaries are:

- `δ̂`: mean(A) - mean(B).
- `σ̂²`: pooled within-group variance, with K-2 residual degrees of freedom.
- `τ̂²`: overall centered sample variance, ignoring labels, with K-1 df.

The raw observations are retained for permutation tests. These summaries are
available for scores, but do not restrict which callable statistics can be used.
"""
struct TwoSample{V<:AbstractVector{<:Real},M<:Real,S<:Real} <: AbstractRandomizationSample{V}
    Z::V
    K_A::Int
    K_B::Int
    δ̂::M
    σ̂²::S
    τ̂²::S

    function TwoSample(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
        !isempty(a) && !isempty(b) || throw(ArgumentError("both groups must be nonempty"))
        K_A, K_B = length(a), length(b)
        K = K_A + K_B
        K > 2 || throw(ArgumentError("at least three observations are required"))
        Z = vcat(a, b)
        all(isfinite, Z) || throw(ArgumentError("observations must be finite"))
        # Canonical summation order preserves within-group permutation ties.
        # The stored raw observations retain their original order for custom scores.
        A, B = sort(Z[1:K_A]), sort(Z[K_A+1:end])
        μA, μB = mean(A), mean(B)
        δ̂ = μA - μB
        σ̂² = (sum(abs2, A .- μA) + sum(abs2, B .- μB)) / (K - 2)
        τ̂² = var(sort(Z); corrected=true)
        new{typeof(Z),typeof(δ̂),typeof(σ̂²)}(Z, K_A, K_B, δ̂, σ̂², τ̂²)
    end
end

nobs(x::TwoSample) = length(x.Z)

function checked_samples(samples::AbstractVector{<:TwoSample})
    isempty(samples) && throw(ArgumentError("at least one hypothesis is required"))
    (; K_A, K_B) = first(samples)
    all(x -> x.K_A == K_A && x.K_B == K_B, samples) ||
        throw(DimensionMismatch("group sizes must agree across hypotheses"))
    all(x -> all(isfinite, x.Z), samples) || throw(ArgumentError("observations must be finite"))
    samples
end

"""The coefficient and residual-variance summaries in equation (31), with K-2 df."""
function Empirikos.NormalChiSquareSample(x::TwoSample)
    v = inv(x.K_A) + inv(x.K_B)
    coefficient, variance = promote(x.δ̂ / sqrt(v), x.σ̂²)
    Empirikos.NormalChiSquareSample(coefficient, variance, nobs(x) - 2)
end
