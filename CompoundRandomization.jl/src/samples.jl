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

"""
    RegressionDesign(W, X)

Shared fixed design for Z_i = W*delta_i + X*theta_i + epsilon_i (Section 7.2,
equation (34)). `W` is the vector of interest; columns of `X` are nuisance
covariates, including an intercept if desired. No intercept is added implicitly.
The full design [W X] must have full column rank and positive residual df.

Copies the inputs and caches their QR factorization.
`v = 1/norm((I-P_X)W)^2` is the contrast
variance factor. `ν` is the full-model residual df K-p-1. Treat fields as read-only.
"""
struct RegressionDesign{F}
    W::Vector{Float64}
    X::Matrix{Float64}
    qr::F
    v::Float64
    ν::Int

    function RegressionDesign(W::AbstractVector{<:Real}, X::AbstractMatrix{<:Real})
        length(W) == size(X, 1) || throw(DimensionMismatch("W and X must have the same observation count"))
        W, X = Vector{Float64}(W), Matrix{Float64}(X)
        all(isfinite, W) && all(isfinite, X) || throw(ArgumentError("design must be finite"))
        A = hcat(W, X)
        ν = size(A, 1) - size(A, 2)
        ν > 0 || throw(ArgumentError("positive residual degrees of freedom are required"))
        rank(A) == size(A, 2) || throw(ArgumentError("[W X] must have full column rank"))
        r = isempty(X) ? copy(W) : W - X * (X \ W)
        v = inv(sum(abs2, r))
        factorization = qr(A)
        new{typeof(factorization)}(W, X, factorization, v, ν)
    end
end

"""
    RegressionSample(z, design::RegressionDesign)
    RegressionSample(z, W, X)

The input `z` contains the K responses for one hypothesis, in the same observation
order as the rows of `design.W` and `design.X`; `Z` stores a copy of this vector.
Tests delta_i = 0 in Section 7.2. `β̂` contains all
OLS coefficients in [W X] order: delta-hat followed by theta-hat. `δ̂` caches
the first coefficient. `σ̂²` is the full-model residual variance with ν df;
`τ̂²` is norm((I-P_X)z)^2/(ν+1), the orbit variance for nuisance-fixed rotations.
In particular, (ν+1)*τ̂² = ν*σ̂² + δ̂^2/v.

Copies the response; shares `design` across hypotheses. Retains raw observations
for other statistics. Treat stored observations, coefficients and design as read-only.
"""
struct RegressionSample{D<:RegressionDesign} <: AbstractRandomizationSample{Vector{Float64}}
    Z::Vector{Float64}
    design::D
    β̂::Vector{Float64}
    δ̂::Float64
    σ̂²::Float64
    τ̂²::Float64

    function RegressionSample(z::AbstractVector{<:Real}, design::RegressionDesign)
        length(z) == length(design.W) || throw(DimensionMismatch("response and design lengths differ"))
        all(isfinite, z) || throw(ArgumentError("response must be finite"))
        Z = Vector{Float64}(z)
        β̂ = design.qr \ Z
        δ̂ = first(β̂)
        residual = Z - design.W * δ̂ - design.X * view(β̂, 2:length(β̂))
        rss = sum(abs2, residual)
        σ̂² = rss / design.ν
        nuisance_residual = isempty(design.X) ? Z : Z - design.X * (design.X \ Z)
        τ̂² = sum(abs2, nuisance_residual) / (design.ν + 1)
        new{typeof(design)}(Z, design, β̂, δ̂, σ̂², τ̂²)
    end
end

RegressionSample(z::AbstractVector{<:Real}, W::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}) =
    RegressionSample(z, RegressionDesign(W, X))

nobs(x::RegressionSample) = length(x.Z)

function checked_samples(samples::AbstractVector{<:RegressionSample})
    isempty(samples) && throw(ArgumentError("at least one hypothesis is required"))
    design = first(samples).design
    all(x -> x.design === design || (x.design.W == design.W && x.design.X == design.X), samples) ||
        throw(ArgumentError("regression samples must use the same design"))
    samples
end

"""Ordinary coefficient summary with full-model residual df, not orbit df (Section 7.2)."""
Empirikos.NormalChiSquareSample(x::RegressionSample) =
    Empirikos.NormalChiSquareSample(x.δ̂ / sqrt(x.design.v), x.σ̂², x.design.ν)
