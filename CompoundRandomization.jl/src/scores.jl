"""Absolute sample mean, abs(mean(z)), using the cached one-sample mean."""
struct AbsMean end
(::AbsMean)(s::ReplicatedSample) = abs(mean(s))

"""Absolute difference in sample means, abs(mean(A) - mean(B))."""
struct AbsMeanDifference end
(::AbsMeanDifference)(x::TwoSample) = abs(x.δ̂)

"""
    ModeratedTScore(prior)

Absolute moderated t-statistic, using a fitted variance prior.
A point-mass prior gives the absolute z-score.
An inverse-scaled-chi-square prior with nu = Inf is represented by its
point-mass limit at the prior scale.

For a new sample type, implement `Empirikos.NormalChiSquareSample(x)` returning
the standardized coefficient estimate delta-hat/sqrt(v), residual variance,
and residual degrees of freedom (Section 7, equations (31)-(32)). This adapter
is specific to moderated scores, not a requirement on other statistics.

To estimate the prior, call `fit_statistic(group, ModeratedTScore(estimator), samples)`.
This uses `orbit_variance(group, x)`, not the score's residual variance.
Neither summary is assumed to identify the entire orbit. This estimator learns
one shared prior; covariate-dependent priors need their own estimator and score.
For one-sample data, prior fitting uses tau_i^2 = sum(z_i.^2)/K
with K degrees of freedom, invariant under both sign flips and full rotations.
"""
struct ModeratedTScore{P}
    prior::P
end

function ModeratedTScore(prior::Empirikos.InverseScaledChiSquare)
    prior.ν == Inf && return ModeratedTScore(Dirac(prior.σ²))
    ModeratedTScore{typeof(prior)}(prior)
end

"""
    sign_symmetric(score)

Whether S(-z) = S(z) for every z, with the fitted score held fixed. This is
global sign reversal, not invariance under arbitrary coordinatewise sign flips.
When true, `SignFlips(reduce_symmetry=true)` keeps one representative of each
pair epsilon, -epsilon, preserving the exact score distribution.

Defaults to false for custom callables. Declare it only when this identity is
known mathematically, not merely observed on the input data. The trait does not
certify null symmetry or that the score was learned from valid orbit information.
"""
sign_symmetric(::Any) = false
sign_symmetric(::AbsMean) = true
sign_symmetric(::ModeratedTScore) = true

"""
    within_group_symmetric(score)

Whether the score is unchanged by reordering observations separately within
groups A and B. Enables exact allocation enumeration for `Permutations`.
Defaults to false for custom callables; no symmetry is inferred from their code.
"""
within_group_symmetric(::Any) = false
within_group_symmetric(::AbsMeanDifference) = true
within_group_symmetric(::ModeratedTScore) = true

"""
    label_symmetric(score)

Whether exchanging equal-sized groups A and B leaves the score unchanged.
Together with `within_group_symmetric`, permits retaining only allocations
with observation 1 in A: one representative of each complementary pair.
Defaults to false for custom callables.
"""
label_symmetric(::Any) = false
label_symmetric(::AbsMeanDifference) = true
label_symmetric(::ModeratedTScore) = true

(score::ModeratedTScore)(x::AbstractRandomizationSample) =
    score(Empirikos.NormalChiSquareSample(x))

function (score::ModeratedTScore{<:Empirikos.InverseScaledChiSquare})(x::Empirikos.NormalChiSquareSample)
    iszero(x.Z) && return 0.0
    variance = Empirikos.ScaledChiSquareSample(x)
    post = Empirikos.posterior(variance, score.prior)
    abs(x.Z) / sqrt(post.σ²)
end

(score::ModeratedTScore{<:Dirac})(x::Empirikos.NormalChiSquareSample) =
    abs(x.Z) / sqrt(score.prior.value)
