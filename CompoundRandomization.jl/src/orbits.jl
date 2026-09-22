"""A symmetry group used to construct conditional randomization distributions."""
abstract type AbstractRandomizationGroup end

"""
    SignFlips(; reduce_symmetry=true)

The coordinatewise sign-flip group H_symm = {±1}^K (Section 4.1).
Enumeration is exact, not Monte Carlo. If `sign_symmetric(score)` is true,
fixing the first sign to +1 gives one representative from each pair epsilon,
-epsilon. Averaging over these 2^(K-1) representatives equals the full-group
average. Set `reduce_symmetry=false` to enumerate all 2^K signs.
"""
Base.@kwdef struct SignFlips <: AbstractRandomizationGroup
    reduce_symmetry::Bool = true
end

"""
    OrthogonalRotations()

The orthogonal group H_orth = O(K) (Section 4.1). Haar randomization makes HZ
uniform on the sphere of radius norm(Z). For `AbsMean` and fitted
`ModeratedTScore`, orbit tails are computed analytically; no rotations are sampled.
"""
struct OrthogonalRotations <: AbstractRandomizationGroup end

"""
    Permutations(; reduce_symmetry=true)

The full coordinate-permutation group S_K from Section 7.1. Null observations
must be exchangeable; equality of means alone is not sufficient.

For a `within_group_symmetric` score, enumerate the binomial(K, K_A) allocations
instead of K! permutations: each allocation represents K_A! K_B! group elements
with the same score. When K_A = K_B and `label_symmetric(score)` also holds,
retain only allocations containing observation 1 in A, halving their number.
Each represents an allocation and its complement with equal scores.
Otherwise, without within-group symmetry, enumerate the full group.
Set `reduce_symmetry=false` to disable both reductions. Enumeration is exact,
not Monte Carlo. Full enumeration becomes infeasible quickly as K grows.
"""
Base.@kwdef struct Permutations <: AbstractRandomizationGroup
    reduce_symmetry::Bool = true
end

"""
    CenteredRotations()

The group {H in O(K): H*1 = 1} from Section 7.1. Fix the overall mean and rotate
the centered observations in dimension K-1. Null errors must be spherically
symmetric in this subspace, as in the equal-variance Gaussian two-sample model.
This is not the full `OrthogonalRotations()` group, which would move the nuisance mean.
"""
struct CenteredRotations <: AbstractRandomizationGroup end

"""
    ResidualRotations()

The subgroup H_orth,X = {H in O(K): H*X = X} of Section 7.2, for
`RegressionSample`. Fix the nuisance projection P_X*z and rotate its orthogonal
complement in dimension K-p = ν+1. The design X comes from each sample's shared
`RegressionDesign(W, X)`. Null errors must be invariant under these rotations;
the homoskedastic Gaussian regression model suffices. 
"""
struct ResidualRotations <: AbstractRandomizationGroup end

"""
    InvolutionGroup(H)
    InvolutionGroup()

The fixed group {id, H} of order two in Section 3.4. `H` is a callable taking
a sample and returning its transformed sample. The caller must establish that
H is not the identity, H(H(x)) = x, and the null law is invariant under H.
These properties cannot be certified for an arbitrary callable from observed data.

The default is the half/half sign flip for `ReplicatedSample`: keep the first
ceil(K/2) replicates and negate the rest. For example, a different fixed sign
pattern can be supplied as `InvolutionGroup(x -> ReplicatedSample(x.Z .* epsilon))`.
Custom groups need their own `orbit_variance` method to learn a moderated score;
alternatively supply a score already learned from valid invariant information.
"""
struct InvolutionGroup{H} <: AbstractRandomizationGroup
    transformation::H
end

"""
    HalfSplit()

The fixed sign-flip involution retaining the first ceil(K/2) replicates and
negating the rest. Calling it on a `ReplicatedSample` returns the transformed
sample; applying it twice recovers the original observations.
"""
struct HalfSplit end

function (::HalfSplit)(x::ReplicatedSample)
    K = nobs(x)
    epsilon = [j <= cld(K, 2) ? 1.0 : -1.0 for j in 1:K]
    ReplicatedSample(x.Z .* epsilon)
end

InvolutionGroup() = InvolutionGroup(HalfSplit())

"""
    fit_statistic(group, statistic, samples)

Prepare the fixed score S-hat conditional on the group orbits (Section 3.2).
For `ModeratedTScore` wrapping an Empirikos Limma estimator, learn the prior
from `orbit_variance(group, sample)`. For one-sample data this is
tau-hat_i^2 = norm(Z_i)^2/K with K degrees of freedom (Section 4.2), invariant
under both `SignFlips` and `OrthogonalRotations`.

Other statistics are returned unchanged and treated as fixed callables.
The caller must ensure any prior learning used only the group orbits or
independent data; the provenance of an arbitrary callable cannot be checked.
"""
fit_statistic(::AbstractRandomizationGroup, statistic,
    samples::AbstractVector{<:AbstractRandomizationSample}) = statistic

function fit_statistic(group::AbstractRandomizationGroup,
    statistic::ModeratedTScore{<:Empirikos.LimmaMethod}, samples::AbstractVector{<:AbstractRandomizationSample})
    checked_samples(samples)
    variances = orbit_variance.(Ref(group), samples)
    ModeratedTScore(Empirikos.fit_prior(statistic.prior, variances))
end

"""
    orbit_variance(group, sample)

Return an `Empirikos.ScaledChiSquareSample` for learning the variance prior.
Its value must be invariant under `group`; its degrees of freedom describe
its Gaussian-null distribution, not necessarily the score's residual df.

For one-sample sign flips and rotations this is tau-hat^2 with K df, equation (19).
Section 7 sample/group combinations supply their own methods: two-sample permutations and
rotations use the overall centered variance with K-1 df; regression rotations
fixing X use norm((I-P_X)z)^2/(K-p) with K-p df. This interface is specific to variance-prior estimation;
other score estimators need not use it.

No fallback is supplied: invariance must be established for each sample/group
combination. The summary need not identify the full orbit.
"""
orbit_variance(::SignFlips, x::ReplicatedSample) =
    Empirikos.ScaledChiSquareSample(x.τ̂², nobs(x))

orbit_variance(::OrthogonalRotations, x::ReplicatedSample) =
    Empirikos.ScaledChiSquareSample(x.τ̂², nobs(x))

orbit_variance(::Permutations, x::TwoSample) =
    Empirikos.ScaledChiSquareSample(x.τ̂², nobs(x) - 1)
orbit_variance(::CenteredRotations, x::TwoSample) =
    Empirikos.ScaledChiSquareSample(x.τ̂², nobs(x) - 1)

orbit_variance(::ResidualRotations, x::RegressionSample) =
    Empirikos.ScaledChiSquareSample(x.τ̂², x.design.ν + 1)

orbit_variance(::InvolutionGroup{HalfSplit}, x::ReplicatedSample) =
    Empirikos.ScaledChiSquareSample(x.τ̂², nobs(x))

"""
    AbstractRandomizationReference

Conditional distributions of the fixed score under group randomization, one
per hypothesis. Implement `orbit_tail(reference, i, t)` to return xi_i(t) from
equations (13)-(15), including equality at t. A reference represents score
distributions, not the raw group orbits themselves.
"""
abstract type AbstractRandomizationReference end

"""
Observed and transformed scores for {id, H}, retaining their roles for
Selective SeqStep+ (Procedure 3). Each orbit tail assigns mass 1/2 to each
score, including both contributions when the scores tie.
"""
struct InvolutionReference <: AbstractRandomizationReference
    observed::Vector{Float64}
    calibration::Vector{Float64}
end

"""
    FiniteRandomizationScores(sorted_scores)

Score distributions induced by a finite uniform randomization group.
These store the terms S-hat(HZ_i) in (13), with (17) as the sign-flip case.
`sorted_scores` is L by n. Each column represents the same number L of
equally weighted transformations, or equally weighted representatives when
the score is invariant under the omitted transformations.
Column i holds the Float64 scores for hypothesis i in ascending order, retaining
multiplicities: xi_i(t) = count(scores >= t)/L. Rows index score order, not signs.
Columns keep each hypothesis's scores contiguous for sorting and tail lookup.
"""
struct FiniteRandomizationScores <: AbstractRandomizationReference
    sorted_scores::Matrix{Float64}
end

"""
    RotationReference(statistic, norm2, dimension, contrast_variance)

Analytic score distributions under H_orth = O(K), conditional on the orbits
(Sections 4.1-4.2).

- `statistic`: the fixed score S-hat, e.g. the learned absolute moderated t-score.
- `norm2[i]`: squared radius of the rotating component; norm(Z_i)^2 in (19).
- `dimension`: dimension of that component, equal to K in the one-sample setting.
  With p nuisance coefficients (Section 7.2), it is K-p = nu_res+1 instead.
- `contrast_variance`: v in Var(delta-hat) = v*sigma^2, needed for raw contrast
  scores. It is 1/K for a sample mean and 1/K_A + 1/K_B for a mean difference.

The squared norm identifies the spherical orbit O_i. Together with the score,
it determines xi_i(t) = P_H(S-hat(HZ_i) >= t), with the learned score held fixed
conditional on O_1:n as in Section 3.2.
For the moderated score, nuisance-fixed rotations use the same Beta formula
with shape parameters 1/2 and (dimension-1)/2, equation (33). Raw absolute-mean
scores have no such generic interpretation outside the one-sample setting.
"""
struct RotationReference{S} <: AbstractRandomizationReference
    statistic::S
    norm2::Vector{Float64}
    dimension::Int
    contrast_variance::Float64
end

"""
    RandomizationFit

Result of `fit_reference(group, score, samples)`, preparing step 2 of Procedure 1.

- `group`: the chosen symmetry group from Section 4.1 or Section 7.
- `statistic`: S-hat, learned from O_1:n in step 1 and then held fixed.
- `observed[i]`: S-hat(Z_i), the threshold at which P_i^cmp is evaluated in (13).
- `reference`: the score distributions induced by Haar randomization within O_i.

Supply a vector of samples and fit any learned score beforehand.
The score is not refitted on transformations. `reference` implements
`AbstractRandomizationReference`, currently `FiniteRandomizationScores`,
`RotationReference`, or `InvolutionReference`; index i always refers to hypothesis i.
This result can be reused for compound BH, separate BH, or DDR.
"""
struct RandomizationFit{G,S,R<:AbstractRandomizationReference}
    group::G
    statistic::S
    observed::Vector{Float64}
    reference::R
end

function Base.show(io::IO, r::RandomizationFit)
    # Do not print the potentially exponential-sized reference score matrix.
    print(io, "RandomizationFit(", nameof(typeof(r.group)), ", ",
        nameof(typeof(r.statistic)), ", n=", length(r.observed), ")")
end

# Orbit tails and p-values: equations (13)-(15).

"""
    orbit_tail(r::RandomizationFit, i, t)

The tail xi_i(t) = P_H(S-hat(HZ_i) >= t), conditional on O_1:n with S-hat fixed
(Section 3.2). This is the Haar integral appearing in (13)-(15).
Equality is included: score ties and repeated randomized scores retain their full mass.
For sign flips this is a finite average; for rotations it is the Beta survival
probability described in `rotation_argument` below.
"""
orbit_tail(r::RandomizationFit, i, t) = orbit_tail(r.reference, i, t)

orbit_tail(reference::InvolutionReference, i, t) =
    ((reference.observed[i] >= t) + (reference.calibration[i] >= t)) / 2

function orbit_tail(reference::FiniteRandomizationScores, i, t)
    scores = view(reference.sorted_scores, :, i)
    first_ge = searchsortedfirst(scores, t)
    (length(scores) - first_ge + 1) / length(scores)
end

"""
    rotation_argument(reference::RotationReference, i, t)

Under H_orth (Section 4.1), r2 = norm(Z_i)^2 = K*tau-hat_i^2 from (19).
For a uniform point on this sphere,
B = K*mean(HZ)^2/r2 ~ Beta(1/2, (K-1)/2).
Return the b for which S-hat(HZ) >= t is equivalent to B >= b, for t > 0, r2 > 0.

The reference supplies the fixed score and the geometry of hypothesis i.
Writing d = reference.dimension and v = reference.contrast_variance:
- Raw absolute contrast: b = t^2/(v*r2). For a mean, v = 1/K;
  for a two-sample mean difference, v = 1/K_A + 1/K_B.
- Absolute z-score with fixed variance s0^2: b = s0^2*t^2/r2.
- Absolute moderated t from (8): b = t^2*(1 + nu0*s0^2/r2)/(d-1 + nu0 + t^2).

Here nu0 and s0^2 are the fitted prior parameters, not the sample df nu = K-1.
For two-sample rotations, r2 = sum((Z_i - mean(Z_i)).^2).
The moderated-t implementation divides by t^2 to also handle t = Inf.
This b is only a change of variable for integration; the score remains (8).
For the standardized coefficient scores of Section 7, d = nu_res+1 and r2
is the squared norm after removing nuisance effects. The moderated formula
then gives (33); v cancels through coefficient standardization.
"""
rotation_argument(r::RotationReference{AbsMean}, i, t) =
    abs2(t) / (r.contrast_variance * r.norm2[i])
rotation_argument(r::RotationReference{AbsMeanDifference}, i, t) =
    abs2(t) / (r.contrast_variance * r.norm2[i])
rotation_argument(r::RotationReference{AbsCoefficient}, i, t) =
    abs2(t) / (r.contrast_variance * r.norm2[i])
rotation_argument(r::RotationReference{<:ModeratedTScore{<:Dirac}}, i, t) =
    abs2(t) * r.statistic.prior.value / r.norm2[i]
function rotation_argument(r::RotationReference{<:ModeratedTScore{<:Empirikos.InverseScaledChiSquare}}, i, t)
    ν0, s0² = r.statistic.prior.ν, r.statistic.prior.σ²
    (1 + ν0 * s0² / r.norm2[i]) / (1 + (r.dimension - 1 + ν0) / abs2(t))
end

function orbit_tail(reference::RotationReference, i, t)
    t <= 0 && return 1.0
    r2 = reference.norm2[i]
    iszero(r2) && return 0.0 # A zero-radius orbit is a point mass at score zero.
    b = rotation_argument(reference, i, t)
    b >= 1 && return 0.0
    ccdf(Beta(0.5, (reference.dimension - 1) / 2), b)
end

"""
    compound_pvalues(r::RandomizationFit)

Equation (13): P_i^cmp = (1/n) sum_j xi_j(S-hat(Z_i)).
Each observed score is calibrated against all n orbit distributions, including
its own. 
"""
compound_pvalues(r::RandomizationFit) = pooled_pvalues(r, ones(length(r.observed)))

"""
    separate_pvalues(r::RandomizationFit)

Equation (15): P_i^sep = xi_i(S-hat(Z_i)). Each score is calibrated against only its
own orbit. With orthogonal rotations and the moderated-t score, these equal
ordinary two-sided t-test p-values for nondegenerate samples (Proposition 8).
"""
separate_pvalues(r::RandomizationFit) = [orbit_tail(r, i, t) for (i, t) in enumerate(r.observed)]

"""
    pooled_pvalues(r, weights)

Compute (1/n) sum_j weights[j] * xi_j(S-hat(Z_i)) for every i.
Unit weights give (13); DDR supplies weights[j] = 1/(1 - xi_j(s_tau)) in (14).
Supply one weight per hypothesis. The denominator is n, not the sum of the weights.
This is the direct formula; the finite-group specialization at the end of this file
computes the same sum without looking up every pair of observed score and column.
"""
function pooled_pvalues(r::RandomizationFit, weights)
    n = length(r.observed)
    [sum(weights[j] * orbit_tail(r, j, t) for j in 1:n) / n for t in r.observed]
end

# Reference score distribution construction.

"""
    fit_reference(group::InvolutionGroup, score, samples)

Compute S-hat(Z_i) and S-hat(HZ_i) for Procedure 3, with the score fixed.
Only two score vectors are stored; no larger group is enumerated.
"""
function fit_reference(group::InvolutionGroup, score, samples::AbstractVector{<:AbstractRandomizationSample})
    checked_samples(samples)
    observed = Float64.(score.(samples))
    calibration = [Float64(score(group.transformation(x))) for x in samples]
    !any(isnan, observed) && !any(isnan, calibration) || throw(ArgumentError("scores must not be NaN"))
    reference = InvolutionReference(observed, calibration)
    RandomizationFit(group, score, observed, reference)
end

"""
    fit_reference(SignFlips(), score, samples)

Prepare the finite averages in (17). Each hypothesis contributes the same number
of sign patterns. The global-reversal reduction is used only when both requested
by the group specification and justified by `sign_symmetric(score)`.

A callable receives a `ReplicatedSample`; larger scores mean more evidence.
Compute its observed and transformed values with the same fixed S-hat, then sort
the transformed scores within each hypothesis. Infinite scores are allowed;
NaNs are not. `AbsMean` selects the BLAS implementation of `sign_statistics`.
"""
function fit_reference(group::SignFlips, score, samples::AbstractVector{<:ReplicatedSample})
    checked_samples(samples)
    paired = group.reduce_symmetry && sign_symmetric(score)
    signs = sign_grid(nobs(first(samples)), paired)
    observed = Float64.(score.(samples))
    randomized_scores = sign_statistics(score, samples, signs)
    # Reuse the observed value for known ties, avoiding scalar/BLAS rounding differences.
    # sign_grid puts the identity last and (in the full group) global reversal first.
    randomized_scores[end, :] .= observed
    if !group.reduce_symmetry && sign_symmetric(score)
        randomized_scores[1, :] .= observed
    end
    !any(isnan, observed) && !any(isnan, randomized_scores) ||
        throw(ArgumentError("scores must not be NaN"))
    foreach(sort!, eachcol(randomized_scores))
    RandomizationFit(group, score, observed, FiniteRandomizationScores(randomized_scores))
end

"""
    fit_reference(OrthogonalRotations(), score, samples)

Prepare the Haar integrals in (13)-(15) using the spherical orbits of Section 4.1.
Store norm(Z_i)^2 = K*tau-hat_i^2 from (19) and the observed S-hat(Z_i); evaluating
an integral later uses only these radii, K, and the fixed score.

The analytic formula supports `AbsMean` and `ModeratedTScore` with a fitted
inverse-scaled-chi-square or point-mass variance prior. Other callables do not
have a supplied analytic rotation formula.
"""
function fit_reference(group::OrthogonalRotations, score::Union{AbsMean,ModeratedTScore},
    samples::AbstractVector{<:ReplicatedSample})
    checked_samples(samples)
    observed = Float64.(score.(samples))
    !any(isnan, observed) || throw(ArgumentError("scores must not be NaN"))
    K = nobs(first(samples))
    reference = RotationReference(score, [K * x.τ̂² for x in samples], K, inv(K))
    RandomizationFit(group, score, observed, reference)
end

"""
    fit_reference(Permutations(), score, samples::AbstractVector{<:TwoSample})

Evaluate the fixed score on exact permutations (or equally weighted allocations),
then retain the sorted score distributions. A callable receives a `TwoSample`.
All multiplicities and ties are retained, including the observed allocation.
"""
function fit_reference(group::Permutations, score, samples::AbstractVector{<:TwoSample})
    checked_samples(samples)
    (; K_A, K_B) = first(samples)
    K = K_A + K_B
    patterns = if group.reduce_symmetry && within_group_symmetric(score)
        allocations = if K_A == K_B && label_symmetric(score)
            (vcat(1, A) for A in combinations(2:K, K_A - 1))
        else
            combinations(1:K, K_A)
        end
        (vcat(A, setdiff(1:K, A)) for A in allocations)
    else
        permutations(1:K)
    end
    observed = Float64.(score.(samples))
    randomized_scores = [Float64(score(TwoSample(x.Z[p[1:K_A]], x.Z[p[K_A+1:end]])))
        for p in patterns, x in samples]
    !any(isnan, observed) && !any(isnan, randomized_scores) ||
        throw(ArgumentError("scores must not be NaN"))
    foreach(sort!, eachcol(randomized_scores))
    RandomizationFit(group, score, observed, FiniteRandomizationScores(randomized_scores))
end

"""
    fit_reference(CenteredRotations(), score, samples::AbstractVector{<:TwoSample})

Analytic Haar tails for `AbsMeanDifference` or a fitted `ModeratedTScore`.
Writing r_i^2 = (K-1)*tau-hat_i^2 and v = 1/K_A + 1/K_B,
delta-hat(HZ_i)^2/(v*r_i^2) ~ Beta(1/2, (K-2)/2).
Thus the absolute-difference tail at t is ccdf(Beta(1/2,(K-2)/2), t^2/(v*r_i^2)).
Moderated scores use equation (33). No random rotations are sampled.
"""
function fit_reference(group::CenteredRotations, score::Union{AbsMeanDifference,ModeratedTScore},
    samples::AbstractVector{<:TwoSample})
    checked_samples(samples)
    (; K_A, K_B) = first(samples)
    K = K_A + K_B
    v = inv(K_A) + inv(K_B)
    observed = Float64.(score.(samples))
    !any(isnan, observed) || throw(ArgumentError("scores must not be NaN"))
    reference = RotationReference(score, [(K - 1) * x.τ̂² for x in samples], K - 1, v)
    RandomizationFit(group, score, observed, reference)
end

"""
    fit_reference(ResidualRotations(), score, samples::AbstractVector{<:RegressionSample})

Analytic Haar tails for the absolute coefficient or moderated t-score in Section
7.2. The nuisance projection is fixed, not randomized or discarded
from the sample. Prior learning uses orbit df ν+1, whereas the score uses ν.
"""
function fit_reference(group::ResidualRotations, score::Union{AbsCoefficient,ModeratedTScore},
    samples::AbstractVector{<:RegressionSample})
    checked_samples(samples)
    design = first(samples).design
    observed = Float64.(score.(samples))
    !any(isnan, observed) || throw(ArgumentError("scores must not be NaN"))
    d = design.ν + 1
    reference = RotationReference(score, [d * x.τ̂² for x in samples], d, design.v)
    RandomizationFit(group, score, observed, reference)
end

# Finite-group enumeration and fast paths.

"""
    sign_grid(K, paired)

Columns are sign vectors epsilon in H_symm (Section 4.1), or representatives
with epsilon[1] = +1 when `paired`. A column acts by coordinatewise multiplication;
the returned matrix is not itself a group transformation.
The result is K by L. The identity is last; global reversal is first in the full
group. `fit_reference(SignFlips(), ...)` relies on this order to preserve their exact ties.
"""
function sign_grid(K, paired)
    d = paired ? K - 1 : K
    d < Sys.WORD_SIZE - 1 || throw(ArgumentError("sign group overflows Int"))
    patterns = Iterators.product(ntuple(_ -> (-1.0, 1.0), d)...)
    signs = stack(patterns; dims=2)
    paired ? vcat(ones(1, size(signs, 2)), signs) : signs
end

"""
    sign_statistics(score, samples, signs)

Return the L by n table with entry [h, i] = S-hat(epsilon_h .* Z_i), the terms
of (17), before sorting. The generic method evaluates the supplied callable.
No transformed sample is used to refit S-hat.
"""
function sign_statistics(score, samples, signs)
    [Float64(score(ReplicatedSample(samples[i].Z .* view(signs, :, h))))
        for h in axes(signs, 2), i in eachindex(samples)]
end

"""
For S-hat(z) = abs(mean(z)), (signs' * X)[h, i] is the signed sum for hypothesis i.
Here X[:, i] = Z_i, so X is K by n. Divide by K and take absolute values to obtain
the same table as generic enumeration, using BLAS matrix multiplication.
"""
function sign_statistics(::AbsMean, samples, signs)
    X = stack(x.Z for x in samples)
    abs.(transpose(signs) * X) ./ size(X, 1)
end

"""
Finite-group implementation of the weighted average in (13)-(14).
Each score in reference column j contributes weights[j]/(n*L) to every observed
threshold at or below it. Binning at the last such threshold and accumulating
from the right computes these contributions, including all tied thresholds.
The permutation is undone so the returned p-values retain hypothesis order.
"""
function pooled_pvalues(r::RandomizationFit{G,S,FiniteRandomizationScores}, weights) where {G,S}
    n = length(r.observed)
    order = sortperm(r.observed)
    thresholds = r.observed[order]
    mass = zeros(n)
    for (j, scores) in enumerate(eachcol(r.reference.sorted_scores)), u in scores
        k = searchsortedlast(thresholds, u)
        k > 0 && (mass[k] += weights[j])
    end
    p = zeros(n)
    p[order] = reverse(cumsum(reverse(mass))) ./ (n * size(r.reference.sorted_scores, 1))
    p
end
