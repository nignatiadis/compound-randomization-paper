"""
A procedure making rejection decisions across multiple hypotheses.
Compound BH, separate BH, and DDR act on a randomization reference fit through
`fit(procedure, ::RandomizationFit)`.
"""
abstract type AbstractMultipleTestingProcedure end

Base.@kwdef struct CompoundBH <: AbstractMultipleTestingProcedure
    α::Float64 = 0.1
end

Base.@kwdef struct SeparateBH <: AbstractMultipleTestingProcedure
    α::Float64 = 0.1
end

"""
    DDR(; α=0.1, τ=α/10)

Procedure 2: find s_tau from the mean orbit-tail odds, then compute
P_i = mean_j[xi_j(S_i) / (1 - xi_j(s_tau))] as in equation (14).
Apply BH at level alpha after censoring p-values above tau to one.
"""
Base.@kwdef struct DDR <: AbstractMultipleTestingProcedure
    α::Float64 = 0.1
    τ::Float64 = α / 10
end

"""
Selective SeqStep+ for a fixed group of order two (Procedure 3).
Observed and calibration scores must be finite and nonnegative.
Thresholds are strictly positive, so zero signed scores are never rejected.
"""
Base.@kwdef struct SeqStepPlus <: AbstractMultipleTestingProcedure
    α::Float64 = 0.1
end

function check_level(alpha)
    0 < alpha < 1 || throw(ArgumentError("level must lie strictly between zero and one"))
end

function bh_result(method, p; tau = nothing)
    check_level(method.α)
    isnothing(tau) || check_level(tau)
    adjp = adjust(isnothing(tau) ? p : ifelse.(p .<= tau, p, 1.0), BenjaminiHochberg())
    rejected = adjp .<= method.α
    cutoff = any(rejected) ? maximum(p[rejected]) : 0.0
    (; method, pvalue = p, adjp, cutoff,
        rj_idx = rejected, total_rejections = count(rejected))
end

fit(method::CompoundBH, r::RandomizationFit) = bh_result(method, compound_pvalues(r))
fit(method::SeparateBH, r::RandomizationFit) = bh_result(method, separate_pvalues(r))

"""
Mean tail odds in the definition of s_tau (Procedure 2).
For finite references, evaluate the right limit: xi_j(s+) = P(S_j > s).
This locates the infimum even when it is not attained. The final DDR weights
use `orbit_tail` instead, retaining the atom at s_tau.
"""
function ddr_tail_odds(reference::FiniteRandomizationScores, s)
    mean(eachcol(reference.sorted_scores)) do scores
        nat_or_below = searchsortedlast(scores, s)
        nabove = length(scores) - nat_or_below
        nabove / nat_or_below
    end
end

function ddr_tail_odds(reference::RotationReference, s)
    mean(eachindex(reference.norm2)) do i
        xi = orbit_tail(reference, i, s)
        xi / (1 - xi)
    end
end

"""
Scan the distinct finite-reference scores from largest to smallest.
The last qualifying score before the tail odds exceed tau is s_tau.
"""
function ddr_cutoff(reference::FiniteRandomizationScores, tau)
    support = sort!(unique(vec(reference.sorted_scores)); rev=true)
    cutoff = Inf
    for s in support
        ddr_tail_odds(reference, s) > tau && break
        cutoff = s
    end
    cutoff
end

"""Largest attainable score on hypothesis i's rotation orbit, used to bracket DDR."""
rotation_maximum(r::RotationReference{AbsMean}, i) =
    sqrt(r.contrast_variance * r.norm2[i])
rotation_maximum(r::RotationReference{AbsMeanDifference}, i) =
    sqrt(r.contrast_variance * r.norm2[i])
rotation_maximum(r::RotationReference{<:ModeratedTScore{<:Dirac}}, i) =
    sqrt(r.norm2[i] / r.statistic.prior.value)
function rotation_maximum(r::RotationReference{<:ModeratedTScore{<:Empirikos.InverseScaledChiSquare}}, i)
    (; ν, σ²) = r.statistic.prior
    r2 = r.norm2[i]
    iszero(r2) ? 0.0 : sqrt(r2 * (r.dimension - 1 + ν) / (ν * σ²))
end

"""For continuous rotation scores, solve mean_j[xi_j(s)/(1-xi_j(s))] = tau."""
function ddr_cutoff(reference::RotationReference, tau)
    upper = maximum(i -> rotation_maximum(reference, i), eachindex(reference.norm2))
    iszero(upper) && return upper
    if isinf(upper)
        upper = 1.0
        while ddr_tail_odds(reference, upper) > tau
            upper *= 2
            isfinite(upper) || error("could not bracket the rotation DDR cutoff")
        end
    end
    Roots.find_zero(s -> ddr_tail_odds(reference, s) - tau, (0.0, upper), Roots.Bisection())
end

function fit(method::DDR, r::RandomizationFit)
    check_level(method.α)
    check_level(method.τ)
    cutoff = ddr_cutoff(r.reference, method.τ)
    weights = [inv(1 - orbit_tail(r, i, cutoff)) for i in eachindex(r.observed)]
    # Degenerate cutoff atoms may give a zero denominator. Conservatively reject
    # nothing, rather than silently turn an undefined 0/0 contribution into zero.
    p = if all(isfinite, weights)
        pooled_pvalues(r, weights)
    else
        fill(Inf, length(weights))
    end
    (; bh_result(method, p; tau = method.τ)..., orbit_cutoff = cutoff, weights)
end

"""Score ties take -calibration; positive magnitude ties enter together."""
function fit(method::SeqStepPlus, r::RandomizationFit{G,S,R}) where {G,S,R<:InvolutionReference}
    check_level(method.α)
    observed, calibration = r.reference.observed, r.reference.calibration
    all(isfinite, observed) && all(isfinite, calibration) || throw(ArgumentError("SeqStep+ needs finite scores"))
    all(>=(0), observed) && all(>=(0), calibration) || throw(ArgumentError("SeqStep+ needs nonnegative scores"))
    # Equality takes -calibration, as in Procedure 3.
    signed = ifelse.(observed .> calibration, observed, .-calibration)
    order = sortperm(abs.(signed); rev = true)
    magnitude = abs.(signed[order])
    nrejections = cumsum(signed[order] .> 0)
    ncalibration = cumsum(signed[order] .< 0)
    fdr_hat = (1 .+ ncalibration) ./ max.(nrejections, 1)
    # Evaluate thresholds only after including every score tied at that magnitude.
    k = findlast(eachindex(order)) do j
        magnitude[j] > 0 && fdr_hat[j] <= method.α &&
            (j == length(order) || magnitude[j] != magnitude[j + 1])
    end
    cutoff = isnothing(k) ? Inf : magnitude[k]
    rejected = isnothing(k) ? falses(length(observed)) : signed .>= cutoff
    (; method, statistic = r.statistic, observed, calibration, cutoff,
        rj_idx = rejected, total_rejections = count(rejected))
end
