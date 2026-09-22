using CompoundRandomization
using Distributions
using Empirikos
using LinearAlgebra
using MultipleTesting
using Random
using Statistics
using Test

const CR = CompoundRandomization

"""Procedure 1': independent threshold formulation used to check compound BH."""
function threshold_rejections(r::RandomizationFit, alpha)
    CR.check_level(alpha)
    admissible = [s for s in unique(r.observed) if
        sum(orbit_tail(r, j, s) for j in eachindex(r.observed)) /
            max(count(>=(s), r.observed), 1) <= alpha]
    isempty(admissible) ? falses(length(r.observed)) : r.observed .>= minimum(admissible)
end

include("testing.jl")
include("interfaces.jl")
include("two_sample.jl")
include("regression.jl")

@testset "Sample and baseline interface" begin
    @test isfile(SENS(seed=123).script)
    @test SENS(seed=123, script="custom.R").script == "custom.R"
    x = ReplicatedSample([1.0, 2, 3])
    @test x isa Empirikos.EBayesSample
    @test nobs(x) == 3
    @test mean(x) == 2
    @test var(x) == 1
    ordinary = Empirikos.SimultaneousTTest(α=0.1)
    @test fit(ordinary, [x]).pvalue ≈ [2ccdf(TDist(2), 2sqrt(3))]
    s = [x, ReplicatedSample([-2.0, 0, 1])]
    Z = [Empirikos.NormalChiSquareSample(sqrt(3) * mean(y), var(y), 2) for y in s]
    @test fit(ordinary, s) == fit(ordinary, Z)
    oracle = Empirikos.EmpiricalPartiallyBayesTTest(
        prior=Empirikos.InverseScaledChiSquare(1.0, 10.0), α=0.1, solver=nothing)
    @test fit(oracle, s) == fit(oracle, Z)
    @test fit(oracle, [x]) == fit(oracle, Z[1:1])
    @test_throws ArgumentError fit(ordinary, [ReplicatedSample([0.0])])
    @test_throws ArgumentError fit(ordinary, typeof(x)[])
    @test_throws ArgumentError fit(ordinary, [ReplicatedSample([NaN, 1.0])])
    @test_throws DimensionMismatch fit(ordinary, [x, ReplicatedSample(ones(2))])
    @test_throws ArgumentError fit(CompoundBH(α=0), fit_reference(SignFlips(), AbsMean(), [x]))
    @test_throws MethodError fit_reference(OrthogonalRotations(), s -> sum(s.Z), [x])
    X = [1.0 2 3; -2 0 1]
    wrapped = ReplicatedSample.(eachrow(X))
    @test fit(ordinary, wrapped).pvalue == fit(ordinary, s).pvalue
    @test_throws MethodError fit_reference(SignFlips(), AbsMean(), X)
    @test_throws MethodError fit_reference(SignFlips(), AbsMean(), x)
    @test_throws MethodError fit_reference(OrthogonalRotations(), AbsMean(), X)
    @test_throws MethodError fit(SeqStepPlus(), X, AbsMean())
    @test_throws MethodError fit_statistic(SignFlips(), ModeratedTScore(Empirikos.QuantileLimma()), X)
end

@testset "Cached sample summaries" begin
    for z in ([1, 2, 3], Float32[1, 2, 3], [1e8 - 1, 1e8, 1e8 + 1])
        x = ReplicatedSample(z)
        @test mean(x) == mean(z)
        @test var(x) == var(z; corrected=true)
        @test x.τ̂² == sum(abs2, z) / length(z)
        @test x.Z == z
        @test x.Z !== z
        z[1] = 0
        @test mean(x) == mean(x.Z)
        @test var(x) == var(x.Z; corrected=true)
        @test x.τ̂² == sum(abs2, x.Z) / nobs(x)
    end
    X = [1.0 2 3; -1 0 1]
    s = ReplicatedSample.(eachrow(X))
    X .= 0
    @test mean.(s) == [2.0, 0.0]
    @test var.(s) == [1.0, 1.0]
    @test getproperty.(s, :τ̂²) == [14 / 3, 2 / 3]
end

@testset "Actual moderated t-score" begin
    rng = MersenneTwister(240)
    for K in (3, 5, 13), nu in (0.0, 2.0, 10.0)
        x = ReplicatedSample(randn(rng, K) .+ 0.5)
        score = ModeratedTScore(Empirikos.InverseScaledChiSquare(2.0, nu))
        expected = sqrt(K) * abs(mean(x)) /
            sqrt(((K - 1) * var(x) + nu * 2.0) / (K - 1 + nu))
        @test score(x) ≈ expected
        @test ModeratedTScore(Dirac(2.0))(x) ≈ sqrt(K) * abs(mean(x)) / sqrt(2.0)
    end
    score = ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0, 0.0))
    @test score(ReplicatedSample(ones(3))) == Inf
    @test score(ReplicatedSample(zeros(3))) == 0
    x = ReplicatedSample(1e8 .+ [-1.0, 0, 1])
    @test score(x) ≈ sqrt(3) * mean(x) / std(x.Z)
    constant = [ReplicatedSample(ones(3))]
    r = fit_reference(SignFlips(), score, constant)
    @test separate_pvalues(r) == [0.25]
    @test_throws ArgumentError fit(SeqStepPlus(), fit_reference(InvolutionGroup(), score, constant))
    @test_throws ArgumentError fit(SeqStepPlus(), fit_reference(InvolutionGroup(), score, repeat(constant, 20)))
    r = fit_reference(OrthogonalRotations(), score, ReplicatedSample.(eachrow([1.0 2 3; -1 1 2])))
    result = fit(DDR(), r)
    tails = [orbit_tail(r, i, result.orbit_cutoff) for i in 1:2]
    @test mean(tails ./ (1 .- tails)) ≈ 0.01 atol=1e-12
end

@testset "Fixed-variance scores give equivalent randomization tests" begin
    rng = MersenneTwister(813)
    for K in (3, 5, 9), variance in (0.5, 2.0)
        X = randn(rng, 24, K)
        X[1:12, :] .+= 3.0
        samples = ReplicatedSample.(eachrow(X))
        scores = (AbsMean(), ModeratedTScore(Dirac(variance)),
            ModeratedTScore(Empirikos.InverseScaledChiSquare(variance, Inf)))
        @test scores[3].prior isa Dirac
        scale = sqrt(K / variance)
        @test scores[2].(samples) ≈ scale .* scores[1].(samples)
        @test scores[3].(samples) == scores[2].(samples)
        large_nu = ModeratedTScore(Empirikos.InverseScaledChiSquare(variance, 1e10))
        @test large_nu.(samples) ≈ scores[2].(samples) rtol=1e-7
        for group in (SignFlips(), OrthogonalRotations())
            references = [fit_reference(group, score, samples) for score in scores]
            for reference in references[2:3]
                @test separate_pvalues(reference) ≈ separate_pvalues(references[1])
                @test compound_pvalues(reference) ≈ compound_pvalues(references[1])
            end
            limiting = fit_reference(group, large_nu, samples)
            @test separate_pvalues(limiting) ≈ separate_pvalues(references[2]) rtol=1e-7
            @test compound_pvalues(limiting) ≈ compound_pvalues(references[2]) rtol=1e-7
            for alpha in (0.1, 0.2), procedure in
                (CompoundBH(α=alpha), SeparateBH(α=alpha), DDR(α=alpha))
                results = [fit(MultipleRandomizationTest(; group, statistic=score, procedure), samples)
                    for score in scores]
                for result in results[2:3]
                    @test result.pvalue ≈ results[1].pvalue
                    @test result.rj_idx == results[1].rj_idx
                end
            end
        end
        for alpha in (0.1, 0.2)
            results = [fit(MultipleRandomizationTest(group=InvolutionGroup(),
                statistic=score, procedure=SeqStepPlus(α=alpha)), samples) for score in scores]
            for result in results[2:3]
                @test result.rj_idx == results[1].rj_idx
                @test result.cutoff ≈ scale * results[1].cutoff
            end
        end
    end
end

@testset "Exact sign group" begin
    for K in (2, 3, 5, 13)
        H = CR.sign_grid(K, false)
        paired = CR.sign_grid(K, true)
        @test size(H) == (K, 2^K)
        @test size(paired) == (K, 2^(K - 1))
        @test all(x -> x == -1 || x == 1, H)
        @test length(Set(Tuple.(eachcol(H)))) == 2^K
        @test H[:, 1] == -ones(K)
        @test H[:, end] == ones(K)
        @test all(==(1), paired[1, :])
        @test paired[:, end] == ones(K)
        @test Set(Tuple.(eachcol(hcat(paired, -paired)))) == Set(Tuple.(eachcol(H)))
    end
end

@testset "Sign symmetry reduction and dispatched AbsMean versus full enumeration" begin
    rng = MersenneTwister(241)
    for K in (2, 3, 4, 5, 6, 13), score in (AbsMean(),
        ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0, 10.0)),
        ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0, 0.0)),
        ModeratedTScore(Dirac(2.0)))
        X = Float64.(rand(rng, -3:3, 8, K))
        X[1, :] .= 1
        X[2, :] .= -1
        X[3, :] .= X[1, :]
        X[4, :] .= 0
        samples = ReplicatedSample.(eachrow(X))
        fast = fit_reference(SignFlips(), score, samples)
        slow = fit_reference(SignFlips(), s -> score(s), samples)
        full = fit_reference(SignFlips(reduce_symmetry=false), score, samples)
        @test size(fast.reference.sorted_scores, 1) * 2 == size(slow.reference.sorted_scores, 1)
        @test compound_pvalues(fast) == compound_pvalues(slow) == compound_pvalues(full)
        @test separate_pvalues(fast) == separate_pvalues(slow) == separate_pvalues(full)
        reference = [mean([orbit_tail(fast, j, t) for j in eachindex(fast.observed)])
            for t in fast.observed]
        @test compound_pvalues(fast) ≈ reference
        weights = collect(range(0.2, 1.2; length=length(fast.observed)))
        weighted_reference = [mean([weights[j] * orbit_tail(fast, j, t)
            for j in eachindex(weights)]) for t in fast.observed]
        @test CR.pooled_pvalues(fast, weights) ≈ weighted_reference
        @test CR.pooled_pvalues(fast, 2ones(length(weights))) ≈ 2compound_pvalues(fast)
        for alpha in (0.01, 0.1 / 1.93, 0.1, 0.25, 0.5)
            @test fit(CompoundBH(α=alpha), fast).rj_idx == threshold_rejections(slow, alpha)
        end
    end
    signed = fit_reference(SignFlips(), s -> sum(s.Z), ReplicatedSample.(eachrow([1.0 1; 1 -1])))
    @test compound_pvalues(signed) == [1 / 4, 3 / 4]
    @test size(signed.reference.sorted_scores, 1) == 4
    tied = fit_reference(SignFlips(), AbsMean(), ReplicatedSample.(eachrow(vcat(ones(2, 3), zeros(2, 3)))))
    @test compound_pvalues(tied) == [1 / 8, 1 / 8, 1, 1]
    @test !any(fit(CompoundBH(α=prevfloat(0.25)), tied).rj_idx)
    @test fit(CompoundBH(α=0.25), tied).rj_idx == [true, true, false, false]
end

@testset "Orbit prior uses K df; ordinary Limma uses K-1" begin
    rng = MersenneTwister(242)
    X = randn(rng, 150, 5) .* exp.(randn(rng, 150))
    s = ReplicatedSample.(eachrow(X))
    for estimator in (Empirikos.MomentLimma(),
        Empirikos.QuantileLimma(quantile_p=(0.2, 0.8)), Empirikos.WinsorizedLimma())
        score = fit_statistic(SignFlips(), ModeratedTScore(estimator), s)
        reference = Empirikos.fit_prior(estimator,
            Empirikos.ScaledChiSquareSample.([sum(abs2, x.Z) / 5 for x in s], 5))
        @test score.prior == reference
        @test_throws MethodError fit_reference(SignFlips(), ModeratedTScore(estimator), s)
        Z = [Empirikos.NormalChiSquareSample(sqrt(5) * mean(x), var(x), 4) for x in s]
        expected = fit(Empirikos.EmpiricalPartiallyBayesTTest(
            prior=Empirikos.Limma(method=estimator), α=0.1, solver=nothing), Z)
        actual = fit(Empirikos.EmpiricalPartiallyBayesTTest(
            prior=Empirikos.Limma(method=estimator), α=0.1, solver=nothing), s)
        @test actual.pvalue == expected.pvalue
        @test actual.rj_idx == expected.rj_idx
    end
end

@testset "Rotations recover t-tests, including constant rows" begin
    rng = MersenneTwister(243)
    for K in (2, 3, 4, 5, 6, 7, 9, 11, 13, 16)
        X = randn(rng, 30, K) .* exp.(randn(rng, 30))
        X[1:5, :] .+= 4
        X[1, :] .= 1
        X[2, :] .= -1
        X[3, :] .= 0
        X[3, 1:2] .= [1, -1]
        s = ReplicatedSample.(eachrow(X))
        expected = fit(Empirikos.SimultaneousTTest(α=0.1), s).pvalue
        for prior in (Empirikos.InverseScaledChiSquare(0.01, 0.0),
            Empirikos.InverseScaledChiSquare(100.0, 0.0))
            r = fit_reference(OrthogonalRotations(), ModeratedTScore(prior), s)
            @test compound_pvalues(r) ≈ expected atol=2e-12
            @test separate_pvalues(r) ≈ expected atol=2e-12
            @test compound_pvalues(r)[1:3] == [0, 0, 1]
            for alpha in (0.01, 0.1 / 1.93, 0.1, 0.5)
                @test fit(CompoundBH(α=alpha), r).rj_idx == fit(Empirikos.SimultaneousTTest(α=alpha), s).rj_idx
                @test fit(CompoundBH(α=alpha), r).rj_idx == threshold_rejections(r, alpha)
            end
        end
        for score in (AbsMean(), ModeratedTScore(Dirac(2.0)),
            ModeratedTScore(Empirikos.InverseScaledChiSquare(2.0, 10.0)))
            r = fit_reference(OrthogonalRotations(), score, s)
            @test separate_pvalues(r) ≈ expected atol=2e-12
            @test fit(CompoundBH(), r).rj_idx == threshold_rejections(r, 0.1)
        end
    end
    zero = fit_reference(OrthogonalRotations(), AbsMean(), ReplicatedSample.(eachrow(zeros(3, 4))))
    @test compound_pvalues(zero) == ones(3)
    @test separate_pvalues(zero) == ones(3)
end

@testset "BH and tau-censored BH" begin
    for p in ([0.0, 0.01, 0.01, nextfloat(0.01), 0.2, 1.0],
        [0.2, 0.3, 0.8], [0.0, 0.0, 0.0]), alpha in (0.01, 0.1, 0.5)
        ordinary = CR.bh_result(CompoundBH(α=alpha), p)
        @test ordinary.adjp == adjust(p, BenjaminiHochberg())
        @test ordinary.rj_idx == (ordinary.adjp .<= alpha)
        for tau in (0.01, 0.1, 0.3)
            result = CR.bh_result(DDR(α=alpha, τ=tau), p; tau)
            sorted = sort(p)
            k = findlast(i -> sorted[i] <= min(alpha * i / length(p), tau), eachindex(sorted))
            expected = isnothing(k) ? falses(length(p)) : p .<= sorted[k]
            @test result.rj_idx == expected
            @test result.adjp == adjust(ifelse.(p .<= tau, p, 1.0), BenjaminiHochberg())
            @test result.pvalue == p
            @test result.cutoff == (any(expected) ? maximum(p[expected]) : 0.0)
        end
    end
    result = CR.bh_result(DDR(), [Inf, Inf]; tau=0.01)
    @test result.adjp == ones(2)
    @test !any(result.rj_idx)
end

@testset "DDR finite cutoff atom and rotation Beta formula" begin
    rng = MersenneTwister(244)
    # Ascending support enumeration independently checks the descending scan.
    fixtures = [zeros(4, 3), fill(2.0, 4, 3),
        [0.0 0.0; 1.0 1.0; 1.0 nextfloat(1.0); 2.0 2.0],
        [0.0 1.0; 1.0 2.0; Inf Inf],
        [0.0 0.0; 1e-100 1e100; 1e200 1e300]]
    append!(fixtures, [sort(Float64.(rand(rng, 0:5, 8, 3)); dims=1) for _ in 1:20])
    for scores in fixtures, tau in (0.01, 0.1, 0.5)
        reference = CR.FiniteRandomizationScores(scores)
        support = sort(unique(vec(scores)))
        odds = [mean(eachcol(scores)) do column
            xi = count(>(s), column) / length(column)
            xi / (1 - xi)
        end for s in support]
        @test CR.ddr_tail_odds.(Ref(reference), support) ≈ odds
        @test CR.ddr_cutoff(reference, tau) == support[findfirst(<=(tau), odds)]
    end
    for K in (3, 4, 5, 6), tau in (0.01, 0.1)
        X = Float64.(rand(rng, -4:4, 12, K))
        samples = ReplicatedSample.(eachrow(X))
        r = fit_reference(SignFlips(), ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0, 10.0)), samples)
        values = r.reference.sorted_scores
        support = sort(unique(vec(values)))
        odds(t) = mean([begin
            xi = count(>(t), col) / length(col)
            xi / (1 - xi)
        end for col in eachcol(values)])
        cutoff = first(s for s in support if odds(s) <= tau)
        weights = [inv(1 - count(>=(cutoff), col) / length(col)) for col in eachcol(values)]
        result = fit(DDR(τ=tau), r)
        @test result.orbit_cutoff == cutoff
        if all(isfinite, weights)
            expected = [mean([weights[i] * count(>=(s), col) / length(col)
                for (i, col) in enumerate(eachcol(values))]) for s in r.observed]
            @test result.pvalue ≈ expected atol=1e-12
        else
            @test all(isinf, result.pvalue)
        end
        @test all(result.pvalue .>= compound_pvalues(r) .- 1e-12)
    end
    for K in (3, 5, 6, 13)
        X = randn(rng, 30, K)
        X[1:6, :] .+= 4
        score = ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0, 10.0))
        samples = ReplicatedSample.(eachrow(X))
        r = fit_reference(OrthogonalRotations(), score, samples)
        result = fit(DDR(), r)
        tails = [orbit_tail(r, i, result.orbit_cutoff) for i in 1:30]
        @test mean(tails ./ (1 .- tails)) ≈ 0.01 atol=1e-12
        expected = [mean([orbit_tail(r, j, s) / (1 - tails[j]) for j in 1:30]) for s in r.observed]
        @test result.pvalue ≈ expected atol=1e-12
        @test !any(result.rj_idx .& (result.pvalue .> 0.01))
    end
end

@testset "SeqStep+ direct threshold, ties and half split" begin
    rng = MersenneTwister(245)
    for K in (3, 4, 5, 13), alpha in (0.1, 0.5)
        X = Float64.(rand(rng, -3:3, 30, K))
        score = ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0, 10.0))
        samples = ReplicatedSample.(eachrow(X))
        method = MultipleRandomizationTest(group=InvolutionGroup(), statistic=score,
            procedure=SeqStepPlus(α=alpha))
        result = fit(method, samples)
        signs = [j <= cld(K, 2) ? 1 : -1 for j in 1:K]
        observed = [score(ReplicatedSample(collect(row))) for row in eachrow(X)]
        calibration = [score(ReplicatedSample(row .* signs)) for row in eachrow(X)]
        signed = ifelse.(observed .> calibration, observed, .-calibration)
        admissible = [t for t in unique(abs.(signed)) if t > 0 &&
            (1 + count(signed .<= -t)) / max(count(signed .>= t), 1) <= alpha]
        expected = isempty(admissible) ? falses(30) : signed .>= minimum(admissible)
        @test result.rj_idx == expected
    end
    @test !any(fit(SeqStepPlus(), fit_reference(InvolutionGroup(), AbsMean(),
        ReplicatedSample.(eachrow(zeros(20, 4))))).rj_idx)
end

@testset "Population local-fdr oracle calibration" begin
    X = randn(MersenneTwister(246), 100, 5)
    samples = ReplicatedSample.(eachrow(X))
    for pi1 in (0.005, 0.025, 0.1)
        oracle = LocalFDROracle(π1=pi1, λ=10, ν0=10, s0²=1)
        result = fit(oracle, samples)
        dist = TDist(14)
        null = (1 - pi1) * 2ccdf(dist, result.t_cutoff)
        alt = pi1 * 2ccdf(dist, result.t_cutoff / sqrt(11))
        @test null / (null + alt) ≈ 0.1 atol=1e-12
        @test result.rj_idx == (result.lfdr .<= result.lfdr_cutoff)
        score = ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0, 10.0))
        @test result.rj_idx == (score.(samples) .>= result.t_cutoff)
        # Compute (5) independently from the full K-dimensional data likelihood.
        f0 = MvTDist(10.0, zeros(5), Matrix{Float64}(I, 5, 5))
        f1 = MvTDist(10.0, zeros(5), Matrix{Float64}(I, 5, 5) + 2ones(5, 5))
        expected = [(1 - pi1) * pdf(f0, z) /
            ((1 - pi1) * pdf(f0, z) + pi1 * pdf(f1, z)) for z in eachrow(X)]
        @test result.lfdr ≈ expected
    end
    none = fit(LocalFDROracle(π1=1e-10, λ=0.01, ν0=10, s0²=1), samples)
    @test none.total_rejections == 0
    @test none.lfdr_cutoff == 0
    above_pi0 = fit(LocalFDROracle(π1=0.95, λ=10, ν0=10, s0²=1), samples)
    @test above_pi0.lfdr_cutoff == 1
    @test above_pi0.total_rejections == 100
    all_rejected = fit(LocalFDROracle(π1=0.5, λ=10, ν0=10, s0²=1, α=0.5), samples)
    @test all_rejected.lfdr_cutoff == 1
    @test all_rejected.total_rejections == 100

end
