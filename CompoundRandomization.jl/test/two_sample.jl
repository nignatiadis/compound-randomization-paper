@testset "Two-sample summaries" begin
    a, b = [1.0, 3], [4.0, 6, 8]
    x = TwoSample(a, b)
    @test x isa AbstractRandomizationSample
    @test x isa Empirikos.EBayesSample
    @test nobs(x) == 5
    @test x.K_A == 2
    @test x.K_B == 3
    @test x.K_A + x.K_B == nobs(x)
    @test x.δ̂ == -4
    @test AbsMeanDifference()(x) == 4
    @test x.σ̂² ≈ (var(a) + 2var(b)) / 3
    @test x.τ̂² ≈ var(vcat(a, b); corrected=true)
    @test 4x.τ̂² ≈ 3x.σ̂² + x.δ̂^2 / (1/2 + 1/3)
    summary = Empirikos.NormalChiSquareSample(x)
    @test summary.Z ≈ x.δ̂ / sqrt(1/2 + 1/3)
    @test summary.S² == x.σ̂²
    @test summary.ν == 3
    a .= 0
    b .= 0
    @test x.Z == [1, 3, 4, 6, 8]
    singleton = TwoSample([1.0], [2.0, 4])
    @test singleton.σ̂² == 2
    @test isfinite(ModeratedTScore(Dirac(1.0))(TwoSample(Float32[1, 2], Float32[3, 4])))
    @test_throws ArgumentError TwoSample(Float64[], [1.0, 2, 3])
    @test_throws ArgumentError TwoSample([1.0], [2.0])
    @test_throws ArgumentError TwoSample([NaN], [1.0, 2])
    @test_throws DimensionMismatch fit_reference(Permutations(), AbsMeanDifference(), [x, singleton])
    @test_throws ArgumentError fit_reference(CenteredRotations(), AbsMeanDifference(), typeof(x)[])
end

@testset "Exact two-sample permutations and ties" begin
    rng = MersenneTwister(513)
    score = AbsMeanDifference()
    for (K_A, K_B) in ((1, 2), (2, 2), (2, 3), (3, 2))
        samples = [TwoSample(rand(rng, -2:2, K_A), rand(rng, -2:2, K_B)) for _ in 1:5]
        reduced = fit_reference(Permutations(), score, samples)
        full = fit_reference(Permutations(reduce_symmetry=false), score, samples)
        L = binomial(K_A+K_B, K_A) ÷ (K_A == K_B ? 2 : 1)
        @test size(reduced.reference.sorted_scores) == (L, 5)
        @test size(full.reference.sorted_scores) == (factorial(K_A+K_B), 5)
        @test separate_pvalues(reduced) == separate_pvalues(full)
        @test compound_pvalues(reduced) == compound_pvalues(full)
        @test compound_pvalues(reduced) ≈ [
            mean(full.reference.sorted_scores .>= t) for t in full.observed]
        for i in 1:5, t in unique(full.reference.sorted_scores[:, i])
            @test orbit_tail(reduced, i, t) == mean(full.reference.sorted_scores[:, i] .>= t)
        end
        @test fit(CompoundBH(), reduced).rj_idx == threshold_rejections(reduced, 0.1)
    end
    samples = [TwoSample([0, 0], [1, 1]), TwoSample([0, 1], [0, 1]), TwoSample([2, 2], [2, 2])]
    reference = fit_reference(Permutations(), score, samples)
    @test reference.reference.sorted_scores[:, 1] == [0, 0, 1]
    @test separate_pvalues(reference) == [1/3, 1, 1]
    @test compound_pvalues(reference) == [2/9, 1, 1]
    @test orbit_tail(reference, 1, 1.0) == 1/3
    @test orbit_tail(reference, 1, nextfloat(1.0)) == 0
    @test orbit_tail(reference, 1, 0.0) == 1

    order_sensitive = x -> abs(x.Z[1])
    custom = fit_reference(Permutations(), order_sensitive, samples)
    @test size(custom.reference.sorted_scores, 1) == 24
    @test separate_pvalues(custom) == [mean(abs.(x.Z) .>= order_sensitive(x)) for x in samples]
    # Canonical summary evaluation preserves known ties even for ill-scaled values.
    ill_scaled = [TwoSample([1e16, 1.0, -1e16], [2.0, 3])]
    @test separate_pvalues(fit_reference(Permutations(), score, ill_scaled)) ==
        separate_pvalues(fit_reference(Permutations(reduce_symmetry=false), score, ill_scaled))
end

# Keep within-group symmetry, but deliberately do not declare label symmetry.
struct AllocationOnlyScore{S}
    score::S
end
(score::AllocationOnlyScore)(x) = score.score(x)
CR.within_group_symmetric(::AllocationOnlyScore) = true

# Label symmetry alone must not enable allocation enumeration.
struct LabelOnlyScore end
(::LabelOnlyScore)(x) = abs(x.Z[1] - x.Z[x.K_A + 1])
CR.label_symmetric(::LabelOnlyScore) = true

@testset "Complementary permutation allocations" begin
    rng = MersenneTwister(711)
    for K_A in (2, 3), score in (AbsMeanDifference(),
        ModeratedTScore(Dirac(1.0)),
        ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0, 8.0)),
        ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0, 0.0)))
        samples = [TwoSample(rand(rng, -2:2, K_A), rand(rng, -2:2, K_A)) for _ in 1:4]
        push!(samples, TwoSample(zeros(Int, K_A), ones(Int, K_A)))
        push!(samples, TwoSample(ones(Int, K_A), ones(Int, K_A)))
        reduced = fit_reference(Permutations(), score, samples)
        allocations = fit_reference(Permutations(), AllocationOnlyScore(score), samples)
        full = fit_reference(Permutations(reduce_symmetry=false), score, samples)
        @test label_symmetric(score)
        @test !label_symmetric(AllocationOnlyScore(score))
        @test size(reduced.reference.sorted_scores, 1) == binomial(2K_A, K_A) ÷ 2
        @test repeat(reduced.reference.sorted_scores; inner=(2, 1)) == allocations.reference.sorted_scores
        @test repeat(allocations.reference.sorted_scores; inner=(factorial(K_A)^2, 1)) == full.reference.sorted_scores
        @test separate_pvalues(reduced) == separate_pvalues(allocations) == separate_pvalues(full)
        @test compound_pvalues(reduced) == compound_pvalues(allocations) == compound_pvalues(full)
        weights = collect(range(0.5, 1.5; length=length(samples)))
        @test CR.pooled_pvalues(reduced, weights) ≈ CR.pooled_pvalues(full, weights)
        @test fit(CompoundBH(), reduced).rj_idx == fit(CompoundBH(), full).rj_idx
    end
    samples = [TwoSample([0, 1], [2, 3])]
    @test !within_group_symmetric(LabelOnlyScore())
    @test size(fit_reference(Permutations(), LabelOnlyScore(), samples).reference.sorted_scores, 1) == 24
    @test !label_symmetric(x -> abs(x.δ̂))
end

@testset "Centered rotations for two samples" begin
    rng = MersenneTwister(881)
    for (K_A, K_B) in ((1, 2), (2, 3), (3, 3), (2, 5))
        K, v = K_A + K_B, inv(K_A) + inv(K_B)
        samples = [TwoSample(randn(rng, K_A), randn(rng, K_B)) for _ in 1:15]
        expected = [2ccdf(TDist(K-2), abs(x.δ̂) / sqrt(v*x.σ̂²)) for x in samples]
        for score in (AbsMeanDifference(), ModeratedTScore(Dirac(1.3)),
            ModeratedTScore(Empirikos.InverseScaledChiSquare(1.2, 8.0)),
            ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0, 0.0)))
            reference = fit_reference(CenteredRotations(), score, samples)
            @test reference.reference.dimension == K - 1
            @test reference.reference.contrast_variance == v
            @test separate_pvalues(reference) ≈ expected atol=1e-12
            @test fit(CompoundBH(), reference).rj_idx == threshold_rejections(reference, 0.1)
        end
        reference = fit_reference(CenteredRotations(), AbsMeanDifference(), samples)
        t = 0.7
        manual = [ccdf(Beta(0.5, (K-2)/2), min(1, t^2 / (v * sum(abs2, x.Z .- mean(x.Z)))))
            for x in samples]
        @test [orbit_tail(reference, i, t) for i in eachindex(samples)] ≈ manual
    end

    # Uniform angles parametrize the centered rotation orbit when K=3.
    x = TwoSample([1.0], [2.0, 4])
    r = fit_reference(CenteredRotations(), AbsMeanDifference(), [x])
    u, w = [1, -1, 0] / sqrt(2), [1, 1, -2] / sqrt(6)
    radius = norm(x.Z .- mean(x.Z))
    differences = map(0:4095) do j
        angle = 2pi * j / 4096
        z = mean(x.Z) .+ radius .* (cos(angle) .* u .+ sin(angle) .* w)
        abs(z[1] - mean(z[2:3]))
    end
    for t in (0.25, 1.0, 2.0)
        @test orbit_tail(r, 1, t) ≈ mean(differences .>= t) atol=0.001
    end
    constant = fit_reference(CenteredRotations(), AbsMeanDifference(), [TwoSample([4, 4], [4, 4])])
    @test separate_pvalues(constant) == [1]
    @test orbit_tail(constant, 1, 0.1) == 0
    @test_throws MethodError fit_reference(OrthogonalRotations(), AbsMeanDifference(), [x])
    @test_throws MethodError fit_reference(CenteredRotations(), s -> abs(s.Z[1]), [x])
end

@testset "Two-sample end-to-end fitting and orbit learning" begin
    rng = MersenneTwister(31)
    A, B = randn(rng, 40, 2), randn(rng, 40, 3)
    samples = TwoSample.(eachrow(A), eachrow(B))
    estimator = ModeratedTScore(Empirikos.QuantileLimma())
    prior = Empirikos.fit_prior(estimator.prior,
        [Empirikos.ScaledChiSquareSample(x.τ̂², 4) for x in samples])
    for group in (Permutations(), CenteredRotations())
        @test all(orbit_variance(group, x).ν == 4 for x in samples)
        learned = fit_statistic(group, estimator, samples)
        @test learned.(samples) == ModeratedTScore(prior).(samples)
        for statistic in (AbsMeanDifference(), estimator)
            fixed = statistic === estimator ? learned : statistic
            reference = fit_reference(group, fixed, samples)
            for procedure in (CompoundBH(), SeparateBH(), DDR())
                method = MultipleRandomizationTest(; group, statistic, procedure)
                @test fit(method, samples) == fit(procedure, reference)
            end
        end
    end
    integer_samples = [TwoSample([0, 2], [3, 4, 6]), TwoSample([1, 2], [0, 0, 4])]
    shifted = [TwoSample(x.Z[1:2] .+ 10, x.Z[3:5] .+ 10) for x in integer_samples]
    for group in (Permutations(), CenteredRotations())
        @test compound_pvalues(fit_reference(group, AbsMeanDifference(), integer_samples)) ≈
            compound_pvalues(fit_reference(group, AbsMeanDifference(), shifted)) atol=1e-14
    end
end
