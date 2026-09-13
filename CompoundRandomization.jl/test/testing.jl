@testset "End-to-end multiple randomization tests" begin
    @test SignFlips <: AbstractRandomizationGroup
    @test OrthogonalRotations <: AbstractRandomizationGroup
    @test CompoundBH <: AbstractMultipleTestingProcedure
    @test SeparateBH <: AbstractMultipleTestingProcedure
    @test DDR <: AbstractMultipleTestingProcedure
    @test CompoundRandomization.FiniteRandomizationScores <: AbstractRandomizationReference
    @test CompoundRandomization.RotationReference <: AbstractRandomizationReference
    @test_throws MethodError CompoundRandomization.RotationReference(AbsMean(), [1.0], 3)
    rng = MersenneTwister(761)
    samples = ReplicatedSample.(eachrow(randn(rng, 40, 5)))
    estimator = ModeratedTScore(Empirikos.QuantileLimma())
    @test_throws MethodError fit(estimator, SignFlips(), samples)
    @test_throws MethodError fit(SignFlips(), samples, AbsMean())
    @test_throws MethodError fit_statistic(estimator, samples)
    learned = fit_statistic(OrthogonalRotations(), estimator, samples)
    fixed = ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0, 10.0))
    for group in (SignFlips(), OrthogonalRotations())
        grouped = fit_statistic(group, estimator, samples)
        @test grouped.(samples) == learned.(samples)
        @test fit_statistic(group, fixed, samples) === fixed
        for statistic in (AbsMean(), fixed, estimator)
            score = statistic === estimator ? learned : statistic
            reference = fit_reference(group, score, samples)
            for procedure in (CompoundBH(α=0.1), SeparateBH(α=0.1), DDR(α=0.1))
                method = MultipleRandomizationTest(; group, statistic, procedure)
                @test fit(method, samples) == fit(procedure, reference)
            end
        end
    end
    custom = x -> abs(mean(x))
    method = MultipleRandomizationTest(group=SignFlips(), statistic=custom)
    @test fit(method, samples) == fit(CompoundBH(), fit_reference(SignFlips(), custom, samples))
    @test_throws MethodError fit(method, randn(rng, 3, 5))
    @test_throws ArgumentError fit(method, typeof(first(samples))[])
    @test_throws ArgumentError fit(MultipleRandomizationTest(
        group=SignFlips(), statistic=AbsMean(), procedure=CompoundBH(α=0)), samples)
end

@testset "SeqStep+ requires finite scores" begin
    for invalid in (Inf, -Inf, NaN), calibration_invalid in (false, true)
        observed = calibration_invalid ? [1.0] : [invalid]
        calibration = calibration_invalid ? [invalid] : [0.0]
        reference = CompoundRandomization.InvolutionReference(observed, calibration)
        r = RandomizationFit(InvolutionGroup(), AbsMean(), observed, reference)
        @test_throws ArgumentError fit(SeqStepPlus(), r)
    end
end

@testset "SeqStep+ excludes zero thresholds" begin
    for zeros_count in (1, 4)
        observed = vcat(ones(10), zeros(zeros_count))
        calibration = zeros(length(observed))
        reference = CompoundRandomization.InvolutionReference(observed, calibration)
        r = RandomizationFit(InvolutionGroup(), AbsMean(), observed, reference)
        result = fit(SeqStepPlus(α=0.2), r)
        @test result.cutoff == 1.0
        @test result.rj_idx == vcat(trues(10), falses(zeros_count))
        @test result.total_rejections == 10
        if zeros_count == 1
            boundary = 1/10
            @test fit(SeqStepPlus(α=boundary), r).cutoff == 1.0
            @test fit(SeqStepPlus(α=prevfloat(boundary)), r).cutoff == Inf
        end
    end
    observed = vcat(ones(10), 0.0, 0.0)
    calibration = vcat(zeros(10), 1.0, 0.0)
    r = RandomizationFit(InvolutionGroup(), AbsMean(), observed,
        CompoundRandomization.InvolutionReference(observed, calibration))
    @test fit(SeqStepPlus(α=0.2), r).cutoff == 1.0
    @test fit(SeqStepPlus(α=0.2), r).rj_idx == vcat(trues(10), falses(2))
end

@testset "SeqStep+ through the common interface" begin
    @test SeqStepPlus <: AbstractMultipleTestingProcedure
    @test InvolutionGroup <: AbstractRandomizationGroup
    @test CompoundRandomization.InvolutionReference <: AbstractRandomizationReference
    rng = MersenneTwister(616)
    samples = ReplicatedSample.(eachrow(randn(rng, 40, 5)))
    group = InvolutionGroup()
    @test group.transformation isa HalfSplit
    @test InvolutionGroup(HalfSplit()) == group
    estimator = ModeratedTScore(Empirikos.QuantileLimma())
    learned = fit_statistic(group, estimator, samples)
    @test learned.(samples) == fit_statistic(SignFlips(), estimator, samples).(samples)
    @test all(orbit_variance(group, x).ν == 5 for x in samples)
    for x in samples
        @test group.transformation(group.transformation(x)).Z == x.Z
    end
    for statistic in (AbsMean(), estimator)
        score = statistic === estimator ? learned : statistic
        reference = fit_reference(group, score, samples)
        method = MultipleRandomizationTest(; group, statistic, procedure=SeqStepPlus())
        result = fit(method, samples)
        @test result == fit(SeqStepPlus(), reference)
        @test result.observed == score.(samples)
        @test result.calibration == [score(ReplicatedSample(x.Z .* [1, 1, 1, -1, -1])) for x in samples]
        @test !hasproperty(result, :pvalue)
        @test separate_pvalues(reference) == ifelse.(result.observed .> result.calibration, 0.5, 1.0)
    end

    custom = InvolutionGroup(x -> ReplicatedSample(x.Z .* [1, -1, 1, -1, 1]))
    reference = fit_reference(custom, AbsMean(), samples)
    result = fit(MultipleRandomizationTest(group=custom, statistic=AbsMean(),
        procedure=SeqStepPlus(α=0.5)), samples)
    @test result.calibration == [abs(mean(x.Z .* [1, -1, 1, -1, 1])) for x in samples]
    for i in eachindex(samples), t in (0.0, reference.observed[i], reference.reference.calibration[i])
        @test orbit_tail(reference, i, t) == mean([reference.observed[i], reference.reference.calibration[i]] .>= t)
    end
    @test_throws MethodError fit_statistic(custom, estimator, samples)
    @test_throws MethodError fit(SeqStepPlus(), samples, AbsMean())
    @test_throws ArgumentError fit(MultipleRandomizationTest(group=group,
        statistic=x -> -1.0, procedure=SeqStepPlus()), samples)

    tied = ReplicatedSample.(eachrow(zeros(5, 4)))
    r = fit_reference(group, AbsMean(), tied)
    @test separate_pvalues(r) == ones(5)
    @test !any(fit(SeqStepPlus(), r).rj_idx)

    # Another sample type and a fixed coordinate-swap involution.
    two_samples = [TwoSample([1.0, 2], [4.0, 8]), TwoSample([0.0, 1], [0.0, 1])]
    swap = InvolutionGroup(x -> TwoSample(x.Z[[3, 2]], x.Z[[1, 4]]))
    method = MultipleRandomizationTest(group=swap, statistic=AbsMeanDifference(), procedure=SeqStepPlus())
    result = fit(method, two_samples)
    @test result == fit(SeqStepPlus(), fit_reference(swap, AbsMeanDifference(), two_samples))
    @test swap.transformation(swap.transformation(first(two_samples))).Z == first(two_samples).Z
end
