# A summary-only test fixture, not an implementation of regression fitting.
struct CoefficientSummary <: AbstractRandomizationSample{Float64}
    estimate::Float64
    variance::Float64
    residual_df::Int
    observations::Int
end

struct SummaryRotations <: AbstractRandomizationGroup end

Empirikos.NormalChiSquareSample(x::CoefficientSummary) =
    Empirikos.NormalChiSquareSample(x.estimate, x.variance, x.residual_df)
CR.nobs(x::CoefficientSummary) = x.observations

CR.orbit_variance(::SummaryRotations, x::CoefficientSummary) =
    Empirikos.ScaledChiSquareSample(
        (abs2(x.estimate) + x.residual_df * x.variance) / (x.residual_df + 1),
        x.residual_df + 1)

function CR.fit_reference(group::SummaryRotations, score, samples::AbstractVector{<:CoefficientSummary})
    radius2 = [abs2(x.estimate) + x.residual_df * x.variance for x in samples]
    reference = CR.RotationReference(score, radius2, first(samples).residual_df + 1, 1.0)
    RandomizationFit(group, score, score.(samples), reference)
end

@testset "Sample extension contract" begin
    rng = MersenneTwister(882)
    samples = [CoefficientSummary(randn(rng), exp(randn(rng)), 4, 9) for _ in 1:40]
    group = SummaryRotations()
    prior = Empirikos.InverseScaledChiSquare(1.5, 6.0)
    score = ModeratedTScore(prior)
    @test first(samples) isa Empirikos.EBayesSample
    for x in samples
        expected = abs(x.estimate) / sqrt((6 * 1.5 + 4 * x.variance) / 10)
        @test score(x) ≈ expected
        @test ModeratedTScore(Dirac(2.0))(x) ≈ abs(x.estimate) / sqrt(2)
        @test orbit_variance(group, x).ν == 5 != nobs(x)
    end

    estimator = ModeratedTScore(Empirikos.QuantileLimma())
    learned = fit_statistic(group, estimator, samples)
    variances = [Empirikos.ScaledChiSquareSample(
        (abs2(x.estimate) + 4 * x.variance) / 5, 5) for x in samples]
    manual = ModeratedTScore(Empirikos.fit_prior(estimator.prior, variances))
    @test learned.(samples) == manual.(samples)
    reference = fit_reference(group, manual, samples)
    @test reference.reference.dimension == 5
    for procedure in (CompoundBH(), SeparateBH(), DDR())
        method = MultipleRandomizationTest(; group, statistic=estimator, procedure)
        @test fit(method, samples) == fit(procedure, reference)
    end

    unmoderated = ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0, 0.0))
    p = separate_pvalues(fit_reference(group, unmoderated, samples))
    @test p ≈ [2ccdf(TDist(4), abs(x.estimate) / sqrt(x.variance)) for x in samples]

    # New groups/sample types cannot inherit the one-sample prior-fitting formula.
    one_sample = [ReplicatedSample([1.0, 2, 3])]
    @test_throws MethodError fit_statistic(group, estimator, one_sample)
    @test_throws MethodError fit_statistic(OrthogonalRotations(), estimator, samples)
    @test_throws MethodError fit_reference(OrthogonalRotations(), score, samples)
end
