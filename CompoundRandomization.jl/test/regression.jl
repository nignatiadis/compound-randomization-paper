@testset "Regression samples and scores: Section 7.2" begin
    rng = MersenneTwister(740)
    K = 10
    W = randn(rng, K)
    X = hcat(ones(K), randn(rng, K))
    design = RegressionDesign(W, X)
    A = hcat(W, X)
    U = nullspace(X')
    @test design.ν == 7
    @test design.v ≈ inv(A' * A)[1, 1]
    @test_throws DimensionMismatch RegressionDesign(W[2:end], X)
    @test_throws ArgumentError RegressionDesign(ones(K), X)
    @test_throws ArgumentError RegressionDesign([1., 0], ones(2, 1))
    @test_throws ArgumentError RegressionDesign(fill(NaN, K), X)
    samples = [RegressionSample(randn(rng, K), design) for _ in 1:30]
    prior = Empirikos.InverseScaledChiSquare(1.3, 6.0)
    for s in samples
        @test s.β̂ ≈ A \ s.Z
        @test s.δ̂ == s.β̂[1]
        @test s.σ̂² ≈ sum(abs2, s.Z - A * (A \ s.Z)) / design.ν
        @test s.τ̂² ≈ sum(abs2, U' * s.Z) / (design.ν + 1)
        @test (design.ν + 1) * s.τ̂² ≈ design.ν * s.σ̂² + abs2(s.δ̂) / design.v
        @test AbsCoefficient()(s) == abs(s.δ̂)
        @test ModeratedTScore(prior)(s) ≈ abs(s.δ̂) / sqrt(design.v *
            (prior.ν * prior.σ² + design.ν * s.σ̂²) / (prior.ν + design.ν))
        shifted = RegressionSample(s.Z + X * [2., -3.], design)
        @test shifted.δ̂ ≈ s.δ̂
        @test shifted.τ̂² ≈ s.τ̂²
    end
    t = fit(Empirikos.SimultaneousTTest(), samples)
    summaries = Empirikos.NormalChiSquareSample.(samples)
    @test t == fit(Empirikos.SimultaneousTTest(), summaries)
    @test t.pvalue ≈ [2ccdf(TDist(design.ν), abs(s.δ̂)/sqrt(design.v*s.σ̂²)) for s in samples]
    limma = Empirikos.EmpiricalPartiallyBayesTTest(prior=Empirikos.Limma(), solver=nothing)
    @test fit(limma, samples) == fit(limma, summaries)
    # The one-sample model is W=1 with no nuisance covariates.
    one_design = RegressionDesign(ones(K), zeros(K, 0))
    one = [RegressionSample(s.Z, one_design) for s in samples]
    reps = [ReplicatedSample(s.Z) for s in samples]
    @test [s.τ̂² for s in one] ≈ [s.τ̂² for s in reps]
    @test ModeratedTScore(prior).(one) ≈ ModeratedTScore(prior).(reps)
    @test fit(Empirikos.SimultaneousTTest(), one).pvalue ≈
        fit(Empirikos.SimultaneousTTest(), reps).pvalue
    # Equal-variance two-sample testing is W=group indicator, X=intercept.
    two_design = RegressionDesign(vcat(ones(4), zeros(6)), ones(K, 1))
    two = [RegressionSample(s.Z, two_design) for s in samples]
    paired = [TwoSample(s.Z[1:4], s.Z[5:end]) for s in samples]
    @test [s.τ̂² for s in two] ≈ [s.τ̂² for s in paired]
    @test ModeratedTScore(prior).(two) ≈ ModeratedTScore(prior).(paired)
    @test fit(Empirikos.SimultaneousTTest(), two).pvalue ≈
        fit(Empirikos.SimultaneousTTest(), paired).pvalue
    @test_throws ArgumentError RegressionSample(fill(Inf, K), design)
    @test_throws DimensionMismatch RegressionSample(zeros(K-1), design)
    @test_throws ArgumentError fit(Empirikos.SimultaneousTTest(), [samples[1], one[1]])
    @test_throws ArgumentError fit(Empirikos.SimultaneousTTest(), samples[1:0])
end
