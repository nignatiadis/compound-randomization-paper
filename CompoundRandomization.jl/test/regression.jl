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

@testset "Regression rotations: Section 7.2" begin
    rng = MersenneTwister(741)
    K = 10
    X = hcat(ones(K), randn(rng,K))
    design = RegressionDesign(randn(rng,K),X)
    U = nullspace(X')
    samples = [RegressionSample(randn(rng,K),design) for _ in 1:30]
    group = ResidualRotations()
    prior = Empirikos.InverseScaledChiSquare(1.3,6.0)
    for s in samples
        @test orbit_variance(group,s) == Empirikos.ScaledChiSquareSample(s.τ̂²,8)
        Q = Matrix(qr(randn(rng,8,8)).Q)
        rotated = RegressionSample(s.Z + U*(Q*(U'*s.Z) - U'*s.Z),design)
        @test X'*rotated.Z ≈ X'*s.Z
        @test rotated.τ̂² ≈ s.τ̂²
    end
    t = fit(Empirikos.SimultaneousTTest(),samples)
    for score in (AbsCoefficient(),ModeratedTScore(prior),ModeratedTScore(Dirac(1.3)),
        ModeratedTScore(Empirikos.InverseScaledChiSquare(1.3,0.0)))
        r = fit_reference(group,score,samples)
        @test separate_pvalues(r) ≈ t.pvalue
        for α in (.05,.1,.5)
            direct = fit(CompoundBH(α=α),r)
            @test direct.rj_idx == threshold_rejections(r,α)
            @test fit(MultipleRandomizationTest(group=group,statistic=score,
                procedure=CompoundBH(α=α)),samples).rj_idx == direct.rj_idx
            @test length(fit(DDR(α=α),r).pvalue) == length(samples)
        end
    end
    score = ModeratedTScore(prior)
    one_design = RegressionDesign(ones(K),zeros(K,0))
    one = [RegressionSample(s.Z,one_design) for s in samples]
    reps = [ReplicatedSample(s.Z) for s in samples]
    @test compound_pvalues(fit_reference(group,score,one)) ≈
        compound_pvalues(fit_reference(OrthogonalRotations(),score,reps))
    two_design = RegressionDesign(vcat(ones(4),zeros(6)),ones(K,1))
    two = [RegressionSample(s.Z,two_design) for s in samples]
    paired = [TwoSample(s.Z[1:4],s.Z[5:end]) for s in samples]
    @test compound_pvalues(fit_reference(group,score,two)) ≈
        compound_pvalues(fit_reference(CenteredRotations(),score,paired))
    learned = fit_statistic(group,ModeratedTScore(Empirikos.QuantileLimma()),samples)
    @test learned.prior == Empirikos.fit_prior(Empirikos.QuantileLimma(),
        orbit_variance.(Ref(group),samples))
    # Compare the analytic tail with independent Haar draws in residual coordinates.
    s = first(samples)
    reference = fit_reference(group,score,[s])
    exceedances = 0
    for _ in 1:4000
        direction = randn(rng,size(U,2))
        direction *= norm(U'*s.Z)/norm(direction)
        z = X*(X\s.Z) + U*direction
        exceedances += score(RegressionSample(z,design)) >= 1.0
    end
    @test abs(exceedances/4000 - orbit_tail(reference,1,1.0)) < .04
    zero = RegressionSample(zeros(K),design)
    reference = fit_reference(group,ModeratedTScore(prior),[zero])
    @test orbit_tail(reference,1,0.) == 1
    @test orbit_tail(reference,1,1.) == 0
    @test_throws ArgumentError fit_reference(group,AbsCoefficient(),[samples[1],one[1]])
end

@testset "Nuisance-stratified regression permutations" begin
    rng = MersenneTwister(814)
    donor = [1,1,1,2,2,2,3,3,3,3]
    cell = [:naive,:treg,:act_naive,:naive,:act_naive,:act_treg,
        :naive,:treg,:act_naive,:act_treg]
    group = StratifiedPermutations()
    # An intercept-only nuisance model gives the full two-sample group.
    design = RegressionDesign([1.,1,1,0,0,0], ones(6,1))
    samples = [RegressionSample(randn(rng,6),design) for _ in 1:4]
    two = [TwoSample(s.Z[1:3],s.Z[4:6]) for s in samples]
    @test length(CompoundRandomization.stratified_permutations(design)) == factorial(6)
    score = ModeratedTScore(Dirac(1.0))
    r = fit_reference(group,score,samples)
    two_r = fit_reference(Permutations(reduce_symmetry=false),score,two)
    @test r.reference.sorted_scores ≈ two_r.reference.sorted_scores
    @test separate_pvalues(r) == separate_pvalues(two_r)
    @test compound_pvalues(r) == compound_pvalues(two_r)
    weights = [1.,2,3,4]
    @test CompoundRandomization.pooled_pvalues(r,weights) ≈
        [mean(weights[j]*orbit_tail(r,j,t) for j in 1:4) for t in r.observed]
    # Distinct nuisance rows leave only identity, even with varying W.
    design = RegressionDesign([1.,0,1,0],hcat(ones(4),1:4))
    @test CompoundRandomization.stratified_permutations(design) == [collect(1:4)]
    @test separate_pvalues(fit_reference(group,AbsCoefficient(),
        [RegressionSample(randn(rng,4),design)])) == [1.0]
    for (resting, activated, L) in ((:naive,:act_naive,8), (:treg,:act_treg,2))
        others = setdiff(unique(cell), [resting, activated])
        X = hcat(ones(10), donor .== 2, donor .== 3, [cell .== c for c in others]...)
        design = RegressionDesign(Float64.(cell .== activated), X)
        patterns = CompoundRandomization.stratified_permutations(design)
        @test length(patterns) == L
        @test length(unique(patterns)) == L
        @test collect(1:10) in patterns
        @test all(p -> sort(p) == 1:10 && X[p,:] == X, patterns)
        @test all(p -> invperm(p) in patterns, patterns)
        @test all(p[q] in patterns for p in patterns, q in patterns)
        fixed_patterns = CompoundRandomization.stratified_permutations(design; fixed=[1])
        @test Set(fixed_patterns) == Set(p for p in patterns if p[1] == 1)
        @test CompoundRandomization.stratified_permutations(design; fixed=collect(1:10)) == [collect(1:10)]
        @test_throws ArgumentError CompoundRandomization.stratified_permutations(design; fixed=[0])
        samples = [RegressionSample(randn(rng,10), design) for _ in 1:12]
        fixed_score = ModeratedTScore(Empirikos.InverseScaledChiSquare(1.3,6.0))
        fixed_ref = fit_reference(StratifiedPermutations(fixed=[1]),fixed_score,samples)
        @test fixed_ref.reference.sorted_scores ≈ sort(
            [fixed_score(RegressionSample(s.Z[p],design)) for p in fixed_patterns, s in samples]; dims=1)
        for p in patterns, s in samples
            transformed = RegressionSample(s.Z[p], design)
            @test transformed.τ̂² ≈ s.τ̂²
            @test orbit_variance(group,s) == orbit_variance(ResidualRotations(),s)
        end
        learned = fit_statistic(group, ModeratedTScore(Empirikos.QuantileLimma()), samples)
        @test learned.prior == fit_statistic(ResidualRotations(),
            ModeratedTScore(Empirikos.QuantileLimma()), samples).prior
        for score in (AbsCoefficient(), ModeratedTScore(Dirac(1.0)), learned, x -> sum(x.Z))
            r = fit_reference(group, score, samples)
            direct = [score(RegressionSample(s.Z[p],design)) for p in patterns, s in samples]
            for (i, scores) in enumerate(eachcol(direct))
                @test r.reference.sorted_scores[:,i] ≈ sort(scores)
                @test orbit_tail(r,i,r.observed[i]) >= 1/L
            end
            @test compound_pvalues(r) ≈ [mean(orbit_tail(r,j,t) for j in eachindex(samples))
                for t in r.observed]
            @test fit(MultipleRandomizationTest(group=group,statistic=score),samples).rj_idx ==
                fit(CompoundBH(),r).rj_idx
            if resting == :naive && score isa Union{AbsCoefficient,ModeratedTScore}
                @test all(scores -> scores[1:2:end] == scores[2:2:end], eachcol(r.reference.sorted_scores))
                @test all(>=(1/4), separate_pvalues(r))
                @test !any(fit(CompoundBH(α=0.24),r).rj_idx)
            end
        end
        zero = RegressionSample(zeros(10),design)
        r = fit_reference(group, AbsCoefficient(), [zero,zero])
        @test separate_pvalues(r) == [1,1]
        @test compound_pvalues(r) == [1,1]
        @test !any(fit(DDR(),r).rj_idx)
    end
end
