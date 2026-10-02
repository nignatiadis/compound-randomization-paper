function check_decisions(r; levels=(0.01, 0.05, 0.1, 0.5), taus=(0.01, 0.2))
    for alpha in levels
        procedures = Tuple[(CompoundBH(α=alpha), CompoundBH(α=alpha, compute_pvalues=false))]
        append!(procedures, [(DDR(α=alpha, τ=tau),
            DDR(α=alpha, τ=tau, compute_pvalues=false)) for tau in taus])
        for (ordinary, fast) in procedures
            expected, result = fit(ordinary, r), fit(fast, r)
            @test result.rj_idx == expected.rj_idx
            @test result.total_rejections == expected.total_rejections
            @test result.cutoff == expected.cutoff
            @test isnothing(result.pvalue) && isnothing(result.adjp)
            @test result.method == fast
            if ordinary isa DDR
                @test result.orbit_cutoff == expected.orbit_cutoff
                @test result.weights == expected.weights
            end
        end
    end
end

@testset "Decisions without p-value vectors" begin
    @test CompoundBH().compute_pvalues
    @test DDR().compute_pvalues
    @test CompoundBH(0.2) == CompoundBH(α=0.2)
    @test DDR(0.2, 0.03) == DDR(α=0.2, τ=0.03)
    rng = MersenneTwister(923)
    for shift in (0.0, 2.0, 8.0), K in (3, 6)
        Z = randn(rng, 24, K)
        Z[1:12,:] .+= shift
        Z[13:16,:] .= Z[1:4,:] # repeated scores must enter together
        samples = ReplicatedSample.(eachrow(Z))
        for score in (AbsMean(), ModeratedTScore(Dirac(1.0)),
            ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0, 5.0)),
            ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0, 0.0))),
            group in (OrthogonalRotations(), SignFlips())
            check_decisions(fit_reference(group, score, samples))
        end
    end
    # All/none rejected, singleton, zero radii and infinite unmoderated scores.
    for Z in (zeros(8,6), ones(8,6), ones(1,6), zeros(1,6),
        vcat(ones(8,6),zeros(8,6))),
        score in (AbsMean(), ModeratedTScore(Empirikos.InverseScaledChiSquare(1.0,0.0)))
        samples = ReplicatedSample.(eachrow(Z))
        for group in (OrthogonalRotations(), SignFlips())
            check_decisions(fit_reference(group,score,samples))
        end
    end
    all_ref = fit_reference(OrthogonalRotations(),AbsMean(),fill(ReplicatedSample(ones(6)),8))
    @test all(fit(CompoundBH(compute_pvalues=false),all_ref).rj_idx)
    zero_ref = fit_reference(OrthogonalRotations(),AbsMean(),fill(ReplicatedSample(zeros(6)),8))
    @test !any(fit(DDR(compute_pvalues=false),zero_ref).rj_idx)
    for alpha in (0.0, 1.0), procedure in
        (CompoundBH(α=alpha,compute_pvalues=false), DDR(α=alpha,compute_pvalues=false))
        @test_throws ArgumentError fit(procedure,all_ref)
    end
    @test_throws ArgumentError fit(DDR(τ=0,compute_pvalues=false),all_ref)

    # Two-sample and regression references use the same rotation search.
    two = [TwoSample(randn(rng,3) .+ 2,randn(rng,3)) for _ in 1:20]
    for group in (CenteredRotations(),Permutations())
        check_decisions(fit_reference(group,AbsMeanDifference(),two))
    end
    X = hcat(ones(8),repeat([0.,1.];inner=4))
    design = RegressionDesign(repeat([0.,1.],4),X)
    samples = [RegressionSample(randn(rng,8) .+ 2 .* design.W,design) for _ in 1:20]
    for group in (ResidualRotations(),StratifiedPermutations())
        check_decisions(fit_reference(group,AbsCoefficient(),samples))
        for procedure in (CompoundBH(compute_pvalues=false),DDR(compute_pvalues=false))
            statistic = ModeratedTScore(Empirikos.QuantileLimma())
            reference = fit_reference(group,fit_statistic(group,statistic,samples),samples)
            @test fit(MultipleRandomizationTest(;group,statistic,procedure),samples) ==
                fit(procedure,reference)
        end
    end
end

@testset "BH boundaries and tied rotation scores" begin
    rng = MersenneTwister(924)
    for n in (7,20), trial in 1:5
        observed = rand(rng,n)
        observed[2:3] .= observed[1]
        reference = CR.RotationReference(AbsMean(),ones(n),3,1.0)
        r = RandomizationFit(OrthogonalRotations(),AbsMean(),observed,reference)
        # Test at the actual adjusted p-values and one floating-point step away.
        full = fit(CompoundBH(),r)
        levels = unique(a for p in full.adjp for a in (prevfloat(p),p,nextfloat(p)) if 0 < a < 1)
        check_decisions(r;levels,taus=())
        for tau in (0.05,0.5)
            full = fit(DDR(τ=tau),r)
            levels = unique(a for p in full.adjp for a in (prevfloat(p),p,nextfloat(p)) if 0 < a < 1)
            check_decisions(r;levels,taus=(tau,))
        end
    end
    # A tied group must be included in full, even if its first rank fails BH.
    r = RandomizationFit(OrthogonalRotations(),AbsMean(),[0.9,0.9,0.1],
        CR.RotationReference(AbsMean(),ones(3),3,1.0))
    @test fit(CompoundBH(α=0.2,compute_pvalues=false),r).rj_idx == [true,true,false]
    check_decisions(r)
end

@testset "Finite DDR cutoff matches exhaustive support search" begin
    rng = MersenneTwister(925)
    for L in (1,2,8,24), n in (1,5,30), tau in (0.01,0.1,0.5)
        scores = sort(Float64.(rand(rng,0:5,L,n));dims=1)
        reference = CR.FiniteRandomizationScores(scores)
        support = sort(unique(vec(scores)))
        expected = first(s for s in support if CR.ddr_tail_odds(reference,s) <= tau)
        @test CR.ddr_cutoff(reference,tau) == expected
    end
end
