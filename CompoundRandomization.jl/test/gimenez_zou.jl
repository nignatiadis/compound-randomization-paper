@testset "Gimenez-Zou: manuscript threshold formula" begin
    rng = MersenneTwister(831)
    for m in (2,4,8), n in (1,8,30), trial in 1:10
        raw = Float64.(rand(rng,0:4,m,n))
        observed = raw[1,:]
        M = vec(maximum(raw; dims=1))
        signed = [M[i] * sign(raw[1,i] - maximum(raw[2:end,i])) for i in 1:n]
        B = signed .> 0
        scores = copy(raw)
        foreach(sort!,eachcol(scores))
        reference = RandomizationFit(StratifiedPermutations(),AbsCoefficient(),observed,
            CR.FiniteRandomizationScores(scores))
        for α in (0.01,0.05,0.1,0.3,0.75)
            qualifying = [s for s in unique(abs.(signed)) if s > 0 &&
                (1+count(signed .<= -s)) / ((m-1)*max(1,count(signed .>= s))) <= α]
            cutoff = isempty(qualifying) ? Inf : minimum(qualifying)
            result = fit(GimenezZou(α=α),reference)
            @test result.cutoff == cutoff
            @test result.unique_maximum == B
            @test result.maxima == M
            @test result.rj_idx == (signed .>= cutoff)
            @test result.group_size == m
            if m == 2
                involution = RandomizationFit(InvolutionGroup(),AbsMean(),observed,
                    CR.InvolutionReference(observed,raw[2,:]))
                @test result.rj_idx == fit(SeqStepPlus(α=α),involution).rj_idx
            end
        end
    end
    # A qualifying partial tie block must not give any discoveries.
    raw = [2. 0 0; 0 2 1; 0 0 0; 0 0 0]
    observed = raw[1,:]
    foreach(sort!,eachcol(raw))
    reference = RandomizationFit(StratifiedPermutations(),AbsCoefficient(),observed,
        CR.FiniteRandomizationScores(raw))
    @test !any(fit(GimenezZou(α=0.5),reference).rj_idx)
    # Zero scores are excluded from the threshold set.
    reference = RandomizationFit(StratifiedPermutations(),AbsCoefficient(),[0.],
        CR.FiniteRandomizationScores(zeros(8,1)))
    result = fit(GimenezZou(α=0.3),reference)
    @test result.cutoff == Inf
    @test !any(result.rj_idx)
    # An observed tie is zero; a loss remains negative even if others tie.
    raw = [2. 2 1; 1 2 2; 0 1 2; 0 0 0]
    observed = raw[1,:]
    foreach(sort!,eachcol(raw))
    reference = RandomizationFit(StratifiedPermutations(),AbsCoefficient(),observed,
        CR.FiniteRandomizationScores(raw))
    @test fit(GimenezZou(α=0.7),reference).rj_idx == [true,false,false]
    @test !any(fit(GimenezZou(α=0.5),reference).rj_idx)
    # A positive observed tie must not count against an order-two winner.
    raw = [2. 2; 1 2]
    reference = RandomizationFit(StratifiedPermutations(),AbsCoefficient(),raw[1,:],
        CR.FiniteRandomizationScores(sort(raw; dims=1)))
    @test fit(GimenezZou(α=0.5),reference).cutoff == Inf
    raw = [2. 2 2; 1 1 2]
    reference = RandomizationFit(StratifiedPermutations(),AbsCoefficient(),raw[1,:],
        CR.FiniteRandomizationScores(sort(raw; dims=1)))
    @test fit(GimenezZou(α=0.5),reference).rj_idx == [true,true,false]
    involution = RandomizationFit(InvolutionGroup(),AbsMean(),raw[1,:],
        CR.InvolutionReference(raw[1,:],raw[2,:]))
    @test fit(SeqStepPlus(α=0.5),involution).rj_idx == [true,true,false]
    for bad in (-1., Inf, NaN)
        reference = RandomizationFit(StratifiedPermutations(),AbsCoefficient(),[bad],
            CR.FiniteRandomizationScores(fill(bad,4,1)))
        @test_throws ArgumentError fit(GimenezZou(),reference)
    end
    samples = [ReplicatedSample([1.,2,3])]
    @test_throws ArgumentError fit(MultipleRandomizationTest(group=SignFlips(),
        statistic=AbsMean(),procedure=GimenezZou()),samples)
    @test !any(fit(MultipleRandomizationTest(group=SignFlips(reduce_symmetry=false),
        statistic=AbsMean(),procedure=GimenezZou()),samples).unique_maximum)
    @test_throws ArgumentError fit(MultipleRandomizationTest(group=Permutations(),
        statistic=AbsMeanDifference(),procedure=GimenezZou()),[TwoSample([1.,2],[3.,4])])
end

@testset "Fixing the first methylation pair gives a genuine subgroup" begin
    donor = [1,1,1,2,2,2,3,3,3,3]
    cell = [:naive,:treg,:act_naive,:naive,:act_naive,:act_treg,
        :naive,:treg,:act_naive,:act_treg]
    X = hcat(ones(10),donor .== 2,donor .== 3,cell .== :treg,cell .== :act_treg)
    design = RegressionDesign(Float64.(cell .== :act_naive),X)
    group = StratifiedPermutations(fixed=[1])
    patterns = CR.stratified_permutations(design; fixed=group.fixed)
    @test length(patterns) == 4
    @test all(p -> p[1] == 1 && p[3] == 3 && X[p,:] == X,patterns)
    @test collect(1:10) in patterns
    @test all(invperm(p) in patterns for p in patterns)
    @test all(p[q] in patterns for p in patterns,q in patterns)
    @test_throws ArgumentError CR.stratified_permutations(design; fixed=[11])
    @test CR.stratified_permutations(design; fixed=collect(1:10)) == [collect(1:10)]
    samples = [RegressionSample([0.,0,2,0,3,0,0,0,4,0],design) for _ in 1:20]
    for score in (AbsCoefficient(),ModeratedTScore(Dirac(1.)),
        ModeratedTScore(Empirikos.InverseScaledChiSquare(1.,10.)))
        full = fit_reference(StratifiedPermutations(),score,samples)
        fixed = fit_reference(group,score,samples)
        @test full.reference.sorted_scores[1:2:end,:] ≈ fixed.reference.sorted_scores
        @test separate_pvalues(full) == separate_pvalues(fixed)
        @test !any(fit(GimenezZou(),full).unique_maximum)
        result = fit(MultipleRandomizationTest(group=group,statistic=score,
            procedure=GimenezZou(α=0.05)),samples)
        @test all(result.unique_maximum)
        @test result.total_rejections == 20
        @test result.group_size == 4
        for p in patterns, s in samples
            @test RegressionSample(s.Z[p],design).τ̂² ≈ s.τ̂²
            @test orbit_variance(group,s) == orbit_variance(ResidualRotations(),s)
        end
    end
    identity = fit_reference(StratifiedPermutations(fixed=collect(1:10)),AbsCoefficient(),samples)
    @test_throws ArgumentError fit(GimenezZou(),identity)
end
