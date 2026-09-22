# Run from VS Code or with: julia code/02_analyze.jl
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CSV, CompoundRandomization, LinearAlgebra, MultipleTesting, Statistics
import Empirikos
include("calibration.jl")
BLAS.set_num_threads(1)

input_dir = joinpath(@__DIR__, "..", "data")
output_dir = joinpath(@__DIR__, "..", "results")
mkpath(output_dir)
metadata = CSV.File(joinpath(input_dir, "samples.csv"))
arrays = CSV.File(joinpath(input_dir, "arrays.csv"))
Y = hcat([getproperty(arrays, Symbol(key)) for key in metadata.sample]...)
@assert size(Y) == (439918,10) && all(isfinite,Y)

# All contrasts use the same ten arrays and additive donor/cell-type model.
contrasts = [("naive","rTreg"), ("naive","act_naive"),
             ("rTreg","act_rTreg"), ("act_naive","act_rTreg")]
counts, priors = NamedTuple[], NamedTuple[]
for (resting,activated) in contrasts
    contrast = "$(activated)_vs_$(resting)"
    println(contrast); flush(stdout)
    others = setdiff(sort(unique(metadata.cell_type)),[resting,activated])
    W = Float64.(metadata.cell_type .== activated)
    X = hcat(ones(length(W)),metadata.donor .== "M29",metadata.donor .== "M30",
        [metadata.cell_type .== cell for cell in others]...)
    design = RegressionDesign(W,X)
    samples = [RegressionSample(y,design) for y in eachrow(Y)]
    @assert design.ν == 4

    ordinary_t = fit(Empirikos.SimultaneousTTest(),samples)
    ordinary_limma = fit(Empirikos.EmpiricalPartiallyBayesTTest(
        prior=Empirikos.Limma(),solver=nothing),samples)
    group = ResidualRotations()
    score = fit_statistic(group,ModeratedTScore(Empirikos.QuantileLimma(
        quantile_p=(.25,.75),scale_p=.5)),samples)
    for (method,prior) in (("limma",ordinary_limma.prior),("orbit",score.prior))
        push!(priors,(;contrast,method,s0sq=prior.σ²,nu0=prior.ν))
    end

    rotations = fit_reference(group,score,samples)
    separate = fit(SeparateBH(),rotations)
    @assert separate.pvalue ≈ ordinary_t.pvalue
    permutations = fit_reference(StratifiedPermutations(),score,samples)
    permutation_q = adjust(compound_pvalues(permutations),BenjaminiHochberg())
    separate_permutation_q = adjust(separate_pvalues(permutations),BenjaminiHochberg())
    # For naive activation this fixes M28, leaving the four-element M29/M30 subgroup.
    fixed_first = fit_reference(StratifiedPermutations(fixed=[1]),score,samples)

    # Enumerate every available single-donor swap for SeqStep+.
    swaps = Pair[]
    for donor in unique(metadata.donor)
        a = findfirst((metadata.donor .== donor) .& (metadata.cell_type .== resting))
        b = findfirst((metadata.donor .== donor) .& (metadata.cell_type .== activated))
        (isnothing(a) || isnothing(b)) && continue
        p = collect(eachindex(W))
        p[a],p[b] = p[b],p[a]
        @assert X[p,:] == X
        involution = InvolutionGroup(s -> RegressionSample(s.Z[p],s.design))
        push!(swaps,donor => fit_reference(involution,score,samples))
    end

    decisions = Pair{Symbol,BitVector}[]
    for alpha in (.05,.10)
        println("  alpha=$alpha"); flush(stdout)
        # Exact shortcuts avoid computing all pairs of ~440,000 probes.
        cutoff = CompoundRandomization.ddr_cutoff(rotations.reference,alpha/10)
        weights = [inv(1-orbit_tail(rotations,i,cutoff)) for i in eachindex(samples)]
        permutation_cutoff = finite_ddr_cutoff(permutations.reference,alpha/10)
        permutation_weights = [inv(1-orbit_tail(permutations,i,permutation_cutoff))
            for i in eachindex(samples)]
        permutation_ddr_p = all(isfinite,permutation_weights) ?
            CompoundRandomization.pooled_pvalues(permutations,permutation_weights) :
            fill(Inf,length(samples))
        permutation_ddr_q = adjust(ifelse.(permutation_ddr_p .<= alpha/10,
            permutation_ddr_p,1.0),BenjaminiHochberg())
        results = [
            "ttest" => (ordinary_t.adjp .<= alpha),
            "limma" => (ordinary_limma.adjp .<= alpha),
            "separate_rotation" => (separate.adjp .<= alpha),
            "compound_rotation" => rotation_bh(rotations; α=alpha),
            "ddr_rotation" => rotation_bh(rotations; α=alpha,τ=alpha/10,weights),
            "compound_rotation_corrected" => rotation_bh(rotations; α=alpha/1.93),
            "compound_permutation" => (permutation_q .<= alpha),
            "compound_permutation_corrected" => (permutation_q .<= alpha/1.93),
            "separate_permutation" => (separate_permutation_q .<= alpha),
            "ddr_permutation" => (permutation_ddr_q .<= alpha),
            "gimenez_zou_fixed_first" => fit(GimenezZou(α=alpha),fixed_first).rj_idx,
            "gimenez_zou_full" => fit(GimenezZou(α=alpha),permutations).rj_idx]
        for (donor,reference) in swaps
            push!(results,"seqstep_swap_$donor" => fit(SeqStepPlus(α=alpha),reference).rj_idx)
        end
        for (method,rejected) in results
            push!(counts,(;contrast,method,alpha,rejections=count(rejected)))
            push!(decisions,Symbol(method,"_",string(alpha)) => BitVector(rejected))
        end
    end
    CSV.write(joinpath(output_dir,"decisions_$contrast.csv"),(;probe=arrays.probe,decisions...))
end
CSV.write(joinpath(output_dir,"counts.csv"),counts)
CSV.write(joinpath(output_dir,"variance_priors.csv"),priors)
foreach(println,counts)
