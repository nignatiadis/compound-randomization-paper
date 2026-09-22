# Run from VS Code or with: julia code/04_analyze.jl
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))

using CSV, Combinatorics, CompoundRandomization, LinearAlgebra, Statistics
using Distributions: pdf, ccdf, TDist
import Plots
import Empirikos

input_file = joinpath(@__DIR__, "..", "data", "paired_differences_complete6.csv")
output_dir = joinpath(@__DIR__, "..", "results")
mkpath(output_dir)
isfile(input_file) || error("Run code/02_tests.R from the analysis directory first.")

data = CSV.File(input_file)
donors = propertynames(data)[2:end]
Z = hcat((getproperty(data,c) for c in donors)...)
samples = ReplicatedSample.(eachrow(Z))
n, K = length(samples), length(donors)
@assert (n,K)==(286,6) && length(unique(data.feature_id))==n
@assert all(s -> all(isfinite, s.Z), samples)

score = fit_statistic(OrthogonalRotations(), ModeratedTScore(
    Empirikos.QuantileLimma(quantile_p=(.25,.75),scale_p=.5)), samples)

rotations = fit_reference(OrthogonalRotations(),score,samples)

flips = fit_reference(SignFlips(),score,samples)

ordinary_t = fit(Empirikos.SimultaneousTTest(),samples)
Plots.histogram(ordinary_t.pvalue; bins = range(0,1; length=11), normalize=:pdf)
@assert fit(SeparateBH(),rotations).pvalue ≈ ordinary_t.pvalue

ordinary_limma = fit(Empirikos.EmpiricalPartiallyBayesTTest(
    prior=Empirikos.Limma(),solver=nothing),samples)
ordinary_limma.prior.σ², ordinary_limma.prior.ν
ordinary_limma.total_rejections
Plots.histogram(ordinary_limma.pvalue; bins = range(0,1; length=11), normalize=:pdf)


CSV.write(joinpath(output_dir, "baselines.csv"), [(;feature_id=data.feature_id[i],
    mean=mean(samples[i]),variance=var(samples[i]),
    t_p=ordinary_t.pvalue[i],t_q=ordinary_t.adjp[i],
    limma_p=ordinary_limma.pvalue[i],limma_q=ordinary_limma.adjp[i]) for i in 1:n])
CSV.write(joinpath(output_dir, "variance_priors.csv"),
    [(;method,s0sq=prior.σ²,nu0=prior.ν) for (method,prior) in
        (("limma",ordinary_limma.prior),("orbit",score.prior))])

# As in the methylation analysis: compare observed variances with their fitted
# marginal distribution (including sampling noise with K-1 degrees of freedom).
sample_variances = var.(samples)
variance_grid = range(0, 1.02 * maximum(sample_variances); length=600)
marginal_pdf_limma = pdf.(ordinary_limma.prior,
    Empirikos.ScaledChiSquareSample.(variance_grid, K-1))
# Use K-1 df for both curves, since the histogram shows sample variances.
marginal_pdf_orbit = pdf.(score.prior,
    Empirikos.ScaledChiSquareSample.(variance_grid, K-1))
variance_plot = Plots.histogram(sample_variances;
    bins=range(0, last(variance_grid); length=61), normalize=:pdf,
    fillcolor=:lightgrey, fillalpha=0.7, linewidth=0.3,
    label="Observed sample variances", xlabel="Sample variance of paired differences",
    ylabel="Density", size=(700, 350), legend=:topright)
Plots.plot!(variance_plot, variance_grid, marginal_pdf_limma;
    label="Fitted limma", color=:purple, linewidth=2)
Plots.plot!(variance_plot, variance_grid, marginal_pdf_orbit;
    label="Orbit-fitted prior", color=:darkorange, linestyle=:dash, linewidth=2)
display(variance_plot)

compound_rotation_pvalues = fit(CompoundBH(),rotations).pvalue
compound_pvalue_plot = Plots.histogram(compound_rotation_pvalues;
    bins=range(0,1; length=11), normalize=:pdf, label=false,
    xlabel="Compound BH p-value (rotations)", ylabel="Density")
display(compound_pvalue_plot)


# Global sign reversal leaves this score unchanged: 10 distinct balanced splits.
patterns = [[j==1 || j in positive ? 1 : -1 for j in 1:K]
            for positive in combinations(2:K,2)]
split_refs = [fit_reference(InvolutionGroup(s -> ReplicatedSample(s.Z .* e)),
                           score,samples) for e in patterns]
counts, results = NamedTuple[], NamedTuple[]
function record!(method,alpha,result; split="")
    has_p = hasproperty(result,:pvalue)
    rejected = result.rj_idx
    push!(counts,(;method,alpha,split,rejections=count(rejected)))
    for i in 1:n
        push!(results,(;feature_id=data.feature_id[i],method,alpha,split,
            pvalue=has_p ? result.pvalue[i] : missing,
            adjp=has_p ? result.adjp[i] : missing,rejected=rejected[i]))
    end
end
for alpha in (.05,.10)
    record!("ttest",alpha,fit(Empirikos.SimultaneousTTest(α=alpha),samples))
    record!("limma",alpha,fit(Empirikos.EmpiricalPartiallyBayesTTest(
        prior=Empirikos.Limma(),solver=nothing,α=alpha),samples))
    record!("compound_rotation",alpha,fit(CompoundBH(α=alpha),rotations))
    record!("compound_signflip",alpha,fit(CompoundBH(α=alpha),flips))
    record!("separate_signflip",alpha,fit(SeparateBH(α=alpha),flips))
    record!("ddr_rotation",alpha,fit(DDR(α=alpha),rotations))
    record!("ddr_signflip",alpha,fit(DDR(α=alpha),flips))
    for (e,reference) in zip(patterns,split_refs)
        record!("seqstep",alpha,fit(SeqStepPlus(α=alpha),reference);
               split=join(s==1 ? "+" : "-" for s in e))
    end
end
CSV.write(joinpath(output_dir, "counts.csv"),counts)
CSV.write(joinpath(output_dir, "results.csv"),results)

# Paired differences in donor order: inspect B2M, GCG, or any row of Z directly.
b = only(findall(==("B2M_3"),data.feature_id))
g = only(findall(==("GCG_36"),data.feature_id))
B2M = Z[b,:]
GCG = Z[g,:]
GCG_flipped = GCG .* [1,1,-1,-1,1,-1]

# Evaluate all three vectors against the same fitted score and orbits.
diagnostics = NamedTuple[]
for (label,i,z) in (("B2M",b,B2M),("GCG",g,GCG),("GCG flipped",g,GCG_flipped))
    s = ReplicatedSample(z)
    t = score(s)
    rotation_p = orbit_tail(rotations,i,t)
    ttest_p = 2 * ccdf(TDist(K-1),abs(mean(z)) / sqrt(var(z)/K))
    @assert rotation_p ≈ ttest_p
    own_count = count(>=(t),flips.reference.sorted_scores[:,i])
    pooled_count = count(>=(t),flips.reference.sorted_scores)
    push!(diagnostics,(;label,NamedTuple{Tuple(donors)}(Tuple(s.Z))...,
        mean=mean(s),variance=var(s),score=t,own_count,pooled_count,
        rotation_p,
        compound_rotation_p=mean(orbit_tail(rotations,j,t) for j in 1:n),
        signflip_p=own_count/size(flips.reference.sorted_scores,1),
        compound_signflip_p=pooled_count/length(flips.reference.sorted_scores)))
end
@assert diagnostics[1].pooled_count==2 && diagnostics[3].pooled_count==1
@assert diagnostics[3].score > diagnostics[1].score
CSV.write(joinpath(output_dir, "diagnostics.csv"),diagnostics)
foreach(println,diagnostics)
open(joinpath(output_dir, "Julia_session.txt"),"w") do io
    println(io,"Julia $VERSION; project=$(Base.active_project())")
    println(io,"CompoundRandomization: $(pathof(CompoundRandomization))")
    println(io,"Empirikos: $(pathof(Empirikos)); n=$n; K=$K")
end
foreach(println,counts)
