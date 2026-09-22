# LaTeX table rows at 5% and 10%; no separate document or PDF build.
import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", ".."))
using CSV, Printf

output_dir = joinpath(@__DIR__, "..", "results")
counts = CSV.File(joinpath(output_dir,"counts.csv"))
priors = CSV.File(joinpath(output_dir,"variance_priors.csv"))
contrasts = ["rTreg_vs_naive","act_naive_vs_naive",
             "act_rTreg_vs_rTreg","act_rTreg_vs_act_naive"]
row(io,values) = println(io,join(values," & ")," \\\\")
methods = ["ttest" => raw"$t$-test / separate rotations",
    "limma" => "Limma", "compound_rotation" => "Compound rotations",
    "ddr_rotation" => "DDR rotations", "compound_rotation_corrected" => "Compound rotations (corrected)",
    "separate_permutation" => "Separate permutations", "compound_permutation" => "Compound permutations",
    "ddr_permutation" => "DDR permutations"]
for alpha in (.05,.10)
    open(joinpath(output_dir,"discoveries_$(Int(100alpha)).tex"),"w") do io
        for (method,label) in methods
            row(io,[label,[string(only(r.rejections for r in counts
                if r.contrast==c && r.method==method && r.alpha==alpha)) for c in contrasts]...])
        end
        gz = [only(r.rejections for r in counts if r.contrast==c && r.alpha==alpha &&
            r.method==(c=="act_naive_vs_naive" ? "gimenez_zou_fixed_first" : "gimenez_zou_full"))
            for c in contrasts]
        row(io,["GZ / SeqStep+",string.(gz)...])
        swaps = [join([string(r.rejections) for r in counts if r.contrast==c &&
            r.alpha==alpha && startswith(r.method,"seqstep_swap_")],", ") for c in contrasts]
        row(io,["SeqStep+ (all donor swaps)",swaps...])
    end
end
open(joinpath(output_dir,"variance_priors.tex"),"w") do io
    for method in ("limma","orbit"), parameter in (:nu0,:s0sq)
        values = [@sprintf("%.5f",getproperty(only(r for r in priors if
            r.contrast==c && r.method==method),parameter)) for c in contrasts]
        row(io,["$method: $(parameter==:nu0 ? raw"$\widehat\nu_0$" : raw"$\widehat s_0^2$")",values...])
    end
end
