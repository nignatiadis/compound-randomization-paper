using CSV
using Random
using Statistics

include("main.jl")

"""Validate one cluster replicate; partial or mismatched files are never averaged."""
function read_replicate(path, s, rep)
    rows = collect(CSV.File(path; types=Dict(:rng_jump=>String), strict=true))
    length(rows) == length(MainSimulation.METHOD_NAMES) &&
        Set(row.method for row in rows) == Set(MainSimulation.METHOD_NAMES) ||
        error("Expected exactly one row for each of the 14 methods: $path")
    expected = (; scenario=s.id, rep, master_seed=MainSimulation.MASTER_SEED,
        rng_jump=string(MainSimulation.RNG_JUMP),
        split_seed=rand(copy(MainSimulation.REPLICATE_RNGS[rep]), 0:Int(typemax(Int32))),
        m=s.n, K=s.K, noise=lowercase(string(nameof(typeof(s.noise)))),
        pi1=s.π1, alpha=s.α, lambda=s.λ, nu0=s.ν0, s0sq=s.s0²,
        quantile_low=MainSimulation.QUANTILES[1], quantile_high=MainSimulation.QUANTILES[2],
        scale_quantile=0.5)
    length(unique(row.nnonnull for row in rows)) == 1 || error("Different datasets across methods: $path")
    for row in rows
        all(isequal(row[key], value) for (key,value) in pairs(expected)) ||
            error("Settings or RNG metadata mismatch: $path")
        level = endswith(row.method, "_corrected") ? s.α/1.93 : s.α
        tau = startswith(row.method, "ddr_") ? s.α/10 : missing
        isequal(row.bh_level,level) && isequal(row.tau,tau) || error("Incorrect calibration: $path")
        A, R, V, T = row.nnonnull, row.discoveries, row.false_discoveries, row.true_discoveries
        all(x -> x isa Integer, (A,R,V,T)) && 0 <= A <= s.n &&
            0 <= T <= A && 0 <= V <= s.n-A && R == V+T || error("Invalid counts: $path")
        row.fdp isa Real && isfinite(row.fdp) &&
            isapprox(row.fdp,V/max(R,1); rtol=1e-12,atol=0) || error("Invalid FDP: $path")
        row.power isa Real && isfinite(row.power) &&
            isapprox(row.power,T/max(A,1); rtol=1e-12,atol=0) || error("Invalid power: $path")
    end
    NamedTuple.(rows)
end

"""Monte Carlo standard error across replicates, undefined for one replicate."""
mcse(values) = length(values) > 1 ? std(values; corrected=true)/sqrt(length(values)) : missing

"""
    summarize_main(input, output)

Validate the expected scenario/replicate files and average replicate FDP and
power, not pooled discovery ratios. Use every available complete replicate and
record its actual count. Incomplete coverage is marked in every summary row,
without preventing aggregation. Missing files are listed in `missing.csv`; temporary files are
ignored. No simulation is run and no input file is modified.
"""
function summarize_main(input, output;
    settings=MainSimulation.SETTINGS, nreps=MainSimulation.NREPS)
    isdir(input) || error("Result directory not found: $input")
    abspath(input) != abspath(output) || error("Use a separate summary directory")
    1 <= nreps <= MainSimulation.NREPS || error("Invalid replicate count")
    expected = Set("scenario_$(s.id)_rep_$rep.csv" for s in settings for rep in 1:nreps)
    actual = filter(name -> startswith(name,"scenario_") && endswith(name,".csv"), readdir(input))
    unexpected = setdiff(Set(actual),expected)
    isempty(unexpected) || error("Unexpected result filenames: $(join(sort!(collect(unexpected)), ", "))")
    missing_rows = [(scenario=s.id,rep) for s in settings
        for rep in 1:nreps if !isfile(joinpath(input,"scenario_$(s.id)_rep_$rep.csv"))]
    mkpath(output)
    CSV.write(joinpath(output,"missing.csv"), missing_rows; header=[:scenario,:rep])
    preview = !isempty(missing_rows)
    summary = NamedTuple[]
    for s in settings
        methods = Dict(method=>NamedTuple[] for method in MainSimulation.METHOD_NAMES)
        for rep in 1:nreps
            path = joinpath(input,"scenario_$(s.id)_rep_$rep.csv")
            isfile(path) || continue
            for row in read_replicate(path,s,rep)
                push!(methods[row.method],row)
            end
        end
        reps = length(methods[first(MainSimulation.METHOD_NAMES)])
        println("Scenario $(s.id): $reps/$nreps validated replicates"); flush(stdout)
        reps == 0 && continue
        noise = lowercase(string(nameof(typeof(s.noise))))
        panels = String[]
        noise == "normal" && s.π1 == 0.025 && push!(panels,"A")
        noise == "normal" && s.K == 5 && push!(panels,"B")
        noise == "uniform" && push!(panels,"C")
        noise == "laplace" && push!(panels,"D")
        for method in MainSimulation.METHOD_NAMES
            rows = methods[method]
            fdp, power = getproperty.(rows,:fdp), getproperty.(rows,:power)
            discoveries = getproperty.(rows,:discoveries)
            for panel in panels
                push!(summary,(; panel,scenario=s.id,method,source="cluster",reps,
                    expected_reps=nreps,preview,m=s.n,K=s.K,pi1=s.π1,
                    alpha=s.α,lambda=s.λ,nu0=s.ν0,s0sq=s.s0²,noise,
                    master_seed=MainSimulation.MASTER_SEED,rng_jump=string(MainSimulation.RNG_JUMP),
                    tau=first(rows).tau,bh_level=first(rows).bh_level,
                    quantile_low=MainSimulation.QUANTILES[1],quantile_high=MainSimulation.QUANTILES[2],
                    mean_nonnull=mean(getproperty.(rows,:nnonnull)),
                    mean_discoveries=mean(discoveries),se_discoveries=mcse(discoveries),
                    fdr=mean(fdp),se_fdr=mcse(fdp),power=mean(power),se_power=mcse(power)))
            end
        end
    end
    isempty(summary) && error("No completed replicates found")
    sort!(summary; by=row->(row.panel,row.K,row.pi1,row.method))
    path = joinpath(output, preview ? "summary_preview.csv" : "summary.csv")
    CSV.write(path,summary)
    println("Saved $path; $(length(expected)-length(missing_rows)) / $(length(expected)) replicates")
    summary
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    length(ARGS) == 2 || error("Usage: simulations/summarize.jl INPUT_DIRECTORY OUTPUT_DIRECTORY")
    summarize_main(ARGS[1],ARGS[2])
end
