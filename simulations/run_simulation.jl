using CSV
using LinearAlgebra
using RCall

include("main.jl")
using .MainSimulation

function main(args)
    length(args) in (2, 3) || error("Usage: simulations/run_simulation.jl GROUP TASK_ID [OUTPUT_DIRECTORY]")
    BLAS.set_num_threads(1)
    scenario, reps = MainSimulation.task_replicates(Symbol(args[1]), parse(Int, args[2]))
    directory = length(args) == 3 ? args[3] : joinpath(@__DIR__, "..", "results", "main")
    mkpath(directory)
    for rep in reps
        path = joinpath(directory, "scenario_$(scenario.id)_rep_$rep.csv")
        isfile(path) && error("Result already exists: $path. Use a fresh output directory for reruns.")
        println("Scenario $(scenario.id), replicate $rep")
        flush(stdout)
        rows = MainSimulation.run_replicate(scenario, rep)
        CSV.write(path * ".tmp", rows)
        mv(path * ".tmp", path)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
