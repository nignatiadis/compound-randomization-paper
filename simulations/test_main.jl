using Test
using Random
using Distributions
using Future: randjump
using CompoundRandomization
include("main.jl")
using .MainSimulation

@testset "Main simulation settings and task allocation" begin
    @test length(MainSimulation.SETTINGS) == 22
    @test MainSimulation.QUANTILES == (0.25, 0.75)
    for s in MainSimulation.SETTINGS
        @test mean(s.noise) ≈ 0 atol=1e-15
        @test var(s.noise) ≈ 1
        @test isconcretetype(typeof(s))
    end
    assignments = Tuple{Int,Int}[]
    job_counts = (k3_5=20, k7_9=30, k11=30, k13=60)
    memory = (k3_5="4G", k7_9="6G", k11="8G", k13="16G")
    time_limits = (k3_5="06:00:00", k7_9="06:00:00", k11="06:00:00", k13="10:00:00")
    submission = read(joinpath(@__DIR__, "submit_main.sh"), String)
    for (group, count) in pairs(job_counts)
        config = MainSimulation.TASK_GROUPS[group]
        line = only(filter(line -> endswith(line, "main.slurm $group"), split(submission, '\n')))
        @test occursin("--array=1-$count ", line)
        @test occursin("--mem=$(memory[group]) ", line)
        @test occursin("--time=$(time_limits[group]) ", line)
        for task in 1:count
            scenario, reps = MainSimulation.task_replicates(group, task)
            @test scenario.K in config.K
            @test length(reps) == config.reps
            append!(assignments, [(scenario.id, rep) for rep in reps])
        end
        @test_throws ArgumentError MainSimulation.task_replicates(group, 0)
        @test_throws ArgumentError MainSimulation.task_replicates(group, count + 1)
    end
    @test sum(values(job_counts)) == 140
    @test length(assignments) == 22 * 500
    @test length(unique(assignments)) == 22 * 500
    @test Set(assignments) == Set((s, rep) for s in 1:22 for rep in 1:500)
    @test_throws ArgumentError MainSimulation.task_replicates(:unknown, 1)
    @test_throws ArgumentError MainSimulation.task_replicates(:k13, 1; nreps=0)
    scenario, reps = MainSimulation.task_replicates(:k13, 2; nreps=26)
    @test scenario.noise isa Normal && scenario.K == 13 && scenario.π1 == 0.025
    @test reps == 26:26
end

@testset "Jumped replicate streams" begin
    rngs = MainSimulation.REPLICATE_RNGS
    @test length(rngs) == 500
    @test first(rngs) == MersenneTwister(MainSimulation.MASTER_SEED)
    for rep in 2:length(rngs)
        @test rngs[rep] == randjump(rngs[rep - 1], MainSimulation.RNG_JUMP)
    end
    @test length(unique(rand(copy(rng), UInt64) for rng in rngs)) == 500
    @test rand(copy(rngs[7]), 10) == rand(copy(rngs[7]), 10)
end

@testset "Shared datasets and metrics" begin
    for noise in (Normal(), Uniform(-sqrt(3.0), sqrt(3.0)),
        Laplace(0.0, inv(sqrt(2.0))), sqrt(3/5) * TDist(5))
        s = MainSimulation.Scenario(id=1, noise=noise, K=3, π1=0.025, n=100)
        data = MainSimulation.simulate(MersenneTwister(123), s)
        @test data == MainSimulation.simulate(MersenneTwister(123), s)
        @test size(data.X) == (100, 3)
        @test all(iszero, data.μ[.!data.nonnull])
        @test all(>(0), data.σ²)
        if noise isa Uniform
            @test all(abs.(data.X .- data.μ) .<= sqrt.(3 .* data.σ²))
        end
    end
    metrics = MainSimulation.metrics([true, true, false, false], [true, false, true, false])
    @test metrics.discoveries == 2
    @test metrics.fdp == 0.5
    @test metrics.power == 0.5
    @test MainSimulation.metrics(falses(4), falses(4)).power == 0
    @test MainSimulation.metrics(falses(4), falses(4)).fdp == 0
end

@testset "Small package-based main simulation" begin
    s = MainSimulation.Scenario(id=1, noise=Normal(), K=3, π1=0.025, n=100)
    original_rng = copy(MainSimulation.REPLICATE_RNGS[1])
    rows = MainSimulation.run_replicate(s, 1; include_sens=false)
    @test length(rows) == 12
    @test all(row -> row.noise == "normal", rows)
    @test Set(row.method for row in rows) == Set(MainSimulation.METHOD_NAMES[1:12])
    @test all(row -> 0 <= row.fdp <= 1 && 0 <= row.power <= 1, rows)
    @test length(unique(row.nnonnull for row in rows)) == 1
    @test all(row -> row.quantile_low == 0.25 && row.quantile_high == 0.75, rows)
    @test isequal(rows, MainSimulation.run_replicate(s, 1; include_sens=false))
    @test MainSimulation.REPLICATE_RNGS[1] == original_rng
    @test all(row -> row.master_seed == MainSimulation.MASTER_SEED, rows)
    @test length(unique(row.split_seed for row in rows)) == 1
    @test all(row -> row.bh_level == s.α / 1.93,
        filter(row -> endswith(row.method, "_corrected"), rows))
    @test all(row -> row.tau == s.α / 10,
        filter(row -> startswith(row.method, "ddr_"), rows))
    for group in ("rotation", "signflip")
        ordinary = only(filter(row -> row.method == "compound_$(group)_quantile", rows))
        corrected = only(filter(row -> row.method == "compound_$(group)_quantile_corrected", rows))
        @test corrected.discoveries <= ordinary.discoveries
    end
end
