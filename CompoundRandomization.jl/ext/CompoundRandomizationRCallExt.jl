module CompoundRandomizationRCallExt

using CompoundRandomization
import CompoundRandomization: fit, check_level, checked_samples
using RCall

function fit(method::SENS, data::AbstractVector{<:ReplicatedSample})
    check_level(method.α)
    method.variant in (:general, :gaussian) || throw(ArgumentError("SENS variant must be :general or :gaussian"))
    0 <= method.seed <= typemax(Int32) || throw(ArgumentError("R seed must be a nonnegative Int32"))
    isfile(method.script) || throw(ArgumentError("SENS script does not exist"))
    samples = checked_samples(data)
    Threads.threadid() == 1 || throw(ArgumentError("run SENS on a process's main Julia thread"))
    X = permutedims(stack(s.Z for s in samples))
    runner = R"""
    function(X, alpha, option, script, seed) {
        env <- new.env(parent = globalenv())
        sys.source(script, envir = env)
        set.seed(seed)
        env[["SENS"]](X, alpha, option = option)
    }
    """
    result = rcall(runner, X, method.α,
        method.variant == :gaussian ? "Gaussian" : "General", abspath(method.script), method.seed)
    rejected = Bool.(rcopy(result[:de]))
    (; method, cutoff = rcopy(result[:th]), rj_idx = rejected,
        total_rejections = count(rejected))
end

end
