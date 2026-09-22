module CompoundRandomization

using Combinatorics: combinations, permutations
using Distributions
import Empirikos
using LinearAlgebra
using MultipleTesting
import Roots
using Statistics
import StatsBase: fit, nobs

export AbstractRandomizationSample, ReplicatedSample, AbsMean, ModeratedTScore, sign_symmetric,
    RegressionDesign, RegressionSample, AbsCoefficient, ResidualRotations, StratifiedPermutations,
    TwoSample, AbsMeanDifference, Permutations, CenteredRotations, within_group_symmetric, label_symmetric,
    AbstractRandomizationGroup, AbstractMultipleTestingProcedure, AbstractRandomizationReference,
    SignFlips, OrthogonalRotations, HalfSplit, InvolutionGroup, RandomizationFit, MultipleRandomizationTest,
    CompoundBH, SeparateBH, DDR, SeqStepPlus,
    LocalFDROracle, SENS,
    fit, fit_statistic, fit_reference, nobs, orbit_variance, orbit_tail, compound_pvalues, separate_pvalues

include("samples.jl")
include("scores.jl")
include("orbits.jl")
include("calibration.jl")
include("testing.jl")
include("baselines.jl")

end
