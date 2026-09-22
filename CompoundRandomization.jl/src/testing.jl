"""
    MultipleRandomizationTest(; group, statistic, procedure=CompoundBH())

End-to-end multiple testing with a fixed or orbit-learned statistic S-hat.
`group` implements `AbstractRandomizationGroup`; `procedure` implements
`AbstractMultipleTestingProcedure` (`CompoundBH`, `SeparateBH`, `DDR`, `SeqStepPlus`, or `GimenezZou`).

`fit(method, samples)` prepares the statistic using `fit_statistic(group, statistic, samples)`,
constructs its randomization reference distributions, and applies the procedure.
It returns the procedure's result: all have `rj_idx`; p-value methods also
have `pvalue` and `adjp`. SeqStep+ requires an `InvolutionGroup` and uses the
paired observed/calibration scores, not p-values.
For an Empirikos Limma estimator wrapped in `ModeratedTScore`, prior fitting uses
`orbit_variance(group, sample)`, with the group-appropriate degrees of freedom.
Other callables are treated as fixed; their orbit measurability is the caller's
responsibility. The statistic is never refitted on randomized data.

```julia
samples = ReplicatedSample.(eachrow(X))
method = MultipleRandomizationTest(
    group=SignFlips(),
    statistic=ModeratedTScore(Empirikos.QuantileLimma()),
    procedure=CompoundBH(α=0.1))
result = fit(method, samples)
```

To reuse a reference distribution across procedures or levels, use the
intermediate `fit_reference(group, score, samples)` interface instead.
For SeqStep+, use `group=InvolutionGroup()` for the default half/half sign flip,
or `InvolutionGroup(H)` for another fixed involution (Section 3.4).
For larger finite groups, use `GimenezZou()`, counting every group element.
For example, `StratifiedPermutations(fixed=[1])` fixes the first observation.
Sample types must subtype `AbstractRandomizationSample`; new testing problems
provide their own summary adapters and group-specific reference construction.
"""
Base.@kwdef struct MultipleRandomizationTest{G<:AbstractRandomizationGroup,S,
    P<:AbstractMultipleTestingProcedure}
    group::G
    statistic::S
    procedure::P = CompoundBH()
end

function fit(method::MultipleRandomizationTest, samples::AbstractVector{<:AbstractRandomizationSample})
    checked_samples(samples)
    statistic = fit_statistic(method.group, method.statistic, samples)
    reference = fit_reference(method.group, statistic, samples)
    fit(method.procedure, reference)
end
