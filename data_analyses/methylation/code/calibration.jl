"""
Exact BH decisions from a monotone pooled rotation tail, without computing all
n p-values. Starting at k=n, count p_i <= alpha*k/n and repeat until k stops
decreasing. Each count uses binary search in the score ranking. This is BH's
step-up rejection set, including complete ties. DDR additionally caps the
threshold at tau and supplies its orbit weights.
"""
function rotation_bh(r; α, τ=1.0, weights=ones(length(r.observed)))
    n = length(r.observed)
    all(isfinite, weights) || return falses(n)
    scores = sort(r.observed; rev=true)
    cache = Dict{Int,Float64}()
    k = n
    while k > 0
        bound = min(τ, α * k / n)
        lower, upper = 0, k
        while lower < upper
            j = (lower + upper + 1) ÷ 2
            p = get!(cache, j) do
                sum(weights[i] * orbit_tail(r, i, scores[j]) for i in 1:n) / n
            end
            if p <= bound
                lower = j
            else
                upper = j - 1
            end
        end
        lower == k && return r.observed .>= scores[k]
        k = lower
    end
    falses(n)
end

"""
Find the finite-reference DDR cutoff without copying all randomized scores.
Keep an interval in each sorted column and bisect the largest remaining one.
Right-limit tail odds locate the cutoff; the final weights use inclusive tails.
"""
function finite_ddr_cutoff(reference, τ)
    L, n = size(reference.sorted_scores)
    lower, upper = ones(Int, n), fill(L, n)
    cutoff = Inf
    while any(lower .<= upper)
        j = argmax(upper .- lower)
        s = reference.sorted_scores[(lower[j] + upper[j]) ÷ 2, j]
        if CompoundRandomization.ddr_tail_odds(reference, s) <= τ
            cutoff = s
            for (i, scores) in enumerate(eachcol(reference.sorted_scores))
                upper[i] = searchsortedfirst(scores, s) - 1
            end
        else
            for (i, scores) in enumerate(eachcol(reference.sorted_scores))
                lower[i] = searchsortedlast(scores, s) + 1
            end
        end
    end
    cutoff
end
