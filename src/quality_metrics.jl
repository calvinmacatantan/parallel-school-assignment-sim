"""
    quality_metrics.jl

Assignment quality measurement: blocking pairs, rank statistics, variance.

Blocking pairs are the canonical stability measure for matching markets.
A blocking pair (s, x) exists when:
  1. Student s prefers school x to their current assignment (or is unassigned).
  2. School x would accept s: either x has spare capacity, or x holds a student
     with a worse lottery number than s (lower lottery = higher priority).

DA produces 0 blocking pairs by construction.
Boston (immediate acceptance) can produce >0, especially under high demand.
Batched DA produces blocking pairs only at band boundaries.
"""

using Statistics

"""
    count_blocking_pairs(result, students, schools) -> Int

O(n × pref_length) with O(n) preprocessing.
"""
function count_blocking_pairs(
    result::AssignmentResult,
    students::Vector{Student},
    schools::Vector{School},
)::Int
    n = length(students)
    isempty(students) && return 0

    lottery = [s.lottery_number for s in students]    # indexed by s.id
    prefs   = [s.preferences    for s in students]

    # Per-school: count fills and track worst (highest) lottery held.
    school_fill         = Dict(sch.id => 0      for sch in schools)
    worst_lottery_held  = Dict(sch.id => -Inf   for sch in schools)

    for sid in 1:n
        school_id = result.assignments[sid]
        school_id == 0 && continue
        school_fill[school_id] += 1
        if lottery[sid] > worst_lottery_held[school_id]
            worst_lottery_held[school_id] = lottery[sid]
        end
    end

    cap       = Dict(sch.id => sch.capacity for sch in schools)
    remaining = Dict(id => cap[id] - fill for (id, fill) in school_fill)

    pairs = 0
    for s in students
        assigned      = result.assignments[s.id]
        assigned_rank = _assigned_rank(assigned, prefs[s.id])

        for (rank, school_id) in enumerate(prefs[s.id])
            rank >= assigned_rank && break   # no better school further down list

            if remaining[school_id] > 0
                pairs += 1
            elseif s.lottery_number < worst_lottery_held[school_id]
                pairs += 1
            end
        end
    end

    return pairs
end

"""
    mean_assigned_rank(result, students) -> Float64

Average rank at which assigned students were placed (1.0 = all first choice).
Only counts assigned students in the denominator.
"""
function mean_assigned_rank(
    result::AssignmentResult,
    students::Vector{Student},
)::Float64
    prefs = [s.preferences for s in students]
    total = 0.0
    count = 0
    for sid in 1:length(students)
        school_id = result.assignments[sid]
        school_id == 0 && continue
        rank = findfirst(==(school_id), prefs[sid])
        rank === nothing && continue
        total += rank
        count += 1
    end
    return count > 0 ? total / count : NaN
end

"""
    quality_summary(result, students, schools) -> NamedTuple

Single call returning all quality metrics for one assignment result.
"""
function quality_summary(
    result::AssignmentResult,
    students::Vector{Student},
    schools::Vector{School},
)
    bp       = count_blocking_pairs(result, students, schools)
    mean_rk  = mean_assigned_rank(result, students)
    n        = length(students)
    n_assigned = n - result.unassigned_count

    return (
        blocking_pairs       = bp,
        mean_assigned_rank   = mean_rk,
        first_choice_rate    = result.first_choice_rate,
        unassigned_count     = result.unassigned_count,
        assigned_fraction    = n_assigned / n,
        rank_distribution    = result.rank_distribution,
    )
end

"""
    aggregate_quality(summaries) -> NamedTuple

Cross-trial statistics from a vector of `TrialSummary`. Includes variance of
first-choice rate (match-effect variance) and mean blocking pairs.
"""
function aggregate_quality(summaries::Vector{TrialSummary})
    n = length(summaries)
    @assert n > 0

    fc     = [s.first_choice_rate for s in summaries]
    unassg = [s.unassigned_count  for s in summaries]
    bp     = [s.blocking_pairs    for s in summaries]

    rank_len    = length(summaries[1].rank_distribution)
    rank_matrix = hcat([s.rank_distribution for s in summaries]...)  # rank_len × n

    return (
        mean_first_choice_rate = mean(fc),
        std_first_choice_rate  = std(fc),
        var_first_choice_rate  = var(fc),   # match-effect variance
        mean_unassigned        = mean(unassg),
        std_unassigned         = std(unassg),
        mean_blocking_pairs    = mean(bp),
        std_blocking_pairs     = std(bp),
        mean_rank_distribution = vec(mean(rank_matrix; dims = 2)),
        n_trials               = n,
    )
end

# ── Internal ───────────────────────────────────────────────────────────────────

function _assigned_rank(assigned::Int, pref::Vector{Int})::Int
    assigned == 0 && return length(pref) + 1
    idx = findfirst(==(assigned), pref)
    return idx === nothing ? length(pref) + 1 : idx
end
