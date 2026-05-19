"""
    da_core.jl

True student-optimal deferred acceptance (Gale-Shapley) via batch rounds.

Algorithm
---------
Each round: every free student simultaneously proposes to their next-preferred
school. Each school tentatively keeps its top-capacity applicants by lottery
priority and rejects the rest. Rejected students re-enter the free pool.

This batch formulation produces the SAME stable matching as sequential GS but
exposes parallelism: the school-resolution step is embarrassingly parallel
(schools don't coordinate), and each round completes in O(n/nthreads) with
`parallel_resolve=true`.

Rounds until termination ≤ pref_length (each student proposes to each school
at most once), so the parallel version has at most pref_length synchronisation
barriers — a hard limit set by the market structure, not by n_students.

Contrast with BostonStrategy (serial_sim.jl): Boston runs in one O(n log n)
pass but can leave blocking pairs. DA is O(n × pref_length) but provably stable.
"""

using Random

"""
    run_da(students, schools; parallel_resolve) -> AssignmentResult

Student-optimal DA. Set `parallel_resolve=true` to parallelise the
school-resolution step via Threads.@threads.

Data-race safety: each school's thread writes to disjoint student-id slots of
`assignment` (a student is in at most one school's proposal pool per round).
"""
function run_da(
    students::Vector{Student},
    schools::Vector{School};
    parallel_resolve::Bool = false,
)::AssignmentResult
    isempty(students) && return AssignmentResult(Int[], 0.0, 0, Int[])

    n           = length(students)
    pref_length = length(students[1].preferences)
    max_sid     = maximum(sch.id for sch in schools)

    cap     = zeros(Int, max_sid)
    for sch in schools; cap[sch.id] = sch.capacity; end

    # Dense arrays indexed by student.id (== position in students vector).
    lottery    = [s.lottery_number for s in students]
    prefs      = [s.preferences    for s in students]
    next_prop  = ones(Int, n)      # next rank to propose
    assignment = zeros(Int, n)     # 0 = unassigned / currently free
    # Vector{Bool} (1 byte/element) not BitVector (1 bit/element, packed):
    # concurrent writes to different indices share a word in BitVector → race.
    free       = fill(true, n)

    school_held = [Int[] for _ in 1:max_sid]

    while true
        # ── Proposal step (sequential — each student is independent but writing
        #    to per-school proposal lists needs a single pass to avoid allocating
        #    a lock per school; this step is O(n) and not the bottleneck) ──────
        proposals    = [Int[] for _ in 1:max_sid]
        any_proposal = false

        for sid in 1:n
            free[sid] || continue
            rank = next_prop[sid]
            if rank > pref_length
                free[sid] = false
                continue
            end
            school_id = prefs[sid][rank]
            next_prop[sid] += 1
            push!(proposals[school_id], sid)
            any_proposal = true
        end

        any_proposal || break

        active_schools = findall(!isempty, proposals)
        rejected       = [Int[] for _ in 1:max_sid]  # per-school, merged after

        resolve! = function (school_id)
            pool = vcat(school_held[school_id], proposals[school_id])
            c    = cap[school_id]

            if length(pool) <= c
                school_held[school_id] = pool
                for sid in proposals[school_id]
                    assignment[sid] = school_id
                    free[sid]       = false
                end
            else
                sort!(pool; by = sid -> lottery[sid])
                school_held[school_id] = pool[1:c]

                for sid in pool[1:c]
                    assignment[sid] = school_id
                    free[sid]       = false
                end
                for sid in pool[c+1:end]
                    if assignment[sid] == school_id   # was tentatively held
                        assignment[sid] = 0
                    end
                    push!(rejected[school_id], sid)
                end
            end
        end

        # ── Resolution step: schools are fully independent ───────────────────
        if parallel_resolve
            Threads.@threads for school_id in active_schools
                resolve!(school_id)
            end
        else
            for school_id in active_schools
                resolve!(school_id)
            end
        end

        for school_id in active_schools
            for sid in rejected[school_id]
                free[sid] = true
            end
        end
    end

    rank_dist = _rank_distribution(assignment, prefs, n, pref_length)
    n_assigned = count(!=(0), assignment)

    return AssignmentResult(
        assignment,
        pref_length > 0 ? rank_dist[1] / n : 0.0,
        n - n_assigned,
        rank_dist,
    )
end

"""
    run_da_with_fresh_lottery(students, schools; rng, parallel_resolve)
        -> AssignmentResult

Re-draw lottery numbers and run DA. Used by Monte Carlo orchestration.
"""
function run_da_with_fresh_lottery(
    students::Vector{Student},
    schools::Vector{School};
    rng::AbstractRNG    = Random.GLOBAL_RNG,
    parallel_resolve::Bool = false,
)::AssignmentResult
    redrawn = [Student(s.id, s.preferences, rand(rng)) for s in students]
    return run_da(redrawn, schools; parallel_resolve = parallel_resolve)
end

# ── Internal helper ────────────────────────────────────────────────────────────

function _rank_distribution(
    assignment::Vector{Int},
    prefs::Vector{Vector{Int}},
    n::Int,
    pref_length::Int,
)::Vector{Int}
    rank_dist = zeros(Int, pref_length)
    for sid in 1:n
        school_id = assignment[sid]
        school_id == 0 && continue
        rank = findfirst(==(school_id), prefs[sid])
        rank !== nothing && rank <= pref_length && (rank_dist[rank] += 1)
    end
    return rank_dist
end
