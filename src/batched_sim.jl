"""
    batched_sim.jl

Band-based batched deferred acceptance.

Students are sorted by lottery number and divided into `n_batches` equal-size
priority bands. Each band runs student-optimal DA against whatever school
capacity remains after earlier bands. Schools do NOT displace students across
band boundaries — a student admitted in band k will not be evicted by a
band-k+1 student even if that student has a lower (better) lottery number.

Trade-offs vs true DA
---------------------
- Quality: band boundaries introduce blocking pairs between bands (a band-2
  student may prefer a school that admitted a worse-lottery band-1 student).
  The number of cross-band blocking pairs grows with n_batches.
- Speed: the n_batches band runs are sequential (later bands depend on earlier
  capacity), but the intra-band DA uses parallel school resolution, and with
  n_batches = n_students all bands trivially parallelise (trivial DA per band).
- Scaling: at fixed n_students, increasing n_batches improves intra-band
  parallel efficiency but degrades match quality. This is the core trade-off
  this variant is designed to surface for benchmarking.
"""

using Random

"""
    run_batched_da(students, schools; n_batches, parallel_resolve)
        -> AssignmentResult

Run band-based batched DA with `n_batches` priority bands.
"""
function run_batched_da(
    students::Vector{Student},
    schools::Vector{School};
    n_batches::Int         = 4,
    parallel_resolve::Bool = true,
)::AssignmentResult
    isempty(students) && return AssignmentResult(Int[], 0.0, 0, Int[])

    n           = length(students)
    pref_length = length(students[1].preferences)

    # Sort by lottery (ascending = higher priority).
    sorted = sort(students; by = s -> s.lottery_number)

    # Mutable per-school remaining capacity.
    remaining = Dict(sch.id => sch.capacity for sch in schools)

    all_assignments = zeros(Int, n)   # indexed by original student.id

    band_size  = ceil(Int, n / n_batches)

    for band_idx in 1:n_batches
        lo   = (band_idx - 1) * band_size + 1
        hi   = min(band_idx * band_size, n)
        band = sorted[lo:hi]

        isempty(band) && break
        all(remaining[sch.id] == 0 for sch in schools) && break

        # Snapshot schools with current remaining capacity.
        band_schools = [School(sch.id, remaining[sch.id], sch.name) for sch in schools]

        result = run_da(band, band_schools; parallel_resolve = parallel_resolve)

        # run_da returns a position-indexed vector (1:length(band)), not id-indexed.
        for (pos, s) in enumerate(band)
            school_id = result.assignments[pos]
            school_id == 0 && continue
            all_assignments[s.id] = school_id
            remaining[school_id] -= 1
        end
    end

    # Aggregate stats over all_assignments.
    prefs     = [s.preferences for s in students]
    rank_dist = _rank_distribution(all_assignments, prefs, n, pref_length)
    n_assigned = count(!=(0), all_assignments)

    return AssignmentResult(
        all_assignments,
        pref_length > 0 ? rank_dist[1] / n : 0.0,
        n - n_assigned,
        rank_dist,
    )
end

"""
    run_batched_da_with_fresh_lottery(students, schools; n_batches, rng)
        -> AssignmentResult
"""
function run_batched_da_with_fresh_lottery(
    students::Vector{Student},
    schools::Vector{School};
    n_batches::Int      = 4,
    rng::AbstractRNG    = Random.GLOBAL_RNG,
    parallel_resolve::Bool = true,
)::AssignmentResult
    redrawn = [Student(s.id, s.preferences, rand(rng)) for s in students]
    return run_batched_da(redrawn, schools; n_batches = n_batches, parallel_resolve = parallel_resolve)
end
