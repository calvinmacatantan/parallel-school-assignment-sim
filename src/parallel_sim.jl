"""
    parallel_sim.jl

Trial-level parallelism for Monte Carlo simulation.

Two schedulers are provided:

`run_parallel_mc` (Threads.@threads, static)
  Work is divided into equal chunks before launch. Zero overhead per trial.
  Best when all trials take the same time (they do here — same n_students,
  same pref_length). Kept for backward compatibility.

`run_parallel_mc_spawn` (Threads.@spawn, dynamic)
  Each trial is a separate `Task` submitted to the thread pool. Julia's
  scheduler steals tasks from idle threads. Has slightly more overhead but
  is resilient to trial-time variance and scales better when n_trials is
  not a multiple of nthreads().

Strategy dispatch
-----------------
Both runners accept an `AbstractMatchingStrategy` so the same harness drives
Boston, DA, ParallelDA, and BatchedDA without duplication.
"""

using Random

# ── Strategy → single-trial function ──────────────────────────────────────────

function _run_trial(
    ::BostonStrategy,
    students::Vector{Student},
    schools::Vector{School},
    rng::AbstractRNG,
)::AssignmentResult
    run_serial_with_fresh_lottery(students, schools; rng = rng)
end

function _run_trial(
    ::DAStrategy,
    students::Vector{Student},
    schools::Vector{School},
    rng::AbstractRNG,
)::AssignmentResult
    run_da_with_fresh_lottery(students, schools; rng = rng, parallel_resolve = false)
end

function _run_trial(
    ::ParallelDAStrategy,
    students::Vector{Student},
    schools::Vector{School},
    rng::AbstractRNG,
)::AssignmentResult
    run_da_with_fresh_lottery(students, schools; rng = rng, parallel_resolve = true)
end

function _run_trial(
    s::BatchedDAStrategy,
    students::Vector{Student},
    schools::Vector{School},
    rng::AbstractRNG,
)::AssignmentResult
    run_batched_da_with_fresh_lottery(
        students, schools;
        n_batches = s.n_batches,
        rng       = rng,
        parallel_resolve = true,
    )
end

# ── Static scheduler (original, @threads) ─────────────────────────────────────

"""
    run_parallel_mc([strategy,] students, schools, n_trials; seed) -> Vector{TrialSummary}

Parallel Monte Carlo with static thread partitioning (`Threads.@threads`).
`strategy` defaults to `BostonStrategy()` for backward compatibility.
"""
function run_parallel_mc(
    strategy::AbstractMatchingStrategy,
    students::Vector{Student},
    schools::Vector{School},
    n_trials::Int;
    seed::Int = 0,
)::Vector{TrialSummary}
    results = Vector{TrialSummary}(undef, n_trials)
    rngs    = [MersenneTwister(seed + i) for i in 1:n_trials]

    Threads.@threads for i in 1:n_trials
        result    = _run_trial(strategy, students, schools, rngs[i])
        results[i] = to_trial_summary(result)
    end

    return results
end

# Backward-compatible two-argument form (Boston strategy, no quality metrics).
function run_parallel_mc(
    students::Vector{Student},
    schools::Vector{School},
    n_trials::Int;
    seed::Int = 0,
)::Vector{TrialSummary}
    run_parallel_mc(BostonStrategy(), students, schools, n_trials; seed = seed)
end

# ── Dynamic scheduler (@spawn) ────────────────────────────────────────────────

"""
    run_parallel_mc_spawn([strategy,] students, schools, n_trials; seed)
        -> Vector{TrialSummary}

Parallel Monte Carlo using `Threads.@spawn` (dynamic task scheduling).
Better than @threads when trial times are uneven or n_trials % nthreads() ≠ 0.
Each trial is an independent `Task`; Julia's work-stealing scheduler fills idle
threads without a static partition step.
"""
function run_parallel_mc_spawn(
    strategy::AbstractMatchingStrategy,
    students::Vector{Student},
    schools::Vector{School},
    n_trials::Int;
    seed::Int = 0,
)::Vector{TrialSummary}
    rngs  = [MersenneTwister(seed + i) for i in 1:n_trials]
    tasks = [
        Threads.@spawn _run_trial(strategy, students, schools, rngs[i])
        for i in 1:n_trials
    ]
    return [to_trial_summary(fetch(t)) for t in tasks]
end

function run_parallel_mc_spawn(
    students::Vector{Student},
    schools::Vector{School},
    n_trials::Int;
    seed::Int = 0,
)::Vector{TrialSummary}
    run_parallel_mc_spawn(BostonStrategy(), students, schools, n_trials; seed = seed)
end

# ── Summary conversion ─────────────────────────────────────────────────────────

"""
    to_trial_summary(result) -> TrialSummary

Convert `AssignmentResult` to the lightweight `TrialSummary` used for
MC aggregation. Blocking pairs are NOT computed here (expensive); call
`count_blocking_pairs` explicitly when you need them.
"""
function to_trial_summary(result::AssignmentResult)::TrialSummary
    n_students = length(result.assignments)
    rank_fracs = result.rank_distribution ./ n_students
    return TrialSummary(
        result.first_choice_rate,
        result.unassigned_count,
        rank_fracs,
        -1,   # -1 = not computed
    )
end

"""
    to_trial_summary_with_quality(result, students, schools) -> TrialSummary

Like `to_trial_summary` but also computes blocking pairs. Use only when you
need quality metrics — it adds O(n × pref_length) per trial.
"""
function to_trial_summary_with_quality(
    result::AssignmentResult,
    students::Vector{Student},
    schools::Vector{School},
)::TrialSummary
    n_students = length(result.assignments)
    rank_fracs = result.rank_distribution ./ n_students
    bp         = count_blocking_pairs(result, students, schools)
    return TrialSummary(
        result.first_choice_rate,
        result.unassigned_count,
        rank_fracs,
        bp,
    )
end

# ── Scaling experiment (simulates k-thread budgets via work subdivision) ───────

"""
    run_parallel_scaling(students, schools, n_trials; thread_counts, seed)
        -> Dict{Int, Float64}

Benchmark parallel throughput at different logical thread-count limits by
running `n_trials ÷ k` trials for each k in `thread_counts` and measuring
wall-clock time. Returns thread_count -> elapsed_seconds.

Note: this measures sequential work-per-thread, not true multi-thread speedup
(Julia's thread count is fixed at startup). For real speedup measurement use
`run_all_benchmarks` with `julia -t N`.
"""
function run_parallel_scaling(
    students::Vector{Student},
    schools::Vector{School},
    n_trials::Int;
    thread_counts::Vector{Int} = [1, 2, 4, 8],
    seed::Int = 0,
)::Dict{Int,Float64}
    times = Dict{Int,Float64}()
    for k in thread_counts
        chunk = max(1, n_trials ÷ k)
        t0    = time()
        run_parallel_mc(students, schools, chunk; seed = seed)
        times[k] = time() - t0
    end
    return times
end
