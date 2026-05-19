"""
    monte_carlo.jl

Orchestration and aggregation for Monte Carlo simulations.
"""

using Statistics, Printf

"""
    run_monte_carlo(students, schools, n_trials; strategy, parallel, spawn, seed)
        -> Vector{TrialSummary}

Top-level entry point. Dispatch matrix:

  parallel=false              → serial loop, strategy-aware
  parallel=true,  spawn=false → Threads.@threads (static partition)
  parallel=true,  spawn=true  → Threads.@spawn   (dynamic scheduling)

`strategy` controls the matching algorithm; defaults to `BostonStrategy()` for
backward compatibility with existing call sites.
"""
function run_monte_carlo(
    students::Vector{Student},
    schools::Vector{School},
    n_trials::Int;
    strategy::AbstractMatchingStrategy = BostonStrategy(),
    parallel::Bool = true,
    spawn::Bool    = false,
    seed::Int      = 0,
)::Vector{TrialSummary}
    if !parallel
        return run_serial_mc(strategy, students, schools, n_trials; seed = seed)
    elseif spawn
        return run_parallel_mc_spawn(strategy, students, schools, n_trials; seed = seed)
    else
        return run_parallel_mc(strategy, students, schools, n_trials; seed = seed)
    end
end

"""
    run_serial_mc([strategy,] students, schools, n_trials; seed) -> Vector{TrialSummary}

Single-threaded Monte Carlo loop. Baseline and correctness reference.
"""
function run_serial_mc(
    strategy::AbstractMatchingStrategy,
    students::Vector{Student},
    schools::Vector{School},
    n_trials::Int;
    seed::Int = 0,
)::Vector{TrialSummary}
    results = Vector{TrialSummary}(undef, n_trials)
    rng     = MersenneTwister(seed)
    for i in 1:n_trials
        result     = _run_trial(strategy, students, schools, rng)
        results[i] = to_trial_summary(result)
    end
    return results
end

# Backward-compatible form (Boston strategy).
function run_serial_mc(
    students::Vector{Student},
    schools::Vector{School},
    n_trials::Int;
    seed::Int = 0,
)::Vector{TrialSummary}
    run_serial_mc(BostonStrategy(), students, schools, n_trials; seed = seed)
end

"""
    aggregate_results(summaries) -> NamedTuple

Backward-compatible aggregation (no blocking-pair stats).
"""
function aggregate_results(summaries::Vector{TrialSummary})
    n = length(summaries)
    @assert n > 0 "No trials to aggregate."

    fc_rates    = [s.first_choice_rate for s in summaries]
    unassigned  = [s.unassigned_count  for s in summaries]
    rank_len    = length(summaries[1].rank_distribution)
    rank_matrix = hcat([s.rank_distribution for s in summaries]...)

    return (
        mean_first_choice_rate = mean(fc_rates),
        std_first_choice_rate  = std(fc_rates),
        mean_unassigned        = mean(unassigned),
        std_unassigned         = std(unassigned),
        mean_rank_distribution = vec(mean(rank_matrix; dims = 2)),
        n_trials               = n,
    )
end

"""
    print_summary(agg)

Pretty-print aggregated statistics.
"""
function print_summary(agg)
    println("=" ^ 50)
    println("Monte Carlo Summary  ($(agg.n_trials) trials)")
    println("=" ^ 50)
    @printf "  Mean first-choice rate : %.2f%%  (±%.2f%%)\n" (agg.mean_first_choice_rate * 100) (agg.std_first_choice_rate * 100)
    @printf "  Mean unassigned        : %.1f  (±%.1f)\n" agg.mean_unassigned agg.std_unassigned
    println("  Rank distribution (mean fraction):")
    for (k, frac) in enumerate(agg.mean_rank_distribution)
        @printf "    Rank %d : %.2f%%\n" k frac * 100
    end
    println("=" ^ 50)
end

"""
    compare_strategies(students, schools, n_trials; seed) -> NamedTuple

Run Boston, DA, and BatchedDA(4) on the same problem, return side-by-side
quality and timing. This is the entry point for non-theater benchmarking:
it measures BOTH runtime AND blocking pairs so you can see the quality/speed
frontier directly.
"""
function compare_strategies(
    students::Vector{Student},
    schools::Vector{School},
    n_trials::Int;
    seed::Int = 0,
)
    strategies = [
        ("Boston",       BostonStrategy()),
        ("DA serial",    DAStrategy()),
        ("DA parallel",  ParallelDAStrategy()),
        ("BatchedDA(4)", BatchedDAStrategy(4)),
    ]

    results = []
    for (name, strat) in strategies
        t0       = time()
        summaries = run_monte_carlo(students, schools, n_trials;
                                    strategy = strat, parallel = true, seed = seed)
        elapsed  = time() - t0
        agg      = aggregate_results(summaries)
        bp_mean  = mean(s.blocking_pairs for s in summaries if s.blocking_pairs >= 0;
                        init = -1.0)
        push!(results, (
            name              = name,
            elapsed_s         = elapsed,
            mean_fc_rate      = agg.mean_first_choice_rate,
            std_fc_rate       = agg.std_first_choice_rate,
            mean_unassigned   = agg.mean_unassigned,
            mean_blocking_pairs = bp_mean,
        ))
    end

    _print_comparison(results, n_trials)
    return results
end

function _print_comparison(results, n_trials)
    println("\n", "=" ^ 80)
    println("Strategy Comparison  ($n_trials trials, $(Threads.nthreads()) threads)")
    println("=" ^ 80)
    @printf("  %-16s  %8s  %10s  %10s  %10s  %12s\n",
            "Strategy", "time(s)", "FC rate", "±σ", "unassigned", "block.pairs")
    println("-" ^ 80)
    for r in results
        bp_str = r.mean_blocking_pairs < 0 ? "    n/a" : @sprintf("%10.1f", r.mean_blocking_pairs)
        @printf("  %-16s  %8.3f  %10.3f  %10.4f  %10.1f  %s\n",
                r.name, r.elapsed_s, r.mean_fc_rate, r.std_fc_rate, r.mean_unassigned, bp_str)
    end
    println("=" ^ 80)
end
