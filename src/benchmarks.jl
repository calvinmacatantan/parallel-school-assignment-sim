"""
    benchmarks.jl

BenchmarkTools.jl-based benchmarks for the serial and parallel assignment
simulation. Reports runtime, memory allocations, and thread-scaling behaviour.
"""

using BenchmarkTools
using Printf

"""
    benchmark_serial(students, schools; n_warmup, n_samples)
        -> BenchmarkTools.Trial

Benchmark a single serial assignment run (no Monte Carlo overhead).
"""
function benchmark_serial(
    students::Vector{Student},
    schools::Vector{School};
    n_warmup::Int = 2,
    n_samples::Int = 10,
)
    b = @benchmarkable run_serial($students, $schools) samples = n_samples evals = 1
    tune!(b)
    return run(b)
end

"""
    benchmark_parallel_mc(students, schools, n_trials; n_samples)
        -> BenchmarkTools.Trial

Benchmark the parallel Monte Carlo run with `n_trials` trials.
"""
function benchmark_parallel_mc(
    students::Vector{Student},
    schools::Vector{School},
    n_trials::Int;
    n_samples::Int = 5,
)
    b = @benchmarkable run_parallel_mc($students, $schools, $n_trials) samples=n_samples evals=1
    tune!(b)
    return run(b)
end

"""
    benchmark_serial_mc(students, schools, n_trials; n_samples)
        -> BenchmarkTools.Trial

Benchmark the serial Monte Carlo loop for comparison against the parallel version.
"""
function benchmark_serial_mc(
    students::Vector{Student},
    schools::Vector{School},
    n_trials::Int;
    n_samples::Int = 5,
)
    b = @benchmarkable run_serial_mc($students, $schools, $n_trials) samples=n_samples evals=1
    tune!(b)
    return run(b)
end

"""
    run_all_benchmarks(students, schools; n_trials, n_samples)
        -> NamedTuple

Run the full benchmark suite and return results as a NamedTuple with fields:
- `serial_single`: single-assignment trial
- `serial_mc`: serial Monte Carlo
- `parallel_mc`: parallel Monte Carlo
- `speedup`: ratio of median times (serial_mc / parallel_mc)
"""
function run_all_benchmarks(
    students::Vector{Student},
    schools::Vector{School};
    n_trials::Int  = 50,
    n_samples::Int = 5,
)
    println("Benchmarking single serial assignment …")
    t_serial = benchmark_serial(students, schools; n_samples = n_samples)

    println("Benchmarking serial Monte Carlo ($n_trials trials) …")
    t_serial_mc = benchmark_serial_mc(students, schools, n_trials; n_samples = n_samples)

    println("Benchmarking parallel Monte Carlo ($n_trials trials) …")
    t_parallel_mc = benchmark_parallel_mc(students, schools, n_trials; n_samples = n_samples)

    speedup = median(t_serial_mc).time / median(t_parallel_mc).time

    print_benchmark_report(t_serial, t_serial_mc, t_parallel_mc, speedup, n_trials)

    return (
        serial_single = t_serial,
        serial_mc     = t_serial_mc,
        parallel_mc   = t_parallel_mc,
        speedup       = speedup,
    )
end

"""
    print_benchmark_report(t_serial, t_serial_mc, t_parallel_mc, speedup, n_trials)

Print a formatted benchmark summary table to stdout.
"""
function print_benchmark_report(t_serial, t_serial_mc, t_parallel_mc, speedup, n_trials)
    ms(t) = median(t).time / 1e6   # nanoseconds -> milliseconds
    mb(t) = median(t).memory / 1e6 # bytes -> megabytes

    println("\n", "=" ^ 60)
    println("Benchmark Results")
    println("  Julia threads available: ", Threads.nthreads())
    println("  Monte Carlo trials     : ", n_trials)
    println("=" ^ 60)
    @printf "  %-30s  %8.2f ms   %6.2f MB\n" "Serial (single run)" ms(t_serial) mb(t_serial)
    @printf "  %-30s  %8.2f ms   %6.2f MB\n" "Serial MC" ms(t_serial_mc) mb(t_serial_mc)
    @printf "  %-30s  %8.2f ms   %6.2f MB\n" "Parallel MC" ms(t_parallel_mc) mb(t_parallel_mc)
    @printf "  %-30s  %8.2fx\n" "Parallel speedup" speedup
    println("=" ^ 60)
end

"""
    scaling_experiment(students, schools; trial_counts, n_samples)
        -> (trial_counts, serial_times_ms, parallel_times_ms)

Measure serial vs parallel median runtimes across a range of trial counts.
Returns three vectors for plotting.
"""
function scaling_experiment(
    students::Vector{Student},
    schools::Vector{School};
    trial_counts::Vector{Int} = [10, 25, 50, 100, 200],
    n_samples::Int = 3,
)
    serial_times   = Float64[]
    parallel_times = Float64[]

    for n in trial_counts
        println("  n_trials = $n")
        ts = benchmark_serial_mc(students, schools, n; n_samples = n_samples)
        tp = benchmark_parallel_mc(students, schools, n; n_samples = n_samples)
        push!(serial_times,   median(ts).time / 1e6)
        push!(parallel_times, median(tp).time / 1e6)
    end

    return trial_counts, serial_times, parallel_times
end
