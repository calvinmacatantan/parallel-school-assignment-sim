"""
run_benchmarks.jl

Full benchmark suite: serial vs parallel runtime, memory allocations, and
scaling across trial counts. Results are printed to stdout and saved under
results/benchmark_results/.

Usage:
    julia --project --threads auto scripts/run_benchmarks.jl
    julia --project --threads 4   scripts/run_benchmarks.jl 10000 20 5 50
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "types.jl"))
include(joinpath(@__DIR__, "..", "src", "generate_data.jl"))
include(joinpath(@__DIR__, "..", "src", "serial_sim.jl"))
include(joinpath(@__DIR__, "..", "src", "parallel_sim.jl"))
include(joinpath(@__DIR__, "..", "src", "monte_carlo.jl"))
include(joinpath(@__DIR__, "..", "src", "benchmarks.jl"))
include(joinpath(@__DIR__, "..", "src", "plots.jl"))

using BenchmarkTools, Printf, Dates

# ── CLI args ──────────────────────────────────────────────────────────────────
n_students  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10_000
n_schools   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
pref_length = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5
n_trials    = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 50

println("=" ^ 60)
println("School Assignment Benchmark Suite")
println("  Date          : $(Dates.now())")
println("  Julia threads : $(Threads.nthreads())")
println("  Students      : $n_students")
println("  Schools       : $n_schools")
println("  Pref length   : $pref_length")
println("  MC trials     : $n_trials")
println("=" ^ 60)

# ── Setup ──────────────────────────────────────────────────────────────────────
students, schools = generate_scenario(;
    n_students  = n_students,
    n_schools   = n_schools,
    pref_length = pref_length,
)

# ── Main benchmarks ────────────────────────────────────────────────────────────
bench = run_all_benchmarks(students, schools; n_trials = n_trials, n_samples = 5)

# ── Scaling experiment ─────────────────────────────────────────────────────────
println("\nRunning scaling experiment …")
trial_counts = [10, 25, 50, 100, 200]
tc, st, pt = scaling_experiment(students, schools;
    trial_counts = trial_counts,
    n_samples    = 3,
)

println("\nScaling Results:")
println("-" ^ 45)
@printf "  %-10s  %-14s  %-14s  %s\n" "Trials" "Serial (ms)" "Parallel (ms)" "Speedup"
println("-" ^ 45)
for i in eachindex(tc)
    sp = st[i] / pt[i]
    @printf "  %-10d  %-14.2f  %-14.2f  %.2fx\n" tc[i] st[i] pt[i] sp
end
println("-" ^ 45)

# ── Save figures ───────────────────────────────────────────────────────────────
println("\nGenerating plots …")
from_mc = run_monte_carlo(students, schools, n_trials; parallel = true, seed = 42)
agg     = aggregate_results(from_mc)
save_all_figures(tc, st, pt, from_mc, agg; dir = "results/figures")

# ── Save text results ──────────────────────────────────────────────────────────
mkpath("results/benchmark_results")
open("results/benchmark_results/summary.txt", "w") do io
    redirect_stdout(io) do
        print_benchmark_report(
            bench.serial_single, bench.serial_mc, bench.parallel_mc,
            bench.speedup, n_trials,
        )
    end
end
println("Benchmark summary saved to results/benchmark_results/summary.txt")
