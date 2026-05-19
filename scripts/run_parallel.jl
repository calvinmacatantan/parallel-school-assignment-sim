"""
run_parallel.jl

Run a multithreaded Monte Carlo simulation and print aggregated statistics.

Usage:
    julia --project --threads auto scripts/run_parallel.jl
    julia --project --threads 4   scripts/run_parallel.jl 10000 20 5 100

CLI args (all optional, positional):
    1. n_students  (default 10_000)
    2. n_schools   (default 20)
    3. pref_length (default 5)
    4. n_trials    (default 50)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "types.jl"))
include(joinpath(@__DIR__, "..", "src", "generate_data.jl"))
include(joinpath(@__DIR__, "..", "src", "serial_sim.jl"))
include(joinpath(@__DIR__, "..", "src", "parallel_sim.jl"))
include(joinpath(@__DIR__, "..", "src", "monte_carlo.jl"))

using Printf, Random, Statistics

# ── CLI args ─────────────────────────────────────────────────────────────────
n_students  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10_000
n_schools   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
pref_length = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5
n_trials    = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 50

println("=" ^ 55)
println("Parallel Monte Carlo School Assignment Simulation")
println("  Julia threads : $(Threads.nthreads())")
println("  Students      : $n_students")
println("  Schools       : $n_schools")
println("  Pref length   : $pref_length")
println("  Trials        : $n_trials")
println("=" ^ 55)

# ── Generate base scenario ────────────────────────────────────────────────────
students, schools = generate_scenario(;
    n_students  = n_students,
    n_schools   = n_schools,
    pref_length = pref_length,
)
println("Base scenario generated.")

# ── Parallel MC ───────────────────────────────────────────────────────────────
t0 = time()
summaries = run_monte_carlo(students, schools, n_trials; parallel = true, seed = 1)
elapsed   = time() - t0

agg = aggregate_results(summaries)

# ── Print results ─────────────────────────────────────────────────────────────
println("\nCompleted in $(round(elapsed; digits=3)) s\n")
print_summary(agg)
