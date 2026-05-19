"""
run_serial.jl

Run a single serial school-assignment simulation and print statistics.

Usage:
    julia --project scripts/run_serial.jl
    julia --project scripts/run_serial.jl 10000 20 5
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "src", "types.jl"))
include(joinpath(@__DIR__, "..", "src", "generate_data.jl"))
include(joinpath(@__DIR__, "..", "src", "serial_sim.jl"))

using Printf, Random

# ── CLI args with defaults ───────────────────────────────────────────────────
n_students  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10_000
n_schools   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20
pref_length = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 5

println("=" ^ 50)
println("Serial School Assignment Simulation")
println("  Students    : $n_students")
println("  Schools     : $n_schools")
println("  Pref length : $pref_length")
println("=" ^ 50)

# ── Generate data ────────────────────────────────────────────────────────────
students, schools = generate_scenario(;
    n_students  = n_students,
    n_schools   = n_schools,
    pref_length = pref_length,
)
println("Data generated.")

# ── Run serial assignment ────────────────────────────────────────────────────
t0 = time()
result = run_serial(students, schools)
elapsed = time() - t0

# ── Print results ────────────────────────────────────────────────────────────
println("\nResults:")
@printf "  Elapsed               : %.3f s\n" elapsed
@printf "  First-choice rate     : %.2f%%\n" result.first_choice_rate * 100
@printf "  Unassigned students   : %d / %d\n" result.unassigned_count n_students
println("  Rank distribution:")
for (k, count) in enumerate(result.rank_distribution)
    @printf "    Rank %d : %d students (%.1f%%)\n" k count (count / n_students * 100)
end
