# Parallel School Assignment Simulator

A Julia implementation of lottery-based school assignment with multiple matching
algorithms, parallel Monte Carlo simulation, and stability analysis. Built for
MIT 18.337 (Parallel Computing and Scientific Machine Learning).

---

## Project Overview

Many cities assign students to public schools through a lottery: each student
submits a ranked preference list, receives a random lottery number, and is
matched to a school through one of several algorithms.

This project implements **three matching algorithms** at increasing levels of
optimality and parallelism, runs them under Monte Carlo over thousands of
lottery draws, and measures both runtime and assignment quality — specifically
**blocking pairs**, the standard stability metric from matching market theory.

| Algorithm | Stability | Intra-trial parallelism | Notes |
|---|---|---|---|
| **Boston** | ✗ (can have blocking pairs) | None | O(N log N), one pass |
| **Deferred Acceptance (DA)** | ✓ (zero blocking pairs) | School-resolution step | Batch Gale-Shapley, ≤ `pref_length` rounds |
| **Batched DA** | Partial (zero within bands) | Intra-band resolution | Quality/speed trade-off tunable via `n_batches` |

The core research question: **under what problem scales and demand patterns does
intra-trial parallelism help, and what does it cost in match quality?**

---

## Repository Structure

```
parallel-school-assignment-sim/
├── src/
│   ├── types.jl            # Core structs + AbstractMatchingStrategy hierarchy
│   ├── generate_data.jl    # Synthetic student/school data generation
│   ├── serial_sim.jl       # Boston mechanism (immediate acceptance)
│   ├── da_core.jl          # True student-optimal DA (batch Gale-Shapley)
│   ├── batched_sim.jl      # Band-based batched DA
│   ├── quality_metrics.jl  # Blocking pairs, mean rank, aggregate_quality
│   ├── parallel_sim.jl     # @threads + @spawn MC runners, strategy dispatch
│   ├── monte_carlo.jl      # Orchestration, aggregation, compare_strategies
│   ├── benchmarks.jl       # BenchmarkTools.jl suite
│   └── plots.jl            # Plots.jl visualisations
├── scripts/
│   ├── run_serial.jl
│   ├── run_parallel.jl
│   └── run_benchmarks.jl
├── notebooks/
│   ├── 01_generate_data.ipynb
│   ├── 02_serial_vs_parallel.ipynb
│   ├── 03_benchmarks.ipynb
│   └── 04_final_figures.ipynb
├── test/
│   └── runtests.jl
├── results/
│   ├── figures/
│   └── benchmark_results/
├── Project.toml
└── Manifest.toml
```

---

## Setup

### Prerequisites

- Julia ≥ 1.9
- (Optional) IJulia for notebooks: `julia -e 'using Pkg; Pkg.add("IJulia")'`

### Install dependencies

```bash
julia --project -e 'using Pkg; Pkg.instantiate()'
```

---

## Running the Simulations

### Serial — single assignment run

```bash
julia --project scripts/run_serial.jl
# With custom parameters: n_students  n_schools  pref_length
julia --project scripts/run_serial.jl 10000 20 5
```

### Parallel — Monte Carlo with multiple threads

```bash
julia --project --threads auto scripts/run_parallel.jl
julia --project --threads 4   scripts/run_parallel.jl 10000 20 5 100
```

### Full benchmark suite

```bash
julia --project --threads auto scripts/run_benchmarks.jl
```

### Compare all strategies side-by-side (REPL)

```julia
julia --project --threads 8
```

```julia
include("src/types.jl"); include("src/generate_data.jl")
include("src/serial_sim.jl"); include("src/da_core.jl")
include("src/batched_sim.jl"); include("src/quality_metrics.jl")
include("src/parallel_sim.jl"); include("src/monte_carlo.jl")

students, schools = generate_scenario(; n_students=10_000, n_schools=20)
compare_strategies(students, schools, 50)
```

---

## Algorithms

### Boston mechanism (`serial_sim.jl`)

The original baseline. Students are sorted by lottery number (ascending = higher
priority) and each student is immediately assigned to the best available school
on their list. Fast but **not stable**: blocking pairs exist when a student
prefers a school that would have accepted them over a lower-priority student who
got there first.

Time complexity: **O(N log N + N·P)** where N = students, P = preference list length.

### Deferred Acceptance (`da_core.jl`)

True student-optimal DA via batch Gale-Shapley. In each round, all free students
simultaneously propose to their next-preferred school; each school keeps its
top-capacity applicants by lottery priority and rejects the rest. Rounds repeat
until no proposals remain.

**Key properties:**
- Terminates in at most `pref_length` rounds (5 in the default scenario).
- Produces **zero blocking pairs** by construction (stability theorem).
- School-resolution step is embarrassingly parallel across schools
  (`parallel_resolve=true` enables `@threads` over the resolution step).

Time complexity: **O(pref_length × N)** rounds, each O(N) work.

### Batched DA (`batched_sim.jl`)

Students are divided into `n_batches` equal-size priority bands by lottery rank.
Each band runs DA against the remaining school capacity left by earlier bands.
Within each band's DA, the school-resolution step is parallelised.

**Trade-off:**
- `n_batches = 1` is equivalent to full DA (zero blocking pairs).
- Increasing `n_batches` exposes more intra-trial parallelism but introduces
  blocking pairs at band boundaries (a band-2 student may be blocked by a
  worse-lottery band-1 student). Blocking pairs are bounded by the number of
  cross-band preference conflicts.

---

## Parallelism Model

### Trial-level (across MC trials)

Both `run_parallel_mc` and `run_parallel_mc_spawn` distribute independent
lottery draws across Julia's thread pool. Race conditions are avoided by:
- Pre-allocating one `MersenneTwister` per trial (seeded as `base_seed + i`).
- Writing results into unique pre-allocated slots (no shared mutable state).

`@threads` partitions trials statically (equal chunks). `@spawn` uses dynamic
task scheduling — better when `n_trials % nthreads() ≠ 0` or trial times vary.

### Intra-trial (within a single DA run)

The school-resolution step in each DA round is parallelised with `@threads` over
active schools. Schools are fully independent — each thread writes to disjoint
student-id slots of the assignment vector, so no locking is needed.

> **Thread-safety note:** `free` is stored as `Vector{Bool}` (1 byte/element),
> not `BitVector` (packed bits). Concurrent writes to adjacent bits in a
> `BitVector` share a 64-bit word and race; `Vector{Bool}` writes to different
> indices are independent.

---

## Quality Metrics (`quality_metrics.jl`)

### Blocking pairs

A blocking pair `(student s, school x)` exists when:
1. `s` prefers `x` over their current assignment (or is unassigned).
2. `x` would accept `s`: either `x` has remaining capacity, or `x` holds a
   student with a worse lottery number (lower priority) than `s`.

DA always produces **0 blocking pairs**. Boston typically produces >0 under
tight market conditions. Batched DA produces blocking pairs only at band
boundaries.

```julia
bp = count_blocking_pairs(result, students, schools)
```

### Other metrics

```julia
qs = quality_summary(result, students, schools)
# → blocking_pairs, mean_assigned_rank, first_choice_rate,
#   unassigned_count, assigned_fraction, rank_distribution

aq = aggregate_quality(summaries)   # across MC trials
# → mean/std of first_choice_rate, var_first_choice_rate (match-effect variance),
#   mean/std of blocking_pairs, mean_rank_distribution
```

`var_first_choice_rate` measures how much the lottery draw itself — not the
algorithm — determines outcomes. High variance means the mechanism is sensitive
to luck.

---

## Strategy Dispatch

All MC runners accept an `AbstractMatchingStrategy`:

```julia
run_parallel_mc(BostonStrategy(),       students, schools, 50)
run_parallel_mc(DAStrategy(),           students, schools, 50)
run_parallel_mc(ParallelDAStrategy(),   students, schools, 50)
run_parallel_mc(BatchedDAStrategy(4),   students, schools, 50)

# Or via the top-level entry point:
run_monte_carlo(students, schools, 50;
    strategy = BatchedDAStrategy(8),
    parallel = true,
    spawn    = true)   # use @spawn instead of @threads
```

---

## Benchmark Results

Measured on 8 threads, 50 trials, 10,000 students, 20 schools:

```
============================================================
Benchmark Results
  Julia threads available: 8
  Monte Carlo trials     : 50
============================================================
  Serial (single run)                 0.72 ms     0.57 MB
  Serial MC                          38.85 ms    41.04 MB
  Parallel MC                         7.11 ms    42.06 MB
  Parallel speedup                    5.46x
============================================================
```

---

## Default Parameters

| Parameter       | Default | Description                          |
|-----------------|---------|--------------------------------------|
| `n_students`    | 10,000  | Number of students in the lottery    |
| `n_schools`     | 20      | Number of available schools          |
| `pref_length`   | 5       | Length of each student's rank list   |
| `base_capacity` | 600     | Mean school capacity (±50%)          |
| `n_trials`      | 50      | Monte Carlo replications             |
| `n_batches`     | 4       | Bands for `BatchedDAStrategy`        |

Total system capacity ≈ 20 × 600 = 12,000 seats for 10,000 students.

---

## Tests

```bash
julia --project --threads 4 test/runtests.jl
```

The test suite verifies:
- Preference lists are valid (no duplicates, valid school IDs, lottery in [0,1)).
- No school exceeds its capacity after any algorithm runs.
- DA produces **zero blocking pairs** on all tested inputs.
- `BatchedDA(1)` is equivalent to full DA.
- More bands produce ≥ blocking pairs than fewer bands (monotonicity).
- Parallel-resolve DA produces the same assignment as serial-resolve DA.
- `@threads` and `@spawn` MC runners return valid `TrialSummary` structs.
- All four strategies work through the strategy-dispatch interface.
- Edge cases: zero-capacity schools, single student, ample capacity.
