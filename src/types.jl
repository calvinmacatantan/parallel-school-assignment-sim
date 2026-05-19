"""
    types.jl

Core data structures for the school assignment simulation.
"""

"""
    Student

Represents a student in the lottery-based school assignment system.

# Fields
- `id::Int`: Unique student identifier.
- `preferences::Vector{Int}`: Ranked list of school IDs (index 1 = most preferred).
- `lottery_number::Float64`: Random draw in [0, 1) used to order students for assignment.
"""
struct Student
    id::Int
    preferences::Vector{Int}
    lottery_number::Float64
end

"""
    School

Represents a school in the assignment system.

# Fields
- `id::Int`: Unique school identifier.
- `capacity::Int`: Maximum number of students the school can accept.
- `name::String`: Human-readable label (e.g., "School 3").
"""
struct School
    id::Int
    capacity::Int
    name::String
end

"""
    AssignmentResult

Stores the outcome of a single assignment run.

# Fields
- `assignments::Vector{Int}`: `assignments[student_id]` = school id, or 0 if unassigned.
- `first_choice_rate::Float64`: Fraction of students assigned to their top-ranked school.
- `unassigned_count::Int`: Number of students who received no assignment.
- `rank_distribution::Vector{Int}`: `rank_distribution[k]` = number of students assigned to their k-th choice.
"""
struct AssignmentResult
    assignments::Vector{Int}           # student id -> school id (0 = unassigned)
    first_choice_rate::Float64
    unassigned_count::Int
    rank_distribution::Vector{Int}     # index k -> count assigned to k-th choice
end

"""
    TrialSummary

Summary statistics from one Monte Carlo trial.

# Fields
- `first_choice_rate::Float64`
- `unassigned_count::Int`
- `rank_distribution::Vector{Float64}`: Fraction assigned at each rank.
- `blocking_pairs::Int`: Number of blocking pairs (-1 if not computed).
"""
struct TrialSummary
    first_choice_rate::Float64
    unassigned_count::Int
    rank_distribution::Vector{Float64}
    blocking_pairs::Int
end

# ── Matching strategy dispatch ─────────────────────────────────────────────────

abstract type AbstractMatchingStrategy end

# Original Boston mechanism: sort by lottery, greedily assign — fast, unstable.
struct BostonStrategy <: AbstractMatchingStrategy end

# True student-optimal deferred acceptance (batch Gale-Shapley).
# Resolution rounds are sequential within each trial.
struct DAStrategy <: AbstractMatchingStrategy end

# DA with school-resolution rounds parallelised via Threads.@threads.
# Exposes intra-trial parallelism at the cost of thread overhead per round.
struct ParallelDAStrategy <: AbstractMatchingStrategy end

# Band-based batched DA: students are divided into n_batches priority bands
# by lottery rank; each band runs ParallelDA against remaining capacity.
# Trades some match quality for throughput at high thread counts.
struct BatchedDAStrategy <: AbstractMatchingStrategy
    n_batches::Int
end
BatchedDAStrategy() = BatchedDAStrategy(4)
