"""
    serial_sim.jl

Baseline serial implementation of the lottery-based school assignment algorithm.
Students are processed in ascending lottery-number order; each student is placed
into the highest-ranked school on their preference list that still has remaining
capacity.
"""

"""
    run_serial(students, schools) -> AssignmentResult

Run the serial deferred-acceptance-style lottery assignment.

Algorithm
---------
1. Sort students by lottery number (ascending → lower number = higher priority).
2. For each student, iterate through their ranked preferences.
3. Assign the student to the first school that has remaining capacity.
4. Collect statistics.

Returns an `AssignmentResult` with per-student assignments and summary metrics.
"""
function run_serial(
    students::Vector{Student},
    schools::Vector{School},
)::AssignmentResult
    n_students = length(students)
    n_schools  = length(schools)
    pref_length = isempty(students) ? 0 : length(students[1].preferences)

    # Mutable capacity counters (one per school, indexed by school id)
    remaining = Dict(s.id => s.capacity for s in schools)

    # Sort by lottery number — lower value wins priority
    sorted_students = sort(students; by = s -> s.lottery_number)

    assignments = zeros(Int, n_students)          # student index -> school id
    rank_distribution = zeros(Int, pref_length)   # rank -> count

    for student in sorted_students
        for (rank, school_id) in enumerate(student.preferences)
            if remaining[school_id] > 0
                assignments[student.id] = school_id
                remaining[school_id] -= 1
                rank_distribution[rank] += 1
                break
            end
        end
        # If no school had space, assignments[student.id] stays 0 (unassigned)
    end

    first_choice_count = rank_distribution[1]
    unassigned = count(==(0), assignments)

    return AssignmentResult(
        assignments,
        first_choice_count / n_students,
        unassigned,
        rank_distribution,
    )
end

"""
    run_serial_with_fresh_lottery(students, schools; rng) -> AssignmentResult

Re-draw lottery numbers for each student (preserving preferences) and run the
serial assignment. Used by Monte Carlo orchestration so each trial is independent.
"""
function run_serial_with_fresh_lottery(
    students::Vector{Student},
    schools::Vector{School};
    rng::AbstractRNG = Random.GLOBAL_RNG,
)::AssignmentResult
    redrawn = [Student(s.id, s.preferences, rand(rng)) for s in students]
    return run_serial(redrawn, schools)
end
