using Test, Random, Statistics

const PROJECT_ROOT = joinpath(@__DIR__, "..")
include(joinpath(PROJECT_ROOT, "src", "types.jl"))
include(joinpath(PROJECT_ROOT, "src", "generate_data.jl"))
include(joinpath(PROJECT_ROOT, "src", "serial_sim.jl"))
include(joinpath(PROJECT_ROOT, "src", "da_core.jl"))
include(joinpath(PROJECT_ROOT, "src", "batched_sim.jl"))
include(joinpath(PROJECT_ROOT, "src", "quality_metrics.jl"))
include(joinpath(PROJECT_ROOT, "src", "parallel_sim.jl"))
include(joinpath(PROJECT_ROOT, "src", "monte_carlo.jl"))

const N_STUDENTS  = 500
const N_SCHOOLS   = 10
const PREF_LENGTH = 4

students, schools = generate_scenario(;
    n_students  = N_STUDENTS,
    n_schools   = N_SCHOOLS,
    pref_length = PREF_LENGTH,
    seed        = 99,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

function check_capacity(result, schools_vec, n)
    counts = Dict(sch.id => 0 for sch in schools_vec)
    for sid in 1:n
        sid_school = result.assignments[sid]
        sid_school == 0 && continue
        counts[sid_school] += 1
    end
    cap = Dict(sch.id => sch.capacity for sch in schools_vec)
    all(counts[id] <= cap[id] for id in keys(counts))
end

function check_preference_validity(result, students_vec, schools_vec)
    valid_ids = Set(sch.id for sch in schools_vec)
    for s in students_vec
        sid = result.assignments[s.id]
        sid == 0 && continue
        sid ∉ valid_ids && return false
        sid ∉ s.preferences && return false
    end
    return true
end

@testset "ParallelSchoolAssignmentSim" begin

    # ── Data generation ────────────────────────────────────────────────────────
    @testset "Data generation" begin
        @test length(students) == N_STUDENTS
        @test length(schools)  == N_SCHOOLS
        for s in students
            @test length(s.preferences) == PREF_LENGTH
            @test allunique(s.preferences)
            @test all(1 <= p <= N_SCHOOLS for p in s.preferences)
            @test 0.0 <= s.lottery_number < 1.0
        end
        for sch in schools
            @test sch.capacity > 0
            @test !isempty(sch.name)
        end
    end

    # ── Boston (serial_sim) ────────────────────────────────────────────────────
    @testset "Boston — validity" begin
        result = run_serial(students, schools)

        @test length(result.assignments) == N_STUDENTS
        @test check_capacity(result, schools, N_STUDENTS)
        @test check_preference_validity(result, students, schools)

        n_assigned = count(!=(0), result.assignments)
        @test result.unassigned_count == N_STUDENTS - n_assigned
        @test 0.0 <= result.first_choice_rate <= 1.0
        @test sum(result.rank_distribution) == n_assigned
    end

    # ── DA correctness ────────────────────────────────────────────────────────
    @testset "DA — validity and stability" begin
        result = run_da(students, schools)

        @test length(result.assignments) == N_STUDENTS
        @test check_capacity(result, schools, N_STUDENTS)
        @test check_preference_validity(result, students, schools)

        n_assigned = count(!=(0), result.assignments)
        @test result.unassigned_count == N_STUDENTS - n_assigned
        @test 0.0 <= result.first_choice_rate <= 1.0

        # DA must produce zero blocking pairs (stability theorem).
        bp = count_blocking_pairs(result, students, schools)
        @test bp == 0
    end

    @testset "DA — no worse unassigned count than Boston under ample capacity" begin
        # With enough total capacity, both algorithms should fully assign everyone.
        big_schools = [School(i, 1000, "S$i") for i in 1:N_SCHOOLS]
        s = students[1:50]
        r_boston = run_serial(s, big_schools)
        r_da     = run_da(s, big_schools)
        @test r_boston.unassigned_count == 0
        @test r_da.unassigned_count == 0
    end

    @testset "DA — parallel resolve matches serial resolve" begin
        result_serial   = run_da(students, schools; parallel_resolve = false)
        result_parallel = run_da(students, schools; parallel_resolve = true)
        # Same match quality (exact assignments may differ only in tie-breaking
        # when two students share the same lottery — they don't, it's Float64).
        @test result_serial.unassigned_count == result_parallel.unassigned_count
        @test result_serial.first_choice_rate ≈ result_parallel.first_choice_rate
        @test sum(result_serial.rank_distribution) == sum(result_parallel.rank_distribution)
        # Both should be stable.
        @test count_blocking_pairs(result_parallel, students, schools) == 0
    end

    # ── Boston vs DA blocking pairs ───────────────────────────────────────────
    @testset "Boston has ≥ blocking pairs than DA (tight market)" begin
        # Use tight capacity (barely enough seats) to stress-test both.
        total = N_STUDENTS
        tight = [School(i, div(total, N_SCHOOLS), "S$i") for i in 1:N_SCHOOLS]
        r_boston = run_serial(students, tight)
        r_da     = run_da(students, tight)

        bp_boston = count_blocking_pairs(r_boston, students, tight)
        bp_da     = count_blocking_pairs(r_da,     students, tight)

        @test bp_da == 0
        # Boston can produce blocking pairs; if market is tight it usually does.
        # We don't assert bp_boston > 0 because in rare cases it may also be 0.
        @test bp_boston >= bp_da
    end

    # ── Batched DA ────────────────────────────────────────────────────────────
    @testset "BatchedDA — validity" begin
        result = run_batched_da(students, schools; n_batches = 4)

        @test length(result.assignments) == N_STUDENTS
        @test check_capacity(result, schools, N_STUDENTS)
        @test check_preference_validity(result, students, schools)

        n_assigned = count(!=(0), result.assignments)
        @test result.unassigned_count == N_STUDENTS - n_assigned
    end

    @testset "BatchedDA(1) ≡ DA" begin
        # One band = DA over all students = student-optimal DA.
        r_da      = run_da(students, schools)
        r_batched = run_batched_da(students, schools; n_batches = 1)
        @test r_da.unassigned_count     == r_batched.unassigned_count
        @test r_da.first_choice_rate    ≈  r_batched.first_choice_rate
        @test count_blocking_pairs(r_batched, students, schools) == 0
    end

    @testset "BatchedDA — more bands ≥ blocking pairs than fewer bands" begin
        bp1 = count_blocking_pairs(
            run_batched_da(students, schools; n_batches = 1), students, schools)
        bp4 = count_blocking_pairs(
            run_batched_da(students, schools; n_batches = 4), students, schools)
        bp8 = count_blocking_pairs(
            run_batched_da(students, schools; n_batches = 8), students, schools)
        @test bp1 == 0
        @test bp4 >= bp1
        @test bp8 >= bp4
    end

    # ── Quality metrics ───────────────────────────────────────────────────────
    @testset "Quality metrics" begin
        result = run_serial(students, schools)

        bp   = count_blocking_pairs(result, students, schools)
        @test bp >= 0

        mrk  = mean_assigned_rank(result, students)
        @test 1.0 <= mrk <= PREF_LENGTH

        qs   = quality_summary(result, students, schools)
        @test qs.blocking_pairs == bp
        @test qs.mean_assigned_rank ≈ mrk
        @test 0.0 <= qs.first_choice_rate <= 1.0
        @test 0.0 <= qs.assigned_fraction <= 1.0
    end

    # ── Parallel MC (legacy @threads) ─────────────────────────────────────────
    @testset "Parallel MC — @threads structure" begin
        summaries = run_parallel_mc(students, schools, 10; seed = 7)
        @test length(summaries) == 10
        for s in summaries
            @test 0.0 <= s.first_choice_rate <= 1.0
            @test s.unassigned_count >= 0
            @test length(s.rank_distribution) == PREF_LENGTH
            @test s.blocking_pairs == -1   # not computed in fast path
        end
    end

    @testset "Parallel MC — @spawn structure" begin
        summaries = run_parallel_mc_spawn(students, schools, 10; seed = 7)
        @test length(summaries) == 10
        for s in summaries
            @test 0.0 <= s.first_choice_rate <= 1.0
        end
    end

    @testset "Parallel MC — DA strategy" begin
        summaries = run_parallel_mc(DAStrategy(), students, schools, 5; seed = 3)
        @test length(summaries) == 5
        for s in summaries
            @test 0.0 <= s.first_choice_rate <= 1.0
        end
    end

    @testset "Parallel MC — BatchedDA strategy" begin
        summaries = run_parallel_mc(BatchedDAStrategy(4), students, schools, 5; seed = 3)
        @test length(summaries) == 5
    end

    # ── Serial MC ─────────────────────────────────────────────────────────────
    @testset "Serial MC — structure" begin
        summaries = run_serial_mc(students, schools, 10; seed = 7)
        @test length(summaries) == 10
        for s in summaries
            @test 0.0 <= s.first_choice_rate <= 1.0
        end
    end

    # ── Aggregation ───────────────────────────────────────────────────────────
    @testset "Aggregation" begin
        summaries = run_serial_mc(students, schools, 20; seed = 5)
        agg = aggregate_results(summaries)
        @test agg.n_trials == 20
        @test 0.0 <= agg.mean_first_choice_rate <= 1.0
        @test agg.std_first_choice_rate >= 0.0
        @test agg.mean_unassigned >= 0.0
        @test length(agg.mean_rank_distribution) == PREF_LENGTH
    end

    @testset "aggregate_quality includes blocking pairs and variance" begin
        # Use to_trial_summary_with_quality on a small set.
        small_s, small_sch = generate_scenario(; n_students=50, n_schools=5,
                                                 pref_length=3, seed=1)
        rng = MersenneTwister(42)
        summaries = [
            to_trial_summary_with_quality(
                run_serial_with_fresh_lottery(small_s, small_sch; rng = rng),
                small_s, small_sch,
            )
            for _ in 1:10
        ]
        aq = aggregate_quality(summaries)
        @test aq.n_trials == 10
        @test aq.mean_blocking_pairs >= 0
        @test aq.var_first_choice_rate >= 0
    end

    # ── Edge cases ────────────────────────────────────────────────────────────
    @testset "Edge case — single student" begin
        one = [Student(1, [1, 2, 3], 0.5)]
        big = [School(i, 1000, "S$i") for i in 1:3]
        for runner in [run_serial, run_da, r -> run_batched_da(r, big)]
            result = run_serial(one, big)
            @test result.assignments[1] == 1
            @test result.first_choice_rate == 1.0
            @test result.unassigned_count == 0
        end

        # DA should also be stable for single student.
        result_da = run_da(one, big)
        @test result_da.assignments[1] == 1
        @test count_blocking_pairs(result_da, one, big) == 0
    end

    @testset "Edge case — zero capacity" begin
        zero_sch = [School(i, 0, "S$i") for i in 1:N_SCHOOLS]
        for result in [run_serial(students, zero_sch), run_da(students, zero_sch)]
            @test all(==(0), result.assignments)
            @test result.unassigned_count == N_STUDENTS
            @test result.first_choice_rate == 0.0
        end
    end

end
