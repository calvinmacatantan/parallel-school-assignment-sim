"""
    generate_data.jl

Functions to generate synthetic students, schools, and preference lists.
"""

using Random

"""
    generate_schools(n_schools; base_capacity, rng) -> Vector{School}

Create `n_schools` schools with randomized capacities drawn around `base_capacity`.
Capacity for each school is sampled uniformly from [base_capacity ÷ 2, 3 * base_capacity ÷ 2].
"""
function generate_schools(
    n_schools::Int;
    base_capacity::Int = 600,
    rng::AbstractRNG = Random.GLOBAL_RNG,
)::Vector{School}
    return [
        School(
            i,
            rand(rng, (base_capacity ÷ 2):(3 * base_capacity ÷ 2)),
            "School $i",
        )
        for i in 1:n_schools
    ]
end

"""
    generate_students(n_students, n_schools, pref_length; rng) -> Vector{Student}

Create `n_students` students, each with:
- a lottery number drawn uniformly from [0, 1),
- a ranked preference list of length `pref_length` sampled without replacement
  from the `n_schools` available schools.

Schools are weighted by a random popularity score so demand is uneven.
"""
function generate_students(
    n_students::Int,
    n_schools::Int,
    pref_length::Int;
    rng::AbstractRNG = Random.GLOBAL_RNG,
)::Vector{Student}
    @assert pref_length <= n_schools "Preference list length cannot exceed number of schools."

    # Random popularity weights — makes some schools highly demanded
    weights = rand(rng, n_schools) .^ 0.5   # skew toward popular schools
    weights ./= sum(weights)

    students = Vector{Student}(undef, n_students)
    for i in 1:n_students
        prefs = sample_preferences(weights, pref_length, rng)
        lottery = rand(rng)
        students[i] = Student(i, prefs, lottery)
    end
    return students
end

"""
    sample_preferences(weights, k, rng) -> Vector{Int}

Sample `k` school indices without replacement, weighted by `weights`.
Uses reservoir / Fisher-Yates–style weighted sampling.
"""
function sample_preferences(
    weights::Vector{Float64},
    k::Int,
    rng::AbstractRNG,
)::Vector{Int}
    n = length(weights)
    # Weighted sampling without replacement via reservoir trick
    keys = [rand(rng) ^ (1.0 / w) for w in weights]   # Efraimidis–Spirakis
    return partialsortperm(keys, 1:k, rev = true)
end

"""
    generate_scenario(; n_students, n_schools, pref_length, base_capacity, seed)

Convenience function returning `(students, schools)` with a fixed random seed
for reproducibility.

# Keyword Arguments
- `n_students = 10_000`
- `n_schools = 20`
- `pref_length = 5`
- `base_capacity = 600`   # total system capacity ~= n_students when varied ±50%
- `seed = 42`
"""
function generate_scenario(;
    n_students::Int = 10_000,
    n_schools::Int = 20,
    pref_length::Int = 5,
    base_capacity::Int = 600,
    seed::Int = 42,
)
    rng = MersenneTwister(seed)
    schools = generate_schools(n_schools; base_capacity = base_capacity, rng = rng)
    students = generate_students(n_students, n_schools, pref_length; rng = rng)
    return students, schools
end
