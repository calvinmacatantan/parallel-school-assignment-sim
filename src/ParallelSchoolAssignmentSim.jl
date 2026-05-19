module ParallelSchoolAssignmentSim

include("types.jl")
include("generate_data.jl")
include("serial_sim.jl")
include("da_core.jl")
include("parallel_sim.jl")
include("batched_sim.jl")
include("monte_carlo.jl")
include("quality_metrics.jl")
include("benchmarks.jl")
include("plots.jl")

end # module
