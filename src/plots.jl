"""
    plots.jl

Visualization functions using Plots.jl.
Generates runtime scaling plots, speedup plots, and assignment distribution charts.
"""

using Plots

# Use a clean default theme
default(; fontfamily = "Computer Modern", dpi = 150, size = (700, 450))

"""
    plot_scaling(trial_counts, serial_times, parallel_times; save_path)

Plot serial vs parallel runtime as a function of trial count.

# Arguments
- `trial_counts::Vector{Int}`
- `serial_times::Vector{Float64}`: Median runtime in ms (serial MC).
- `parallel_times::Vector{Float64}`: Median runtime in ms (parallel MC).
- `save_path::String`: If non-empty, save the figure to this path.
"""
function plot_scaling(
    trial_counts::Vector{Int},
    serial_times::Vector{Float64},
    parallel_times::Vector{Float64};
    save_path::String = "",
)
    p = plot(
        trial_counts, serial_times;
        label     = "Serial",
        marker    = :circle,
        linewidth = 2,
        xlabel    = "Number of Trials",
        ylabel    = "Median Runtime (ms)",
        title     = "Serial vs Parallel Runtime Scaling",
        legend    = :topleft,
    )
    plot!(p, trial_counts, parallel_times;
        label     = "Parallel ($(Threads.nthreads()) threads)",
        marker    = :square,
        linewidth = 2,
    )

    isempty(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_speedup(trial_counts, serial_times, parallel_times; save_path)

Plot parallel speedup (serial / parallel) as a function of trial count.
Includes a dashed line at speedup = 1 for reference.
"""
function plot_speedup(
    trial_counts::Vector{Int},
    serial_times::Vector{Float64},
    parallel_times::Vector{Float64};
    save_path::String = "",
)
    speedups = serial_times ./ parallel_times
    p = plot(
        trial_counts, speedups;
        label     = "Speedup",
        marker    = :diamond,
        linewidth = 2,
        color     = :green,
        xlabel    = "Number of Trials",
        ylabel    = "Speedup (serial / parallel)",
        title     = "Parallel Speedup  ($(Threads.nthreads()) threads)",
        legend    = :bottomright,
        ylims     = (0, max(Threads.nthreads() + 1, maximum(speedups) * 1.15)),
    )
    hline!(p, [1.0]; label = "No speedup", linestyle = :dash, color = :red)
    hline!(p, [Threads.nthreads()]; label = "Ideal speedup",
           linestyle = :dot, color = :gray)

    isempty(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_rank_distribution(mean_rank_dist; save_path)

Bar chart of the average fraction of students assigned at each preference rank.
"""
function plot_rank_distribution(
    mean_rank_dist::Vector{Float64};
    save_path::String = "",
)
    ranks = 1:length(mean_rank_dist)
    p = bar(
        ranks, mean_rank_dist .* 100;
        xlabel    = "Assigned Rank",
        ylabel    = "Fraction of Students (%)",
        title     = "Assignment Rank Distribution (Monte Carlo Mean)",
        legend    = false,
        color     = :steelblue,
        alpha     = 0.8,
        xticks    = collect(ranks),
    )
    annotate!(p,
        length(ranks) + 0.5, maximum(mean_rank_dist) * 100 * 0.95,
        text("Unassigned\nnot shown", 8, :right, :gray),
    )

    isempty(save_path) || savefig(p, save_path)
    return p
end

"""
    plot_first_choice_distribution(summaries; save_path)

Histogram of per-trial first-choice assignment rates across all Monte Carlo trials.
"""
function plot_first_choice_distribution(
    summaries::Vector{TrialSummary};
    save_path::String = "",
)
    rates = [s.first_choice_rate * 100 for s in summaries]
    p = histogram(
        rates;
        bins      = 20,
        xlabel    = "First-Choice Assignment Rate (%)",
        ylabel    = "Count",
        title     = "Distribution of First-Choice Rates Across Trials",
        legend    = false,
        color     = :salmon,
        alpha     = 0.8,
    )

    isempty(save_path) || savefig(p, save_path)
    return p
end

"""
    save_all_figures(trial_counts, serial_times, parallel_times, summaries, agg;
                     dir)

Generate and save all standard figures to `dir`.
"""
function save_all_figures(
    trial_counts::Vector{Int},
    serial_times::Vector{Float64},
    parallel_times::Vector{Float64},
    summaries::Vector{TrialSummary},
    agg;
    dir::String = "results/figures",
)
    mkpath(dir)
    plot_scaling(trial_counts, serial_times, parallel_times;
        save_path = joinpath(dir, "scaling.png"))
    plot_speedup(trial_counts, serial_times, parallel_times;
        save_path = joinpath(dir, "speedup.png"))
    plot_rank_distribution(agg.mean_rank_distribution;
        save_path = joinpath(dir, "rank_distribution.png"))
    plot_first_choice_distribution(summaries;
        save_path = joinpath(dir, "first_choice_histogram.png"))
    println("Figures saved to $dir/")
end
