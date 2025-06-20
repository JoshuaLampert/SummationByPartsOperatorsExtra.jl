@inline function isfinished(integrator)
    # Checking for floating point equality is OK here as `DifferentialEquations.jl`
    # sets the time exactly to the final time in the last iteration
    return integrator.t == last(integrator.sol.prob.tspan) ||
           isempty(integrator.opts.tstops) ||
           integrator.iter == integrator.opts.maxiters
end

"""
    AnalysisCallback(semi)

"""
mutable struct AnalysisCallback
    semi::AbstractSemidiscretization
    tstops::Vector{Float64}
    quantities::Vector{Vector{Float64}}
end

"""
    quantities(analysis_callback)

Return the computed quantities for each time step.
"""
function quantities(cb::DiscreteCallback{Condition,
                                         Affect!}) where {Condition,
                                                          Affect! <:
                                                          AnalysisCallback}
    analysis_callback = cb.affect!
    return analysis_callback.quantities
end

function quantities(cb::DiscreteCallback{Condition,
                                         Affect!}) where {Condition,
                                                          Affect! <:
                                                          PeriodicCallbackAffect{AnalysisCallback}
                                                          }
    analysis_callback = cb.affect!.affect!
    return analysis_callback.quantities
end

"""
    tstops(analysis_callback)

Return the time values that correspond to the saved values of the [`quantities`](@ref).
"""
function tstops(cb::DiscreteCallback{Condition,
                                     Affect!}) where {Condition,
                                                      Affect! <:
                                                      AnalysisCallback}
    analysis_callback = cb.affect!
    return analysis_callback.tstops
end

function tstops(cb::DiscreteCallback{Condition,
                                     Affect!}) where {Condition,
                                                      Affect! <:
                                                      PeriodicCallbackAffect{AnalysisCallback}
                                                      }
    analysis_callback = cb.affect!.affect!
    return analysis_callback.tstops
end

function AnalysisCallback(semi; interval = 0, dt = nothing)
    if !isnothing(dt) && interval > 0
        throw(ArgumentError("You can either set the number of steps between output (using `interval`) or the time between outputs (using `dt`) but not both simultaneously"))
    end

    if isnothing(dt)
        interval_or_dt = interval
    else # !isnothing(dt)
        interval_or_dt = dt
    end
    # Decide when the callback is activated.
    # With error-based step size control, some steps can be rejected. Thus,
    #   `integrator.iter >= integrator.stats.naccept`
    #    (total #steps)       (#accepted steps)
    # We need to check the number of accepted steps since callbacks are not
    # activated after a rejected step.
    condition = (u, t, integrator) -> interval > 0 &&
        ((integrator.stats.naccept % interval == 0) || isfinished(integrator))

    analysis_callback = AnalysisCallback(semi,
                                         Vector{Float64}(),
                                         Vector{Vector{Float64}}())

    if isnothing(dt)
        # Save every `interval` (accepted) time steps
        # The first one is the condition, the second the affect!
        return DiscreteCallback(condition, analysis_callback,
                                save_positions = (false, false))
    else
        # Add a `tstop` every `dt`, and save the final solution.
        return PeriodicCallback(analysis_callback, dt,
                                save_positions = (false, false),
                                initial_affect = true,
                                final_affect = true)
    end
end

function (analysis_callback::AnalysisCallback)(integrator)
    @unpack semi, tstops, quantities = analysis_callback
    du = first(get_tmp_cache(integrator))
    u = integrator.u
    p = integrator.p
    t = integrator.t
    push!(tstops, t)

    semi(du, u, p, t)

    push!(quantities, analyze_quantities(semi, du, u, p, t))

    # avoid re-evaluating possible FSAL stages
    u_modified!(integrator, false)

    return nothing
end

# Fallback
analyze_quantities(semi::AbstractSemidiscretization, du, u, p, t) = Float64[]
