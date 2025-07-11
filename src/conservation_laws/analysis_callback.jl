@inline function isfinished(integrator)
    # Checking for floating point equality is OK here as `DifferentialEquations.jl`
    # sets the time exactly to the final time in the last iteration
    return integrator.t == last(integrator.sol.prob.tspan) ||
           isempty(integrator.opts.tstops) ||
           integrator.iter == integrator.opts.maxiters
end

"""
    AnalysisCallback(semi; interval = 0, dt = nothing)

Analyze the numerical solution either every `interval` accepted time steps
or every `dt` in terms of integration time. You can only pass either `interval`
or `dt`, but not both at the same time.
The analyzed quantities are computed by `analyze_quantities` defined for each
equation type. The resulting quantities can be accessed via the
[`quantities`](@ref) function, and the corresponding time values via the
[`tstops`](@ref) function.
"""
mutable struct AnalysisCallback{IntervalType}
    semi::AbstractSemidiscretization
    interval_or_dt::IntervalType
    tstops::Vector{Float64}
    quantities::Vector{Vector{Float64}}
end

function Base.show(io::IO, cb::DiscreteCallback{<:Any, <:AnalysisCallback})
    @nospecialize cb # reduce precompilation time

    analysis_callback = cb.affect!
    print(io, "AnalysisCallback(interval=", analysis_callback.interval_or_dt,
          ")")
end

function Base.show(io::IO,
                   cb::DiscreteCallback{<:Any,
                                        <:PeriodicCallbackAffect{<:AnalysisCallback}})
    @nospecialize cb # reduce precompilation time

    analysis_callback = cb.affect!.affect!
    print(io, "AnalysisCallback(dt=", analysis_callback.interval_or_dt, ")")
end

"""
    quantities(analysis_callback)

Return the computed quantities for each time step.
"""
function quantities(cb::DiscreteCallback{<:Any, <:AnalysisCallback})
    analysis_callback = cb.affect!
    return analysis_callback.quantities
end

function quantities(cb::DiscreteCallback{<:Any,
                                         <:PeriodicCallbackAffect{<:AnalysisCallback}})
    analysis_callback = cb.affect!.affect!
    return analysis_callback.quantities
end

"""
    tstops(analysis_callback)

Return the time values that correspond to the saved values of the [`quantities`](@ref).
"""
function tstops(cb::DiscreteCallback{<:Any, <:AnalysisCallback})
    analysis_callback = cb.affect!
    return analysis_callback.tstops
end

function tstops(cb::DiscreteCallback{<:Any,
                                     <:PeriodicCallbackAffect{<:AnalysisCallback}})
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
    condition = (u, t, integrator) -> interval_or_dt > 0 &&
        ((integrator.stats.naccept % interval_or_dt == 0) || isfinished(integrator))

    analysis_callback = AnalysisCallback(semi,
                                         interval_or_dt,
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

function analyze_quantities(semi::VariableLinearAdvectionNonperiodicSemidiscretization, du,
                            u, p, t)
    @unpack a, left_bc, right_bc = semi
    D = semi.derivative
    P = mass_matrix(D)
    mass = sum(P * u)
    mass_rate = sum(P * du)
    fnum_left = SummationByPartsOperators.godunov_flux_variablelinearadvection(left_bc(t),
                                                                               u[1], a[1])
    fnum_right = SummationByPartsOperators.godunov_flux_variablelinearadvection(u[end],
                                                                                right_bc(t),
                                                                                a[end])
    mass_rate_boundary = mass_rate - fnum_left + fnum_right

    energy = 0.5 * sum(P * (u .^ 2)) # = 1/2 ||u||_P^2
    energy_rate = sum(P * (du .* u)) # = 1/2 d/dt||u||_P^2 = u' * P * du
    energy_rate_boundary = energy_rate + 0.5 * (u .^ 2)' * P * (D * a)
    if a[1] > 0.0
        energy_rate_boundary -= (0.5 * a[1] * left_bc(t)^2 - a[end] * u[end]^2)
    end
    if a[end] < 0.0
        energy_rate_boundary -= (0.5 * a[1] * u[1]^2 - a[end] * right_bc(t)^2)
    end
    energy_rate_boundary_dissipation = energy_rate_boundary
    if a[1] > 0.0
        energy_rate_boundary_dissipation += 0.5 * a[1] * (u[1] - left_bc(t))^2
    end
    if a[end] < 0.0
        energy_rate_boundary_dissipation += 0.5 * a[end] * (u[end] - right_bc(t))^2
    end
    return [mass, mass_rate, mass_rate_boundary,
            energy, energy_rate, energy_rate_boundary, energy_rate_boundary_dissipation]
end
