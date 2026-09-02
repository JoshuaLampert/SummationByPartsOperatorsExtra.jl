using SummationByPartsOperatorsExtra
using LinearAlgebra: dot
import Optim, ForwardDiff
using OrdinaryDiffEqSSPRK

# domain
xmin, xmax = 0.0, 1.0
N = 48
nodes = collect(LinRange(xmin, xmax, N))
dx = step(LinRange(xmin, xmax, N))

# upwind FD-FSBP operators for the function space spanned by the `basis_functions`. The central
# operator is banded, as is common for finite difference operators; `bandwidth = 3` is the
# smallest bandwidth for which the optimization converges for this function space. The
# dissipation matrix is built from generalized divided differences and its strength is
# calibrated by a stiffness budget, i.e. it is the largest one for which the spectral radius of
# the semidiscretization grows by at most 20 %.
basis_functions = [one, identity, exp]
D = function_space_operator(basis_functions, nodes, GlaubitzNordströmÖffner2023();
                            bandwidth = 3, size_boundary = 8, different_values = false)
D_upwind = upwind_operators(D, basis_functions,
                            GlaubitzLampertMattssonNiemeläWinters2026FD())

# equation
# The initial condition carries the highest frequency the grid supports on top of a smooth wave.
# That mode is invisible to the function space and is exactly what the dissipation removes.
a(x) = 1.0
u_smooth(x, t) = sinpi(2 * (x - a(x) * t))
u0(x) = u_smooth(x, 0) + 0.1 * cospi((x - xmin) / dx)
# The unresolved mode is not fed back in at the inflow boundary, so its decay measures the
# dissipation in the volume alone.
bc_left(t) = u_smooth(xmin, t)
bc_right(t) = 0.0 # This does not matter as a > 0
# For a positive wave speed and the global Lax-Friedrichs splitting, only `D^-` remains.
# Passing the central operator `D` instead gives the corresponding central scheme.
semi = VariableLinearAdvectionNonperiodicSemidiscretization(D_upwind.minus, nothing, a,
                                                            Val(false), bc_left, bc_right)

# time integration
CFL = 0.9
dt = CFL * dx / a(xmin)
tspan = (0.0, 1.0)
ode = semidiscretize(u0, semi, tspan)
alg = SSPRK53()

analysis_callback = AnalysisCallback(semi; dt = 0.01)
saveat = range(tspan..., length = 100)
kwargs = (; dt = dt, adaptive = false, save_everystep = false, saveat = saveat,
          callback = analysis_callback)
sol = solve(ode, alg; kwargs...)

# Error against the smooth part of the solution, and the amplitude of the highest grid mode
# before and after the simulation. The unresolved mode leaves the domain through the outflow
# boundary in either case, but the volume dissipation of the upwind operators removes it while
# it is still inside, so that it pollutes the smooth solution far less: passing the central
# operator `D` to the semidiscretization above gives an error about two orders of magnitude
# larger here.
x = grid(D)
P = mass_matrix(D)
error_smooth = sol.u[end] - u_smooth.(x, tspan[end])
unresolved_mode = cospi.((x .- xmin) ./ dx)
mode_amplitude(u) = dot(unresolved_mode, P, u) / dot(unresolved_mode, P, unresolved_mode)
println("upwind FD-FSBP operators")
println("  error against the smooth solution: ", sqrt(dot(error_smooth, P, error_smooth)))
println("  amplitude of the highest grid mode: ", mode_amplitude(sol.u[begin]), " -> ",
        mode_amplitude(sol.u[end]))
