using SummationByPartsOperatorsExtra
using OrdinaryDiffEqLowStorageRK
using LinearAlgebra: norm
using Serialization: deserialize

# Note: You need to run the example RBF_MFSBP.jl first to create the file D_rectangle_Wendland_scattered_0.05_sparse_0.08_0.14.jls
# or use another .jls file containing a MultidimensionalFunctionSpaceOperator.
file = "Wendland_scattered_0.05_sparse_0.08_0.14"
D = deserialize(joinpath(@__DIR__, "D_rectangle_$file.jls"))

# equation
advection_velocity = (1.0, 1.0)
u0(x) = exp(-30 * norm(x .- (0.0, 0.0))^2)
u(x, t) = u0(x .- advection_velocity .* t)
# u(x, t) = u0(mod.(x .- a .* t .- xmin, xmax .- xmin) .+ xmin) # periodic boundary conditions
g(x, t) = u(x, t) # boundary condition from the analytical solution
semi = MultidimensionalLinearAdvectionNonperiodicSemidiscretization(D, advection_velocity, g)

# time integration
tspan = (0.0, 0.5)
ode = semidiscretize(u0, semi, tspan)
alg = RDPK3SpFSAL49()
analysis_callback = AnalysisCallback(semi; dt = 0.01)

saveat = range(tspan..., length = 100)
kwargs = (; save_everystep = false, saveat = saveat, callback = analysis_callback)
sol = solve(ode, alg; kwargs...)
