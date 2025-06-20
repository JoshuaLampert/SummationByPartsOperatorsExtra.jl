using SummationByPartsOperatorsExtra
using OrdinaryDiffEqLowStorageRK
using LinearAlgebra: norm
using Serialization: deserialize

file = "Wendland_scattered_0.05_sparse_0.08_0.14"
# file = "thinplate_regular_20_20_sparse_2"
# file = "Wendland_regular_40_40_sparse_2"
# file = "Wendland_regular_noise_40_40_sparse_2"
D = deserialize(joinpath(@__DIR__, "D_rectangle_$file.jls"))
# file = "MattssonNordström2004_tensorproduct"
# xmin, xmax = -1.0, 1.0
# ymin, ymax = -1.0, 1.0
# N_x, N_y = 40, 40

# # Construct one-dimensional SBP operators
# D_1 = derivative_operator(MattssonNordström2004(), derivative_order = 1, accuracy_order = 4,
#                           xmin = xmin, xmax = xmax, N = N_x)
# D_2 = derivative_operator(MattssonNordström2004(), derivative_order = 1, accuracy_order = 4,
#                           xmin = ymin, xmax = ymax, N = N_y)

# # Construct the two-dimensional SBP operator
# D = tensor_product_operator_2D(D_1, D_2)

# equation
a = (1.0, 1.0)
u0(x) = exp(-30 * norm(x .- (0.0, 0.0))^2)
u(x, t) = u0(x .- a .* t)
# u(x, t) = u0(mod.(x .- a .* t .- xmin, xmax .- xmin) .+ xmin) # periodic boundary conditions
g(x, t) = u(x, t) # boundary condition from the analytical solution
semi = MultidimensionalLinearAdvectionNonperiodicSemidiscretization(D, a, g)

# time integration
tspan = (0.0, 0.5)
ode = semidiscretize(u0, semi, tspan)
alg = RDPK3SpFSAL49()
analysis_callback = AnalysisCallback(semi; dt = 0.01)

saveat = range(tspan..., length = 100)
kwargs = (; save_everystep = false, saveat = saveat, callback = analysis_callback)
sol = solve(ode, alg; kwargs...)

using Plots: Plots, scatter, scatter!, pythonplot, mp4, savefig, @animate
import PythonPlot
using Printf
pythonplot()
OUT = joinpath("out", "images", "MFSBP_rectangle")
ispath(OUT) || mkpath(OUT)

nodes = grid(D)
# anim = @animate for i in eachindex(sol)
#     t = sol.t[i]
#     scatter(first.(nodes), last.(nodes), sol[i], label = "numerical")
#     scatter!(first.(nodes), last.(nodes), u.(nodes, Ref(t)), label = "analytical",
#              xlabel = "x", ylabel = "y", zlabel = "u",
#              title = @sprintf("t = %.2f", t), zrange = (-0.5, 1.1), legend = :topright,
#              dpi = 170)
# end
# mp4(anim, joinpath(OUT, "rectangle_advection_$file.mp4"), fps = 10)
t = sol.t[end]
scatter(first.(nodes), last.(nodes), sol[end], label = "numerical")
scatter!(first.(nodes), last.(nodes), u.(nodes, Ref(t)), label = "analytical",
         xlabel = "x", ylabel = "y", zlabel = "u",
         title = @sprintf("t = %.2f", t), zrange = (-0.5, 1.1), legend = :topright,
         dpi = 170)
savefig(joinpath(OUT, "rectangle_advection_$(file)_final.png"))
