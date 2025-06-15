using SummationByPartsOperatorsExtra
using Optim: Optim, BFGS
using KernelInterpolation: WendlandKernel, StandardBasis, NodeSet
import ADTypes
import Mooncake
using OrdinaryDiffEqSSPRK

# domain
xmin, xmax = 0.0, 1.0
N = 100
nodes = LinRange(xmin, xmax, N + 1)
dx = step(nodes)

# RBF-FSBP operator
shape_parameter = 0.5
kernel = WendlandKernel{1}(2, shape_parameter = shape_parameter)
# This is a hack to avoid evaluating derivatives of the kernels at 0,
# see https://github.com/JuliaDiff/ForwardDiff.jl/issues/303.
# We perturb the right node a differently than the left node
# to ensure 0.5 is not in the centers because it is also in the nodes.
centers = NodeSet(LinRange(xmin + eps(), xmax - 2 * eps(), 5))
basis = StandardBasis(centers, kernel)
basis_functions = collect(basis)
push!(basis_functions, one)
D = function_space_operator(basis_functions, collect(nodes), GlaubitzNordströmÖffner2023();
                            opt_alg = BFGS(),
                            options = Optim.Options(g_tol = 1e-16, iterations = 5000),
                            autodiff = ADTypes.AutoMooncake(; config = nothing),
                            verbose = false, bandwidth = 3)

# equation
a(x) = 1.0
u0(x) = tanh(50 * (x - 0.1))
u(x, t) = u0(x - a(x) * t)
bc_left(t) = u(xmin, t)
bc_right(t) = 0.0 # This does not matter as a > 0
semi = VariableLinearAdvectionNonperiodicSemidiscretization(D, nothing, a, Val(true),
                                                            bc_left, bc_right)

# time integration
CFL = 0.9
dt = CFL * dx / a(xmin)
tspan = (0.0, 0.5)
ode = semidiscretize(u0, semi, tspan)
alg = SSPRK53()

saveat = range(tspan..., length = 100)
kwargs = (; dt = dt, adaptive = false, save_everystep = false, saveat = saveat)
sol = solve(ode, alg; kwargs...)
