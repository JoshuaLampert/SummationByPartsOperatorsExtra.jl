# This example takes some time to run.
using SummationByPartsOperatorsExtra
import Random
using Meshes, MeshIntegrals
using KernelInterpolation
import Optim,  ADTypes, Mooncake
Random.seed!(43)

xmin = -1.0
xmax = 1.0
ymin = -1.0
ymax = 1.0
geometry = Box((xmin, ymin), (xmax, ymax))
d = paramdim(geometry)
alpha = 0.05
sampler = MinDistanceSampling(alpha)

N_boundary = 76
sampler_boundary = RegularSampling(N_boundary)

K_RBF = 3
sampler_basis = HomogeneousSampling(K_RBF)
centers = PointSet(sample(geometry, sampler_basis))
kernel = WendlandKernel{2}(2; shape_parameter = 0.4)
basis = LagrangeBasis(NodeSet(centers), kernel; m = 2)

opt_kwargs = (;
              options = Optim.Options(f_abstol = 1e-20, g_tol = 1e-13, iterations = 50000,
                                      show_trace = true), opt_alg = Optim.BFGS(),
              autodiff = ADTypes.AutoMooncake(; config = nothing))

shorter = 0.08
longer = 0.14
ellipsoid_lengths = ((longer, shorter), (shorter, longer))
kwargs = (; ellipsoid_lengths, verbose = true)

D = multidimensional_function_space_operator(basis, geometry, sampler,
                                             sampler_boundary,
                                             GlaubitzIskeLampertÖffner2025();
                                             kwargs..., opt_kwargs...)
