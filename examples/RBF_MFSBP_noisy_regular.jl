using SummationByPartsOperatorsExtra
import Random
using Meshes, MeshIntegrals
using KernelInterpolation: WendlandKernel, LagrangeBasis, NodeSet
import Optim, ADTypes, Mooncake
Random.seed!(43)

xmin = -1.0
xmax = 1.0
ymin = -1.0
ymax = 1.0
N_x = 40
N_y = 40
dx = (xmax - xmin) / (N_x - 1)
dy = (ymax - ymin) / (N_y - 1)
geometry = Box((xmin, ymin), (xmax, ymax))
d = paramdim(geometry)
sampler = RegularSampling(N_x, N_y)

# Use the nodes, which are sampled at the boundary by `sampler` as the boundary nodes
sampler_boundary = nothing

nodes_regular, normals, boundary_indices = SummationByPartsOperatorsExtra.compute_nodes_normals(geometry,
                                                                                                sampler,
                                                                                                sampler_boundary)

nodes = Vector{eltype(nodes_regular)}(undef, length(nodes_regular))
for (i, node) in enumerate(nodes_regular)
    if i in boundary_indices
        nodes[i] = node
    else
        nodes[i] = node .+ (dx / 6, dy / 6) .* (1.0 .- 2.0 .* rand(d))
    end
end
corners = SummationByPartsOperatorsExtra.find_corners(boundary_indices)

K_RBF = 8
sampler_basis = HomogeneousSampling(K_RBF)
centers = PointSet(sample(geometry, sampler_basis))
kernel = WendlandKernel{2}(2; shape_parameter = 0.4)
basis = LagrangeBasis(NodeSet(centers), kernel; m = 2)

opt_kwargs = (;
              options = Optim.Options(f_abstol = 1e-25, g_tol = 1e-15, iterations = 50000,
                                      show_trace = true), opt_alg = Optim.BFGS(),
              autodiff = ADTypes.AutoMooncake(; config = nothing))

tol = 1e-5
shorter = tol # don't include a point in the other direction
longer = 2 * max(dx, dy) + tol # take two points to the left/bottom and two points to the right/top
ellipsoid_lengths = ((longer, shorter), (shorter, longer))
sparsity_patterns = ntuple(i -> neighborhood_sparsity_pattern(nodes_regular, # based on the regularly distributed nodes
                                                              ellipsoid_lengths[i]), d)
moments = compute_moments_boundary(basis, geometry)
vol = Meshes.ustrip(measure(geometry))

OUT = joinpath("out")
ispath(OUT) || mkdir(OUT)
file = "rectangle_Wendland_regular_noise_$(N_x)_$(N_y)_sparse_$(shorter)_$(longer)"

D = open(joinpath(OUT, "out_$file.txt"), "w") do file
    redirect_stdout(file) do
        D = multidimensional_function_space_operator(basis, nodes,
                                                     boundary_indices, normals, moments,
                                                     vol, GlaubitzIskeLampertÖffner2026();
                                                     corners, sparsity_patterns,
                                                     verbose = true, opt_kwargs...)
        return D
    end
end

using Serialization: serialize
serialize(joinpath(OUT, "D_$file.jls"), D)
