@testsnippet Examples begin
    using TrixiBase: trixi_include
    include("test_util.jl")
    examples_dir() = pkgdir(SummationByPartsOperatorsExtra, "examples")
end

@testitem "RBF_FSBP_advection.jl" setup=[Examples] begin
    @test_nowarn include(joinpath(examples_dir(), "RBF_FSBP_advection.jl"))
end

@testitem "RBF_MFSBP.jl" setup=[Examples] begin
    # This example takes a long time to run, so we only test it with a small number of iterations.
    @test_nowarn_mod trixi_include(joinpath(examples_dir(), "RBF_MFSBP.jl"),
                                   iterations = 20)
end

@testitem "RBF_MFSBP.jl with regular sampling" setup=[Examples] begin
    using Meshes: RegularSampling
    xmin = -1.0
    xmax = 1.0
    ymin = -1.0
    ymax = 1.0
    N_x, N_y = 20, 20
    sampler = RegularSampling(N_x, N_y)
    # Use the nodes, which are sampled at the boundary by `sampler` as the boundary nodes
    sampler_boundary = nothing

    dx = (xmax - xmin) / (N_x - 1)
    dy = (ymax - ymin) / (N_y - 1)
    tol = 1e-5
    shorter = tol # don't include a point in the other direction
    longer = 2 * max(dx, dy) + tol # take two points to the left/bottom and two points to the right/top
    kwargs = (; ellipsoid_lengths = ((longer, shorter), (shorter, longer)), verbose = true)

    # This example takes a long time to run, so we only test it with a small number of iterations.
    @test_nowarn_mod trixi_include(joinpath(examples_dir(), "RBF_MFSBP.jl"),
                                   xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax,
                                   sampler = sampler, sampler_boundary = sampler_boundary,
                                   kwargs = kwargs,
                                   iterations = 50)
end
