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
