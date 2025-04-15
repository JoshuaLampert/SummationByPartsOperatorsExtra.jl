@testsnippet Examples begin
    examples_dir() = pkgdir(SummationByPartsOperatorsExtra, "examples")
end

@testitem "RBF_FSBP_advection.jl" setup=[Examples] begin
    @test_nowarn include(joinpath(examples_dir(), "RBF_FSBP_advection.jl"))
end
