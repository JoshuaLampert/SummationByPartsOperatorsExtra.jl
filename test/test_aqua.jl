@testitem "Aqua.jl" begin
    import Aqua
    using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports,
                           check_all_explicit_imports_via_owners,
                           check_all_qualified_accesses_via_owners,
                           check_no_self_qualified_accesses

    Aqua.test_all(SummationByPartsOperatorsExtra,
                  stale_deps = (;
                                ignore = [:PreallocationTools]))
    @test isnothing(check_no_implicit_imports(SummationByPartsOperatorsExtra,
                                              skip = (SummationByPartsOperatorsExtra.SummationByPartsOperators,)))
    @test isnothing(check_no_stale_explicit_imports(SummationByPartsOperatorsExtra,
                                                    ignore = (:AbstractMultidimensionalMatrixDerivativeOperator,)))
    @test isnothing(check_all_qualified_accesses_via_owners(SummationByPartsOperatorsExtra))
    @test isnothing(check_no_self_qualified_accesses(SummationByPartsOperatorsExtra))
end
