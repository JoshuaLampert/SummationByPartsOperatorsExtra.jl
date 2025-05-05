@testitem "Aqua.jl" begin
    import Aqua
    using ExplicitImports: check_no_implicit_imports, check_no_stale_explicit_imports,
                           check_all_explicit_imports_via_owners,
                           check_all_qualified_accesses_via_owners,
                           check_no_self_qualified_accesses

    Aqua.test_all(SummationByPartsOperatorsExtra,
                  stale_deps = (;
                                # These are needed in the extensions, but not in the main module.
                                # Since package extensions cannot have additional dependencies,
                                # we need to load them in the main module.
                                ignore = [:PreallocationTools, :StatsBase]))
    @test isnothing(check_no_implicit_imports(SummationByPartsOperatorsExtra,
                                              skip = (Core, Base,
                                                      SummationByPartsOperatorsExtra.SummationByPartsOperators)))
    @test isnothing(check_no_stale_explicit_imports(SummationByPartsOperatorsExtra))
    @test isnothing(check_all_qualified_accesses_via_owners(SummationByPartsOperatorsExtra))
    @test isnothing(check_no_self_qualified_accesses(SummationByPartsOperatorsExtra))
end
