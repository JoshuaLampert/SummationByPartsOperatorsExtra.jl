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
                                ignore = [:PreallocationTools, :ScopedValues]),
                  # We use type piracy for `mass_matrix_boundary`. This probably doesn't want to live
                  # in SummationByPartsOperators.jl since it has some assumptions. The same holds for
                  # `interpolation_matrix(!)` for general derivative operators.
                  piracies = (;
                              treat_as_own = [PolynomialBases.interpolation_matrix,
                                  PolynomialBases.interpolation_matrix!,
                                  SummationByPartsOperators.mass_matrix_boundary]))

    # Check the package extensions for type piracy. Aqua.jl treats only names defined in an
    # extension itself as own, but methods for functions and types of this package are fine.
    import Optim, ForwardDiff, Manifolds, Manopt, Meshes, CairoMakie
    import LinearAlgebra
    own = Any[]
    for name in names(SummationByPartsOperatorsExtra; all = true)
        isdefined(SummationByPartsOperatorsExtra, name) || continue
        obj = getfield(SummationByPartsOperatorsExtra, name)
        if (obj isa Function || obj isa Type) &&
           parentmodule(obj) === SummationByPartsOperatorsExtra
            push!(own, obj)
        end
    end
    # `eigen` for `ForwardDiff.Dual`s is needed for the eigenvalue property and can be removed
    # once ForwardDiff.jl supports differentiating eigenvalues of non-symmetric matrices.
    push!(own, LinearAlgebra.eigen)
    for name in (:OptimForwardDiffExt, :ManifoldsManoptForwardDiffExt, :MeshesExt,
                 :MeshesMakieExt)
        extension = Base.get_extension(SummationByPartsOperatorsExtra, name)
        @test !isnothing(extension)
        Aqua.test_piracies(extension; treat_as_own = own)
    end
    @test isnothing(check_no_implicit_imports(SummationByPartsOperatorsExtra,
                                              skip = (Core, Base,
                                                      SummationByPartsOperatorsExtra.SummationByPartsOperators,
                                                      SummationByPartsOperatorsExtra.PolynomialBases)))
    @test isnothing(check_no_stale_explicit_imports(SummationByPartsOperatorsExtra,
                                                    ignore = (:derivative_discontinuity!,)))
    @test isnothing(check_all_qualified_accesses_via_owners(SummationByPartsOperatorsExtra;
                                                            ignore = (:ustrip,
                                                                      :AutoForwardDiff,
                                                                      :ManifoldDiff)))
    @test isnothing(check_no_self_qualified_accesses(SummationByPartsOperatorsExtra))
end
