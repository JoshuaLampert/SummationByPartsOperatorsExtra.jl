@testitem "Aqua.jl" begin
    import Aqua
    using ExplicitImports: test_explicit_imports

    Aqua.test_all(SummationByPartsOperatorsExtra,
                  stale_deps = (;
                                # These are needed in the extensions, but not in the main module.
                                # Since package extensions cannot have additional dependencies,
                                # we need to load them in the main module.
                                ignore = [:PreallocationTools, :ScopedValues]),
                  # We use type piracy for `interpolation_matrix(!)` for general derivative operators.
                  # This probably doesn't want to live in PolynomialBases.jl since it has some assumptions.
                  piracies = (;
                              treat_as_own = [PolynomialBases.interpolation_matrix,
                                  PolynomialBases.interpolation_matrix!]))

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
    own_names = Tuple(nameof.(own))
    # `eigen` for `ForwardDiff.Dual`s is needed for the eigenvalue property and can be removed
    # once ForwardDiff.jl supports differentiating eigenvalues of non-symmetric matrices.
    push!(own, LinearAlgebra.eigen)
    extension_names = (:OptimForwardDiffExt, :ManifoldsManoptForwardDiffExt, :MeshesExt,
                       :MeshesMakieExt)
    extensions = Base.get_extension.(Ref(SummationByPartsOperatorsExtra), extension_names)
    for extension in extensions
        @test !isnothing(extension)
        Aqua.test_piracies(extension; treat_as_own = own)
    end

    # Names, which are not public in the upstream packages.
    # `require_one_based_indexing` is only public from Julia v1.11.
    imports_not_public = (:AbstractDerivativeOperator,
                          :AbstractMultidimensionalMatrixDerivativeOperator,
                          :AbstractNonperiodicDerivativeOperator,
                          :AbstractSemidiscretization, :PeriodicCallbackAffect)
    accesses_not_public = (:Line, :barycentric_weights, :get_weight,
                           :godunov_flux_variablelinearadvection, :lower_bandwidth,
                           :require_one_based_indexing, :upper_bandwidth, :xmax, :xmin)
    # The package extensions use names of this package (e.g., the function stubs), which do not
    # need to be public, and some names, which are not public in the upstream packages.
    # `depwarn` is only public from Julia v1.11.
    extension_imports_not_public = (own_names..., :Options, :minimizer)
    extension_accesses_not_public = (own_names..., :ADTypes, :ATOL64, :AutoForwardDiff,
                                     :Dual, :ManifoldDiff, :depwarn, :derivative, :gradient,
                                     :pickchunksize, :tagtype, :ustrip)
    # The owners Unitful.jl, ADTypes.jl, and ManifoldDiff.jl are no (weak) dependencies,
    # so the package extensions cannot access these names via their owners.
    extension_accesses_not_via_owners = (:ustrip, :AutoForwardDiff, :ManifoldDiff)

    # ExplicitImports.jl analyzes the loaded package extensions as part of the package, but it
    # reports their findings only on some Julia versions (e.g., v1.10). Thus, we ignore the names
    # used in the package extensions here and check the package extensions separately below.
    test_explicit_imports(SummationByPartsOperatorsExtra;
                          # We reexport these packages.
                          no_implicit_imports = (;
                                                 skip = (Core, Base,
                                                         SummationByPartsOperatorsExtra.SummationByPartsOperators,
                                                         SummationByPartsOperatorsExtra.PolynomialBases)),
                          # Only imported for newer versions of SciMLBase.jl
                          no_stale_explicit_imports = (;
                                                       ignore = (:derivative_discontinuity!,)),
                          all_explicit_imports_are_public = (;
                                                             ignore = (imports_not_public...,
                                                                       extension_imports_not_public...)),
                          all_qualified_accesses_via_owners = (;
                                                               ignore = extension_accesses_not_via_owners),
                          all_qualified_accesses_are_public = (;
                                                               ignore = (accesses_not_public...,
                                                                         extension_accesses_not_public...)))
    for (name, extension) in zip(extension_names, extensions)
        test_explicit_imports(extension,
                              joinpath(pkgdir(SummationByPartsOperatorsExtra), "ext",
                                       "$name.jl");
                              all_explicit_imports_are_public = (;
                                                                 ignore = extension_imports_not_public),
                              all_qualified_accesses_via_owners = (;
                                                                   ignore = extension_accesses_not_via_owners),
                              all_qualified_accesses_are_public = (;
                                                                   ignore = extension_accesses_not_public))
    end
end
