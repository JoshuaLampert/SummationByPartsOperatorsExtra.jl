"""
    GlaubitzIskeLampertÖffner2026Basic()

Function space SBP operators constructed with optimization procedure given in
- Glaubitz, Iske, Lampert, Öffner (2026):
  Why summation by parts is not enough.
  [arXiv:2602.10786](https://arxiv.org/abs/2602.10786)

See [`function_space_operator`](@ref).
"""
struct GlaubitzIskeLampertÖffner2026Basic <: SourceOfCoefficients end

function Base.show(io::IO, source::GlaubitzIskeLampertÖffner2026Basic)
    if get(io, :compact, false)
        summary(io, source)
    else
        print(io,
              "Glaubitz, Iske, Lampert, Öffner (2026) \n",
              "  Why summation by parts is not enough.\n",
              "  arXiv:2602.10786")
    end
end

"""
    GlaubitzIskeLampertÖffner2026Regularized()

Function space SBP operators constructed with regularized constrained optimization procedure given in
- Glaubitz, Iske, Lampert, Öffner (2026):
  Why summation by parts is not enough.
  [arXiv:2602.10786](https://arxiv.org/abs/2602.10786)

See [`function_space_operator`](@ref).
"""
struct GlaubitzIskeLampertÖffner2026Regularized <: SourceOfCoefficients end

function Base.show(io::IO, source::GlaubitzIskeLampertÖffner2026Regularized)
    if get(io, :compact, false)
        summary(io, source)
    else
        print(io,
              "Glaubitz, Iske, Lampert, Öffner (2026) \n",
              "  Why summation by parts is not enough.\n",
              "  arXiv:2602.10786")
    end
end

"""
    GlaubitzIskeLampertÖffner2026EigenvalueProperty()

Function space SBP operators constructed with constrained optimization procedure satisfying an eigenvalue
property given in
- Glaubitz, Iske, Lampert, Öffner (2026):
  Why summation by parts is not enough.
  [arXiv:2602.10786](https://arxiv.org/abs/2602.10786)

See [`function_space_operator`](@ref).
"""
struct GlaubitzIskeLampertÖffner2026EigenvalueProperty <: SourceOfCoefficients end

function Base.show(io::IO, source::GlaubitzIskeLampertÖffner2026EigenvalueProperty)
    if get(io, :compact, false)
        summary(io, source)
    else
        print(io,
              "Glaubitz, Iske, Lampert, Öffner (2026) \n",
              "  Why summation by parts is not enough.\n",
              "  arXiv:2602.10786")
    end
end

# The construction of the operator is implemented in the package extensions OptimForwardDiffExt and
# ManifoldsManoptForwardDiffExt, which extend `construct_function_space_operator`
"""
    function_space_operator(basis_functions, nodes, source;
                            derivative_order = 1, accuracy_order = 0,
                            basis_functions_weights = ones(length(basis_functions)),
                            regularization_functions = nothing,
                            min_real_eigen = 0.1,
                            bandwidth = length(nodes) - 1, size_boundary = 2 * bandwidth,
                            different_values = true, sparsity_pattern = nothing,
                            opt_alg, options, autodiff = ADTypes.AutoForwardDiff(),
                            x0 = nothing, verbose = false)

Construct an operator that represents a first-derivative operator in a function space spanned by
the `basis_functions`, which is an iterable of functions. The operator is constructed on the
interval `[x_min, x_max]` with the nodes `nodes`, where `x_min` is taken as the minimal value in
`nodes` and `x_max` the maximal value. Note that the `nodes` will be sorted internally. The
`accuracy_order` is the order of the accuracy of the operator, which can optionally be passed,
but does not have any effect on the operator.

The operator is constructed solving an optimization problem with [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
for the source [`SummationByPartsOperators.GlaubitzNordströmÖffner2023`](@extref) and using [Manopt.jl](https://github.com/JuliaManifolds/Manopt.jl)
for the sources [`GlaubitzIskeLampertÖffner2026Basic`](@ref), [`GlaubitzIskeLampertÖffner2026Regularized`](@ref), and
[`GlaubitzIskeLampertÖffner2026EigenvalueProperty`](@ref). You can specify the
optimization algorithm, the options for the optimization problem, and the `autodiff` mode with
the keyword arguments `opt_alg`, `options`, and `autodiff` respectively, see also the documentation of
Optim.jl about [configurable options](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
and [automatic differentiation](https://julianlsolvers.github.io/Optim.jl/stable/user/gradientsandhessians/#Automatic-differentiation)
or of Manopt.jl about [algorithm options for, e.g., `augmented_Lagrangian_method`](https://manoptjl.org/stable/solvers/augmented_Lagrangian_method/#Manopt.augmented_Lagrangian_method)
or [`debug` output](https://manoptjl.org/stable/tutorials/HowToDebug/).
The defaults depend on the `source`: for [`SummationByPartsOperators.GlaubitzNordströmÖffner2023`](@extref),
`opt_alg = Optim.BFGS()` and `options = Optim.Options(g_tol = 1e-16, iterations = 10000, show_trace = verbose)`.
For the other sources, `opt_alg` is `Manopt.quasi_Newton` ([`GlaubitzIskeLampertÖffner2026Basic`](@ref)) or
`Manopt.augmented_Lagrangian_method` ([`GlaubitzIskeLampertÖffner2026Regularized`](@ref) and
[`GlaubitzIskeLampertÖffner2026EigenvalueProperty`](@ref)), and `options` is a `NamedTuple` of keyword arguments
passed to it, by default a `stopping_criterion` and a `debug` output if `verbose` is `true`.
For higher number of nodes, reverse mode automatic differentiation is usually significantly faster than forward mode.
We recommend using `autodiff = ADTypes.AutoMooncake(; config = nothing)` or
`autodiff = ADTypes.AutoEnzyme(; mode = Enzyme.Reverse, function_annotation = Enzyme.Duplicated)`. Note that
you need to import the package ADTypes.jl as well as the corresponding autodiff (i.e., Mooncake.jl or Enzyme.jl)
package to use these modes. Both Optim.jl and Manopt.jl support `autodiff` backends via ADTypes.jl.

The initial guess for the optimization problem can be passed with the keyword argument `x0`, which is optional.
If `nothing` is passed, a default initial guess (zeros for the entries of the differentiation matrix and
equal values for all the weights) is used.

You can weight the basis functions with the keyword argument `basis_functions_weights`, which is a vector
with one weight per basis function. The default is a vector of ones, which means that all basis functions
are equally weighted. Note that the weights are applied after orthonormalizing the basis: the basis functions
``f_1, \\dots, f_K`` are orthonormalized in the given order with respect to the discrete inner product
``\\langle f, g \\rangle = \\sum_n f(x_n) g(x_n) + \\sum_n f'(x_n) g'(x_n)``, which yields functions
``g_k = \\sum_{j \\le k} a_{kj} f_j`` with ``a_{kk} > 0``. The weight `basis_functions_weights[k]` multiplies the
residual of the exactness condition for ``g_k``, not for ``f_k``. Hence, the result depends on the order of
the basis functions. Since ``g_1, \\dots, g_m`` span the same space as ``f_1, \\dots, f_m``, you can enforce
exactness for certain basis functions (high weights), but allow non-exactness for others only minimizing the
error (low weights), by putting the functions to be reproduced at the beginning of `basis_functions` and
assigning the high weights to them.

When using the source [`SummationByPartsOperators.GlaubitzNordströmÖffner2023`](@extref) there are two alternative
ways to enforce sparsity of the resulting operator.

The first is by passing a matrix `sparsity_pattern` that is a matrix of zeros and ones, where the ones indicate
the non-zero entries of the operator. This matrix should be symmetric or `UpperTriangular` and have zeros on the diagonal.

The second way is to use a banded-block structure for the operator as is common, e.g., in finite difference methods.
The keyword arguments `bandwidth` and `size_boundary` specify the bandwidth and the size of the
boundary blocks of the operator, where the default of `bandwidth` is set to `length(nodes) - 1`,
i.e., a dense operator (in this case `size_boundary` is ignored). To construct a sparse operator, you can set the
bandwidth to a smaller value, such that `2 * size_boundary + bandwidth <= length(nodes)`, which is a
requirement for the boundary blocks in the upper left and lower right of the resulting operator.
If `different_values` is set to `true` all the entries in the upper right triangle of S (the skew symmetric
part of D) are different, which is generally meaningful for non-equidistant nodes and general bases, if it
is `false` the entries of the stencil are repeated in the central part and the two boundary closures share
their values (makes sense for uniformly distributed nodes and, e.g., a polynomial basis). The keyword
argument `different_values` is ignored for dense operators.

When using the regularized optimization procedure using [`GlaubitzIskeLampertÖffner2026Regularized`](@ref), you have to pass
additional functions to be regularized with the keyword argument `regularization_functions`, which is an iterable of functions.

When using the optimization procedure with eigenvalue constraint using [`GlaubitzIskeLampertÖffner2026EigenvalueProperty`](@ref),
you can specify the minimal real part of the eigenvalues of the resulting operator with the keyword argument `min_real_eigen`.

The keyword argument `verbose` can be set to `true` to print information about the optimization process.

The operator that is returned follows the general interface. Currently, it is wrapped in a
[`SummationByPartsOperators.MatrixDerivativeOperator`](@extref), but this might change in the future.

In order to use this function, either the packages Optim.jl and ForwardDiff.jl (for [`SummationByPartsOperators.GlaubitzNordströmÖffner2023`](@extref))
or Manifolds.jl, Manopt.jl, and ForwardDiff.jl (for [`GlaubitzIskeLampertÖffner2026Basic`](@ref), [`GlaubitzIskeLampertÖffner2026Regularized`](@ref),
and [`GlaubitzIskeLampertÖffner2026EigenvalueProperty`](@ref)) must be loaded.

See also [`SummationByPartsOperators.GlaubitzNordströmÖffner2023`](@extref), [`GlaubitzIskeLampertÖffner2026Basic`](@ref),
[`GlaubitzIskeLampertÖffner2026Regularized`](@ref), and [`GlaubitzIskeLampertÖffner2026EigenvalueProperty`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function function_space_operator end

function function_space_operator(basis_functions, nodes::Vector{T},
                                 source::SourceOfCoefficients;
                                 derivative_order = 1, accuracy_order = 0,
                                 kwargs...) where {T}
    assert_first_derivative_order(derivative_order)
    nodes = sort(nodes)
    weights, D = construct_function_space_operator(basis_functions, nodes, source;
                                                   kwargs...)
    return MatrixDerivativeOperator(first(nodes), last(nodes), nodes, weights, D,
                                    accuracy_order, source)
end

function assert_first_derivative_order(derivative_order)
    if derivative_order != 1
        throw(ArgumentError("Derivative order $derivative_order not implemented."))
    end
end

# We need to define these functions here, because we dispatch on the type of `source` in the
# different extensions
# Placeholder for constructing a function space operator. This method is extended in backend-specific implementations.
function construct_function_space_operator end

# Placeholder for providing the default optimization algorithm. This method is extended in backend-specific implementations.
function default_opt_alg end

# Placeholder for providing the default optimization options. This method is extended in backend-specific implementations.
function default_options end

# Placeholder for creating the matrix S. This method is extended in backend-specific implementations.
function create_S end
