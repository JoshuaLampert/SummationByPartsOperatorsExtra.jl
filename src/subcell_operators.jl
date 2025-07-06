"""
    SubcellOperator{T}
    SubcellOperator(nodes::Vector{T}, x_M::T,
                    weights_left::Vector{T}, weights_right::Vector{T},
                    Q_left::QType, Q_right::QType,
                    B_left::BType, B_right::BType,
                    accuracy_order::Int,
                    source::SourceOfCoefficients) where {T <: Real,
                                                         QType <: AbstractMatrix{T},
                                                         BType <: AbstractMatrix{T},
                                                         SourceOfCoefficients}

A sub-cell derivative operator on a non-periodic grid with scalar type `T`. A sub-cell operator
consists of two parts, a left and a right part, which are defined on the left and right
sub-cells of the grid. Each of the two parts satisfy a summation-by-parts property on their
respecting sub-cell. The whole operator satisfies a summation-by-parts property on the whole
grid.

The whole operator follows the general interface of a derivative operator, e.g., implementing
matrix-vector multiplication, integration, and the mass matrix. To obtain the derivative matrix
``D = P^{-1}(Q_L + Q_R)`` associated to the sub-cell operator, use the function
[`derivative_matrix`](@ref) or `Matrix`.
The left and right mass matrices can be obtained with the functions
[`mass_matrix_left`](@ref) and [`mass_matrix_right`](@ref), respectively. Similarly, the
boundary mass matrices can be obtained with the functions
[`mass_matrix_boundary_left`](@ref) and [`mass_matrix_boundary_right`](@ref).

See also [`subcell_operator`](@ref) and [`GlaubitzLampertWintersNordström2025`](@ref).

References:
- Glaubitz, Lampert, Winters, Nordström (2025):
  Towards provable energy stable overset grid methods using sub-cell
  summation-by-parts operators.
  TODO
"""
@auto_hash_equals struct SubcellOperator{T, QType <: AbstractMatrix{T},
                                         BType <: AbstractMatrix{T},
                                         ProjectionType <: AbstractVector{T},
                                         SourceOfCoefficients} <:
                         AbstractNonperiodicDerivativeOperator{T}
    grid::Vector{T}
    x_M::T
    weights_left::Vector{T}
    weights_right::Vector{T}
    Q_left::QType
    Q_right::QType
    B_left::BType
    B_right::BType
    e_L::ProjectionType
    e_M_L::ProjectionType
    e_M_R::ProjectionType
    e_R::ProjectionType
    accuracy_order::Int
    source::SourceOfCoefficients

    function SubcellOperator(nodes::Vector{T}, x_M::T,
                             weights_left::Vector{T}, weights_right::Vector{T},
                             Q_left::QType, Q_right::QType,
                             B_left::BType, B_right::BType,
                             e_L::ProjectionType, e_M_L::ProjectionType,
                             e_M_R::ProjectionType, e_R::ProjectionType,
                             accuracy_order::Int,
                             source::SourceOfCoefficients) where {T <: Real,
                                                                  QType <:
                                                                  AbstractMatrix{T},
                                                                  BType <:
                                                                  AbstractMatrix{T},
                                                                  ProjectionType <:
                                                                  AbstractVector{T},
                                                                  SourceOfCoefficients
                                                                  }
        if length(nodes) != length(weights_left) + length(weights_right)
            throw(ArgumentError("If `x_M` is not in `nodes`, then the length of `nodes` must be equal to the sum of the lengths of `weights_left` and `weights_right`."))
        end
        new{T, QType, BType, ProjectionType, SourceOfCoefficients}(nodes, x_M,
                                                                   weights_left,
                                                                   weights_right,
                                                                   Q_left, Q_right,
                                                                   B_left, B_right,
                                                                   e_L, e_M_L, e_M_R, e_R,
                                                                   accuracy_order, source)
    end
end

SummationByPartsOperators.derivative_order(D::SubcellOperator) = 1
LinearAlgebra.issymmetric(D::SubcellOperator) = false

SummationByPartsOperators.source_of_coefficients(D::SubcellOperator) = D.source

function PolynomialBases.integrate(func, u, D::SubcellOperator)
    return integrate(func, u, weights(D))
end

"""
    grid_left(D::SubcellOperator)

Returns the grid associated to the left part of the sub-cell operator `D`.
"""
grid_left(D::SubcellOperator) = grid(D)[1:length(D.weights_left)]
"""
    grid_right(D::SubcellOperator)

Returns the grid associated to the right part of the sub-cell operator `D`.
"""
function grid_right(D::SubcellOperator)
    x = grid(D)
    N = length(x)
    N_R = length(D.weights_right)
    return x[(N - N_R + 1):N]
end

weights_left(D::SubcellOperator) = get_weight_left.(Ref(D), 1:size(D, 2))
weights_right(D::SubcellOperator) = get_weight_right.(Ref(D), 1:size(D, 2))
weights(D::SubcellOperator) = weights_left(D) + weights_right(D)
"""
    mass_matrix_left(D::SubcellOperator)

Returns the mass matrix associated to the left part of the sub-cell operator `D`.
"""
mass_matrix_left(D::SubcellOperator) = Diagonal(weights_left(D))
"""
    mass_matrix_right(D::SubcellOperator)

Returns the mass matrix associated to the right part of the sub-cell operator `D`.
"""
mass_matrix_right(D::SubcellOperator) = Diagonal(weights_right(D))
PolynomialBases.mass_matrix(D::SubcellOperator) = Diagonal(weights(D))

"""
    mass_matrix_boundary_left(D::SubcellOperator)

Returns the mass matrix associated to the left boundary of the sub-cell operator `D`.
"""
mass_matrix_boundary_left(D::SubcellOperator) = D.B_left
"""
    mass_matrix_boundary_right(D::SubcellOperator)

Returns the mass matrix associated to the right boundary of the sub-cell operator `D`.
"""
mass_matrix_boundary_right(D::SubcellOperator) = D.B_right
# If e_L and e_R are the first and last unit vectors and e_M is the same in both sub-cells, this would always be diag(-1, 0, ..., 0, 1).
PolynomialBases.mass_matrix_boundary(D::SubcellOperator) = D.B_left + D.B_right

"""
    left_projection_left(D::SubcellOperator)

Returns the left projection operator ``e_L`` associated to the left part of the sub-cell operator `D`.
"""
left_projection_left(D::SubcellOperator) = D.e_L
"""
    left_projection_right(D::SubcellOperator)

Returns the right projection operator ``e_{M_L}`` associated to the left part of the sub-cell operator `D`.
"""
left_projection_right(D::SubcellOperator) = D.e_M_L
"""
    right_projection_left(D::SubcellOperator)

Returns the left projection operator ``e_{M_R}`` associated to the right part of the sub-cell operator `D`.
"""
right_projection_left(D::SubcellOperator) = D.e_M_R
"""
    right_projection_right(D::SubcellOperator)

Returns the right projection operator ``e_R`` associated to the right part of the sub-cell operator `D`.
"""
right_projection_right(D::SubcellOperator) = D.e_R

"""
    derivative_matrix(Dop::SubcellOperator)

Returns the derivative matrix ``D = P^{-1}(Q_L + Q_R)`` associated to the sub-cell operator `Dop`.
"""
PolynomialBases.derivative_matrix(D::SubcellOperator) = inv(mass_matrix(D)) *
                                                        (D.Q_left + D.Q_right)

Base.Matrix(D::SubcellOperator) = copy(derivative_matrix(D))

Base.eltype(::SubcellOperator{T}) where {T} = T

function SummationByPartsOperators.scale_by_mass_matrix!(u::AbstractVector,
                                                         D::SubcellOperator,
                                                         factor = true)
    Base.require_one_based_indexing(u)
    @boundscheck begin
        length(u) == size(D, 2) ||
            throw(DimensionMismatch("sizes of input vector and operator do not match"))
    end

    @inbounds @simd for i in eachindex(u, weights(D))
        u[i] = factor * u[i] * get_weight(D, i)
    end

    return u
end

function SummationByPartsOperators.scale_by_inverse_mass_matrix!(u::AbstractVector,
                                                                 D::SubcellOperator,
                                                                 factor = true)
    Base.require_one_based_indexing(u)
    @boundscheck begin
        length(u) == size(D, 2) ||
            throw(DimensionMismatch("sizes of input vector and operator do not match"))
    end

    @inbounds @simd for i in eachindex(u, weights(D))
        u[i] = factor * u[i] / get_weight(D, i)
    end

    u
end

function get_weight_left(D::SubcellOperator, i::Int)
    @unpack weights_left = D
    N, _ = size(D)
    N_L = length(weights_left)
    @boundscheck begin
        @argcheck 1 <= i <= N
    end
    if i <= N_L
        @inbounds ω = weights_left[i]
    else
        ω = zero(eltype(D))
    end
    ω
end

function get_weight_right(D::SubcellOperator, i::Int)
    @unpack weights_right = D
    N, _ = size(D)
    N_R = length(weights_right)
    @boundscheck begin
        @argcheck 1 <= i <= N
    end
    if i > N - N_R
        @inbounds ω = weights_right[i - (N - N_R)]
    else
        ω = zero(eltype(D))
    end
    ω
end

get_weight(D::SubcellOperator, i::Int) = get_weight_left(D, i) + get_weight_right(D, i)

function Base.show(io::IO, D::SubcellOperator)
    if get(io, :compact, false)
        summary(io, D)
    else
        x = grid(D)
        x_L = first(x)
        x_R = last(x)
        N_L = length(grid_left(D))
        N_R = length(grid_right(D))
        print(io, "Sub-cell operator {T=", eltype(D), "}")
        print(io, " on ", length(x), " nodes in [", x_L, ", ", x_R, "]")
        print(io, " with ", N_L, " nodes in left sub-cell [", x_L, ", ", D.x_M, "]")
        print(io, " and ", N_R, " nodes in right sub-cell [", D.x_M, ", ", x_R, "]")
    end
end

function mul!(dest::AbstractVector, Dop::SubcellOperator, u::AbstractVector,
              α = true, β = false)
    D = Matrix(Dop)
    N, _ = size(D)
    @boundscheck begin
        @argcheck N == length(u)
        @argcheck N == length(dest)
    end

    mul!(dest, D, u, α, β)
end

function SummationByPartsOperators.lower_bandwidth(D::SubcellOperator)
    size(D, 1) - 1
end

function SummationByPartsOperators.upper_bandwidth(D::SubcellOperator)
    size(D, 1) - 1
end

function SummationByPartsOperators.accuracy_order(D::SubcellOperator)
    D.accuracy_order
end

function SummationByPartsOperators.left_boundary_weight(D::SubcellOperator)
    @inbounds retval = D.weights_left[begin]
    retval
end

function SummationByPartsOperators.right_boundary_weight(D::SubcellOperator)
    @inbounds retval = D.weights_right[end]
    retval
end

"""
    couple_subcell(D_left::AbstractNonperiodicDerivativeOperator,
                   D_right::AbstractNonperiodicDerivativeOperator,
                   x_M)

Construct a [`SubcellOperator`](@ref) from two non-periodic derivative operators `D_left` and `D_right` from
SummationByPartsOperators.jl. `D_left` is defined on the left sub-cell, which is the interval
``[x_L, x_M]`` and `D_right` is defined on the right sub-cell, which is the interval ``[x_M, x_R]``,
where `x_L` and `x_R` are the left and right boundaries of the grid of `D_left` and `D_right`, respectively.
Note that `x_M` must be between the right boundary of `D_left` and the left boundary of `D_right`.

See also [`GlaubitzLampertWintersNordström2025`](@ref).
"""
function couple_subcell(D_left::AbstractNonperiodicDerivativeOperator,
                        D_right::AbstractNonperiodicDerivativeOperator, x_M)
    T = promote_type(eltype(D_left), eltype(D_right))
    grid_left = grid(D_left)
    N_L = length(grid_left)
    grid_right = grid(D_right)
    N_R = length(grid_right)
    if x_M < last(grid_left)
        throw(ArgumentError("Left sub-cell must be to the left of x_M."))
    end
    if x_M > first(grid_right)
        throw(ArgumentError("Right sub-cell must be to the right of x_M."))
    end
    nodes = vcat(collect(grid_left), collect(grid_right))
    weights_left_ = diag(mass_matrix(D_left))
    weights_right_ = diag(mass_matrix(D_right))

    Q_left_ = mass_matrix(D_left) * Matrix(D_left)
    Q_left = [Q_left_ zeros(T, N_L, N_R)
              zeros(T, N_R, N_L) zeros(T, N_R, N_R)]
    Q_right_ = mass_matrix(D_right) * Matrix(D_right)
    Q_right = [zeros(T, N_L, N_L) zeros(T, N_L, N_R)
               zeros(T, N_R, N_L) Q_right_]

    B_left_ = mass_matrix_boundary(D_left)
    B_left = [B_left_ zeros(T, N_L, N_R)
              zeros(T, N_R, N_L) zeros(T, N_R, N_R)]
    B_right_ = mass_matrix_boundary(D_right)
    B_right = [zeros(T, N_L, N_L) zeros(T, N_L, N_R)
               zeros(T, N_R, N_L) B_right_]

    R_left = interpolation_matrix([boundary_left(D_left), boundary_right(D_left)], D_left)
    e_L = [R_left[1, :]; zeros(T, N_R)]
    e_M_L = [R_left[2, :]; zeros(T, N_R)]
    R_right = interpolation_matrix([boundary_left(D_right), boundary_right(D_right)],
                                   D_right)
    e_M_R = [zeros(T, N_L); R_right[1, :]]
    e_R = [zeros(T, N_L); R_right[2, :]]
    acc_order = min(accuracy_order(D_left), accuracy_order(D_right))
    source = GlaubitzLampertWintersNordström2025()
    return SubcellOperator(nodes, x_M, weights_left_, weights_right_,
                           Q_left, Q_right, B_left, B_right,
                           e_L, e_M_L, e_M_R, e_R,
                           acc_order, source)
end

"""
    GlaubitzLampertWintersNordström2025()

Sub-cell SBP operators given in
- Glaubitz, Lampert, Winters, Nordström (2025):
  Towards provable energy stable overset grid methods using sub-cell
  summation-by-parts operators.
  TODO

See [`subcell_operator`](@ref).
"""
struct GlaubitzLampertWintersNordström2025 <: SourceOfCoefficients end

function Base.show(io::IO, source::GlaubitzLampertWintersNordström2025)
    if get(io, :compact, false)
        summary(io, source)
    else
        print(io,
              "Glaubitz, Lampert, Winters, Nordström (2025) \n",
              "  Towards provable energy stable overset grid methods using sub-cell \n",
              "  summation-by-parts operators. \n",
              "  TODO")
    end
end

# This function is extended in the package extension SummationByPartsOperatorsExtraOptimExt
"""
    subcell_operator(basis_functions, nodes, x_M, source;
                     derivative_order = 1, accuracy_order = 0,
                     bandwidths = [N_L - 1, N_R - 1], size_boundaries = 2 .* bandwidths,
                     different_values = [true, true], sparsity_patterns = [nothing, nothing],
                     M_local_approximation = [N_L, N_R],
                     opt_alg = Optim.LBFGS(), options = Optim.Options(g_tol = 1e-14, iterations = 10000),
                     autodiff = :forward, x0 = nothing, verbose = false)

Construct a sub-cell operator in a function space spanned by the `basis_functions`, which is an
iterable of functions. The operator is constructed on the interval `[x_min, x_max]` with the nodes `nodes`,
where `x_min` is taken as the minimal value in `nodes` and `x_max` the maximal value. Note that the `nodes`
will be sorted internally. The left part of the sub-cell operator consists of the `nodes`, which are smaller
than `x_M` and the right part of the `nodes`, which are bigger than `x_M`.
The `accuracy_order` is the order of the accuracy of the operator, which can optionally be passed,
but does not have any effect on the operator.

The operator is constructed solving an optimization problem with Optim.jl. You can specify the
optimization algorithm, the options for the optimization problem, and the `autodiff` mode with
the keyword arguments `opt_alg`, `options`, and `autodiff` respectively, see also the documentation of
Optim.jl about [configurable options](https://julianlsolvers.github.io/Optim.jl/stable/user/config/)
and [automatic differentiation](https://julianlsolvers.github.io/Optim.jl/stable/user/gradientsandhessians/#Automatic-differentiation).
In this case, reverse mode automatic differentiation is usually significantly faster than forward mode.
We recommend using `autodiff = ADTypes.AutoMooncake(; config = nothing)` or
`autodiff = ADTypes.AutoEnzyme(; mode = Enzyme.Reverse, function_annotation = Enzyme.Duplicated)`. Note that
you need to import the package `ADTypes` as well as the corresponding autodiff (i.e., `Mooncake` or `Enzyme`)
package to use these modes.

The initial guess for the optimization problem can be passed with the keyword argument `x0`, which is optional.
If `nothing` is passed, a default initial guess (zeros for the entries of the differentiation matrix and
equal values for all the weights) is used.

There are two alternative ways to enforce sparsity of the resulting left and right operator. The first is by passing
matrices `sparsity_pattern` that are matrices of zeros and ones each, where the ones indicate the non-zero
entries of the left and operator, respectively. The matrices should be symmetric or `UpperTriangular` and have zeros on the diagonal.

The second way is to use a banded-block structure for the parts of the operator as is common, e.g., in finite difference methods.
The keyword arguments `bandwidths` and `size_boundaries` specify the bandwidth and the size of the
boundary blocks of the operators, where the default of `bandwidths` is set to the number of nodes in the left and right
sub-cell minus one, i.e., a dense operator (in this case `size_boundaries` is ignored). To construct a sparse operator, you can set the
bandwidth to a smaller value, such that `2 * size_boundaries[i] + bandwidths[i] < N_{L/R}`, which is a
requirement for the boundary blocks in the upper left and lower right of the resulting operator.
If `different_values` is set to `true` all the entries in the upper right triangle of S (the skew symmetric
parts of the differentiation matrix blocks) are different, which is generally meaningful for non-equidistant nodes and general bases, if it
is `false` the entries of the stencil are repeated in the central part and the two boundary closures share
their values (makes sense for uniformly distributed nodes and, e.g., a polynomial basis). The keyword
argument `different_values` is ignored for dense operators.

You can use the keyword argument `M_local_approximation` to specify the number of points used for local approximations
of the discrete projections. The default is to use the number of nodes in the left and right sub-cell, respectively.
To use an interpolation, you can set `M_local_approximation` to `[K, K]`, where `K` is the number of basis functions.

The keyword argument `verbose` can be set to `true` to print information about the optimization process.

Returns a [`SubcellOperator`](@ref) object.

See also [`GlaubitzLampertWintersNordström2025`](@ref).

!!! compat "Julia 1.9"
    This function requires at least Julia 1.9.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function subcell_operator end

# Just to be able to call them from outside
function create_S_left end
function create_S_right end
