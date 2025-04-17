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

TODO

See also [`subcell_operator`](@ref).

References:
- TODO
"""
@auto_hash_equals struct SubcellOperator{T, QType <: AbstractMatrix{T},
                                         BType <: AbstractMatrix{T},
                                         SourceOfCoefficients} <:
                         SummationByPartsOperators.AbstractNonperiodicDerivativeOperator{T}
    grid::Vector{T}
    x_M::T
    weights_left::Vector{T}
    weights_right::Vector{T}
    Q_left::QType
    Q_right::QType
    B_left::BType
    B_right::BType
    accuracy_order::Int
    source::SourceOfCoefficients

    function SubcellOperator(nodes::Vector{T}, x_M::T,
                             weights_left::Vector{T}, weights_right::Vector{T},
                             Q_left::QType, Q_right::QType,
                             B_left::BType, B_right::BType,
                             accuracy_order::Int,
                             source::SourceOfCoefficients) where {T <: Real,
                                                                  QType <:
                                                                  AbstractMatrix{T},
                                                                  BType <:
                                                                  AbstractMatrix{T},
                                                                  SourceOfCoefficients
                                                                  }
        if x_M in nodes
            if length(nodes) != length(weights_left) + length(weights_right) - 1
                throw(ArgumentError("If `x_M` is in `nodes`, then the length of `nodes` must be equal to the sum of the lengths of `weights_left` and `weights_right` minus 1."))
            end
        else
            if length(nodes) != length(weights_left) + length(weights_right)
                throw(ArgumentError("If `x_M` is not in `nodes`, then the length of `nodes` must be equal to the sum of the lengths of `weights_left` and `weights_right`."))
            end
        end
        new{T, QType, BType, SourceOfCoefficients}(nodes, x_M,
                                                   weights_left, weights_right,
                                                   Q_left, Q_right,
                                                   B_left, B_right,
                                                   accuracy_order, source)
    end
end

SummationByPartsOperators.derivative_order(D::SubcellOperator) = 1
LinearAlgebra.issymmetric(D::SubcellOperator) = false

SummationByPartsOperators.source_of_coefficients(D::SubcellOperator) = D.source

function integrate(func, u, D::SubcellOperator)
    return integrate(func, u, weights(D))
end

grid_left(D::SubcellOperator) = grid(D)[1:length(D.weights_left)]
function grid_right(D::SubcellOperator)
    x = grid(D)
    N = length(x)
    N_R = length(D.weights_right)
    return x[(N - N_R + 1):N]
end

weights_left(D::SubcellOperator) = get_weight_left.(Ref(D), 1:size(D, 2))
weights_right(D::SubcellOperator) = get_weight_right.(Ref(D), 1:size(D, 2))
weights(D::SubcellOperator) = weights_left(D) + weights_right(D)
mass_matrix_left(D::SubcellOperator) = Diagonal(weights_left(D))
mass_matrix_right(D::SubcellOperator) = Diagonal(weights_right(D))
SummationByPartsOperators.mass_matrix(D::SubcellOperator) = Diagonal(weights(D))

mass_matrix_boundary_left(D::SubcellOperator) = D.B_left
mass_matrix_boundary_right(D::SubcellOperator) = D.B_right
# This is always diag(-1, 0, ..., 0, 1) if e_L and e_R are the first and last unit vectors
# SummationByPartsOperators.mass_matrix_boundary(D::SubcellOperator) = D.B_left + D.B_right

"""
    derivative_matrix(Dop::SubcellOperator)

Returns the derivative matrix ``D = P^{-1}(Q_L + Q_R)`` associated to the sub-cell operator `Dop`.
"""
derivative_matrix(D::SubcellOperator) = inv(mass_matrix(D)) * (D.Q_left + D.Q_right)

Base.eltype(::SubcellOperator{T}) where {T} = T

function scale_by_mass_matrix!(u::AbstractVector, D::SubcellOperator,
                               factor = true)
    Base.require_one_based_indexing(u)
    @boundscheck begin
        length(u) == size(D, 2) ||
            throw(DimensionMismatch("sizes of input vector and operator do not match"))
    end

    @inbounds @simd for i in eachindex(u, D.weights)
        u[i] = factor * u[i] * get_weight(D, i)
    end

    return u
end

function scale_by_inverse_mass_matrix!(u::AbstractVector, D::SubcellOperator, factor = true)
    Base.require_one_based_indexing(u)
    @boundscheck begin
        length(u) == size(D, 2) ||
            throw(DimensionMismatch("sizes of input vector and operator do not match"))
    end

    @inbounds @simd for i in eachindex(u, D.weights)
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
    D = derivative_matrix(Dop)
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
    GlaubitzLampertNordströmWinters2025()

Sub-cell SBP operators given in
TODO

See [`subcell_operator`](@ref).
"""
struct GlaubitzLampertNordströmWinters2025 <: SourceOfCoefficients end

function Base.show(io::IO, source::GlaubitzLampertNordströmWinters2025)
    if get(io, :compact, false)
        summary(io, source)
    else
        print(io,
              "Glaubitz, Lampert, Nordström, Winters (2025) \n",
              "  TODO \n",
              "  TODO")
    end
end

# This function is extended in the package extension SummationByPartsOperatorsExtraOptimExt
"""
    subcell_operator(basis_functions, nodes, x_M, source;
                     derivative_order = 1, accuracy_order = 0,
                     bandwidth = length(nodes) - 1, size_boundary = 2 * bandwidth,
                     different_values = true, sparsity_pattern = nothing,
                     opt_alg = Optim.LBFGS(), options = Optim.Options(g_tol = 1e-14, iterations = 10000),
                     autodiff = :forward, x0 = nothing, verbose = false)

Construct a sub-cell operator in a function space spanned by the `basis_functions`, which is an
iterable of functions.

TODO

See also [`GlaubitzLampertNordströmWinters2025`](@ref).

!!! compat "Julia 1.9"
    This function requires at least Julia 1.9.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function subcell_operator end
