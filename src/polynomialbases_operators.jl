"""
    PolynomialBasesDerivativeOperator{T<:Real, BasisType <: NodalBasis{PolynomialBases.Line}}

A derivative operator on a nonperiodic grid with scalar type
`T` based on a basis from PolynomialBases.jl.
"""
@auto_hash_equals struct PolynomialBasesDerivativeOperator{T <: Real,
                                                           BasisType <:
                                                           NodalBasis{PolynomialBases.Line}} <:
                         AbstractNonperiodicDerivativeOperator{T}
    jac::T
    Δx::T
    # We need to store `xmin` and `xmax` because in general the grid does not include boundary nodes.
    xmin::T
    xmax::T
    grid::Vector{T}
    basis::BasisType

    function PolynomialBasesDerivativeOperator(xmin::T, xmax::T,
                                               basis::BasisType) where {
                                                                        T <: Real,
                                                                        BasisType <:
                                                                        NodalBasis{PolynomialBases.Line}
                                                                        }
        grid = map_from_canonical.(basis.nodes, xmin, xmax, basis)
        jac = 2 / (xmax - xmin)
        Δx = inv(jac)

        new{T, BasisType}(jac, Δx, xmin, xmax, grid, basis)
    end
end

"""
    PolynomialBasesDerivativeOperator(basis_type, xmin::T, xmax::T, N::Int) where {T<:Real}

Construct the `PolynomialBasesDerivativeOperator` on a grid between `xmin` and
`xmax` using `N` nodes and `N-1` modes defined by `basis_type` from PolynomialBases.jl.
"""
function PolynomialBasesDerivativeOperator(basis_type, xmin::T, xmax::T,
                                           N::Int) where {T <: Real}
    @argcheck N >= 2

    basis = basis_type(N - 1, T)

    PolynomialBasesDerivativeOperator(xmin, xmax, basis)
end

"""
    polynomialbases_derivative_operator(basis_type, xmin::Real, xmax::Real, N::Integer)
    polynomialbases_derivative_operator(basis_type; xmin::Real, xmax::Real, N::Integer)

Construct the `PolynomialBasesDerivativeOperator` on a uniform grid between `xmin` and
`xmax` using `N` nodes and `N-1` Legendre modes.
"""
function polynomialbases_derivative_operator(basis_type, xmin::Real, xmax::Real, N::Integer)
    PolynomialBasesDerivativeOperator(basis_type, promote(xmin, xmax)..., N)
end

function polynomialbases_derivative_operator(basis_type; xmin::Real, xmax::Real, N::Integer)
    polynomialbases_derivative_operator(basis_type, xmin, xmax, N)
end

SummationByPartsOperators.derivative_order(D::PolynomialBasesDerivativeOperator) = 1
LinearAlgebra.issymmetric(D::PolynomialBasesDerivativeOperator) = false

function PolynomialBases.integrate(func, u, D::PolynomialBasesDerivativeOperator)
    D.Δx * integrate(func, u, D.basis)
end

function PolynomialBases.mass_matrix(D::PolynomialBasesDerivativeOperator)
    Diagonal(D.Δx * D.basis.weights)
end

function PolynomialBases.mass_matrix_boundary(D::PolynomialBasesDerivativeOperator)
    mass_matrix_boundary(D.basis)
end

function Base.eltype(::PolynomialBasesDerivativeOperator{T}) where {T}
    T
end

function SummationByPartsOperators.scale_by_mass_matrix!(u::AbstractVector,
                                                         D::PolynomialBasesDerivativeOperator,
                                                         factor = true)
    Base.require_one_based_indexing(u)
    @boundscheck begin
        length(u) == size(D, 2) ||
            throw(DimensionMismatch("sizes of input vector and operator do not match"))
    end
    @unpack Δx, basis = D

    @inbounds @simd for i in eachindex(u, basis.weights)
        u[i] = factor * u[i] * (Δx * basis.weights[i])
    end

    return u
end

function SummationByPartsOperators.scale_by_inverse_mass_matrix!(u::AbstractVector,
                                                                 D::PolynomialBasesDerivativeOperator,
                                                                 factor = true)
    Base.require_one_based_indexing(u)
    @boundscheck begin
        length(u) == size(D, 2) ||
            throw(DimensionMismatch("sizes of input vector and operator do not match"))
    end
    @unpack Δx, basis = D

    @inbounds @simd for i in eachindex(u, basis.weights)
        u[i] = factor * u[i] / (Δx * basis.weights[i])
    end

    u
end

function SummationByPartsOperators.get_weight(D::PolynomialBasesDerivativeOperator, i::Int)
    @unpack Δx, basis = D
    @unpack weights = basis
    N, _ = size(D)
    @boundscheck begin
        @argcheck 1 <= i <= N
    end
    @inbounds ω = Δx * weights[i]
    ω
end

function Base.show(io::IO, D::PolynomialBasesDerivativeOperator)
    if get(io, :compact, false)
        summary(io, D)
    else
        x = grid(D)
        print(io, "First derivative operator {T=", eltype(D), "}")
        print(io, " on ", length(x), " ", string(nameof(typeof(D.basis))), " nodes in [",
              D.xmin, ", ",
              D.xmax, "]")
    end
end

function mul!(dest::AbstractVector, D::PolynomialBasesDerivativeOperator, u::AbstractVector,
              α = true, β = false)
    @unpack jac, basis = D
    N, _ = size(D)
    @boundscheck begin
        @argcheck N == length(u)
        @argcheck N == length(dest)
    end

    mul!(dest, basis.D, u, α * jac, β)
end

function SummationByPartsOperators.lower_bandwidth(D::PolynomialBasesDerivativeOperator)
    size(D, 1) - 1
end

function SummationByPartsOperators.upper_bandwidth(D::PolynomialBasesDerivativeOperator)
    size(D, 1) - 1
end

function SummationByPartsOperators.accuracy_order(D::PolynomialBasesDerivativeOperator)
    size(D, 1) - 1
end

function SummationByPartsOperators.left_boundary_weight(D::PolynomialBasesDerivativeOperator)
    @inbounds retval = D.Δx * D.basis.weights[1]
    retval
end

function SummationByPartsOperators.right_boundary_weight(D::PolynomialBasesDerivativeOperator)
    @inbounds retval = D.Δx * D.basis.weights[end]
    retval
end

SummationByPartsOperators.xmin(D::PolynomialBasesDerivativeOperator) = D.xmin
SummationByPartsOperators.xmax(D::PolynomialBasesDerivativeOperator) = D.xmax
