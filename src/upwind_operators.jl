"""
    GlaubitzRanochaWintersSchlottkeLakemperÖffnerGassner2025()

Upwind SBP operators given in
- Glaubitz, Ranocha, Winters, Schlottke-Lakemper, Öffner, Gassner (2025):
  Generalized upwind summation-by-parts operators and their
  application to nodal discontinuous Galerkin methods.
  Journal of Computational Physics 529 p. 113841.
  [DOI: 10.1016/j.jcp.2025.113841](https://doi.org/10.1016/j.jcp.2025.113841)

See [`upwind_operators`](@ref).
"""
struct GlaubitzRanochaWintersSchlottkeLakemperÖffnerGassner2025 <: SourceOfCoefficients end

function Base.show(io::IO,
                   source::GlaubitzRanochaWintersSchlottkeLakemperÖffnerGassner2025)
    if get(io, :compact, false)
        summary(io, source)
    else
        print(io,
              "Glaubitz, Ranocha, Winters, Schlottke-Lakemper, Öffner, Gassner (2025) \n",
              "  Generalized upwind summation-by-parts operators and their \n",
              "  application to nodal discontinuous Galerkin methods. \n",
              "  Journal of Computational Physics 529 p. 113841.")
    end
end

"""
    GlaubitzLampertMattssonNiemeläWinters2026DG()

Function space upwind SBP operators of DG type, i.e. with a dense dissipation matrix acting on
the whole unresolved complement of the function space, given in
- Glaubitz, Lampert, Mattsson, Niemelä, Winters (2026):
  Upwind summation-by-parts operators for general function spaces:
  Discontinuous Galerkin-type operators.
  [DOI: TODO](TODO)

See [`dissipation_matrix`](@ref) and [`upwind_operators`](@ref).
"""
struct GlaubitzLampertMattssonNiemeläWinters2026DG <: SourceOfCoefficients end

function Base.show(io::IO, source::GlaubitzLampertMattssonNiemeläWinters2026DG)
    if get(io, :compact, false)
        summary(io, source)
    else
        print(io,
              "Glaubitz, Lampert, Mattsson, Niemelä, Winters (2026) \n",
              "  Upwind summation-by-parts operators for general function spaces: \n",
              "  Discontinuous Galerkin-type operators. \n",
              "  TODO.")
    end
end

"""
    GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized()

Function space upwind SBP operators whose dissipation matrix is determined by minimizing the
error of the upwind operators on a set of test functions, given in
- Glaubitz, Lampert, Mattsson, Niemelä, Winters (2026):
  Upwind summation-by-parts operators for general function spaces.
  [DOI: TODO](TODO)

See [`upwind_operators`](@ref).

!!! warning "Degenerate construction"
    This construction is documented in the reference above as a *negative* result and is only
    kept here for reproducibility. It does not yield useful dissipation, see the discussion in
    the docstring of [`upwind_operators`](@ref).
"""
struct GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized <: SourceOfCoefficients end

function Base.show(io::IO,
                   source::GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized)
    if get(io, :compact, false)
        summary(io, source)
    else
        print(io,
              "Glaubitz, Lampert, Mattsson, Niemelä, Winters (2026) \n",
              "  Upwind summation-by-parts operators for general function spaces. \n",
              "  TODO.")
    end
end

# Vandermonde-like matrix of the `functions` after orthonormalizing them with respect to the
# discrete inner product induced by `nodes`. If `length(functions) == length(nodes)`, the
# result is an orthogonal matrix, i.e., the discrete orthonormal basis (DOB).
function orthonormal_vandermonde(functions, nodes)
    functions_orthonormalized = orthonormalize_gram_schmidt(functions, nodes)
    return vandermonde_matrix(functions_orthonormalized, nodes)
end

# Dissipation matrix `S = V * Diagonal(lambda) * V'` with `lambda = [0, ..., 0, sigma...]` for
# an orthogonal matrix `V` whose first `N - length(sigma)` columns span the nodal values of the
# function space the operators are exact for. The `Symmetric` wrapper makes `S` exactly
# symmetric, which the product of the three factors is not in floating point arithmetic.
function eigen_dissipation_matrix(V, sigma)
    N = size(V, 1)
    K = N - length(sigma)
    lambda = zeros(eltype(sigma), N)
    lambda[(K + 1):N] .= sigma
    return Symmetric(V * Diagonal(lambda) * V')
end

"""
    upwind_operators(D, S::AbstractMatrix, source; accuracy_order = accuracy_order(D))

Create [`UpwindOperators`](@extref SummationByPartsOperators.UpwindOperators) from a central
first-derivative SBP operator `D` and a dissipation matrix `S` as
```math
D^- = D - M^{-1} S / 2, \\qquad D^+ = D + M^{-1} S / 2,
```
where ``M`` is the mass matrix of `D`. The matrix `S` must be symmetric (pass, e.g., a
`Symmetric` matrix) and should be negative semi-definite with ``S \\mathbf{f} = \\mathbf{0}``
for all functions `f` in the function space `D` is exact for. In that case, `D^-` and `D^+`
are upwind SBP operators that are exact for the same function space.
Negative semi-definiteness is not verified since this is comparatively expensive.

The `accuracy_order` is stored in the resulting operators `D^-` and `D^+`. Note that it is
generally *not* the accuracy order of `D`: `D^\\pm` are exact for a function `f` if and only if
`D` is exact for `f` *and* ``S \\mathbf{f} = \\mathbf{0}``.

The central operator `D` is stored unchanged in the resulting
[`UpwindOperators`](@extref SummationByPartsOperators.UpwindOperators).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function upwind_operators(D::AbstractNonperiodicDerivativeOperator, S::AbstractMatrix,
                          source;
                          accuracy_order = SummationByPartsOperators.accuracy_order(D))
    nodes = grid(D)
    N = length(nodes)
    if size(S) != (N, N)
        throw(DimensionMismatch("size(S) = $(size(S)) does not match the number of nodes N = $N"))
    end
    if !issymmetric(S)
        throw(ArgumentError("the dissipation matrix `S` must be symmetric, consider wrapping it in `Symmetric`"))
    end
    x_min = SummationByPartsOperators.xmin(D)
    x_max = SummationByPartsOperators.xmax(D)
    P = mass_matrix(D)
    weights = diag(P)
    D_central = Matrix(D)
    # `M^{-1} S / 2`; the mass matrix is diagonal, so this solve is cheap
    dissipation = (P \ S) / 2
    D_minus = D_central - dissipation
    D_plus = D_central + dissipation
    return UpwindOperators(MatrixDerivativeOperator(x_min, x_max, nodes, weights, D_minus,
                                                    accuracy_order, source),
                           D,
                           MatrixDerivativeOperator(x_min, x_max, nodes, weights, D_plus,
                                                    accuracy_order, source))
end

"""
    upwind_operators(D, sigma, source::GlaubitzRanochaWintersSchlottkeLakemperÖffnerGassner2025)

Create upwind SBP operators with central derivative operator `D` and given negative eigenvalues `sigma`
of the dissipation matrix `S`. The dissipation matrix is constructed as `S = V * Diagonal(lambda) * V'`
with `lambda = [0, ..., 0, sigma...]` and `V` being the Vandermonde matrix of the orthonormalized monomials,
i.e. the number of zero eigenvalues is equal to `N - length(sigma)`, where `N` is the number of nodes in `D`.
The upwind operators are then given as
```math
D^- = D - M^{-1} S / 2, \\qquad D^+ = D + M^{-1} S / 2,
```
where ``M`` is the mass matrix.

The operators `D^-` and `D^+` are exact for a function `f` if and only if `D` is exact for `f`
*and* `S` annihilates `f`. By construction, `S` annihilates all polynomials of degree at most
`K - 1` with `K = N - length(sigma)`. Hence, whenever `accuracy_order(D)` is the degree up to
which `D` is exact for polynomials, the resulting operators are exact for polynomials of degree
at most `min(K - 1, accuracy_order(D))`, which is stored as their `accuracy_order`. This is the
case for the polynomial operators considered in the reference below, but not, e.g., for a
non-polynomial [`function_space_operator`](@ref), whose `accuracy_order` carries no information
about polynomial exactness. There, `min(K - 1, accuracy_order(D))` is still stored in `D^-` and
`D^+`, but their exactness is the one given by the equivalence above.

See also [`GlaubitzRanochaWintersSchlottkeLakemperÖffnerGassner2025`](@ref) for details.
"""
function upwind_operators(D::AbstractNonperiodicDerivativeOperator, sigma::AbstractVector,
                          source::GlaubitzRanochaWintersSchlottkeLakemperÖffnerGassner2025)
    nodes = grid(D)
    N = length(nodes)
    @argcheck length(sigma) < N
    K = N - length(sigma)

    functions = [x -> x^k for k in 0:(N - 1)]
    V = orthonormal_vandermonde(functions, nodes)
    S = eigen_dissipation_matrix(V, sigma)
    accuracy_order = min(K - 1, SummationByPartsOperators.accuracy_order(D))
    return upwind_operators(D, S, source; accuracy_order)
end

"""
    upwind_operators(D, basis_functions, additional_functions, test_functions,
                     source::GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized;
                     autodiff = Optim.ADTypes.AutoForwardDiff(),
                     sigma0 = nothing, verbose = false,
                     opt_alg = LBFGSB(), options = Optim.Options(g_tol = 1e-10, iterations = 10000))

Create upwind function space SBP operators with central derivative operator `D`, e.g. a [`function_space_operator`](@ref)
created by `D = function_space_operator(basis_functions, nodes, GlaubitzNordströmÖffner2023())`. The `basis_functions` is
a list of the basis functions for which `D` is exact, the `additional_functions` are the additional functions `g_j` used to
construct a basis of ``\\mathbf{R}^N``, i.e., the number of additional functions must be equal to `N - length(basis_functions)`,
where `N` is the number of nodes in `D`. A dissipation matrix `S = V * Diagonal(lambda) * V'` is constructed by choosing
the negative eigenvalues `sigma = lambda[(K + 1):N]` such that the error of the upwind operators ``D^\\pm`` on the
`test_functions` is minimized, i.e., by solving
```math
\\min_{\\sigma \\leq 0} \\sum_l \\| D^+ \\varphi_l - \\varphi_l' \\|_2^2 + \\| D^- \\varphi_l - \\varphi_l' \\|_2^2.
```

The optimization problem is solved using [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) with the optimization algorithm `opt_alg` and options `options`.
The `autodiff` method can be set with `autodiff` using the interface of [ADTypes.jl](https://github.com/SciML/ADTypes.jl).
The initial guess for the optimization problem can be set with `sigma0`, which is a vector of negative eigenvalues of the dissipation matrix `S`. If omitted,
a default initial guess of `-1.0` is used for all eigenvalues. The `verbose` flag can be set to `true` to print additional information about the optimization process.

See also [`GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized`](@ref) for details.

!!! warning "This construction is degenerate and only kept for reproducibility"
    The objective above cannot select a dissipation matrix. Using ``D^\\pm \\varphi_l = D \\varphi_l \\pm M^{-1} S \\varphi_l / 2``
    and expanding the squares, the ``\\pm`` cross terms cancel, so the objective equals
    ``2 \\sum_l \\| D \\varphi_l - \\varphi_l' \\|_2^2 + \\frac{1}{2} \\sum_l \\| M^{-1} S \\varphi_l \\|_2^2``, whose first term
    is independent of `sigma`. The objective is therefore monotonically non-increasing as `sigma` tends to zero: its
    minimizers are exactly those `sigma` with ``\\sigma_j (\\mathbf{v}_j^T \\boldsymbol{\\varphi}_l) = 0`` for all `j` and `l`.
    The optimizer thus drives every eigenvalue whose mode overlaps some test function to zero, while modes orthogonal to
    *all* test functions are invisible to the objective and simply retain their initial guess `sigma0`.
    Whatever dissipation remains is an artifact of the initialization.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function upwind_operators(D::AbstractNonperiodicDerivativeOperator, basis_functions,
                          additional_functions, test_functions,
                          source::GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized;
                          kwargs...)
    nodes = grid(D)
    N = length(nodes)
    K = length(basis_functions)
    @argcheck length(additional_functions)==N - K "length(additional_functions) = $(length(additional_functions)) must be equal to N - K = $(N - K)"

    V = orthonormal_vandermonde([basis_functions; additional_functions], nodes)
    sigma = compute_dissipation_eigenvalues(D, V, test_functions, K, source; kwargs...)
    S = eigen_dissipation_matrix(V, sigma)
    return upwind_operators(D, S, source)
end

# Placeholder for computing the negative eigenvalues of the dissipation matrix. This method is
# extended in backend-specific implementations.
function compute_dissipation_eigenvalues end

# Legendre polynomials of the lowest degrees whose nodal values are not already contained in the
# span of the columns of `M`, mapped from the reference interval `[-1, 1]` to the interval spanned
# by the `nodes`. Degrees `0, ..., N - 1` always suffice since the nodal values of the polynomials
# of degree at most `N - 1` span all of `R^N`.
function default_enrichment_functions(M, nodes, n_enrichment, rtol)
    N = length(nodes)
    x_min, x_max = extrema(nodes)
    to_reference(x) = (2 * x - (x_min + x_max)) / (x_max - x_min)
    enrichment_functions = Vector{Any}(undef, 0)
    for degree in 0:(N - 1)
        length(enrichment_functions) == n_enrichment && break
        g(x) = legendre(to_reference(x), degree)
        values = g.(nodes)
        # Skip the degrees whose nodal values are (numerically) already in the span of `M`
        norm(values - M * (M \ values)) > rtol * norm(values) || continue
        push!(enrichment_functions, g)
        M = hcat(M, values)
    end
    if length(enrichment_functions) != n_enrichment
        throw(ArgumentError("could not find $n_enrichment enrichment functions, the nodal values of the basis functions are likely (numerically) linearly dependent"))
    end
    return enrichment_functions
end

# Orthogonal matrix `V` whose first `K` columns span the nodal values of the `basis_functions` and
# whose remaining columns are an orthonormal complement built from the `enrichment_functions`.
# We use a Householder QR decomposition instead of the Gram-Schmidt process of the reference. Only
# the matrix `V` is needed here (in contrast to `orthonormalize_gram_schmidt`, we never need the
# orthonormalized functions themselves), and the QR decomposition is orthogonal to machine
# precision independently of the conditioning of the enriched basis. Both give the same `V` up to
# the signs of the columns, which do not affect `S = V * Diagonal(lambda) * V'`.
function enriched_orthonormal_vandermonde(basis_functions, enrichment_functions, nodes,
                                          rtol)
    V_basis = vandermonde_matrix(basis_functions, nodes)
    M = if isempty(enrichment_functions)
        V_basis
    else
        hcat(V_basis, vandermonde_matrix(enrichment_functions, nodes))
    end
    factorization = qr(M)
    R_diagonal = abs.(diag(factorization.R))
    if minimum(R_diagonal) <= rtol * maximum(R_diagonal)
        throw(ArgumentError("the nodal values of the basis and enrichment functions are (numerically) linearly dependent"))
    end
    return Matrix(factorization.Q)
end

# Vector of the `N - K` negative eigenvalues of the dissipation matrix from the `lambda` argument,
# which is either a scalar (flat choice) or a vector.
function dissipation_eigenvalues(lambda::Real, n_enrichment, ::Type{T}) where {T}
    return fill(convert(T, lambda), n_enrichment)
end

function dissipation_eigenvalues(lambda::AbstractVector, n_enrichment, ::Type{T}) where {T}
    if length(lambda) != n_enrichment
        throw(DimensionMismatch("length(lambda) = $(length(lambda)) must be equal to N - K = $n_enrichment"))
    end
    return convert(Vector{T}, lambda)
end

"""
    dissipation_matrix(basis_functions, nodes, source::GlaubitzLampertMattssonNiemeläWinters2026DG;
                       lambda, enrichment_functions = nothing, rtol = sqrt(eps(eltype(nodes))))

Construct a dissipation matrix `S` on the `nodes` for the function space spanned by the
`basis_functions` following the DG-type construction of the reference in
[`GlaubitzLampertMattssonNiemeläWinters2026DG`](@ref).

The matrix is given by `S = V * Diagonal([0, ..., 0, lambda...]) * V'`, where `V` is an orthogonal
matrix whose first `K = length(basis_functions)` columns span the nodal values of the
`basis_functions` and whose remaining `N - K` columns are an orthonormal complement obtained from
the `enrichment_functions` (`N = length(nodes)`). The resulting `S` is symmetric and negative
semi-definite, satisfies `S * f == 0` for all `f` in the function space, and `f' * S * f < 0`
otherwise, provided all entries of `lambda` are negative.

The `lambda` keyword argument holds the `N - K` eigenvalues of `S` that are not forced to be zero.
It is either a `Real`, which is used for all of them (the flat choice used as the default in the
reference), or an `AbstractVector` of length `N - K`. All values must be non-positive.

The `enrichment_functions` are the functions `g_j` used to complete the basis. If `nothing` is
passed, the Legendre polynomials of the lowest degrees whose nodal values are not already in the
span of the `basis_functions` are used. The `rtol` is the relative tolerance used to detect such
linear dependence.

!!! note "The flat choice does not depend on the enrichment"
    For a flat `lambda = -c`, the construction simplifies to `S = -c * (I - P)` with `P` being the
    orthogonal projection onto the nodal values of the function space, so `S` is independent of the
    `enrichment_functions`. They only matter for a non-constant `lambda`.

See also [`upwind_operators`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function dissipation_matrix(basis_functions, nodes::AbstractVector,
                            source::GlaubitzLampertMattssonNiemeläWinters2026DG;
                            lambda, enrichment_functions = nothing,
                            rtol = sqrt(eps(eltype(nodes))))
    N = length(nodes)
    K = length(basis_functions)
    @argcheck K<=N "length(basis_functions) = $K must not be larger than the number of nodes N = $N"
    n_enrichment = N - K

    if isnothing(enrichment_functions)
        enrichment_functions = default_enrichment_functions(vandermonde_matrix(basis_functions,
                                                                               nodes),
                                                            nodes, n_enrichment, rtol)
    elseif length(enrichment_functions) != n_enrichment
        throw(DimensionMismatch("length(enrichment_functions) = $(length(enrichment_functions)) must be equal to N - K = $n_enrichment"))
    end

    sigma = dissipation_eigenvalues(lambda, n_enrichment, eltype(nodes))
    if !all(<=(0), sigma)
        throw(ArgumentError("all values of `lambda` must be non-positive, got $sigma"))
    end
    V = enriched_orthonormal_vandermonde(basis_functions, enrichment_functions, nodes, rtol)
    return eigen_dissipation_matrix(V, sigma)
end

"""
    upwind_operators(D, basis_functions, source::GlaubitzLampertMattssonNiemeläWinters2026DG;
                     lambda, enrichment_functions = nothing, rtol = sqrt(eps(eltype(grid(D)))))

Create upwind function space SBP operators of DG type with central derivative operator `D`, e.g. a
[`function_space_operator`](@ref) created by
`D = function_space_operator(basis_functions, nodes, GlaubitzNordströmÖffner2023())`, which is
exact for the function space spanned by the `basis_functions`. The dissipation matrix `S` is
constructed with [`dissipation_matrix`](@ref) on the grid of `D`, to which all keyword arguments
are forwarded, and the upwind operators are given as
```math
D^- = D - M^{-1} S / 2, \\qquad D^+ = D + M^{-1} S / 2,
```
where ``M`` is the mass matrix. Since `S` annihilates the nodal values of the function space `D`
is exact for, `D^-` and `D^+` are exact for the same function space.

See also [`GlaubitzLampertMattssonNiemeläWinters2026DG`](@ref) and [`dissipation_matrix`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function upwind_operators(D::AbstractNonperiodicDerivativeOperator,
                          basis_functions::AbstractVector,
                          source::GlaubitzLampertMattssonNiemeläWinters2026DG; kwargs...)
    S = dissipation_matrix(basis_functions, grid(D), source; kwargs...)
    return upwind_operators(D, S, source)
end
