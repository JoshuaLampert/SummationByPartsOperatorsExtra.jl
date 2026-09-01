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
  Upwind summation-by-parts operators for general function spaces.
  [DOI: TODO](TODO)

See [`dissipation_matrix`](@ref), [`upwind_operators`](@ref),
and [`StiffnessBudget`](@ref).
"""
struct GlaubitzLampertMattssonNiemeläWinters2026DG <: SourceOfCoefficients end

function Base.show(io::IO, source::GlaubitzLampertMattssonNiemeläWinters2026DG)
    if get(io, :compact, false)
        summary(io, source)
    else
        print(io,
              "Glaubitz, Lampert, Mattsson, Niemelä, Winters (2026) \n",
              "  Upwind summation-by-parts operators for general function spaces. \n",
              "  TODO.")
    end
end

"""
    GlaubitzLampertMattssonNiemeläWinters2026FD()

Function space upwind SBP operators of FD type, i.e. with a banded dissipation matrix built from
generalized divided differences, which acts locally instead of on the whole unresolved complement
of the function space, given in
- Glaubitz, Lampert, Mattsson, Niemelä, Winters (2026):
  Upwind summation-by-parts operators for general function spaces.
  [DOI: TODO](TODO)

See [`dissipation_matrix`](@ref), [`upwind_operators`](@ref), [`annihilation_matrix`](@ref),
and [`StiffnessBudget`](@ref).
"""
struct GlaubitzLampertMattssonNiemeläWinters2026FD <: SourceOfCoefficients end

function Base.show(io::IO, source::GlaubitzLampertMattssonNiemeläWinters2026FD)
    if get(io, :compact, false)
        summary(io, source)
    else
        print(io,
              "Glaubitz, Lampert, Mattsson, Niemelä, Winters (2026) \n",
              "  Upwind summation-by-parts operators for general function spaces. \n",
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

# The dissipation matrix is negative semi-definite if and only if its non-zero eigenvalues are
# non-positive.
function check_dissipation_eigenvalues(sigma)
    if !all(<=(0), sigma)
        throw(ArgumentError("the eigenvalues of the dissipation matrix must be non-positive, got $sigma"))
    end
    return nothing
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
    # `MatrixDerivativeOperator` needs a `Vector` of nodes, while `grid` may return a range, e.g.
    # for the classical FD-SBP operators of SummationByPartsOperators.jl
    nodes = collect(grid(D))
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
of the dissipation matrix `S`. The dissipation matrix is constructed with
[`dissipation_matrix`](@ref), i.e. `S = V * Diagonal([0, ..., 0, sigma...]) * V'` with `V` an orthogonal matrix
whose first `K = N - length(sigma)` columns span the nodal values of the polynomials of degree at most `K - 1`,
where `N` is the number of nodes in `D`.
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
                          source::GlaubitzRanochaWintersSchlottkeLakemperÖffnerGassner2025;
                          kwargs...)
    S = dissipation_matrix(sigma, D, source; kwargs...)
    K = length(grid(D)) - length(sigma)
    accuracy_order = min(K - 1, SummationByPartsOperators.accuracy_order(D))
    return upwind_operators(D, S, source; accuracy_order)
end

"""
    dissipation_matrix(sigma, D, source::GlaubitzRanochaWintersSchlottkeLakemperÖffnerGassner2025;
                       rtol = sqrt(eps(eltype(grid(D)))))

Construct the dissipation matrix `S = V * Diagonal([0, ..., 0, sigma...]) * V'` on the grid of the
central derivative operator `D` for the given negative eigenvalues `sigma`, following the reference
in [`GlaubitzRanochaWintersSchlottkeLakemperÖffnerGassner2025`](@ref).

In contrast to the other sources, the function space is not passed but derived from the number of
eigenvalues: `V` is an orthogonal matrix whose first `K = N - length(sigma)` columns span the nodal
values of the polynomials of degree at most `K - 1`, where `N` is the number of nodes of `D`. Only
the grid of `D` is used. All values of `sigma` must be non-positive. The `rtol` is the relative
tolerance used to detect linear dependence of the nodal values.

See also [`upwind_operators`](@ref).
"""
function dissipation_matrix(sigma::AbstractVector,
                            D::AbstractNonperiodicDerivativeOperator,
                            source::GlaubitzRanochaWintersSchlottkeLakemperÖffnerGassner2025;
                            rtol = sqrt(eps(eltype(grid(D)))))
    nodes = grid(D)
    N = length(nodes)
    @argcheck length(sigma)<N "length(sigma) = $(length(sigma)) must be less than N = $N"
    K = N - length(sigma)

    # The function space is the space of polynomials of degree at most `K - 1`. We represent it by
    # Legendre polynomials rather than monomials since they span the same space, but are much
    # better conditioned. The orthonormal complement is the default (Legendre) enrichment.
    basis_functions = [reference_legendre(nodes, degree) for degree in 0:(K - 1)]
    enriched_basis = enriched_orthonormal_vandermonde(basis_functions, nodes, nothing, rtol)
    check_dissipation_eigenvalues(sigma)
    return eigen_dissipation_matrix(enriched_basis.V, sigma)
end

"""
    dissipation_matrix(basis_functions, D,
                       source::GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized;
                       test_functions, enrichment_functions = nothing,
                       rtol = sqrt(eps(eltype(grid(D)))),
                       autodiff = Optim.ADTypes.AutoForwardDiff(),
                       sigma0 = nothing, verbose = false,
                       opt_alg = LBFGSB(), options = Optim.Options(g_tol = 1e-10, iterations = 10000))

Construct a dissipation matrix on the grid of the central derivative operator `D`, e.g. a
[`function_space_operator`](@ref) created by
`D = function_space_operator(basis_functions, nodes, GlaubitzNordströmÖffner2023())`. The `basis_functions` is
a list of the basis functions for which `D` is exact and the `enrichment_functions` are the functions `g_j` completing them to
a basis of ``\\mathbf{R}^N``; see [`dissipation_matrix`](@ref) for how they default to the lowest-degree Legendre polynomials
not already contained in the function space. A dissipation matrix `S = V * Diagonal(lambda) * V'` is constructed by choosing
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
function dissipation_matrix(basis_functions, D::AbstractNonperiodicDerivativeOperator,
                            source::GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized;
                            test_functions, enrichment_functions = nothing,
                            rtol = sqrt(eps(eltype(grid(D)))), kwargs...)
    nodes = grid(D)
    enriched_basis = enriched_orthonormal_vandermonde(basis_functions, nodes,
                                                      enrichment_functions, rtol)
    sigma = dissipation_eigenvalues(test_functions, D, enriched_basis,
                                    length(basis_functions), source; kwargs...)
    check_dissipation_eigenvalues(sigma)
    return eigen_dissipation_matrix(enriched_basis.V, sigma)
end

# Legendre polynomials of the lowest degrees whose nodal values are not already contained in the
# span of the columns of `M`, mapped from the reference interval `[-1, 1]` to the interval spanned
# by the `nodes`. Degrees `0, ..., N - 1` always suffice since the nodal values of the polynomials
# of degree at most `N - 1` span all of `R^N`.
# Legendre polynomial of the given `degree`, mapped from the reference interval `[-1, 1]` to the
# interval spanned by the `nodes`
function reference_legendre(nodes, degree)
    x_min, x_max = extrema(nodes)
    return x -> legendre((2 * x - (x_min + x_max)) / (x_max - x_min), degree)
end

# Derivative of `reference_legendre`, including the chain rule factor of the affine map. Having it
# in closed form means the default enrichment needs no automatic differentiation.
function reference_legendre_derivative(nodes, degree)
    x_min, x_max = extrema(nodes)
    scaling = 2 / (x_max - x_min)
    function derivative(x)
        _, derivative_value = legendre_and_derivative((2 * x - (x_min + x_max)) /
                                                      (x_max - x_min), degree)
        return scaling * derivative_value
    end
    return derivative
end

function default_enrichment_functions(M, nodes, n_enrichment, rtol)
    N = length(nodes)
    enrichment_functions = Vector{Any}(undef, 0)
    enrichment_functions_derivatives = Vector{Any}(undef, 0)
    for degree in 0:(N - 1)
        length(enrichment_functions) == n_enrichment && break
        g = reference_legendre(nodes, degree)
        values = g.(nodes)
        # Skip the degrees whose nodal values are (numerically) already in the span of `M`
        norm(values - M * (M \ values)) > rtol * norm(values) || continue
        push!(enrichment_functions, g)
        push!(enrichment_functions_derivatives,
              reference_legendre_derivative(nodes, degree))
        M = hcat(M, values)
    end
    if length(enrichment_functions) != n_enrichment
        throw(ArgumentError("could not find $n_enrichment enrichment functions, the nodal values of the basis functions are likely (numerically) linearly dependent"))
    end
    return enrichment_functions, enrichment_functions_derivatives
end

# Orthogonal matrix `V` whose first `K` columns span the nodal values of the `basis_functions` and
# whose remaining columns are an orthonormal complement built from the `enrichment_functions`.
# We use a Householder QR decomposition instead of the Gram-Schmidt process of the reference: it is
# orthogonal to machine precision independently of the conditioning of the enriched basis. Both
# give the same `V` up to the signs of the columns, which do not affect
# `S = V * Diagonal(lambda) * V'`.
function enriched_orthonormal_vandermonde(basis_functions, nodes, enrichment_functions,
                                          rtol)
    N = length(nodes)
    K = length(basis_functions)
    @argcheck K<=N "length(basis_functions) = $K must not be larger than the number of nodes N = $N"
    n_enrichment = N - K

    V_basis = vandermonde_matrix(basis_functions, nodes)
    enrichment_functions_derivatives = nothing
    if isnothing(enrichment_functions)
        enrichment_functions, enrichment_functions_derivatives = default_enrichment_functions(V_basis,
                                                                                              nodes,
                                                                                              n_enrichment,
                                                                                              rtol)
    elseif length(enrichment_functions) != n_enrichment
        throw(DimensionMismatch("length(enrichment_functions) = $(length(enrichment_functions)) must be equal to N - K = $n_enrichment"))
    end

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
    # `M = V * R`, so the function with nodal values `V[:, j]` is the combination of the enriched
    # basis given by the `j`-th column of `inv(R)`. Its nodal derivative values are therefore the
    # `j`-th column of `M_x / R` with `M_x` the Vandermonde matrix of the derivatives, cf.
    # `enriched_derivative_vandermonde`.
    return (; V = Matrix(factorization.Q), R = factorization.R, enrichment_functions,
            enrichment_functions_derivatives)
end

# Nodal derivative values of the orthonormalized enriched basis, see the comment above.
function enriched_derivative_vandermonde(basis_functions_derivatives,
                                         enrichment_functions_derivatives, nodes,
                                         enriched_basis)
    M_x = if isempty(enrichment_functions_derivatives)
        vandermonde_matrix(basis_functions_derivatives, nodes)
    else
        hcat(vandermonde_matrix(basis_functions_derivatives, nodes),
             vandermonde_matrix(enrichment_functions_derivatives, nodes))
    end
    return M_x / enriched_basis.R
end

"""
    FlatWeights()

Damp all unresolved modes at the same rate, i.e. use `lambda_j = -scale` for all of them. Pass it
to [`StiffnessBudget`](@ref).

By the remark on the flat choice in [`dissipation_matrix`](@ref), the resulting dissipation matrix
is a multiple of the orthogonal projection onto the unresolved complement and hence does not
depend on the enrichment functions at all. It therefore damps every unresolved direction alike and
cannot distinguish between unresolved modes; see [`ErrorWeights`](@ref) for a refinement.
"""
struct FlatWeights end

@doc raw"""
    ErrorWeights(; basis_functions_derivatives, enrichment_functions_derivatives = nothing)

Damp the unresolved modes proportionally to the error the central operator makes on them, i.e. use
`lambda_j = -scale * w_j` with `w_j = e_j / sum(e)` and
```math
e_j = \\| D \\mathbf{v}_j - \\mathbf{v}_j' \\|_P, \\qquad j = K+1, \\dots, N,
```
where ``\\mathbf{v}_j'`` collects the nodal derivative values of the function in the enriched space
with nodal values ``\\mathbf{v}_j``. Pass it to [`StiffnessBudget`](@ref).

Modes that `D` differentiates accurately receive weak damping, while the budget is concentrated on
the modes with the largest errors. The `e_j` are fixed data computed from the central operator, so
nothing is optimized here and the degeneracy of
[`GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized`](@ref) cannot occur.

In contrast to [`FlatWeights`](@ref), the result depends on the enrichment functions, both through
the space they span, which is what defines ``\\mathbf{v}_j'``, and through the orthonormal basis of
the unresolved complement they induce.

Computing the `e_j` requires the derivatives of the basis functions, which have to be passed as
`basis_functions_derivatives`. The derivatives of the enrichment functions are only needed if the
`enrichment_functions` are passed explicitly to [`dissipation_matrix`](@ref); for the default
Legendre enrichment they are known in closed form.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct ErrorWeights{Basis, Enrichment}
    basis_functions_derivatives::Basis
    enrichment_functions_derivatives::Enrichment
end

function ErrorWeights(; basis_functions_derivatives,
                      enrichment_functions_derivatives = nothing)
    return ErrorWeights(basis_functions_derivatives, enrichment_functions_derivatives)
end

"""
    StiffnessBudget(weights = FlatWeights(); tol = 1 // 5, num_elements = 2,
                    max_scale = 100, rtol = 1e-10)

Determine the scale of the dissipation matrix from a *stiffness budget*: the largest scale such
that the spectral radius of the semidiscretization grows by at most a factor `1 + tol` over the
one of the central operator. The eigenvalues are `lambda_j = -scale * w_j` with the `weights` `w_j`
given by [`FlatWeights`](@ref) or [`ErrorWeights`](@ref); their normalization is immaterial since
any rescaling is absorbed by the calibrated scale. Pass it as the `lambda` argument of [`dissipation_matrix`](@ref) or
[`upwind_operators`](@ref).

The semidiscretization is the periodic upwind FSBP-SAT discretization of the linear advection
equation with a positive wave speed and the global Lax-Friedrichs splitting, discretized on
`num_elements` elements. For that splitting only `D^-` and the upwind coupling to the left
neighbor remain, so the assembled matrix is
`-a * couple_discontinuously(D^-, mesh, Val(:minus))`.

The scale is found by bisection on `[0, max_scale]`, which assumes the spectral radius to grow
monotonically with it. The search stops once the bracket is smaller than `rtol` relative to its
upper end. An error is thrown if the budget is not binding below `max_scale`.

See also [`dissipation_matrix`](@ref), [`upwind_operators`](@ref),
and [`GlaubitzLampertMattssonNiemeläWinters2026DG`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
struct StiffnessBudget{Weights, T <: Real}
    weights::Weights
    tol::T
    num_elements::Int
    max_scale::T
    rtol::T
end

function StiffnessBudget(weights = FlatWeights(); tol = 1 // 5, num_elements = 2,
                         max_scale = 100, rtol = 1.0e-10)
    tol, max_scale, rtol = promote(float(tol), float(max_scale), float(rtol))
    return StiffnessBudget(weights, tol, num_elements, max_scale, rtol)
end

# Spectral radius of the periodic upwind FSBP-SAT semidiscretization of linear advection built
# from the pair `D +- P^{-1} S / 2`, see the docstring of `StiffnessBudget`. The wave speed and
# the element width only enter as a common factor, so we use the unit interval and `a = 1`.
# For a negative wave speed one would coup1e `D^+` with `Val(:plus)` instead; the two agree
# whenever grid and operator are symmetric under `x -> -x`, which need not hold in general.
function semidiscretization_spectral_radius(D, S, source, num_elements)
    T = eltype(D)
    D_upwind = upwind_operators(D, S, source)
    mesh = UniformPeriodicMesh1D(xmin = zero(T), xmax = one(T), Nx = num_elements)
    coupled = couple_discontinuously(D_upwind.minus, mesh, Val(:minus))
    return maximum(abs, eigvals(Matrix(coupled)))
end

# Largest `scale >= 0` with `rho(scale * S_shape) <= (1 + tol) * rho(0)`, cf. `StiffnessBudget`.
function calibrate_stiffness_budget(D, S_shape, source, budget::StiffnessBudget)
    (; tol, num_elements, max_scale, rtol) = budget
    rho(scale) = semidiscretization_spectral_radius(D, scale * S_shape, source,
                                                    num_elements)

    target = (1 + tol) * rho(zero(tol))
    if rho(max_scale) <= target
        throw(ArgumentError("the stiffness budget is not binding for any scale up to max_scale = $max_scale; increase `max_scale` or decrease `tol`"))
    end

    lower, upper = zero(max_scale), max_scale
    while upper - lower > rtol * upper
        middle = (lower + upper) / 2
        if rho(middle) <= target
            lower = middle
        else
            upper = middle
        end
    end
    return lower
end

# Non-negative weights `w_j` such that the eigenvalues are `-scale * w_j`
function dissipation_weights(::FlatWeights, D, enriched_basis, K, basis_functions)
    return ones(eltype(enriched_basis.V), size(enriched_basis.V, 1) - K)
end

function dissipation_weights(weights::ErrorWeights, D, enriched_basis, K, basis_functions)
    enrichment_functions_derivatives = if isnothing(weights.enrichment_functions_derivatives)
        enriched_basis.enrichment_functions_derivatives
    else
        weights.enrichment_functions_derivatives
    end
    if isnothing(enrichment_functions_derivatives)
        throw(ArgumentError("`ErrorWeights` needs the derivatives of the `enrichment_functions`; pass them as `enrichment_functions_derivatives` or use the default enrichment"))
    end
    if length(weights.basis_functions_derivatives) != length(basis_functions)
        throw(DimensionMismatch("length(basis_functions_derivatives) = $(length(weights.basis_functions_derivatives)) must be equal to length(basis_functions) = $(length(basis_functions))"))
    end

    V = enriched_basis.V
    V_x = enriched_derivative_vandermonde(weights.basis_functions_derivatives,
                                          enrichment_functions_derivatives, grid(D),
                                          enriched_basis)
    P = mass_matrix(D)
    D_matrix = Matrix(D)
    N = size(V, 1)
    errors = similar(V, N - K)
    for j in (K + 1):N
        residual = D_matrix * view(V, :, j) - view(V_x, :, j)
        errors[j - K] = sqrt(dot(residual, P, residual))
    end
    return errors / sum(errors)
end

"""
    dissipation_matrix(basis_functions, D, source::GlaubitzLampertMattssonNiemeläWinters2026DG;
                       lambda = StiffnessBudget(), enrichment_functions = nothing,
                       rtol = sqrt(eps(eltype(grid(D)))))

Construct a dissipation matrix `S` on the grid of the central derivative operator `D` for the
function space spanned by the `basis_functions` following the DG-type construction of the
reference in [`GlaubitzLampertMattssonNiemeläWinters2026DG`](@ref).

The matrix is given by `S = V * Diagonal([0, ..., 0, lambda...]) * V'`, where `V` is an orthogonal
matrix whose first `K = length(basis_functions)` columns span the nodal values of the
`basis_functions` and whose remaining `N - K` columns are an orthonormal complement obtained from
the `enrichment_functions` (`N` being the number of nodes of `D`). The resulting `S` is symmetric
and negative semi-definite, satisfies `S * f == 0` for all `f` in the function space, and
`f' * S * f < 0` otherwise, provided all entries of `lambda` are negative.

The `lambda` keyword argument determines the `N - K` eigenvalues of `S` that are not forced to be
zero. It is either a [`StiffnessBudget`](@ref), which calibrates them from the semidiscretization
built with `D` and is the choice proposed in the reference, a `Real`, which is used for all of
them (the flat choice), or an `AbstractVector` of length `N - K`. All values must be non-positive.
Note that `D` enters only through its grid unless `lambda` is a [`StiffnessBudget`](@ref).

The `enrichment_functions` are the functions `g_j` used to complete the basis. If `nothing` is
passed, the Legendre polynomials of the lowest degrees whose nodal values are not already in the
span of the `basis_functions` are used, as proposed in the reference. The `rtol` is the relative
tolerance used to detect such linear dependence.

!!! note "The flat choice does not depend on the enrichment"
    For a flat `lambda = -c`, the construction simplifies to `S = -c * (I - P)` with `P` being the
    orthogonal projection onto the nodal values of the function space, so `S` is independent of the
    `enrichment_functions`. They only matter for a non-constant `lambda`.

See also [`upwind_operators`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function dissipation_matrix(basis_functions, D::AbstractNonperiodicDerivativeOperator,
                            source::GlaubitzLampertMattssonNiemeläWinters2026DG;
                            lambda = StiffnessBudget(), enrichment_functions = nothing,
                            rtol = sqrt(eps(eltype(grid(D)))))
    nodes = grid(D)
    enriched_basis = enriched_orthonormal_vandermonde(basis_functions, nodes,
                                                      enrichment_functions, rtol)
    sigma = dissipation_eigenvalues(lambda, D, enriched_basis, length(basis_functions),
                                    source; basis_functions)
    check_dissipation_eigenvalues(sigma)
    return eigen_dissipation_matrix(enriched_basis.V, sigma)
end

# Vector of the `N - K` negative eigenvalues of the dissipation matrix from the `lambda` argument,
# which is either a scalar (flat choice) or a vector.
# The `N - K` eigenvalues of the dissipation matrix that are not forced to be zero. The first
# argument specifies how they are determined, which is why its meaning depends on the `source` -
# just as for `dissipation_matrix`. The method for
# `GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized` needs to solve an optimization
# problem and is therefore provided by a package extension.

# Fixed eigenvalues do not depend on the operator
function dissipation_eigenvalues(lambda::Real, D, enriched_basis, K,
                                 source::GlaubitzLampertMattssonNiemeläWinters2026DG;
                                 basis_functions = nothing)
    V = enriched_basis.V
    return fill(convert(eltype(V), lambda), size(V, 1) - K)
end

function dissipation_eigenvalues(lambda::AbstractVector, D, enriched_basis, K,
                                 source::GlaubitzLampertMattssonNiemeläWinters2026DG;
                                 basis_functions = nothing)
    V = enriched_basis.V
    n_enrichment = size(V, 1) - K
    if length(lambda) != n_enrichment
        throw(DimensionMismatch("length(lambda) = $(length(lambda)) must be equal to N - K = $n_enrichment"))
    end
    return convert(Vector{eltype(V)}, lambda)
end

function dissipation_eigenvalues(budget::StiffnessBudget, D, enriched_basis, K,
                                 source::GlaubitzLampertMattssonNiemeläWinters2026DG;
                                 basis_functions)
    V = enriched_basis.V
    weights = dissipation_weights(budget.weights, D, enriched_basis, K, basis_functions)
    S_shape = eigen_dissipation_matrix(V, -weights)
    scale = calibrate_stiffness_budget(D, S_shape, source, budget)
    return -convert(eltype(V), scale) * weights
end

@doc raw"""
    annihilation_matrix(basis_functions, nodes; normalization = :two_norm,
                        rtol = sqrt(eps(eltype(nodes))))

Banded matrix ``\widetilde{D} \in \mathbb{R}^{(N - K) \times N}`` of generalized divided
differences of the function space ``\mathcal{F}`` spanned by the `K` `basis_functions` on the `N`
`nodes`, following the reference in
[`GlaubitzLampertMattssonNiemeläWinters2026FD`](@ref). The `i`-th row is supported on the window
`i:(i + K)` of `K + 1` consecutive nodes and holds the coefficient vector
``\mathbf{d}^{(i)}``, unique up to scaling, with
```math
\sum_{j = 0}^{K} d^{(i)}_j f(x_{i + j}) = 0
\quad \text{for all } f \in \mathcal{F},
```
i.e. `annihilation_matrix(basis_functions, nodes) * f.(nodes) == 0` for all `f` in the function
space. It is used by [`dissipation_matrix`](@ref) to build the FD-type dissipation matrix.

The coefficients are computed as the left singular vector of the local Vandermonde matrix
associated with its vanishing singular value, which requires the null spaces of `N - K` matrices
of size `(K + 1) x K` and is therefore cheap. This assumes that every window stays
``\mathcal{F}``-unisolvent after removing any single one of its nodes, which holds in particular
whenever ``\mathcal{F}`` is an extended complete Chebyshev system. An `ArgumentError` is thrown if
a window violates it; `rtol` is the relative tolerance of that check.

The `normalization` fixes the remaining scaling of each row:
- `:two_norm` (default, as in the reference) scales it to `norm(d) == 1`.
- `:undivided` scales it to `d[end] == 1`. For ``\mathcal{F}`` the polynomials of degree at most
  `K - 1` on equidistant nodes, this reproduces the classical undivided differences of order `K`
  exactly, e.g. `[-1, 3, -3, 1]` for `K = 3`.
In both cases the sign is fixed by making the last entry positive. If the nodes are equidistant
and the function space is translation invariant, all windows are congruent, the two choices differ
by a single global factor that is absorbed by the scaling parameter of
[`dissipation_matrix`](@ref), and ``\widetilde{D}`` is a Toeplitz matrix. On a non-uniform grid the
choice weights the windows differently and does matter.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function annihilation_matrix(basis_functions, nodes; normalization = :two_norm,
                             rtol = sqrt(eps(eltype(nodes))))
    N = length(nodes)
    K = length(basis_functions)
    @argcheck K<N "length(basis_functions) = $K must be smaller than the number of nodes N = $N"
    V = vandermonde_matrix(basis_functions, nodes)
    D_tilde = zeros(eltype(V), N - K, N)
    for i in 1:(N - K)
        window = i:(i + K)
        D_tilde[i, window] .= divided_difference_coefficients(V[window, :], normalization,
                                                              rtol)
    end
    return D_tilde
end

# Coefficient vector spanning the null space of `V_window'`, normalized as described in
# `annihilation_matrix`.
function divided_difference_coefficients(V_window, normalization, rtol)
    K = size(V_window, 2)
    factorization = svd(V_window; full = true)
    singular_values = factorization.S
    if singular_values[K] <= rtol * singular_values[1]
        throw(ArgumentError("the nodal values of the basis functions are (numerically) linearly dependent on a window of $(K + 1) consecutive nodes"))
    end
    # The last left singular vector spans the orthogonal complement of the range of `V_window`,
    # which is the null space of `V_window'`. It has unit 2-norm.
    d = factorization.U[:, K + 1]
    # By the cofactor representation of `d`, no entry vanishes if and only if the window stays
    # unisolvent after removing any single node. Without it, the rows of the annihilation matrix
    # can be linearly dependent, leaving some unresolved mode undamped.
    if minimum(abs, d) <= rtol
        throw(ArgumentError("a window of $(K + 1) consecutive nodes is not unisolvent for the function space after removing one of its nodes"))
    end
    d = sign(d[end]) * d
    if normalization === :two_norm
        return d
    elseif normalization === :undivided
        return d / d[end]
    else
        throw(ArgumentError("unknown normalization $normalization, use `:two_norm` or `:undivided`"))
    end
end

@doc raw"""
    dissipation_matrix(basis_functions, D, source::GlaubitzLampertMattssonNiemeläWinters2026FD;
                       epsilon = StiffnessBudget(), C = I, normalization = :two_norm,
                       rtol = sqrt(eps(eltype(grid(D)))))

Construct a banded dissipation matrix `S` on the grid of the central derivative operator `D`, e.g.
a banded [`function_space_operator`](@ref), for the function space ``\mathcal{F}`` spanned by the
`basis_functions` following the FD-type construction of the reference in
[`GlaubitzLampertMattssonNiemeläWinters2026FD`](@ref),
```math
S = -\varepsilon \, \widetilde{D}^T C \widetilde{D},
```
with the annihilation matrix ``\widetilde{D}`` of [`annihilation_matrix`](@ref), a diagonal
positive definite weighting matrix `C`, and a single scaling parameter ``\varepsilon > 0``. The
resulting `S` is symmetric and negative semi-definite, satisfies `S * f == 0` for all `f` in the
function space, and `f' * S * f < 0` otherwise, provided the unisolvence assumption of
[`annihilation_matrix`](@ref) holds and ``\varepsilon > 0``.

In contrast to the DG-type construction for
[`GlaubitzLampertMattssonNiemeläWinters2026DG`](@ref), the dissipation acts *locally*: `S` is
banded with bandwidth at most `2K + 1`, and each row of ``\widetilde{D}`` behaves like
``h_\mathrm{loc}^K`` times a fixed differential operator with kernel ``\mathcal{F}`` on functions
that are smooth relative to the local node spacing. Hence ``P^{-1} S`` vanishes under refinement on
all smooth functions, not only on ``\mathcal{F}``, so that `D^-` and `D^+` stay consistent. This is
what makes the construction the appropriate one in the FD regime `N >> K`, where interfaces are
rare and dissipation in the volume cannot be substituted by dissipation at interfaces.

The `epsilon` keyword argument sets the dissipation strength. It is either a
[`StiffnessBudget`](@ref), which calibrates it from the semidiscretization built with `D` and is
the choice proposed in the reference, or a non-negative `Real`. Note that `D` enters only through
its grid unless `epsilon` is a [`StiffnessBudget`](@ref). It plays the role that the flat rate
`lambda` plays for the DG-type construction; per-mode weights have no analogue here, since the
window weighting `C` is the corresponding degree of freedom.

The reference uses `C = I` throughout, which is the default: on an equidistant grid with a
translation invariant function space all interior windows are congruent, so any weighting derived
from the local errors of the central operator is constant in the interior anyway.

See also [`upwind_operators`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function dissipation_matrix(basis_functions, D::AbstractNonperiodicDerivativeOperator,
                            source::GlaubitzLampertMattssonNiemeläWinters2026FD;
                            epsilon = StiffnessBudget(), C = I,
                            normalization = :two_norm,
                            rtol = sqrt(eps(eltype(grid(D)))))
    D_tilde = annihilation_matrix(basis_functions, grid(D); normalization, rtol)
    S_shape = Symmetric(-D_tilde' * (C * D_tilde))
    scale = dissipation_scale(epsilon, D, S_shape, source)
    return scale * S_shape
end

# The scaling parameter `epsilon` of the FD-type dissipation matrix. As for
# `dissipation_eigenvalues`, the first argument specifies how it is determined.
function dissipation_scale(epsilon::Real, D, S_shape,
                           source::GlaubitzLampertMattssonNiemeläWinters2026FD)
    if epsilon < 0
        throw(ArgumentError("the scaling parameter of the dissipation matrix must be non-negative, got $epsilon"))
    end
    return epsilon
end

function dissipation_scale(budget::StiffnessBudget, D, S_shape,
                           source::GlaubitzLampertMattssonNiemeläWinters2026FD)
    if !(budget.weights isa FlatWeights)
        throw(ArgumentError("the FD-type dissipation matrix has a single scaling parameter and no per-mode weights; use `StiffnessBudget()` and the weighting matrix `C` of `dissipation_matrix` instead"))
    end
    return calibrate_stiffness_budget(D, S_shape, source, budget)
end

"""
    upwind_operators(D, basis_functions, source; kwargs...)

Create upwind function space SBP operators with central derivative operator `D`, e.g. a
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

Which construction is used, and which keyword arguments are accepted, is determined by the
`source`; see the corresponding methods of [`dissipation_matrix`](@ref) for
[`GlaubitzLampertMattssonNiemeläWinters2026DG`](@ref),
[`GlaubitzLampertMattssonNiemeläWinters2026FD`](@ref), and
[`GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized`](@ref).

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function upwind_operators(D::AbstractNonperiodicDerivativeOperator,
                          basis_functions::AbstractVector, source; kwargs...)
    S = dissipation_matrix(basis_functions, D, source; kwargs...)
    return upwind_operators(D, S, source)
end
