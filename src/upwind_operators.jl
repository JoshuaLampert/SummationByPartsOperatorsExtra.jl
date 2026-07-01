"""
    GlaubitzEtAl2025()

Upwind SBP operators given in
- Glaubitz, Ranocha, Winters, Schlottke-Lakemper, Öffner, Gassner (2025):
  Generalized upwind summation-by-parts operators and their
  application to nodal discontinuous Galerkin methods.
  Journal of Computational Physics 529 p. 113841.
  [DOI: 10.1016/j.jcp.2025.113841](https://doi.org/10.1016/j.jcp.2025.113841)

See [`upwind_operators`](@ref).
"""
struct GlaubitzEtAl2025 <: SourceOfCoefficients end

function Base.show(io::IO, source::GlaubitzEtAl2025)
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
    upwind_operators(D, sigma, source::GlaubitzEtAl2025)

Create upwind SBP operators with central derivative operator `D` and given negative eigenvalues `sigma`
of the dissipation matrix `S`. The dissipation matrix is constructed as `S = V * Diagonal(lambda) * V'`
with `lambda = [0, ..., 0, sigma]` and `V` being the Vandermonde matrix of the orthonormalized basis functions,
i.e. the number of zero eigenvalues is equal to `N - length(sigma)`, where `N` is the number of nodes in `D`.
The upwind operators are then given as
```math
D^- = D - 0.5 * M^{-1} * S
D^+ = D + 0.5 * M^{-1} * S
```
where `M` is the mass matrix.

See also [`GlaubitzEtAl2025`](@ref) for details.
"""
function upwind_operators(D, sigma, source::GlaubitzEtAl2025)
    xmin = SummationByPartsOperators.xmin(D)
    xmax = SummationByPartsOperators.xmax(D)
    nodes = grid(D)
    N = length(nodes)
    @assert length(sigma)<N "length(sigma) = $(length(sigma)) must be less than N = $N"
    K = N - length(sigma)
    accuracy_order = K - 1 # TODO: This is for D being GLL, but for FD?

    P = mass_matrix(D)
    weights = diag(P)
    P_inv = inv(P)
    Dc = Matrix(D)

    functions = [x -> x^k for k in 0:(N - 1)]
    functions_orthonormalized = orthonormalize_gram_schmidt(functions, nodes)
    V = vandermonde_matrix(functions_orthonormalized, nodes)
    lambda = zeros(real(D), N)
    lambda[(K + 1):N] .= sigma
    S = V * Diagonal(lambda) * V'
    Dm = Dc - 0.5f0 * P_inv * S
    Dp = Dc + 0.5f0 * P_inv * S
    D_upw = UpwindOperators(MatrixDerivativeOperator(xmin, xmax, nodes, weights, Dm,
                                                     accuracy_order, source),
                            D,
                            MatrixDerivativeOperator(xmin, xmax, nodes, weights, Dp,
                                                     accuracy_order, source))
    return D_upw
end

"""
    GlaubitzEtAl2026()

Function space upwind SBP operators given in
- Glaubitz, Lampert, Mattsson, Niemelä, Winters (2026):
  Upwind summation-by-parts operators for general function spaces:
  Discontinuous Galerkin-type operators.
  [DOI: TODO](TODO)

See [`upwind_operators`](@ref).
"""
struct GlaubitzEtAl2026 <: SourceOfCoefficients end

function Base.show(io::IO, source::GlaubitzEtAl2026)
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
    upwind_operators(D, basis_functions, additional_functions, test_functions,
                     source::GlaubitzEtAl2026; autodiff = Optim.ADTypes.AutoForwardDiff(),
                     sigma0 = nothing, verbose = false,
                     opt_alg = LBFGSB(), options = Optim.Options(g_tol = 1e-10, iterations = 10000)))

Create upwind function space SBP operators with central derivative operator `D`, e.g. a [`function_space_operator`](@ref)
created by `D = function_space_operators(basis_functions, nodes, GlaubitzNordströmÖffner2023())`. The `basis_functions` is
a list of the basis functions for which `D` is exact, the `additional_functions` are the additional functions `g_j` used to
construct a basis of ``\\mathbf{R}^N``, i.e., the number of additional functions must be equal to `N - length(basis_functions)`,
where `N` is the number of nodes in `D`. A dissipation matrix `S` is constructed by solving a constrained optimization problem such
that the upwind operators ``D^-`` and ``D^+`` satisfy the upwind SBP property, are exact for the `basis_functions`, and dissipative for
`additional_functions`. They are constructed, such that the error of the upwind operators for the `test_functions` is minimized.

The optimization problem is solved using [`Optim.jl`](https://github.com/JuliaNLSolvers/Optim.jl) with the optimization algorithm `opt_alg` and options `options`.
The `autodiff` method can be set with `autodiff` using the interface of [ADTypes.jl](https://github.com/SciML/ADTypes.jl).
The initial guess for the optimization problem can be set with `sigma0`, which is a vector of negative eigenvalues of the dissipation matrix `S`. If omitted,
a default initial guess of `-1.0` is used for all eigenvalues. The `verbose` flag can be set to `true` to print additional information about the optimization process.

The upwind operators are then given as
```math
D^- = D - 0.5 * M^{-1} * S
D^+ = D + 0.5 * M^{-1} * S
```
where `M` is the mass matrix.

See also [`GlaubitzEtAl2026`](@ref) for details.

!!! warning "Experimental implementation"
    This is an experimental feature and may change in future releases.
"""
function upwind_operators end
