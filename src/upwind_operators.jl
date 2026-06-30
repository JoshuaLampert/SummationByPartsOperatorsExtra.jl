"""
    GlaubitzEtAl2025()

Sub-cell SBP operators given in
- Glaubitz, Ranocha, Winters, Schlottke-Lakemper, Öffner, Gassner (2025):
  Generalized upwind summation-by-parts operators and their
  application to nodal discontinuous Galerkin methods.
  Journal of Computational Physics 529 p. 113841.
  [DOI: 10.1016/j.jcp.2025.113841](https://doi.org/10.1016/j.jcp.2025.113841)

See [`upwind_operator`s](@ref).
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

function SummationByPartsOperators.upwind_operators(D, sigma, source::GlaubitzEtAl2025)
    xmin = SummationByPartsOperators.xmin(D)
    xmax = SummationByPartsOperators.xmax(D)
    nodes = grid(D)
    N = length(nodes)
    @assert length(sigma) < N "length(sigma) = $(length(sigma)) must be less than N = $N"
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
                            # D,
                            MatrixDerivativeOperator(xmin, xmax, nodes, weights, Dc,
                                                     K, source),
                            MatrixDerivativeOperator(xmin, xmax, nodes, weights, Dp,
                                                     accuracy_order, source))
    return D_upw
end
