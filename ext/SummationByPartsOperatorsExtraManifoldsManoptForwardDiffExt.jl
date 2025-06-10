module SummationByPartsOperatorsExtraManifoldsManoptForwardDiffExt

using Manifolds: Manifolds, SkewSymmetricMatrices, PositiveVectors, ProductManifold,
                 check_point
using Manopt: quasi_Newton, StopAfterIteration, StopWhenGradientNormLess, StopWhenCostLess
import RecursiveArrayTools: ArrayPartition
import ForwardDiff
import SummationByPartsOperatorsExtra: construct_function_space_operator,
                                       default_opt_alg,
                                       default_options
using SummationByPartsOperatorsExtra: SummationByPartsOperatorsExtra,
                                      GlaubitzIskeLampertÖffner2025,
                                      MatrixDerivativeOperator

include("utils.jl")
include("function_space_operators.jl")

default_opt_alg(::GlaubitzIskeLampertÖffner2025) = quasi_Newton
function default_options(::GlaubitzIskeLampertÖffner2025, verbose)
    if verbose
        debug = [:Iteration,
            :Time,
            " | ",
            (:Cost, "f(x): %.6e"),
            " | ",
            (:GradientNorm, "∇ f(x): %.6e"),
            "\n",
            :Stop]
    else
        debug = []
    end
    return (;
            debug = debug,
            stopping_criterion = StopAfterIteration(10000) |
                                 StopWhenGradientNormLess(1e-16) |
                                 StopWhenCostLess(1e-28))
end

function construct_function_space_operator(basis_functions, nodes,
                                           source::GlaubitzIskeLampertÖffner2025;
                                           basis_function_weights = ones(eltype(nodes),
                                                                         length(basis_functions)),
                                           bandwidth = length(nodes) - 1,
                                           size_boundary = 2 * bandwidth,
                                           different_values = true,
                                           sparsity_pattern = nothing,
                                           autodiff = :forward,
                                           x0 = nothing, verbose = false,
                                           opt_alg = default_opt_alg(source),
                                           options = default_options(source, verbose))
    T = eltype(nodes)
    K = length(basis_functions)
    N = length(nodes)

    assert_correct_bandwidth(N, bandwidth, size_boundary)
    if !isnothing(sparsity_pattern)
        assert_correct_sparsity_pattern(sparsity_pattern)
        sparsity_pattern = UpperTriangular(sparsity_pattern)
    end
    L = get_nsigma(N; bandwidth, size_boundary, different_values, sparsity_pattern)

    basis_functions_derivatives = [x -> ForwardDiff.derivative(basis_functions[i], x)
                                   for i in 1:K]
    basis_functions_orthonormalized, basis_functions_orthonormalized_derivatives = orthonormalize_gram_schmidt(basis_functions,
                                                                                                               basis_functions_derivatives,
                                                                                                               nodes)
    # This weights column k, i.e. basis function k, with the weight `basis_function_weights[k]`
    V = vandermonde_matrix(basis_functions_orthonormalized, nodes) *
        Diagonal(basis_function_weights)
    V_x = vandermonde_matrix(basis_functions_orthonormalized_derivatives, nodes) *
          Diagonal(basis_function_weights)
    B = spzeros(T, N, N)
    B[1, 1] = -1
    B[N, N] = 1
    R = B * V / 2
    x_length = last(nodes) - first(nodes)

    if isnothing(x0)
        x0 = [zeros(T, L); invsig.(1 / N * ones(T, N))]
    else
        n_total = L + N
        if length(x0) != n_total
            throw(ArgumentError("Initial guess to has be L + N = $n_total long"))
        end
    end

    sigma0, rho0 = split_x_function_space_operator(x0, L)

    M1 = SkewSymmetricMatrices(N)
    M2 = PositiveVectors(N)
    M = ProductManifold(M1, M2)
    S0 = SummationByPartsOperatorsExtra.create_S(sigma0, N, bandwidth, size_boundary,
                                                 different_values, sparsity_pattern)
    p0 = diag(create_P(rho0, x_length))
    x0 = ArrayPartition(S0, p0)

    param = (; V, V_x, R)
    f(M, x) = optimization_function_function_space_operator(M, x, param)

    function optimization_gradient_function_space_operator(M, x, autodiff)
        b = Manifolds.TangentDiffBackend(autodiff)
        return Manifolds.gradient(M, x -> f(M, x), x, b)
    end
    if autodiff == :forward
        autodiff = Manifolds.ManifoldDiff.AutoForwardDiff()
    end
    grad_f(M, x) = optimization_gradient_function_space_operator(M, x, autodiff)
    x = opt_alg(M, f, grad_f, x0; options...)

    S, p = x.x
    weights = p
    Q = S + B / 2
    D = inv(Diagonal(p)) * Q
    return weights, D
end

function optimization_function_function_space_operator(M, x, param)
    (; V, V_x, R) = param
    S, p = x.x

    A = S * V - Diagonal(p) * V_x + R
    return sum(abs2, A)
end

end
