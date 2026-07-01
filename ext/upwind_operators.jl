using SummationByPartsOperatorsExtra: grid, mass_matrix, orthonormalize_gram_schmidt
using LinearAlgebra: norm

function SummationByPartsOperatorsExtra.upwind_operators(D, basis_functions,
                                                         additional_functions,
                                                         test_functions,
                                                         source::GlaubitzEtAl2026;
                                                         kwargs...)
    xmin = SummationByPartsOperators.xmin(D)
    xmax = SummationByPartsOperators.xmax(D)
    nodes = SummationByPartsOperators.grid(D)
    N = length(nodes)
    K = length(basis_functions)
    @assert length(additional_functions)==N - K "length(additional_functions) = $(length(additional_functions)) must be equal to N - K = $N - $K"
    accuracy_order = K - 1 # TODO: This is for D being GLL, but for FD?

    P = mass_matrix(D)
    weights = diag(P)
    P_inv = inv(P)
    Dc = Matrix(D)

    functions = [basis_functions; additional_functions]
    functions_orthonormalized = orthonormalize_gram_schmidt(functions, nodes)
    V = vandermonde_matrix(functions_orthonormalized, nodes)
    sigma = compute_sigma(D, V, test_functions, K, source; kwargs...)
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

default_opt_alg(::GlaubitzEtAl2026) = LBFGSB()
function default_options(::GlaubitzEtAl2026, verbose)
    return Options(g_tol = 1e-10, iterations = 10000, show_trace = verbose)
end

function compute_sigma(D, V, test_functions, K, source;
                       autodiff = Optim.ADTypes.AutoForwardDiff(),
                       sigma0 = nothing, verbose = false,
                       opt_alg = default_opt_alg(source),
                       options = default_options(source, verbose))
    nodes = grid(D)
    N = length(nodes)
    NmK = N - K
    sigma0 = isnothing(sigma0) ? fill(real(D)(-1.0), NmK) : sigma0

    P = mass_matrix(D)
    P_inv = inv(P)
    Dc = Matrix(D)
    test_functions_derivatives = [x -> ForwardDiff.derivative(test_function, x)
                                  for test_function in test_functions]
    test_functions_values = [test_function.(nodes) for test_function in test_functions]
    test_functions_derivatives_values = [test_function_derivative.(nodes)
                                         for test_function_derivative in test_functions_derivatives]
    p = (; P_inv, Dc, V, test_functions_values, test_functions_derivatives_values)
    f(sigma) = optimization_function_upwind_function_space_operator(sigma, p)
    result = optimize(f, fill(-Inf, NmK), fill(0.0, NmK), sigma0,
                      opt_alg, options; autodiff)
    verbose && show(stdout, "text/plain", result)
    sigma = minimizer(result)
    return sigma
end

function optimization_function_upwind_function_space_operator(sigma, p)
    (; P_inv, Dc, V, test_functions_values, test_functions_derivatives_values) = p
    N = size(V, 1)
    K = N - length(sigma)
    @info ForwardDiff.value.(sigma)

    lambda = zeros(eltype(sigma), N)
    lambda[(K + 1):N] .= sigma
    S = V * Diagonal(lambda) * V'
    Dm = Dc - 0.5f0 * P_inv * S
    Dp = Dc + 0.5f0 * P_inv * S

    # Compute the residuals for the test functions
    residuals = zeros(eltype(sigma), length(test_functions_values))
    for (i, (test_function_values, test_function_derivative_values)) in enumerate(zip(test_functions_values,
                                                                                      test_functions_derivatives_values))
        residuals[i] = norm(Dp * test_function_values - test_function_derivative_values)^2 +
                       norm(Dm * test_function_values - test_function_derivative_values)^2
    end
    return sum(residuals)
end
