function SummationByPartsOperatorsExtra.function_space_operator(basis_functions,
                                                                nodes::Vector{T},
                                                                source::SourceOfCoefficients;
                                                                derivative_order = 1,
                                                                accuracy_order = 0,
                                                                bandwidth = length(nodes) -
                                                                            1,
                                                                size_boundary = 2 *
                                                                                bandwidth,
                                                                different_values = true,
                                                                sparsity_pattern = nothing,
                                                                opt_alg = LBFGS(),
                                                                options = Options(g_tol = 1e-14,
                                                                                  iterations = 10000),
                                                                autodiff = :forward,
                                                                x0 = nothing,
                                                                verbose = false) where {T,
                                                                                        SourceOfCoefficients
                                                                                        }
    assert_first_derivative_order(derivative_order)
    sort!(nodes)
    weights, D = construct_function_space_operator(basis_functions, nodes, source;
                                                   bandwidth, size_boundary,
                                                   different_values, sparsity_pattern,
                                                   opt_alg, options, autodiff, x0, verbose)
    return MatrixDerivativeOperator(first(nodes), last(nodes), nodes, weights, D,
                                    accuracy_order, source)
end

function construct_function_space_operator(basis_functions, nodes,
                                           ::GlaubitzNordströmÖffner2023;
                                           bandwidth = length(nodes) - 1,
                                           size_boundary = 2 * bandwidth,
                                           different_values = true,
                                           sparsity_pattern = nothing,
                                           opt_alg = LBFGS(),
                                           options = Options(g_tol = 1e-14,
                                                             iterations = 10000),
                                           autodiff = :forward,
                                           x0 = nothing, verbose = false)
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
    V = vandermonde_matrix(basis_functions_orthonormalized, nodes)
    V_x = vandermonde_matrix(basis_functions_orthonormalized_derivatives, nodes)
    # Here, W satisfies W'*W = I
    # W = [V; -V_x]

    B = spzeros(T, N, N)
    B[1, 1] = -1
    B[N, N] = 1
    R = B * V / 2

    x_length = last(nodes) - first(nodes)
    S = zeros(T, N, N)
    SV = zeros(T, N, K)
    PV_x = zeros(T, N, K)
    A = zeros(T, N, K)
    S_cache = DiffCache(S)
    SV_cache = DiffCache(SV)
    PV_x_cache = DiffCache(PV_x)
    A_cache = DiffCache(A)
    p = (; L, x_length, V, V_x, R, S_cache, SV_cache, PV_x_cache, A_cache,
         bandwidth, size_boundary, different_values, sparsity_pattern)

    if isnothing(x0)
        x0 = [zeros(T, L); invsig.(1 / N * ones(T, N))]
    else
        n_total = L + N
        if length(x0) != n_total
            throw(ArgumentError("Initial guess to has be L + N = $n_total long"))
        end
    end

    f(x) = optimization_function_function_space_operator(x, p)
    result = optimize(f, x0, opt_alg, options; autodiff)
    verbose && show(stdout, "text/plain", result)

    x = minimizer(result)
    sigma, rho = split_x_function_space_operator(x, L)
    S = SummationByPartsOperatorsExtra.create_S(sigma, N, bandwidth, size_boundary,
                                                different_values, sparsity_pattern)
    P = create_P(rho, x_length)
    weights = diag(P)
    Q = S + B / 2
    D = inv(P) * Q
    return weights, D
end

@views function optimization_function_function_space_operator(x, p)
    (; L, x_length, V, V_x, R, S_cache, SV_cache, PV_x_cache, A_cache,
    bandwidth, size_boundary, different_values, sparsity_pattern) = p

    N = size(R, 1)
    sigma, rho = split_x_function_space_operator(x, L)

    S = get_tmp(S_cache, x)
    SV = get_tmp(SV_cache, x)
    PV_x = get_tmp(PV_x_cache, x)
    A = get_tmp(A_cache, x)
    set_S!(S, sigma, bandwidth, size_boundary, different_values, sparsity_pattern)
    mul!(SV, S, V)
    P = create_P(rho, x_length)
    mul!(PV_x, P, V_x)
    @. A = SV - PV_x + R
    return sum(abs2, A)
end

# x = [sigma; rho]
split_x_function_space_operator(x, L) = x[1:L], x[(L + 1):end]
