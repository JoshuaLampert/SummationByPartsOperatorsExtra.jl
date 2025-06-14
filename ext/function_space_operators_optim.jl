include("function_space_operators.jl")

default_opt_alg(::GlaubitzNordströmÖffner2023) = LBFGS()
function default_options(::GlaubitzNordströmÖffner2023, verbose)
    Options(g_tol = 1e-16, iterations = 10000, show_trace = verbose)
end

function construct_function_space_operator(basis_functions, nodes,
                                           source::GlaubitzNordströmÖffner2023;
                                           basis_functions_weights = ones(eltype(nodes),
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
    assert_correct_length_basis_functions_weights(basis_functions_weights, basis_functions)
    L = get_nsigma(N; bandwidth, size_boundary, different_values, sparsity_pattern)

    basis_functions_derivatives = [x -> ForwardDiff.derivative(basis_functions[i], x)
                                   for i in 1:K]
    basis_functions_orthonormalized, basis_functions_orthonormalized_derivatives = orthonormalize_gram_schmidt(basis_functions,
                                                                                                               basis_functions_derivatives,
                                                                                                               nodes)

    # This weights column k, i.e. basis function k, with the weight `basis_functions_weights[k]`
    V = vandermonde_matrix(basis_functions_orthonormalized, nodes) *
        Diagonal(basis_functions_weights)
    V_x = vandermonde_matrix(basis_functions_orthonormalized_derivatives, nodes) *
          Diagonal(basis_functions_weights)
    # Here, W satisfies W'*W = I
    # W = [V; -V_x]

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

    chunksize = ForwardDiff.pickchunksize(length(x0))
    S = zeros(T, N, N)
    SV = zeros(T, N, K)
    PV_x = zeros(T, N, K)
    A = zeros(T, N, K)
    S_cache = DiffCache(S, chunksize)
    SV_cache = DiffCache(SV, chunksize)
    PV_x_cache = DiffCache(PV_x, chunksize)
    A_cache = DiffCache(A, chunksize)
    p = (; L, x_length, V, V_x, R, S_cache, SV_cache, PV_x_cache, A_cache,
         bandwidth, size_boundary, different_values, sparsity_pattern)

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
