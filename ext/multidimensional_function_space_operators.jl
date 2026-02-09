function SummationByPartsOperatorsExtra.multidimensional_function_space_operator(basis_functions,
                                                                                 nodes,
                                                                                 boundary_indices,
                                                                                 normals,
                                                                                 moments,
                                                                                 vol,
                                                                                 source::SourceOfCoefficients;
                                                                                 derivative_order = 1,
                                                                                 accuracy_order = 0,
                                                                                 kwargs...) where {SourceOfCoefficients}
    assert_first_derivative_order(derivative_order)
    weights, weights_boundary, Ds = construct_multidimensional_function_space_operator(basis_functions,
                                                                                       nodes,
                                                                                       boundary_indices,
                                                                                       normals,
                                                                                       moments,
                                                                                       vol,
                                                                                       source;
                                                                                       kwargs...)
    return MultidimensionalMatrixDerivativeOperator(nodes, boundary_indices, normals,
                                                    weights, weights_boundary, Ds,
                                                    accuracy_order, source)
end

function construct_multidimensional_function_space_operator(basis_functions, nodes,
                                                            boundary_indices, normals,
                                                            moments, vol,
                                                            ::GlaubitzIskeLampertÖffner2026Multidimensional;
                                                            basis_functions_weights = ones(typeof(basis_functions[1](nodes[1])),
                                                                                           length(basis_functions)),
                                                            bandwidth = length(nodes) - 1,
                                                            size_boundary = 2 * bandwidth,
                                                            different_values = true,
                                                            sparsity_patterns = nothing,
                                                            corners = nothing,
                                                            opt_alg = LBFGS(),
                                                            options = Options(g_tol = 1e-14,
                                                                              iterations = 10000),
                                                            autodiff = Optim.ADTypes.AutoForwardDiff(),
                                                            x0 = nothing, verbose = false)
    T = typeof(basis_functions[1](nodes[1]))
    d = length(first(nodes))
    K = length(basis_functions)
    N = length(nodes)
    N_boundary = length(boundary_indices)

    @assert length(normals)==N_boundary "You must provide normals for all boundary nodes (length(normals) = $(length(normals)), N_boundary = $N_boundary)."
    assert_correct_bandwidth(N, bandwidth, size_boundary)
    if !isnothing(sparsity_patterns)
        for sparsity_pattern in sparsity_patterns
            assert_correct_sparsity_pattern(sparsity_pattern)
        end
        sparsity_patterns = UpperTriangular.(sparsity_patterns)
    else
        d = length(first(nodes))
        sparsity_patterns = ntuple(_ -> nothing, d)
    end
    if isnothing(corners)
        corners = ntuple(_ -> T[], d)
    end
    assert_correct_length_basis_functions_weights(basis_functions_weights, basis_functions)
    Ls = ntuple(i -> get_nsigma(N; bandwidth, size_boundary, different_values,
                                sparsity_pattern = sparsity_patterns[i]), d)
    basis_functions_gradients = [x -> ForwardDiff.gradient(basis_functions[i], x)
                                 for i in 1:K]
    # TODO: Orthonormalize? What happens with moments? Need moments with respect to orthonormalized basis functions?
    # This weights column k, i.e. basis function k, with the weight `basis_functions_weights[k]`
    V = vandermonde_matrix(basis_functions, nodes) *
        Diagonal(basis_functions_weights)
    V_xis = ntuple(j -> vandermonde_matrix([x -> basis_functions_gradients[i](x)[j]
                                            for i in 1:K], nodes) *
                        Diagonal(basis_functions_weights), d)

    if isnothing(x0)
        # x0 = zeros(T, sum(Ls) + N + N_boundary)
        x0 = [zeros(T, sum(Ls)); invsig.(convert(T, 1 / N) * ones(T, N));
              invsig_b.(convert(T, 1 / N_boundary) * ones(T, N_boundary))]
    else
        n_total = sum(Ls) + N + N_boundary
        if length(x0) != n_total
            throw(ArgumentError("Initial guess has to be sum(Ls) + N + N_boundary = $n_total long"))
        end
    end

    chunksize = ForwardDiff.pickchunksize(length(x0))
    S = zeros(T, N, N)
    A = zeros(T, N, K)
    M = zeros(T, K, K)
    S_cache = DiffCache(S, chunksize)
    SV_cache = DiffCache(copy(A), chunksize)
    PV_xi_cache = DiffCache(copy(A), chunksize)
    B_cache = DiffCache(copy(S), chunksize)
    BV_cache = DiffCache(copy(A), chunksize)
    A_cache = DiffCache(A, chunksize)
    VTBV_cache = DiffCache(M, chunksize)
    C_cache = DiffCache(copy(M), chunksize)
    p = (; Ls, vol, normals, moments, boundary_indices, V, V_xis,
         S_cache, SV_cache, PV_xi_cache, B_cache, BV_cache, A_cache, VTBV_cache, C_cache,
         bandwidth, size_boundary, different_values, sparsity_patterns, corners)

    f(x) = optimization_function_multidimensional_function_space_operator(x, p)
    result = optimize(f, x0, opt_alg, options; autodiff)
    verbose && show(stdout, "text/plain", result)

    x = minimizer(result)
    sigmas, rho, phi = split_x_multidimensional_function_space_operator(x, Ls, d, N,
                                                                        N_boundary)
    P = create_P(rho, vol)
    weights = diag(P)
    weights_boundary = sig_b.(phi)
    function create_D(i)
        sigma = sigmas[i]
        S = SummationByPartsOperatorsExtra.create_S(sigma, N, bandwidth, size_boundary,
                                                    different_values, sparsity_patterns[i])
        B = create_B(N, weights_boundary, normals, boundary_indices, i; corners)
        Q = S + B / 2
        D = inv(P) * Q
        return D
    end
    Ds = ntuple(i -> create_D(i), d)
    return weights, weights_boundary, Ds
end

@views function optimization_function_multidimensional_function_space_operator(x, p)
    (; Ls, vol, normals, moments, boundary_indices, V, V_xis,
    S_cache, SV_cache, PV_xi_cache, B_cache, BV_cache, A_cache, VTBV_cache, C_cache,
    bandwidth, size_boundary, different_values, sparsity_patterns, corners) = p
    d = length(V_xis)
    N = size(V, 1)
    N_boundary = length(normals)
    sigmas, rho, phi = split_x_multidimensional_function_space_operator(x, Ls, d, N,
                                                                        N_boundary)

    S = get_tmp(S_cache, x)
    SV = get_tmp(SV_cache, x)
    PV_xi = get_tmp(PV_xi_cache, x)
    B = get_tmp(B_cache, x)
    BV = get_tmp(BV_cache, x)
    A = get_tmp(A_cache, x)
    VTBV = get_tmp(VTBV_cache, x)
    C = get_tmp(C_cache, x)

    P = create_P(rho, vol)
    res = 0.0
    for i in 1:d
        M = moments[i]
        V_xi = V_xis[i]
        sigma = sigmas[i]
        set_S!(S, sigma, bandwidth, size_boundary, different_values,
               sparsity_patterns[i])
        mul!(SV, S, V)
        mul!(PV_xi, P, V_xi)
        set_B!(B, phi, normals, boundary_indices, i; corners)
        mul!(BV, B, V)
        @. A = SV - PV_xi + 0.5 * BV
        mul!(VTBV, V', BV)
        @. C = VTBV - M
        res += sum(abs2, A) + sum(abs2, C)
    end
    return res
end

# x = [sigma_1; ...; sigma_d; rho; phi]
function split_x_multidimensional_function_space_operator(x, Ls, d, N, N_boundary)
    sigmas = ntuple(i -> x[(sum(Ls[1:(i - 1)], init = 0) + 1):sum(Ls[1:i])], d)
    rho = x[(end - N - N_boundary + 1):(end - N_boundary)]
    phi = x[(end - N_boundary + 1):end]
    return sigmas, rho, phi
end
