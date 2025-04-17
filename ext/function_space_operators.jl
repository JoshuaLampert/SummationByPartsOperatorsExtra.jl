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
    if derivative_order != 1
        throw(ArgumentError("Derivative order $derivative_order not implemented."))
    end
    if !isnothing(sparsity_pattern)
        if !(sparsity_pattern isa UpperTriangular || issymmetric(sparsity_pattern)) ||
           !all(diag(sparsity_pattern) .== 0)
            throw(ArgumentError("Sparsity pattern has to be symmetric with all diagonal entries being false or `UpperTriangular`."))
        end
        sparsity_pattern = UpperTriangular(sparsity_pattern)
    end
    if (length(nodes) < 2 * size_boundary + bandwidth || bandwidth < 1) &&
       (bandwidth != length(nodes) - 1)
        throw(ArgumentError("2 * size_boundary + bandwidth = $(2 * size_boundary + bandwidth) needs to be smaller than or equal to N = $(length(nodes)) and bandwidth = $bandwidth needs to be at least 1."))
    end
    sort!(nodes)
    weights, D = construct_function_space_operator(basis_functions, nodes, source;
                                                   bandwidth, size_boundary,
                                                   different_values, sparsity_pattern,
                                                   opt_alg, options, autodiff, x0, verbose)
    return MatrixDerivativeOperator(first(nodes), last(nodes), nodes, weights, D,
                                    accuracy_order, source)
end

function create_P(rho, vol)
    P = Diagonal(sig.(rho))
    P *= vol / sum(P)
    return P
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
        x0 = [zeros(T, L); invsig.(x_length / N * ones(T, N))]
    else
        @assert length(x0)==L + N "Initial guess to has be L + N = $(L + N) long"
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

    (N, _) = size(R)
    sigma, rho = split_x_function_space_operator(x, L)

    S = get_tmp(S_cache, x)
    SV = get_tmp(SV_cache, x)
    PV_x = get_tmp(PV_x_cache, x)
    A = get_tmp(A_cache, x)
    set_S!(S, sigma, N, bandwidth, size_boundary, different_values, sparsity_pattern)
    P = create_P(rho, x_length)
    mul!(SV, S, V)
    mul!(PV_x, P, V_x)
    @. A = SV - PV_x + R
    return sum(abs2, A)
end

# x = [sigma; rho]
split_x_function_space_operator(x, L) = x[1:L], x[(L + 1):end]
