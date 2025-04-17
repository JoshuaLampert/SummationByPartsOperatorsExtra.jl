function SummationByPartsOperatorsExtra.subcell_operator(basis_functions,
                                                         nodes::Vector{T},
                                                         x_M::T,
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
    x_L = first(nodes)
    x_R = last(nodes)
    if x_M < x_L || x_M > x_R
        throw(ArgumentError("x_M = $x_M needs to be in the interval [$x_L, $x_R]."))
    end
    weights_left, weights_right, Q_left, Q_right, B_left, B_right = construct_subcell_operator(basis_functions,
                                                                                               nodes,
                                                                                               x_M,
                                                                                               source;
                                                                                               bandwidth,
                                                                                               size_boundary,
                                                                                               different_values,
                                                                                               sparsity_pattern,
                                                                                               opt_alg,
                                                                                               options,
                                                                                               autodiff,
                                                                                               x0,
                                                                                               verbose)
    return SubcellOperator(nodes, x_M, weights_left, weights_right, Q_left, Q_right,
                           B_left, B_right, accuracy_order, source)
end

function create_P_left(rho, N, vol)
    N_L = length(rho)
    P_L = Diagonal(zeros(eltype(rho), N))
    for i in 1:N_L
        P_L[i, i] = sig(rho[i])
    end
    # P_L *= vol / sum(P_L)
    return P_L
end

function create_P_right(rho, N, vol)
    N_R = length(rho)
    P_R = Diagonal(zeros(eltype(rho), N))
    for i in (N - N_R + 1):N
        P_R[i, i] = sig(rho[i - (N - N_R)])
    end
    # P_R *= vol / sum(P_R)
    return P_R
end

function construct_subcell_operator(basis_functions, nodes, x_M,
                                    ::GlaubitzLampertNordströmWinters2025;
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
    N_L = findfirst(x -> x > x_M, nodes) - 1
    # If x_M is in the nodes, x_M will be part of both sub-cell, which means we have one more node
    N_R = x_M in nodes ? N - N_L + 1 : N - N_L
    L = get_nsigma(N; bandwidth, size_boundary, different_values, sparsity_pattern)

    basis_functions_derivatives = [x -> ForwardDiff.derivative(basis_functions[i], x)
                                   for i in 1:K]

    V = vandermonde_matrix(basis_functions, nodes)
    V_x = vandermonde_matrix(basis_functions_derivatives, nodes)

    # We assume that x_1 = x_L and x_N = x_R
    e_L = spzeros(T, N)
    e_L[1] = 1
    e_R = spzeros(T, N)
    e_R[end] = 1
    # e_L = ([basis_functions[i](first(nodes)) for i in 1:K]' / V)'
    # e_R = ([basis_functions[i](last(nodes)) for i in 1:K]' / V)'
    e_M = ([basis_functions[i](x_M) for i in 1:K]' / V)'
    # If x_M is in the nodes, we can also use
    # e_M = spzeros(T, N)
    # e_M[N_L] = 1
    E_L = [e_L e_M]
    E_R = [e_M e_R]

    B_P = [-one(T) zero(T)
           zero(T) one(T)]
    B_left = E_L * B_P * E_L'
    B_right = E_R * B_P * E_R'

    # Here, B is also just B_left + B_right
    B = spzeros(T, N, N)
    B[1, 1] = -1
    B[N, N] = 1
    R = B * V / 2

    x_length_left = x_M - first(nodes)
    x_length_right = last(nodes) - x_M
    S_L = zeros(T, N, N)
    S_R = zeros(T, N, N)
    S = zeros(T, N, N)
    SV = zeros(T, N, K)
    PV_x = zeros(T, N, K)
    A = zeros(T, N, K)
    P_LS_R = zeros(T, N, N)
    P_RS_L = zeros(T, N, N)
    P_LB_R = zeros(T, N, N)
    P_RB_L = zeros(T, N, N)
    C = zeros(T, N, N)
    S_L_cache = DiffCache(S_L)
    S_R_cache = DiffCache(S_R)
    S_cache = DiffCache(S)
    SV_cache = DiffCache(SV)
    PV_x_cache = DiffCache(PV_x)
    A_cache = DiffCache(A)
    P_LS_R_cache = DiffCache(P_LS_R)
    P_RS_L_cache = DiffCache(P_RS_L)
    P_LB_R_Cache = DiffCache(P_LB_R)
    P_RB_L_Cache = DiffCache(P_RB_L)
    C_cache = DiffCache(C)
    p = (; L, N_L, x_length_left, x_length_right, V, V_x, R, B_left, B_right,
         S_L_cache, S_R_cache, S_cache, SV_cache, PV_x_cache, A_cache,
         P_LS_R_cache, P_RS_L_cache, P_LB_R_Cache, P_RB_L_Cache, C_cache,
         bandwidth, size_boundary, different_values, sparsity_pattern)

    if isnothing(x0)
        x0 = [zeros(T, 2 * L); invsig.(x_length_left / N_L * ones(T, N_L)); invsig.(x_length_right / N_R * ones(T, N_R))]
    else
        n_x = 2 * L + N_L + N_R
        @assert length(x0)==n_x "Initial guess has to be 2L + N_L + N_R = $n_x long"
    end

    f(x) = optimization_function_subcell_operator(x, p)
    result = optimize(f, x0, opt_alg, options; autodiff)
    verbose && show(stdout, "text/plain", result)

    x = minimizer(result)
    sigma_L, sigma_R, rho_L, rho_R = split_x_subcell_operator(x, L, N_L)
    S_L = SummationByPartsOperatorsExtra.create_S(sigma_L, N, bandwidth, size_boundary,
                                                  different_values, sparsity_pattern)
    S_R = SummationByPartsOperatorsExtra.create_S(sigma_R, N, bandwidth, size_boundary,
                                                  different_values, sparsity_pattern)
    P_L = create_P_left(rho_L, N, x_length_left)
    P_R = create_P_right(rho_R, N, x_length_right)
    weights_left = diag(P_L)[1:N_L]
    weights_right = diag(P_R)[(N - N_R + 1):end]
    Q_left = S_L + B_left / 2
    Q_right = S_R + B_right / 2
    return weights_left, weights_right, Q_left, Q_right, B_left, B_right
end

@views function optimization_function_subcell_operator(x, p)
    (; L, N_L, x_length_left, x_length_right, V, V_x, R, B_left, B_right,
    S_L_cache, S_R_cache, S_cache, SV_cache, PV_x_cache, A_cache,
    P_LS_R_cache, P_RS_L_cache, P_LB_R_Cache, P_RB_L_Cache, C_cache,
    bandwidth, size_boundary, different_values, sparsity_pattern) = p

    (N, _) = size(R)
    sigma_L, sigma_R, rho_L, rho_R = split_x_subcell_operator(x, L, N_L)
    # first part (exactness of derivative operator)
    S_L = get_tmp(S_L_cache, x)
    S_R = get_tmp(S_R_cache, x)
    S = get_tmp(S_cache, x)
    SV = get_tmp(SV_cache, x)
    PV_x = get_tmp(PV_x_cache, x)
    A = get_tmp(A_cache, x)
    set_S!(S_L, sigma_L, N, bandwidth, size_boundary, different_values, sparsity_pattern)
    set_S!(S_R, sigma_R, N, bandwidth, size_boundary, different_values, sparsity_pattern)
    @. S = S_L + S_R
    mul!(SV, S, V)
    P_L = create_P_left(rho_L, N, x_length_left)
    P_R = create_P_right(rho_R, N, x_length_right)
    P = P_L + P_R
    mul!(PV_x, P, V_x)
    @. A = SV - PV_x + R

    # second part (P_{L/R}D = Q_{L/R})
    P_LS_R = get_tmp(P_LS_R_cache, x)
    P_RS_L = get_tmp(P_RS_L_cache, x)
    P_LB_R = get_tmp(P_LB_R_Cache, x)
    P_RB_L = get_tmp(P_RB_L_Cache, x)
    C = get_tmp(C_cache, x)
    mul!(P_LS_R, P_L, S_R)
    mul!(P_RS_L, P_R, S_L)
    mul!(P_LB_R, P_L, B_right)
    mul!(P_RB_L, P_R, B_left)
    @. C = P_LS_R - P_RS_L + 0.5 * (P_LB_R - P_RB_L)
    return sum(abs2, A) + sum(abs2, C)
end

# x = [sigma_L; sigma_R; rho_L; rho_R]
function split_x_subcell_operator(x, L, N_L)
    return x[1:L], x[(L + 1):(2 * L)], x[(2 * L + 1):(2 * L + N_L)],
           x[(2 * L + N_L + 1):end]
end
