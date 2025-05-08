function SummationByPartsOperatorsExtra.subcell_operator(basis_functions,
                                                         nodes::Vector{T},
                                                         x_M::T,
                                                         source::SourceOfCoefficients;
                                                         derivative_order = 1,
                                                         accuracy_order = 0,
                                                         bandwidths = [0, 0],
                                                         size_boundaries = [0, 0],
                                                         different_values = [true, true],
                                                         sparsity_patterns = [nothing,
                                                             nothing],
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
    x_L = first(nodes)
    x_R = last(nodes)
    if x_M < x_L || x_M > x_R
        throw(ArgumentError("x_M = $x_M needs to be in the interval [$x_L, $x_R]."))
    end
    weights_left, weights_right, Q_left, Q_right, B_left, B_right = construct_subcell_operator(basis_functions,
                                                                                               nodes,
                                                                                               x_M,
                                                                                               source;
                                                                                               bandwidths,
                                                                                               size_boundaries,
                                                                                               different_values,
                                                                                               sparsity_patterns,
                                                                                               opt_alg,
                                                                                               options,
                                                                                               autodiff,
                                                                                               x0,
                                                                                               verbose)
    return SubcellOperator(nodes, x_M, weights_left, weights_right, Q_left, Q_right,
                           B_left, B_right, accuracy_order, source)
end

function SummationByPartsOperatorsExtra.create_S_left(sigma_L, N, N_L, bandwidth,
                                                      size_boundary,
                                                      different_values, sparsity_pattern)
    S_L = zeros(eltype(sigma_L), N, N)
    set_S_left!(S_L, sigma_L, N_L, bandwidth, size_boundary, different_values,
                sparsity_pattern)
    return S_L
end

function SummationByPartsOperatorsExtra.create_S_right(sigma_R, N, N_R, bandwidth,
                                                       size_boundary,
                                                       different_values, sparsity_pattern)
    S_R = zeros(eltype(sigma_R), N, N)
    set_S_right!(S_R, sigma_R, N_R, bandwidth, size_boundary, different_values,
                 sparsity_pattern)
    return S_R
end

function set_S_left!(S_L, sigma_L, N_L, bandwidth, size_boundary,
                     different_values, sparsity_pattern)
    set_S!(view(S_L, 1:N_L, 1:N_L), sigma_L, bandwidth, size_boundary,
           different_values, sparsity_pattern)
end

function set_S_right!(S_R, sigma_R, N_R, bandwidth, size_boundary,
                      different_values, sparsity_pattern)
    N = size(S_R, 1)
    set_S!(view(S_R, (N - N_R + 1):N, (N - N_R + 1):N), sigma_R, bandwidth,
           size_boundary, different_values, sparsity_pattern)
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
                                    bandwidths = [0, 0],
                                    size_boundaries = [0, 0],
                                    different_values = [true, true],
                                    sparsity_patterns = [nothing, nothing],
                                    opt_alg = LBFGS(),
                                    options = Options(g_tol = 1e-14,
                                                      iterations = 10000),
                                    autodiff = :forward,
                                    x0 = nothing, verbose = false)
    T = eltype(nodes)
    K = length(basis_functions)
    N = length(nodes)
    N_L = findfirst(x -> x > x_M, nodes) - 1
    # If x_M is in the nodes, x_M will be part of both sub-cells, which means we have one more node
    # N_R = x_M in nodes ? N - N_L + 1 : N - N_L
    # If x_M is in the nodes, we just put it in the left sub-cell, but not in the right one
    N_R = N - N_L

    if bandwidths[1] == 0
        bandwidths[1] = N_L - 1
    end
    if bandwidths[2] == 0
        bandwidths[2] = N_R - 1
    end
    if size_boundaries[1] == 0
        size_boundaries[1] = 2 * bandwidths[1]
    end
    if size_boundaries[2] == 0
        size_boundaries[2] = 2 * bandwidths[2]
    end
    assert_correct_bandwidth(N_L, bandwidths[1], size_boundaries[1])
    assert_correct_bandwidth(N_R, bandwidths[2], size_boundaries[2])
    for (i, sparsity_pattern) in enumerate(sparsity_patterns)
        if !isnothing(sparsity_pattern)
            assert_correct_sparsity_pattern(sparsity_patterns)
            sparsity_patterns[i] = UpperTriangular(sparsity_patterns[i])
        end
    end
    L_L = get_nsigma(N_L; bandwidth = bandwidths[1], size_boundary = size_boundaries[1],
                     different_values = different_values[1],
                     sparsity_pattern = sparsity_patterns[1])
    L_R = get_nsigma(N_R; bandwidth = bandwidths[2], size_boundary = size_boundaries[2],
                     different_values = different_values[2],
                     sparsity_pattern = sparsity_patterns[2])

    basis_functions_derivatives = [x -> ForwardDiff.derivative(basis_functions[i], x)
                                   for i in 1:K]

    V = vandermonde_matrix(basis_functions, nodes)
    V_x = vandermonde_matrix(basis_functions_derivatives, nodes)

    # We assume that x_1 = x_L and x_N = x_R
    e_L = spzeros(T, N)
    e_L[1] = 1
    e_R = spzeros(T, N)
    e_R[end] = 1
    # e_L = V' \ [basis_functions[i](first(nodes)) for i in 1:K]
    # e_R = V' \ [basis_functions[i](last(nodes)) for i in 1:K]
    f_M = [basis_functions[i](x_M) for i in 1:K]
    # e_M = V' \ f_M
    # We need e_M_L and e_M_R to be supported only in their sub-cells
    e_M_L_1 = view(V', :, 1:N_L) \ f_M
    e_M_L = [e_M_L_1; zeros(T, N - N_L)]
    e_M_R_1 = view(V', :, (N - N_R + 1):N) \ f_M
    e_M_R = [zeros(T, N - N_R); e_M_R_1]

    # If x_M is in the nodes, we can also use
    # e_M = spzeros(T, N)
    # e_M[N_L] = 1
    E_L = [e_L e_M_L]
    E_R = [e_M_R e_R]

    B_P = [-one(T) zero(T)
           zero(T) one(T)]
    B_left = E_L * B_P * E_L' # = e_M_L * e_M_L' - e_L * e_L'
    B_right = E_R * B_P * E_R' # = e_R * e_R' - e_M_R * e_M_R'

    # B needs to be a block matrix
    # B_left has a non-zero upper left block and B_right has a non-zero lower right block
    # B = spzeros(T, N, N)
    # B[1, 1] = -1
    # B[N, N] = 1
    B = B_left + B_right
    R = B * V / 2
    x_length_left = x_M - first(nodes)
    x_length_right = last(nodes) - x_M

    if isnothing(x0)
        x0 = [zeros(T, L_L + L_R); invsig.(1 / N_L * ones(T, N_L));
              invsig.(1 / N_R * ones(T, N_R))]
    else
        n_total = L_L + L_R + N_L + N_R
        if length(x0) != n_total
            throw(ArgumentError("Initial guess has to be L_L + L_R + N_L + N_R = $n_total long"))
        end
    end

    chunksize = ForwardDiff.pickchunksize(length(x0))
    S_L = zeros(T, N, N)
    S_R = zeros(T, N, N)
    S = zeros(T, N, N)
    SV = zeros(T, N, K)
    PV_x = zeros(T, N, K)
    A = zeros(T, N, K)
    S_L_cache = DiffCache(S_L, chunksize)
    S_R_cache = DiffCache(S_R, chunksize)
    S_cache = DiffCache(S, chunksize)
    SV_cache = DiffCache(SV, chunksize)
    PV_x_cache = DiffCache(PV_x, chunksize)
    A_cache = DiffCache(A, chunksize)
    p = (; L_L, L_R, N_L, N_R, x_length_left, x_length_right, V, V_x, R,
         S_L_cache, S_R_cache, S_cache, SV_cache, PV_x_cache, A_cache,
         bandwidths, size_boundaries, different_values, sparsity_patterns)

    f(x) = optimization_function_subcell_operator(x, p)
    result = optimize(f, x0, opt_alg, options; autodiff)
    verbose && show(stdout, "text/plain", result)

    x = minimizer(result)
    sigma_L, sigma_R, rho_L, rho_R = split_x_subcell_operator(x, L_L, L_R, N_L)
    S_L = SummationByPartsOperatorsExtra.create_S_left(sigma_L, N, N_L, bandwidths[1],
                                                       size_boundaries[1],
                                                       different_values[1],
                                                       sparsity_patterns[1])
    S_R = SummationByPartsOperatorsExtra.create_S_right(sigma_R, N, N_R, bandwidths[2],
                                                        size_boundaries[2],
                                                        different_values[2],
                                                        sparsity_patterns[2])
    P_L = create_P_left(rho_L, N, x_length_left)
    P_R = create_P_right(rho_R, N, x_length_right)
    weights_left = diag(P_L)[1:N_L]
    weights_right = diag(P_R)[(N - N_R + 1):end]
    Q_left = S_L + B_left / 2
    Q_right = S_R + B_right / 2
    return weights_left, weights_right, Q_left, Q_right, B_left, B_right
end

function optimization_function_subcell_operator(x, p)
    (; L_L, L_R, N_L, N_R, x_length_left, x_length_right, V, V_x, R,
    S_L_cache, S_R_cache, S_cache, SV_cache, PV_x_cache, A_cache,
    bandwidths, size_boundaries, different_values, sparsity_patterns) = p

    (N, _) = size(R)
    sigma_L, sigma_R, rho_L, rho_R = split_x_subcell_operator(x, L_L, L_R, N_L)

    S_L = get_tmp(S_L_cache, x)
    S_R = get_tmp(S_R_cache, x)
    S = get_tmp(S_cache, x)
    SV = get_tmp(SV_cache, x)
    PV_x = get_tmp(PV_x_cache, x)
    A = get_tmp(A_cache, x)
    set_S_left!(S_L, sigma_L, N_L, bandwidths[1], size_boundaries[1], different_values[1],
                sparsity_patterns[1])
    set_S_right!(S_R, sigma_R, N_R, bandwidths[2], size_boundaries[2], different_values[2],
                 sparsity_patterns[2])
    @. S = S_L + S_R
    mul!(SV, S, V)
    P_L = create_P_left(rho_L, N, x_length_left)
    P_R = create_P_right(rho_R, N, x_length_right)
    P = P_L + P_R
    mul!(PV_x, P, V_x)
    @. A = SV - PV_x + R

    return sum(abs2, A)
end

# x = [sigma_L; sigma_R; rho_L; rho_R]
function split_x_subcell_operator(x, L_L, L_R, N_L)
    return x[1:L_L], x[(L_L + 1):(L_L + L_R)],
           x[(L_L + L_R + 1):(L_L + L_R + N_L)], x[(L_L + L_R + N_L + 1):end]
end
