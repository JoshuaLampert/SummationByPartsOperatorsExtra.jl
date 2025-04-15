module SummationByPartsOperatorsExtraOptimForwardDiffExt

using Optim: Optim, Options, LBFGS, optimize, minimizer
import ForwardDiff

using SummationByPartsOperatorsExtra: SummationByPartsOperatorsExtra,
                                      GlaubitzNordströmÖffner2023,
                                      MatrixDerivativeOperator
using SummationByPartsOperatorsExtra: get_nsigma
using LinearAlgebra: Diagonal, UpperTriangular, LowerTriangular, diag, mul!,
                     issymmetric
using SparseArrays: spzeros
using PreallocationTools: DiffCache, get_tmp

function vandermonde_matrix(functions, nodes)
    N = length(nodes)
    K = length(functions)
    T = typeof(functions[1](nodes[1]))
    V = zeros(T, N, K)
    for i in 1:N
        for j in 1:K
            V[i, j] = functions[j](nodes[i])
        end
    end
    return V
end

function SummationByPartsOperatorsExtra.create_S(sigma, N, bandwidth, size_boundary,
                                                 different_values, sparsity_pattern)
    S = zeros(eltype(sigma), N, N)
    set_S!(S, sigma, N, bandwidth, size_boundary, different_values, sparsity_pattern)
    return S
end

# M has to be square
function set_skew_symmetric!(M, sigma, init_k = 1)
    N = size(M, 1)
    k = init_k
    for i in 1:N
        for j in (i + 1):N
            M[i, j] = sigma[k]
            M[j, i] = -sigma[k]
            k += 1
        end
    end
    return k
end

# D has to be square
function set_banded!(D, sigma, bandwidth, init_k = 1, different_values = true)
    N = size(D, 1)
    k = init_k
    for i in 1:N
        for j in (i + 1):N
            if j - i <= bandwidth
                if different_values
                    l = k
                    k += 1
                else
                    l = init_k + j - i - 1
                end
                D[i, j] = sigma[l]
                D[j, i] = -sigma[l]
            end
        end
    end
    return k
end

function set_triangular!(C, sigma, bandwidth, size_boundary = 2 * bandwidth, init_k = 1,
                         different_values = true)
    N = size(C, 1)
    k = init_k
    if different_values
        start_i = N - bandwidth + 1
    else
        start_i = size_boundary - bandwidth + 1
    end
    for i in start_i:N
        for j in 1:(i - start_i + 1)
            if different_values
                l = k
            else
                l = init_k - 1 + bandwidth + j - (i - start_i + 1)
            end
            C[i, j] = sigma[l]
            k += 1
        end
    end
    return k
end

function set_S!(S, sigma, N, bandwidth, size_boundary = 2 * bandwidth,
                different_values = true, sparsity_pattern = nothing)
    fill!(S, zero(eltype(sigma)))
    if isnothing(sparsity_pattern)
        set_S_block_banded!(S, sigma, N, bandwidth, size_boundary, different_values)
    else
        set_S_sparsity_pattern!(S, sigma, N, sparsity_pattern)
    end
end

permute_rows_and_cols(P) = P[size(P, 1):-1:1, size(P, 2):-1:1]

@views function set_S_block_banded!(S, sigma, N, bandwidth, size_boundary = 2 * bandwidth,
                                    different_values = true)
    if bandwidth == N - 1
        set_skew_symmetric!(S, sigma)
    else
        b = bandwidth
        c = size_boundary
        # upper left boundary block
        M1 = S[1:c, 1:c]
        k = set_skew_symmetric!(M1, sigma, 1)
        # lower right boundary block
        M2 = S[(N - c + 1):N, (N - c + 1):N]
        if different_values
            k = set_skew_symmetric!(M2, sigma, k)
        else
            M2 .= -permute_rows_and_cols(M1)
        end

        # banded matrix in the middle
        D = S[(c + 1):(N - c), (c + 1):(N - c)]
        k = set_banded!(D, sigma, b, k, different_values)

        # upper central block with triangular part
        C1 = S[1:c, (c + 1):(N - c)]
        k = set_triangular!(C1, sigma, b, c, k, different_values)
        # central left block with triangular part
        S[(c + 1):(N - c), 1:c] = -C1'
        # central right block with triangular part
        C2 = S[(c + 1):(N - c), (N - c + 1):N]
        if different_values
            k = set_triangular!(C2, sigma, b, c, k, different_values)
            # lower central block with triangular part
            S[(N - c + 1):N, (c + 1):(N - c)] = -C2'
        else
            C1_bar = permute_rows_and_cols(C1)
            C2 .= C1_bar'
            S[(N - c + 1):N, (c + 1):(N - c)] = -C1_bar
        end
    end
end

function set_S_sparsity_pattern!(S, sigma, N, sparsity_pattern)
    k = 1
    for i in 1:N
        for j in (i + 1):N
            if sparsity_pattern[i, j]
                S[i, j] = sigma[k]
                S[j, i] = -sigma[k]
                k += 1
            end
        end
    end
end

# sig(x) = x
# invsig(p) = p

sig(x) = 1 / (1 + exp(-x))
invsig(p) = log(p / (1 - p))

# leading to softmax
# sig(x) = exp(x)
# invsig(p) = log(p)

# For b, no sigmoid function seems to perform better
# sig_b(x) = x
# invsig_b(p) = p

# sig_b(x) = 1 / (1 + exp(-x))
# invsig_b(p) = log(p / (1 - p))

function create_P(rho, vol)
    P = Diagonal(sig.(rho))
    P *= vol / sum(P)
    return P
end

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

function inner_H1(f, g, f_derivative, g_derivative, nodes)
    return sum(f.(nodes) .* g.(nodes) + f_derivative.(nodes) .* g_derivative.(nodes))
end
norm_H1(f, f_derivative, nodes) = sqrt(inner_H1(f, f, f_derivative, f_derivative, nodes))

function call_orthonormal_basis_function(A, basis_functions, k, x)
    sum([basis_functions[i](x) * A[k, i] for i in 1:k])
end

# This will orthonormalize the basis functions using the Gram-Schmidt process to reduce the condition
# number of the Vandermonde matrix. The matrix A transfers the old basis functions to the new orthonormalized by
# g(x) = A * f(x), where f(x) is the vector of old basis functions and g(x) is the vector of the new orthonormalized
# basis functions. Analogously, we have g'(x) = A * f'(x).
function orthonormalize_gram_schmidt(basis_functions, basis_functions_derivatives, nodes)
    K = length(basis_functions)

    A = LowerTriangular(zeros(eltype(nodes), K, K))

    basis_functions_orthonormalized = Vector{Function}(undef, K)
    basis_functions_orthonormalized_derivatives = Vector{Function}(undef, K)

    for k in 1:K
        A[k, k] = 1
        for j in 1:(k - 1)
            g(x) = call_orthonormal_basis_function(A, basis_functions, j, x)
            function g_derivative(x)
                call_orthonormal_basis_function(A, basis_functions_derivatives, j, x)
            end
            inner_product = inner_H1(basis_functions[k], g, basis_functions_derivatives[k],
                                     g_derivative, nodes)
            norm_squared = inner_H1(g, g, g_derivative, g_derivative, nodes)
            A[k, :] = A[k, :] - inner_product / norm_squared * A[j, :]
        end

        basis_functions_orthonormalized[k] = x -> call_orthonormal_basis_function(A,
                                                                                  basis_functions,
                                                                                  k, x)
        basis_functions_orthonormalized_derivatives[k] = x -> call_orthonormal_basis_function(A,
                                                                                              basis_functions_derivatives,
                                                                                              k,
                                                                                              x)
        # Normalization
        r = norm_H1(basis_functions_orthonormalized[k],
                    basis_functions_orthonormalized_derivatives[k], nodes)
        A[k, :] = A[k, :] / r
    end
    return basis_functions_orthonormalized, basis_functions_orthonormalized_derivatives
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

    B = spzeros(eltype(nodes), N, N)
    B[1, 1] = -1
    B[N, N] = 1

    R = B * V / 2
    x_length = last(nodes) - first(nodes)
    S = zeros(eltype(nodes), N, N)
    A = zeros(eltype(nodes), N, K)
    SV = zeros(eltype(nodes), N, K)
    PV_x = zeros(eltype(nodes), N, K)
    S_cache = DiffCache(S)
    A_cache = DiffCache(A)
    SV_cache = DiffCache(SV)
    PV_x_cache = DiffCache(PV_x)
    daij_dsigmak = zeros(eltype(nodes), N, K, L)
    daij_drhok = zeros(eltype(nodes), N, K, N)
    p = (; V, V_x, R, x_length, S_cache, A_cache, SV_cache, PV_x_cache, bandwidth,
         size_boundary, different_values, sparsity_pattern, daij_dsigmak, daij_drhok)

    if isnothing(x0)
        x0 = [zeros(L); invsig.(1 / N * ones(N))]
    else
        @assert length(x0)==L + N "Initial guess has be L + N = $(L + N) long"
    end

    f(x) = optimization_function(x, p)
    result = optimize(f, x0, opt_alg, options; autodiff)
    verbose && show(stdout, "text/plain", result)

    x = minimizer(result)
    sigma = x[1:L]
    rho = x[(L + 1):end]
    S = SummationByPartsOperatorsExtra.create_S(sigma, N, bandwidth, size_boundary,
                                                different_values, sparsity_pattern)
    P = create_P(rho, x_length)
    weights = diag(P)
    Q = S + B / 2
    D = inv(P) * Q
    return weights, D
end

@views function optimization_function(x, p)
    (; V, V_x, R, x_length, S_cache, A_cache, SV_cache, PV_x_cache, bandwidth, size_boundary, different_values, sparsity_pattern) = p
    S = get_tmp(S_cache, x)
    A = get_tmp(A_cache, x)
    SV = get_tmp(SV_cache, x)
    PV_x = get_tmp(PV_x_cache, x)
    (N, _) = size(R)
    L = get_nsigma(N; bandwidth, size_boundary, different_values, sparsity_pattern)
    sigma = x[1:L]
    rho = x[(L + 1):end]
    set_S!(S, sigma, N, bandwidth, size_boundary, different_values, sparsity_pattern)
    P = create_P(rho, x_length)
    mul!(SV, S, V)
    mul!(PV_x, P, V_x)
    @. A = SV - PV_x + R
    return sum(abs2, A)
end
end