using SummationByPartsOperatorsExtra: get_nsigma, vandermonde_matrix
using LinearAlgebra: Diagonal, UpperTriangular, I, diag, issymmetric, qr
using SparseArrays: spzeros

function SummationByPartsOperatorsExtra.create_S(sigma, N, bandwidth, size_boundary,
                                                 different_values, sparsity_pattern)
    S = zeros(eltype(sigma), N, N)
    set_S!(S, sigma, bandwidth, size_boundary, different_values, sparsity_pattern)
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

function set_S!(S, sigma, bandwidth, size_boundary = 2 * bandwidth,
                different_values = true, sparsity_pattern = nothing)
    N = size(S, 1)
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
sig_b(x) = x
invsig_b(p) = p

# sig_b(x) = 1 / (1 + exp(-x))
# invsig_b(p) = log(p / (1 - p))

function create_P(rho, vol)
    P = Diagonal(sig.(rho))
    P *= vol / sum(P)
    return P
end

function create_B(N, phi, normals, boundary_indices, dim;
                  corners = ntuple(_ -> eltype(phi)[], dim))
    b = zeros(eltype(phi), N)
    B = Diagonal(b)
    set_B!(B, phi, normals, boundary_indices, dim; corners)
    return B
end

function set_B!(B, phi, normals, boundary_indices, dim;
                corners = ntuple(_ -> eltype(phi)[], dim))
    fill!(B, zero(eltype(B)))
    for j in eachindex(boundary_indices)
        k = boundary_indices[j]
        # If we have corners, we store multiple weights (boundary_indices is not unique)
        # and we need to make sure to not overwrite the corner weights
        if !(j in corners[dim])
            B[k, k] = sig_b(phi[j]) * normals[j][dim]
        end
    end
end

# Nodal values of the `basis_functions` and of their derivatives after orthonormalizing the basis
# with respect to the discrete H^1 inner product induced by the `nodes`,
#     <f, g>_{H^1} = sum_n f(x_n) * g(x_n) + sum_n f'(x_n) * g'(x_n),
# which reduces the condition number of the Vandermonde matrix. That is, the returned matrices
# `V` and `V_x` satisfy `V' * V + V_x' * V_x = I`.
#
# The orthonormalization is linear: the orthonormalized basis is `g(x) = A * f(x)` with the
# derivatives `g'(x) = A * f'(x)` for the same lower triangular matrix `A`, where `f(x)` is the
# vector of the old basis functions. Moreover, the H^1 inner product above is just the Euclidean
# inner product of the stacked vectors `[f(x); f'(x)]`. Hence, `A` is determined by a QR
# decomposition of the stacked Vandermonde matrix `W = [V; V_x]`: if `W = Q * R`, then
# `A = inv(R)'` is lower triangular and satisfies
#     [V; V_x] * A' = W * inv(R) = Q,
# i.e. the two blocks of `Q` are exactly the nodal values we are after. We therefore never have to
# form `A` (or the orthonormalized basis functions) explicitly.
#
# We use a Householder QR decomposition rather than the Gram-Schmidt process used previously: it is
# orthogonal to machine precision independently of the conditioning of `W`, whereas the classical
# Gram-Schmidt process loses orthogonality proportionally to the *squared* condition number of `W`
# - which partly defeats the purpose of orthonormalizing in the first place. It is also cheaper
# since the basis functions are evaluated once instead of once per inner product.
function orthonormal_vandermonde_matrices(basis_functions, basis_functions_derivatives,
                                          nodes)
    T = eltype(nodes)
    N = length(nodes)
    K = length(basis_functions)
    W = [vandermonde_matrix(basis_functions, nodes);
         vandermonde_matrix(basis_functions_derivatives, nodes)]

    factorization = qr(W)
    # `Matrix(factorization.Q)` would form the full `2N x 2N` factor, so we apply the Householder
    # reflectors to the first `K` unit vectors instead to obtain the thin factor directly
    Q = factorization.Q * Matrix{T}(I, 2 * N, K)
    # The QR decomposition is unique only up to the signs of the columns of `Q`. We normalize them
    # such that the diagonal of `R` (and hence of `A`) is positive, as for the Gram-Schmidt process.
    for k in 1:K
        if factorization.R[k, k] < 0
            @views Q[:, k] .= -Q[:, k]
        end
    end
    return Q[1:N, :], Q[(N + 1):(2 * N), :]
end

function assert_first_derivative_order(derivative_order)
    if derivative_order != 1
        throw(ArgumentError("Derivative order $derivative_order not implemented."))
    end
end

function assert_correct_bandwidth(N, bandwidth, size_boundary)
    if (N < 2 * size_boundary + bandwidth || bandwidth < 1) &&
       (bandwidth != N - 1)
        throw(ArgumentError("2 * size_boundary + bandwidth = $(2 * size_boundary + bandwidth) needs to be smaller than or equal to N = $N and bandwidth = $bandwidth needs to be at least 1."))
    end
end

function assert_correct_sparsity_pattern(sparsity_pattern)
    if !(sparsity_pattern isa UpperTriangular || issymmetric(sparsity_pattern)) ||
       !all(diag(sparsity_pattern) .== 0)
        throw(ArgumentError("Sparsity pattern has to be symmetric with all diagonal entries being false or `UpperTriangular`."))
    end
end

function assert_correct_length_basis_functions_weights(basis_functions_weights,
                                                       basis_functions)
    if length(basis_functions_weights) != length(basis_functions)
        throw(ArgumentError("Length of basis function weights $(length(basis_functions_weights)) does not match number of basis functions $(length(basis_functions))."))
    end
end

# x = [sigma; rho]
split_x_function_space_operator(x, L) = x[1:L], x[(L + 1):end]
