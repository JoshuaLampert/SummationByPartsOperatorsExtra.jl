function get_nsigma(N; bandwidth = N - 1,
                    size_boundary = 2 * bandwidth, different_values = true,
                    sparsity_pattern = nothing)
    if isnothing(sparsity_pattern)
        if bandwidth == N - 1
            # whole upper right triangle
            return div(N * (N - 1), 2)
        else
            if different_values
                # upper right corner for boundary blocks cxc each: c*(c - 1)/2
                # lower triangle including diagonal for two different upper and right central blocks bxb each: b*(b + 1)/2
                # non-repeating stencil for diagonal block: (N - 2c - b)b + b*(b - 1)/2 = Nb - 1/2(4c*b + b^2 + b)
                # => in total: Nb + 1/2b^2 + c^2 - 2c*b - c + 1/2b
                # return N * bandwidth + div(bandwidth * (bandwidth - 3), 2) # for c = 2b
                b = bandwidth
                c = size_boundary
                return N * b + div(b * (b + 1), 2) + c^2 - 2 * b * c - c
            else
                # upper right corner for boundary blocks cxc: c*(c - 1)/2 plus b from repeating stencil
                # => in total: c*(c - 1)/2 + b
                # return 2 * bandwidth^2 # for c = 2b
                return div(size_boundary * (size_boundary - 1), 2) + bandwidth
            end
        end
    else
        # the sparsity_pattern matrix is a `UpperTriangular` matrix with zeros on the diagonal
        return count(sparsity_pattern)
    end
end

# This requires sig(x) = 1 / (1 + exp(-x))
invsig(p) = log(p / (1 - p))

function reconstruct_sparsity_pattern!(sigma, S, sparsity_pattern)
    N = size(S, 1)
    k = 1
    for i in 1:N
        for j in (i + 1):N
            if sparsity_pattern[i, j]
                sigma[k] = S[i, j]
                k += 1
            end
        end
    end
end

function reconstruct_skew_symmetric!(sigma, S, init_k = 1)
    N = size(S, 1)
    k = init_k
    for i in 1:N
        for j in (i + 1):N
            sigma[k] = S[i, j]
            k += 1
        end
    end
    return k
end

# Only for different_values = true
function reconstruct_banded!(sigma, D, bandwidth, init_k = 1)
    N = size(D, 1)
    k = init_k
    for i in 1:N
        for j in (i + 1):N
            if j - i <= bandwidth
                sigma[k] = D[i, j]
                k += 1
            end
        end
    end
    return k
end

# Only for different_values = true
function reconstruct_triangular!(sigma, C, bandwidth, init_k = 1)
    N = size(C, 1)
    k = init_k
    start_i = N - bandwidth + 1
    for i in start_i:N
        for j in 1:(i - start_i + 1)
            sigma[k] = C[i, j]
            k += 1
        end
    end
    return k
end

# Helper function to get the entries to optimize for from other operators.
# These can, e.g., be used to initialize the optimization problem.
function get_optimization_entries(D;
                                  bandwidth = div(accuracy_order(D), 2),
                                  size_boundary = SummationByPartsOperators.lower_bandwidth(D) +
                                                  1,
                                  different_values = false,
                                  sparsity_pattern = nothing)
    p = diag(mass_matrix(D))
    # for sig = exp this is only equal to the values from the optimization up to a constant, but they give the same P
    # if sig is the logistic function, inverting the normalized logistic function is harder, but this still works
    # (eventhough it is not the exaxt inverse)
    rho = invsig.(p)
    Matrix_D = Matrix(D)
    Q = mass_matrix(D) * Matrix_D
    S = 0.5 * (Q - Q')
    if isnothing(sparsity_pattern)
        sigma = get_optimization_entries_block_banded(S; bandwidth, size_boundary,
                                                      different_values)
    else
        sigma = get_optimization_entries_sparsity_pattern(S; sparsity_pattern)
    end
    return [sigma; rho]
end

function get_optimization_entries_sparsity_pattern(S; sparsity_pattern)
    N = size(S, 1)
    L = get_nsigma(N; sparsity_pattern)
    sigma = zeros(L)
    reconstruct_sparsity_pattern!(sigma, S, sparsity_pattern)
    return sigma
end

function get_optimization_entries_block_banded(S;
                                               bandwidth = div(accuracy_order(D), 2),
                                               size_boundary = SummationByPartsOperators.lower_bandwidth(D) +
                                                               1,
                                               different_values = false)
    b = bandwidth
    c = size_boundary
    N = size(S, 1)
    L = get_nsigma(N; bandwidth = b, size_boundary = c, different_values)
    sigma = zeros(L)
    if b == N - 1 # dense operator
        reconstruct_skew_symmetric!(sigma, S)
    else # sparse operator
        if different_values
            k = 1
            # upper left boundary block
            M1 = S[1:c, 1:c]
            k = reconstruct_skew_symmetric!(sigma, M1, k)
            # lower right boundary block
            M2 = S[(N - c + 1):N, (N - c + 1):N]
            k = reconstruct_skew_symmetric!(sigma, M2, k)

            # banded matrix in the middle
            D = S[(c + 1):(N - c), (c + 1):(N - c)]
            k = reconstruct_banded!(sigma, D, b, k)

            # upper central block with triangular part
            C1 = S[1:c, (c + 1):(N - c)]
            k = reconstruct_triangular!(sigma, C1, b, k)
            # central right block with triangular part
            C2 = S[(c + 1):(N - c), (N - c + 1):N]
            k = reconstruct_triangular!(sigma, C2, b, k)
        else
            k = 1
            # upper left boundary block
            M1 = S[1:c, 1:c]
            k = reconstruct_skew_symmetric!(sigma, M1, k)
            # values from triangle block (repeating stencil)
            for i in c:-1:(c - b + 1)
                sigma[k] = S[i, c + 1]
                k += 1
            end
        end
    end
    return sigma
end
