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

function inner_L2(f, g, nodes)
    return sum(f.(nodes) .* g.(nodes))
end
norm_L2(f, nodes) = sqrt(inner_L2(f, f, nodes))

function call_orthonormal_basis_function(A, basis_functions, k, x)
    return sum(basis_functions[i](x) * A[k, i] for i in 1:k)
end

# This will orthonormalize the basis functions using the Gram-Schmidt process to reduce the condition
# number of the Vandermonde matrix. The matrix A transfers the old basis functions to the new orthonormalized by
# g(x) = A * f(x), where f(x) is the vector of old basis functions and g(x) is the vector of the new orthonormalized
# basis functions.
function orthonormalize_gram_schmidt(functions, nodes)
    K = length(functions)
    A = LowerTriangular(zeros(eltype(nodes), K, K))

    functions_orthonormalized = Vector{Function}(undef, K)

    for k in 1:K
        A[k, k] = 1
        for j in 1:(k - 1)
            g(x) = call_orthonormal_basis_function(A, functions, j, x)
            inner_product = inner_L2(functions[k], g, nodes)
            norm_squared = inner_L2(g, g, nodes)
            A[k, :] = A[k, :] - inner_product / norm_squared * A[j, :]
        end

        functions_orthonormalized[k] = x -> call_orthonormal_basis_function(A, functions, k, x)
        # Normalization
        r = norm_L2(functions_orthonormalized[k], nodes)
        A[k, :] = A[k, :] / r
    end
    return functions_orthonormalized
end

include("interpolation.jl")
include("optimization.jl")
include("sparsity_patterns.jl")
include("moments.jl")
include("corners.jl")
include("visualization.jl")
