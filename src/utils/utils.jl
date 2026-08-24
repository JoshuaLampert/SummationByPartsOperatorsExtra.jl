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

# Placeholder for orthonormalizing basis functions with the Gram-Schmidt process. This method is
# extended in backend-specific implementations, where the orthonormalized functions themselves are
# needed (in contrast to just their nodal values, cf. `enriched_orthonormal_vandermonde`).
function orthonormalize_gram_schmidt end

include("interpolation.jl")
include("optimization.jl")
include("sparsity_patterns.jl")
include("moments.jl")
include("corners.jl")
include("visualization.jl")
