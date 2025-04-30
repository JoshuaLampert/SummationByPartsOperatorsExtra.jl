"""
    compute_moments_boundary(functions, nodes, normals)
    compute_moments_boundary(functions, D::AbstractMultidimensionalMatrixDerivativeOperator)

Compute the moments, i.e., the integrals of the product of two basis functions weighted by the normal direction
of the direction. For each direction, it computes a ``K \times K`` matrix, where ``K`` is the number of `functions`
and returns a tuple of these matrices.
"""
function compute_moments_boundary(functions, nodes, normals)
    K = length(functions)
    M = zeros(K, K)
    for i in 1:K
        for j in 1:K
            fi = functions[i]
            fj = functions[j]
            M[i, j] = fi(nodes[1]) * fj(nodes[1]) * normals[1][1] +
                      fi(nodes[end]) * fj(nodes[end]) * normals[end][1]
        end
    end
    return (M,)
end

function compute_moments_boundary(functions,
                                  D::AbstractMultidimensionalMatrixDerivativeOperator{Dim,
                                                                                      T}) where {
                                                                                                 Dim,
                                                                                                 T
                                                                                                 }
    K = length(functions)
    nodes = grid(D)
    moments = ntuple(Dim) do i
        M = zeros(T, K, K)
        for k in 1:K
            for l in 1:K
                f = x -> functions[k](x) * functions[l](x)
                M[k, l] = integrate_boundary(f, nodes, D, i)
            end
        end
        M
    end
    return moments
end
