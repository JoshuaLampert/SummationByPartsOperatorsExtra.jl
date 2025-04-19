function compute_moments_boundary(basis_functions, nodes, normals)
    K = length(basis_functions)
    M = zeros(K, K)
    for i in 1:K
        for j in 1:K
            fi = basis_functions[i]
            fj = basis_functions[j]
            M[i, j] = fi(nodes[1]) * fj(nodes[1]) * normals[1][1] +
                      fi(nodes[end]) * fj(nodes[end]) * normals[end][1]
        end
    end
    return (M,)
end

function compute_moments_boundary(functions,
                                  D::SummationByPartsOperators.AbstractMultidimensionalMatrixDerivativeOperator{Dim,
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
