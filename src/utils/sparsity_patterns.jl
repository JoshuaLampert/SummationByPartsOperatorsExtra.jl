function to_S(P, D)
    Q = P * D
    S = 0.5 * (Q - Q')
    return S
end

function to_S(D::AbstractNonperiodicDerivativeOperator)
    to_S(mass_matrix(D), Matrix(D))
end

function to_S(D::AbstractMultidimensionalMatrixDerivativeOperator{1})
    to_S(mass_matrix(D), Matrix(D[1]))
end

"""
    get_sparsity_pattern(S)
    get_sparsity_pattern(D::AbstractNonperiodicDerivativeOperator)
    get_sparsity_pattern(D::AbstractMultidimensionalMatrixDerivativeOperator{2})

If `S` is a (skew-symmetric) matrix, this function returns the sparsity pattern of `S` as a `UpperTriangular` matrix.
If `D` is a one-dimensional derivative operator, this function returns the sparsity pattern of the skew-symmetric part of `D`.
If `D` is a two-dimensional derivative operator, this function returns a tuple of the sparsity patterns of the skew-symmetric parts of `D` in each direction.
"""
function get_sparsity_pattern(S)
    return UpperTriangular(S .!= 0.0)
end

function get_sparsity_pattern(D::AbstractNonperiodicDerivativeOperator)
    return get_sparsity_pattern(to_S(D))
end

function get_sparsity_pattern(D::AbstractMultidimensionalMatrixDerivativeOperator{2})
    P = mass_matrix(D)
    return (get_sparsity_pattern(to_S(P, D[1])), get_sparsity_pattern(to_S(P, D[2])))
end

"""
    neighborhood_sparsity_pattern(nodes, lengths)

For a given set of `nodes` in a multi-dimensional space, this function computes the sparsity pattern of
the differentiation matrices, which only includes non-zero entries at nodes, which are within a certain
ellipsoid neighborhood.
`lengths` is a tuple of length `d` (dimension) representing the lengths of an ellipsoid
indicating, which nodes are counted as neighbors.
For example, for a differentiation matrix in x direction, it makes sense to use
a larger length in x direction than in y direction.
"""
function neighborhood_sparsity_pattern(nodes, lengths)
    N = length(nodes)
    sparsity_pattern = zeros(Bool, N, N)
    for i in 1:N
        for j in i:N # Only need to compute the upper triangular part since the matrix is symmetric
            if norm((nodes[i] - nodes[j]) ./ lengths) < 1.0
                sparsity_pattern[i, j] = true
            end
        end
    end
    sparsity_pattern[diagind(sparsity_pattern)] .= false
    return UpperTriangular(sparsity_pattern)
end
