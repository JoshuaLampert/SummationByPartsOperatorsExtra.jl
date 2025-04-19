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
