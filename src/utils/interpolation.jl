# This is type piracy
function PolynomialBases.interpolation_matrix(dest, D::AbstractNonperiodicDerivativeOperator)
    nodes = grid(D)
    baryweights = PolynomialBases.barycentric_weights(nodes)
    interpolation_matrix(dest, nodes, baryweights)
end

function PolynomialBases.interpolation_matrix!(mat, dest, D::AbstractNonperiodicDerivativeOperator)
    nodes = grid(D)
    baryweights = PolynomialBases.barycentric_weights(nodes)
    interpolation_matrix!(mat, dest, nodes, baryweights)
end
