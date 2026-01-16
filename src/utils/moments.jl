@doc raw"""
    compute_moments_boundary(functions, nodes, normals)
    compute_moments_boundary(functions, D::AbstractDerivativeOperator)
    compute_moments_boundary(functions, geometry::Meshes.Geometry)

Compute the moments, i.e., the integrals of the product of two basis functions weighted by the normal direction
of the direction. For each direction, it computes a ``K \times K`` matrix, where ``K`` is the number of `functions`
and returns a tuple of these matrices.
In one dimension, `nodes` and `normals` can be passed. You can also pass a derivative operator `D`
or a `Geometry` object from Meshes.jl. Note that the latter is defined in a package extension of MeshIntegrals.jl and
therefore requires loading that package before.
"""
function compute_moments_boundary(functions, nodes, normals)
    T = eltype(nodes[1])
    K = length(functions)
    M = zeros(T, K, K)
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
        return M
    end
    return moments
end

function compute_moments_boundary(functions,
                                  D::AbstractDerivativeOperator{T}) where {T}
    K = length(functions)
    nodes = grid(D)
    M = zeros(T, K, K)
    for k in 1:K
        for l in 1:K
            f = x -> functions[k](x) * functions[l](x)
            M[k, l] = integrate_boundary(f, nodes, D)
        end
    end
    return (M,)
end
