# This function finds corners based on `boundary_indices`, which are assumed
# to have duplicates for each corner.
# Works only for 2D for now.
# This assumes that the first corner index (and therefore the corresponding normal!) is in x-direction
# and the second corner index are in y-direction.
function find_corners(boundary_indices)
    all_corners = sort(findall(>(1), countmap(boundary_indices)))
    corners_x = Int[]
    corners_y = Int[]
    for corner in all_corners
        corner_inds = findall(==(corner), boundary_indices)
        if length(corner_inds) == 2
            push!(corners_x, corner_inds[1])
            push!(corners_y, corner_inds[2])
        end
    end
    return (corners_x, corners_y)
end

# Note: This is type piracy!
# In SummationByPartsOperators.jl `mass_matrix_boundary` is only allowed
# if `D` doesn't contain corners or if the operator is a `TensorProductOperator`
# with very specific corner indices. If we have another operator with other corner
# indices, we can detect them and use a similar approach as done in SummationByPartsOperators.jl
# to compute the boundary mass matrix.
# Note that this assumes that the indices of the normals in x-direction are before (i.e., smaller than) the
# indices of the normals in y-direction! This is, e.g., not the case for the `TensorProductOperator`, but it is
# the case if we use `find_corners!` from the Meshes.jl extension.
function SummationByPartsOperators.mass_matrix_boundary(D::MultidimensionalMatrixDerivativeOperator{2},
                                                        dim::Int)
    boundary_weights = SummationByPartsOperators.weights_boundary_scaled(D, dim)
    boundary_indices = copy(D.boundary_indices)
    corners = find_corners(boundary_indices)
    deleteat!(boundary_weights, corners[dim])
    deleteat!(boundary_indices, corners[dim])
    b = zeros(eltype(D), length(grid(D)))
    b[boundary_indices] .= boundary_weights
    return Diagonal(b)
end
