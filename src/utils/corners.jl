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
