module SummationByPartsOperatorsExtraMeshesExt

using Meshes: Meshes, PointSet, Point, Ring,
              paramdim, sample, boundary, measure, to

using SummationByPartsOperatorsExtra: SummationByPartsOperatorsExtra,
                                      multidimensional_function_space_operator,
                                      neighborhood_sparsity_pattern,
                                      compute_moments_boundary,
                                      find_corners,
                                      SVector
include("normals.jl")

uto(p::Point) = Meshes.ustrip.(to(p))
Meshes.Point(x::AbstractVector) = Point(Tuple(x))
Meshes.PointSet(xs::Vector{<:AbstractVector}) = PointSet(Point.(xs))
# This function samples in the whole geometry using `sampler` and uses
# the boundary nodes as nodes on the boundary.
function compute_nodes_normals(geometry, sampler, ::Nothing)
    nodes = PointSet(sample(geometry, sampler))
    _, _, boundary_indices = divide_into_inner_and_boundary(geometry, nodes)
    # Note: This will always be the same normal for all boundary nodes with the same location (corners),
    # but we usually want different normals for the different directions for the corners.
    # Therefore, we need to fix the corner normals with `fix_corner_normals!` below.
    normals = outer_normal.(Ref(geometry), nodes[boundary_indices])
    corners = find_corners(boundary_indices)
    fix_corner_normals!(normals, corners, boundary(geometry), nodes, boundary_indices)
    nodes = uto.(nodes)
    return nodes, normals, boundary_indices
end

# This function uses `sampler_boundary` to sample the boundary nodes
function compute_nodes_normals(geometry, sampler, sampler_boundary)
    geometry_boundary = boundary(geometry)
    # Allow sampling boundary nodes, but exclude them for `nodes_inner`
    nodes_inner_ = sample(geometry, sampler)
    nodes_inner = Point[]
    for node in nodes_inner_
        if !(node in geometry_boundary)
            push!(nodes_inner, node)
        end
    end
    nodes_inner = PointSet(nodes_inner)
    N = length(nodes_inner)

    nodes_boundary = PointSet(sample(geometry_boundary, sampler_boundary))
    N_boundary = length(nodes_boundary)
    normals = outer_normal.(Ref(geometry), nodes_boundary)
    nodes = [uto.(nodes_inner); uto.(nodes_boundary)]
    # First, we set the boundary nodes without respecting the corners
    boundary_indices = [i for i in (N + 1):(N + N_boundary)]
    # Next, we fix the boundary indices for the corners and compute the corners (this will make boundary_indices longer)
    corners = find_corners!(boundary_indices, geometry_boundary, nodes)
    # Lastly, we fix the normals for the corners (for this, we need to make `normals` longer)
    d = paramdim(geometry)
    extended_normals = Vector{SVector{d, Float64}}(undef,
                                                   length(boundary_indices) - N_boundary)
    append!(normals, extended_normals)
    fix_corner_normals!(normals, corners, geometry_boundary, nodes, boundary_indices)
    return nodes, normals, boundary_indices
end

# This function assumes one has sampled a set of `nodes` including boundary nodes,
# but `boundary_indices` only contains the indices of the boundary nodes once
# independent of whether it is a corner or not. The function finds corners based
# on the geometry and fixes `boundary_indices` by doubling indices corresponding to corners.
# It also returns the corners similar to `find_corners`.
function find_corners!(boundary_indices, ring::Ring, nodes)
    if !allunique(boundary_indices)
        throw(ArgumentError("The boundary indices should be unique"))
    end
    nodes = PointSet(nodes)
    corners_x = Int[]
    corners_y = Int[]
    counter_boundary_nodes = 0
    for (i, node) in enumerate(nodes)
        if node in ring # or i in boundary_indices
            counter_boundary_nodes += 1
            n_segments = 0
            for seg in segments(ring)
                if node in seg
                    n_segments += 1
                end
            end
            # Found a corner
            if n_segments > 1
                # We add the corner a second time to the boundary indices
                push!(boundary_indices, i)
                # Corner in x-direction is the one we already had
                push!(corners_x, counter_boundary_nodes)
                # Corner in y-direction is the one we just added
                push!(corners_y, length(boundary_indices))
            end
        end
    end
    return (corners_x, corners_y)
end

# Only in 2D
# In `compute_nodes_normals` above, we set the same normal for each corner.
# However, for each representative of each corner we need the normal in the
# direction of the correct segment.
# `outer_normal` finds the first segment in the ring, which contains the point.
# By reversing the ring, we can find the other segment.
# I'm not 100% sure this works in any case, but for a `Box` it seems to work.
function fix_corner_normals!(normals, corners, ring::Ring, nodes, boundary_indices)
    nodes = PointSet(nodes)
    (corners_x, corners_y) = corners
    for corner_x in corners_x
        normal_1 = outer_normal(ring, nodes[boundary_indices[corner_x]])
        normal_2 = -outer_normal(reverse(ring), nodes[boundary_indices[corner_x]])
        # For the x-corners, we choose the normal, which goes into y-direction
        if abs(normal_1[1]) < abs(normal_2[1])
            normals[corner_x] = normal_1
        else
            normals[corner_x] = normal_2
        end
    end
    for corner_y in corners_y
        normal_1 = outer_normal(ring, nodes[boundary_indices[corner_y]])
        normal_2 = -outer_normal(reverse(ring), nodes[boundary_indices[corner_y]])
        # For the y-corners, we choose the normal, which goes into x-direction
        if abs(normal_1[1]) < abs(normal_2[1])
            normals[corner_y] = normal_2
        else
            normals[corner_y] = normal_1
        end
    end
end

# We need to specialize this for `Ring` because we need corners multiple times.
# We detect corners by lying in multiple segments of the `Ring`.
function push_to_boundary_indices!(boundary_indices, boundary_geometry::Ring, node, i)
    for seg in segments(boundary_geometry)
        if node in seg
            push!(boundary_indices, i)
        end
    end
end

function push_to_boundary_indices!(boundary_indices, boundary_geometry, node, i)
    push!(boundary_indices, i)
end

function divide_into_inner_and_boundary(geometry, nodes)
    nodes = PointSet(nodes)
    boundary_geometry = boundary(geometry)
    nodes_inner = eltype(nodes)[]
    nodes_boundary = eltype(nodes)[]
    boundary_indices = Int[]
    for (i, node) in enumerate(nodes)
        if node in boundary_geometry
            push!(nodes_boundary, node)
            push_to_boundary_indices!(boundary_indices, boundary_geometry, node, i)
        else
            push!(nodes_inner, node)
        end
    end
    return PointSet(nodes_inner), PointSet(nodes_boundary), boundary_indices
end

function SummationByPartsOperatorsExtra.multidimensional_function_space_operator(basis_functions,
                                                                                 geometry,
                                                                                 sampler,
                                                                                 sampler_boundary,
                                                                                 source;
                                                                                 ellipsoid_lengths = nothing,
                                                                                 kwargs...)
    d = paramdim(geometry)
    nodes, normals, boundary_indices = compute_nodes_normals(geometry, sampler,
                                                             sampler_boundary)
    corners = find_corners(boundary_indices)
    if !isnothing(ellipsoid_lengths)
        @assert length(ellipsoid_lengths)==d "ellipsoid_lengths has to be of length $(d)"
        sparsity_patterns = ntuple(i -> neighborhood_sparsity_pattern(nodes,
                                                                      ellipsoid_lengths[i]),
                                   d)
    else
        sparsity_patterns = nothing
    end

    moments = compute_moments_boundary(basis_functions, geometry)
    vol = Meshes.ustrip(measure(geometry))
    D = multidimensional_function_space_operator(basis_functions, nodes,
                                                 boundary_indices, normals, moments, vol,
                                                 source; corners, sparsity_patterns,
                                                 kwargs...)
    return D
end

end
