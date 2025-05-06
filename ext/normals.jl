using Meshes: Vec, Ngon, Box, Ball, Ring, Segment, segments
using LinearAlgebra: normalize
using Base.ScopedValues: with

function outer_normal(geometry::Union{Ngon, Box, Ball}, point)
    outer_normal(boundary(geometry), point)
end

# This assumes that the vertices are ordered counter-clockwise!
function outer_normal(ring::Ring, point)
    # When sampling points on the boundary, they might not be exactly on the boundary
    with(Meshes.ATOL64 => 1e-8) do
        @assert point in ring "point has to be on the boundary"
        segs = segments(ring)
        for seg in segs
            if point in seg
                return outer_normal(seg, point)
            end
        end
    end
end

# It's hard to say what "outer" means in general for just a segment.
# We assume that the segment is part of the boundary of a polygon (ring)
# and the outer normal is the normal pointing outwards of the polygon.
# Otherwise one might need to take the negative of the normal.
function outer_normal(segment::Segment, point)
    @assert point in segment "point has to be on the boundary"
    x1, x2 = extrema(segment)
    d = x2 - x1
    n = Vec(d[2], -d[1])
    return normalize(n)
end
