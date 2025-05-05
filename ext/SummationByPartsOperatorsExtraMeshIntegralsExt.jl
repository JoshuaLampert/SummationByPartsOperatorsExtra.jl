module SummationByPartsOperatorsExtraMeshIntegralsExt

using SummationByPartsOperatorsExtra: SummationByPartsOperatorsExtra, integrate_boundary

using MeshIntegrals: MeshIntegrals, integral
using Meshes: Meshes, Geometry, Point, Ring, Sphere, boundary, paramdim, to

include("normals.jl")

uto(p::Point) = Meshes.ustrip(to(p))
# if geometry is already the boundary
function SummationByPartsOperatorsExtra.integrate_boundary(func, geometry::Union{Ring, Sphere})
    return Meshes.ustrip(integral(func, geometry, MeshIntegrals.GaussLegendre(100)))
end
function SummationByPartsOperatorsExtra.integrate_boundary(func, geometry::Geometry)
    integrate_boundary(func, boundary(geometry))
end

function SummationByPartsOperatorsExtra.compute_moments_boundary(functions, geometry::Geometry)
    K = length(functions)
    moments = ntuple(paramdim(geometry)) do i
        M = zeros(K, K)
        for k in 1:K
            for l in 1:K
                f = x -> functions[k](uto(x)) * functions[l](uto(x)) *
                         outer_normal(geometry, x)[i]
                M[k, l] = integrate_boundary(f, geometry)
            end
        end
        return M
    end
    return moments
end
end
