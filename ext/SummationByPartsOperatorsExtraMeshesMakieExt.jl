module SummationByPartsOperatorsExtraMeshesMakieExt

using LinearAlgebra: Symmetric
using Meshes: Meshes, PointSet, coords, viz, viz!
import Makie
using SummationByPartsOperatorsExtra: SummationByPartsOperatorsExtra, grid,
                                      boundary_indices, restrict_boundary

function SummationByPartsOperatorsExtra.plot_nodes(nodes_inner, nodes_boundary;
                                                   corners = nothing, kwargs...)
    viz(nodes_inner; color = :blue, kwargs...)
    viz!(nodes_boundary; color = :green, kwargs...)
    if !isnothing(corners)
        viz!(corners; color = :red, kwargs...)
    end
    Makie.current_figure()
end

function SummationByPartsOperatorsExtra.plot_nodes(nodes, boundary_indices::Vector{Int};
                                                   corner_indices = nothing,
                                                   kwargs...)
    nodes_inner = PointSet(nodes[setdiff(1:end, boundary_indices)])
    nodes_boundary = PointSet(nodes[boundary_indices])
    if !isnothing(corner_indices)
        corners = PointSet(nodes[boundary_indices[corner_indices]])
    else
        corners = nothing
    end
    SummationByPartsOperatorsExtra.plot_nodes(nodes_inner, nodes_boundary; corners,
                                              kwargs...)
end

function SummationByPartsOperatorsExtra.plot_nodes(D; kwargs...)
    SummationByPartsOperatorsExtra.plot_nodes(grid(D), boundary_indices(D); kwargs...)
end

function SummationByPartsOperatorsExtra.plot_normals(nodes_boundary, normals; kwargs...)
    nodes_boundary = PointSet(nodes_boundary)
    N_boundary = length(nodes_boundary)
    x_vals = [Meshes.ustrip(coords(nodes_boundary[i]).x) for i in 1:N_boundary]
    y_vals = [Meshes.ustrip(coords(nodes_boundary[i]).y) for i in 1:N_boundary]
    u = [normals[i].x for i in 1:N_boundary]
    v = [normals[i].y for i in 1:N_boundary]
    Makie.arrows2d(x_vals, y_vals, u, v)
    viz!(nodes_boundary; color = :green, kwargs...)
    Makie.current_figure()
end

function SummationByPartsOperatorsExtra.plot_normals(D; kwargs...)
    nodes = grid(D)
    nodes_boundary = PointSet(Tuple.(restrict_boundary(nodes, D)))
    normals = D.normals
    SummationByPartsOperatorsExtra.plot_normals(nodes_boundary, normals; kwargs...)
end

function SummationByPartsOperatorsExtra.plot_sparsity_pattern(sparsity_pattern, nodes, node_index)
    sparsity_pattern = Symmetric(sparsity_pattern)
    nodes = PointSet(nodes)
    viz(nodes[node_index], color=:red, pointsize=10)
    for i in eachindex(nodes)
        if i == node_index
            continue
        end
        if sparsity_pattern[node_index, i]
            viz!(nodes[i], color=:green, pointsize=10)
        else
            viz!(nodes[i], color=:blue, pointsize=10)
        end
    end
    Makie.current_figure()
end

end
