# These functions are defined in the Makie.jl extension SummationByPartsOperatorsExtraMeshesMakieExt
"""
    plot_nodes(nodes_inner, nodes_boundary; corners = nothing, kwargs...)
    plot_nodes(nodes, boundary_indices::Vector{Int}; corner_indices = nothing,
               kwargs...)
    plot_nodes(D; kwargs...)

Plot the nodes of a multidimensional derivative operator `D`. The interior nodes `nodes_inner` are plotted and
the boundary nodes `nodes_boundary` are plotted in different colors. If `corner_indices` are provided, the corners are also plotted
in a different color. Additional keyword arguments are passed to `viz` from Meshes.jl, see [the documentation](https://juliageometry.github.io/MeshesDocs/stable/visualization/).
The function returns the current figure.
"""
function plot_nodes end

"""
    plot_normals(nodes_boundary, normals; kwargs...)
    plot_normals(D; kwargs...)

Plot the normals of a multidimensional derivative operator `D`. The boundary nodes `nodes_boundary` are plotted and
the normals are plotted as arrows. Additional keyword arguments are passed to `viz` from Meshes.jl, see [the documentation](https://juliageometry.github.io/MeshesDocs/stable/visualization/).
The function returns the current figure.
"""
function plot_normals end

"""
    plot_sparsity_pattern(sparsity_pattern, nodes, node_index)

Plot the `sparsity_pattern`, which is a boolean matrix of length `length(nodes) x length(nodes)`, where `nodes` is a
set of nodes. The `node_index` is the index of the node in `nodes` that is used to plot the sparsity pattern around that node,
i.e., it will be plotted as red, while all neighboring nodes according to the `sparsity_pattern` are plotted as green dots and
the remaining nodes are plotted as blue dots.
"""
function plot_sparsity_pattern end
