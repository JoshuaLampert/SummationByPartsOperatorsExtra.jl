"""
    MultidimensionalLinearAdvectionNonperiodicSemidiscretization(D, a, bc)

A semidiscretization of the linear advection equation
    \$\\partial_t u(x, t) + a\\cdot \\nabla u(x, t) = 0\$
with boundary conditions `bc(x, t)`.

`D` is a multidimensional SBP derivative operator, and `a` is a tuple of the constant coefficients.

The boundary condition is imposed weakly by a simultaneous approximation term at the boundary
nodes `j` with `τ_j < 0`, where `τ = diag(a[1] * B_1 + a[2] * B_2)` and `B_i` are the boundary
mass matrices of `D`. For a boundary node with a single normal `n`, `τ_j < 0` is the usual
inflow condition `a ⋅ n < 0`. At a corner, `τ_j` combines the contributions of both adjacent
faces, and deciding by the sign of `τ_j` rather than by one of the normals is what keeps the
semidiscretization energy stable when one face is an inflow and the other an outflow boundary.
"""
@auto_hash_equals struct MultidimensionalLinearAdvectionNonperiodicSemidiscretization{Dim,
                                                                                      T,
                                                                                      Derivative <:
                                                                                      AbstractDerivativeOperator{T},
                                                                                      BC,
                                                                                      Cache} <:
                         AbstractSemidiscretization
    derivative::Derivative
    a::NTuple{Dim, T}
    bc::BC
    cache::Cache

    function MultidimensionalLinearAdvectionNonperiodicSemidiscretization(derivative::Derivative,
                                                                          a::NTuple{Dim, T},
                                                                          bc::BC) where {
                                                                                         Dim,
                                                                                         T,
                                                                                         Derivative <:
                                                                                         AbstractDerivativeOperator{T},
                                                                                         BC}
        # Only support 2D for now
        D_x = derivative[1]
        D_y = derivative[2]
        A = a[1] * D_x + a[2] * D_y
        invP = inv(mass_matrix(derivative))
        B_x = mass_matrix_boundary(derivative, 1)
        B_y = mass_matrix_boundary(derivative, 2)
        B = invP * (a[1] * B_x + a[2] * B_y)
        # Nodes receiving boundary data: those with τ_j < 0. At a corner `τ_j` sums the
        # contributions of both adjacent faces, so this also decides mixed inflow/outflow
        # corners consistently with the boundary matrix that enters the SBP property.
        tau = diag(a[1] * B_x + a[2] * B_y)
        inflow_indices = findall(<(0), tau)
        tmp1 = zeros(T, size(derivative, 1))
        cache = (; A, B, tmp1, inflow_indices)
        return new{Dim, T, Derivative, BC, typeof(cache)}(derivative, a, bc, cache)
    end
end

function Base.show(io::IO,
                   semi::MultidimensionalLinearAdvectionNonperiodicSemidiscretization)
    if get(io, :compact, false)
        print(io,
              "Semidiscretization of the multidimensional linear advection equation (nonperiodic)")
    else
        println(io, "Semidiscretization of the linear advection equation")
        println(io, "  \$ \\partial_t u(x, t) + a\\cdot \\nabla u(x, t) = 0 \$")
        println(io, "with nonperiodic boundaries using")
        print(io, semi.derivative)
    end
end

# This semidiscretization is based on equation (6.2) in
# Jan Glaubitz, Simon-Christian Klein, Jan Nordström, Philipp Öffner (2023)
# Multi-dimensional summation-by-parts operators for general function spaces: Theory and construction
# with one change at corners: (6.2) decides inflow or outflow per boundary node by its normal, which
# is ambiguous at a corner with two normals. Previously, the last of the two entries of a corner in
# `boundary_indices` won, while `B` contains the contributions of both faces; at a corner adjacent
# to one inflow and one outflow face this made the semidiscretization energy unstable. Here, the
# data is imposed exactly at the nodes with `τ_j < 0`, `τ = diag(a[1] * B_1 + a[2] * B_2)`, which
# gives `d/dt ‖u‖²_P ≤ Σ_{τ_j < 0} |τ_j| g_j²` for any diagonal boundary matrices. `tmp1` holds the
# values entering the boundary terms: the boundary data at those nodes and `u` elsewhere.
function set_bc!(tmp1, u, bc, D, inflow_indices, t)
    copyto!(tmp1, u)
    nodes = grid(D)
    for j in inflow_indices
        tmp1[j] = bc(nodes[j], t)
    end
    return nothing
end

function (disc::MultidimensionalLinearAdvectionNonperiodicSemidiscretization)(du, u, p, t)
    @unpack bc, derivative, cache = disc
    @unpack A, B, tmp1, inflow_indices = cache

    set_bc!(tmp1, u, bc, derivative, inflow_indices, t)
    du .= -A * u + B * (u - tmp1)

    return nothing
end

function analyze_quantities(disc::MultidimensionalLinearAdvectionNonperiodicSemidiscretization,
                            du, u, p, t)
    D = disc.derivative
    P = mass_matrix(D)
    mass = sum(P * u)
    mass_rate = sum(P * du)
    # P * disc.cache.B = a[1] * B_1 + a[2] * B_2
    # tmp1 are the boundary values of u
    mass_rate_boundary = mass_rate + sum(P * disc.cache.B * disc.cache.tmp1)

    energy = 0.5 * sum(P * (u .^ 2)) # = 1/2 ||u||_P^2
    energy_rate = sum(P * (du .* u)) # = 1/2 d/dt||u||_P^2 = u' * P * du
    energy_rate_boundary_dissipation = energy_rate -
                                       0.5 * u' * P * disc.cache.B *
                                       (u - 2 * disc.cache.tmp1)
    # With τ = diag(P * B) = diag(a[1] * B_1 + a[2] * B_2) and `du` from the semidiscretization,
    # energy_rate = 1/2 Σ_j τ_j u_j (u_j - 2 tmp1_j). Since tmp1_j = u_j wherever τ_j ≥ 0,
    # energy_rate_boundary = energy_rate + 1/2 Σ_j τ_j tmp1_j²
    #                      = 1/2 Σ_{τ_j < 0} τ_j (u_j - g_j)² ≤ 0,
    # the dissipation of the boundary terms.
    tmp1 = disc.cache.tmp1
    energy_rate_boundary = energy_rate + 0.5 * dot(tmp1, P * (disc.cache.B * tmp1))
    return [mass, mass_rate, mass_rate_boundary,
            energy, energy_rate, energy_rate_boundary, energy_rate_boundary_dissipation]
end
