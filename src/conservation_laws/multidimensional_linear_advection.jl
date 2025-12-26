"""
    MultidimensionalLinearAdvectionNonperiodicSemidiscretization(D, a, bc)

A semidiscretization of the linear advection equation
    \$\\partial_t u(x, t) + a\\cdot \\nabla u(x, t) = 0\$
with boundary conditions `bc(x, t)`.

`D` is a multidimensional SBP derivative operator, and `a` is a tuple of the constant coefficients.
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
        tmp1 = zeros(T, size(derivative, 1))
        cache = (; A, B, tmp1)
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
function set_bc!(tmp1, u, a, bc, D, t)
    fill!(tmp1, zero(eltype(tmp1)))
    for (i, node) in enumerate(restrict_boundary(grid(D), D))
        j = D.boundary_indices[i]
        normal = D.normals[i]
        if dot(normal, a) < 0 # inflow
            tmp1[j] = bc(node, t)
        else # outflow
            tmp1[j] = u[j]
        end
    end
end

function (disc::MultidimensionalLinearAdvectionNonperiodicSemidiscretization)(du, u, p, t)
    @unpack a, bc, derivative, cache = disc
    @unpack A, B, tmp1 = cache

    set_bc!(tmp1, u, a, bc, derivative, t)
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
    energy_rate_boundary = energy_rate
    for (i, node) in enumerate(restrict_boundary(grid(D), D))
        j = D.boundary_indices[i]
        normal = D.normals[i]
        an = dot(normal, disc.a)
        tau = SummationByPartsOperators.get_weight_boundary(D, i) * an
        energy_rate_boundary += 0.5 * tau * disc.cache.tmp1[j]^2
    end
    return [mass, mass_rate, mass_rate_boundary,
            energy, energy_rate, energy_rate_boundary, energy_rate_boundary_dissipation]
end
