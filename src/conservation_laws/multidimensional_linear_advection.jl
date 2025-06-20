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
        new{Dim, T, Derivative, BC, typeof(cache)}(derivative, a, bc, cache)
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

    nothing
end

function analyze_quantities(disc::MultidimensionalLinearAdvectionNonperiodicSemidiscretization,
                            du, u, p, t)
    D = disc.derivative
    P = mass_matrix(D)
    m = sum(P * du)
    # P * disc.cache.B = a[1] * B_1 + a[2] * B_2
    # tmp1 are the boundary values of u
    m_boundary = m + sum(P * disc.cache.B * disc.cache.tmp1)

    e = 2.0 * sum(P * (du .* u)) # = ||u||_P^2 = 2.0 * u' * P * du
    # This should compute the same as below, but we need the explicit loop for `e_boundary` anyway
    # e_boundary_dissipation = e - u' * P * disc.cache.B * (u - 2 * disc.cache.tmp1)
    e_boundary = e
    e_boundary_dissipation = e
    for (i, node) in enumerate(restrict_boundary(grid(D), D))
        j = D.boundary_indices[i]
        normal = D.normals[i]
        an = dot(normal, disc.a)
        tau = SummationByPartsOperators.get_weight_boundary(D, i) * an
        e_boundary += tau * disc.cache.tmp1[j]^2
        e_boundary_dissipation += tau * disc.cache.tmp1[j]^2
        if an < 0 # inflow
            e_boundary_dissipation -= tau * (u[j] - disc.bc(node, t))^2
        end
    end
    return [m, m_boundary, e, e_boundary, e_boundary_dissipation]
end
