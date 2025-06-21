@testsnippet ReproducingSubcell1D begin
    using Optim: Optim
    using LinearAlgebra: diag
    using PolynomialBases: GaussRadauLeft, GaussRadauRight

    # Map interval [a, b] (reference interval) to [c, d] (physical interval)
    linear_map(x, a, b, c, d) = c + (x - a) / (b - a) * (d - c)
end

@testitem "Reproducing Gauss-Radau" setup=[ReproducingSubcell1D] begin
    # Gauss-Radau nodes are defined on [-1, 1]
    a = -1.0
    b = 1.0
    # Construct `D` on [c, d] with subcells [c, x_M] and [x_M, d]
    c = -1.0
    d = 1.0
    x_M = 0.1
    jac_left = (b - a) / (x_M - c)
    jac_right = (b - a) / (d - x_M)
    source = GlaubitzLampertNordströmWinters2025()
    for p in 1:5
        n = p + 1  # Number of nodes in the Gauss-Radau basis
        gr_basis_left = GaussRadauLeft(p)
        gr_basis_right = GaussRadauRight(p)
        basis = [x -> x^(i - 1) for i in 1:n]
        # Map the reference nodes to the left part and use symmetry
        # to construct the right part of the interval
        nodes = [
            linear_map.(grid(gr_basis_left), a, b, c, x_M)...,
            linear_map.(grid(gr_basis_right), a, b, x_M, d)...
        ]
        D = subcell_operator(basis, nodes, x_M, source;
                             options = Optim.Options(; iterations = 10000,
                                                     g_tol = 1e-16))

        weights_D = diag(mass_matrix(D))
        @test all(jac_left * weights_D[1:n] .≈ diag(mass_matrix(gr_basis_left)))
        @test all(jac_right * weights_D[(n + 1):end] .≈ diag(mass_matrix(gr_basis_right)))

        D_D = derivative_matrix(D)
        @test all(1 / jac_left * D_D[1:n, 1:n] .≈ derivative_matrix(gr_basis_left))
        @test all(1 / jac_right * D_D[(n + 1):end, (n + 1):end] .≈
                  derivative_matrix(gr_basis_right))

        B_D = mass_matrix_boundary(D)
        @test all(B_D[1:n, 1:n] .≈ mass_matrix_boundary(gr_basis_left))
        @test all(B_D[(n + 1):end, (n + 1):end] .≈ mass_matrix_boundary(gr_basis_right))
    end
end
