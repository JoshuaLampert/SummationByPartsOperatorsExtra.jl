@testsnippet ReproducingSubcell1D begin
    import Optim, ForwardDiff
    using LinearAlgebra: diag
    using PolynomialBases: GaussRadauLeft, GaussRadauRight

    # Map interval [a, b] (reference interval) to [c, d] (physical interval)
    linear_map(x, a, b, c, d) = c + (x - a) / (b - a) * (d - c)

    function test_reproducing(basis_left_type, basis_right_type, x_M; atol = 1e-12)
        # Nodes from PolynomialBases.jl are always defined on [-1, 1]
        a = -1.0
        b = 1.0
        # Construct `D` on [x_L, x_M] with subcells [x_L, x_M] and [x_M, x_R]
        x_L = -1.0
        x_R = 1.0
        jac_left = (b - a) / (x_M - x_L)
        jac_right = (b - a) / (x_R - x_M)
        source = GlaubitzLampertWintersNordström2025()
        for p in 1:5
            n = p + 1  # Number of nodes
            basis_left = basis_left_type(p)
            basis_right = basis_right_type(p)

            D_left = polynomialbases_derivative_operator(basis_left_type, x_L, x_M, n)
            D_right = polynomialbases_derivative_operator(basis_right_type, x_M, x_R, n)
            D_coupled = couple_subcell(D_left, D_right, x_M)
            nodes = grid(D_coupled)
            basis = [x -> x^(i - 1) for i in 1:n]

            D = subcell_operator(basis, nodes, x_M, source;
                                 options = Optim.Options(; iterations = 10000,
                                                         g_tol = 1e-16),
                                 opt_alg = Optim.BFGS())

            # Test if all three options (optimization, directly the bases, and the coupled bases)
            # give the same result
            weights_D = diag(mass_matrix(D))
            @test all(isapprox.(jac_left * weights_D[1:n],
                                diag(mass_matrix(basis_left)), atol = atol))
            @test all(isapprox.(jac_right * weights_D[(n + 1):end],
                                diag(mass_matrix(basis_right)), atol = atol))
            @test all(isapprox.(weights_D, diag(mass_matrix(D_coupled)), atol = atol))

            D_D = Matrix(D)
            @test all(isapprox.(1 / jac_left * D_D[1:n, 1:n],
                                Matrix(basis_left), atol = atol))
            @test all(isapprox.(1 / jac_right * D_D[(n + 1):end, (n + 1):end],
                                Matrix(basis_right), atol = atol))
            @test all(isapprox.(D_D, Matrix(D_coupled), atol = atol))

            B_D = mass_matrix_boundary(D)
            @test all(isapprox.(B_D[1:n, 1:n],
                                mass_matrix_boundary(basis_left), atol = atol))
            @test all(isapprox.(B_D[(n + 1):end, (n + 1):end],
                                mass_matrix_boundary(basis_right), atol = atol))
            @test all(isapprox.(B_D, mass_matrix_boundary(D_coupled), atol = atol))
        end
    end
end

@testitem "Reproducing GaussRadauLeft and GaussRadauRight" setup=[ReproducingSubcell1D] begin
    test_reproducing(GaussRadauLeft, GaussRadauRight, 0.1; atol = 1e-11)
end

@testitem "Reproducing LobattoLegendre and GaussRadauRight" setup=[ReproducingSubcell1D] begin
    test_reproducing(LobattoLegendre, GaussRadauRight, 0.0; atol = 1e-10)
end
