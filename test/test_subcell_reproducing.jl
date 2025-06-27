@testsnippet ReproducingSubcell1D begin
    using Optim: Optim
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
        source = GlaubitzLampertNordströmWinters2025()
        for p in 1:5
            n = p + 1  # Number of nodes
            basis_left = basis_left_type(p)
            basis_right = basis_right_type(p)
            basis = [x -> x^(i - 1) for i in 1:n]
            # Map the reference nodes to the left part and
            # right parts, respectively
            nodes = [
                linear_map.(grid(basis_left), a, b, x_L, x_M)...,
                linear_map.(grid(basis_right), a, b, x_M, x_R)...
            ]
            D = subcell_operator(basis, nodes, x_M, source;
                                 options = Optim.Options(; iterations = 10000,
                                                         g_tol = 1e-16))
            D_coupled = couple_subcell(basis_left, basis_right, x_L, x_M, x_R)

            # Test if all three options (optimization, directly the bases, and the coupled bases)
            # give the same result
            weights_D = diag(mass_matrix(D))
            @test all(isapprox.(jac_left * weights_D[1:n],
                                diag(mass_matrix(basis_left)), atol = atol))
            @test all(isapprox.(jac_right * weights_D[(n + 1):end],
                                diag(mass_matrix(basis_right)), atol = atol))
            @test all(isapprox.(weights_D, diag(mass_matrix(D_coupled)), atol = atol))

            D_D = derivative_matrix(D)
            @test all(isapprox.(1 / jac_left * D_D[1:n, 1:n],
                                derivative_matrix(basis_left), atol = atol))
            @test all(isapprox.(1 / jac_right * D_D[(n + 1):end, (n + 1):end],
                                derivative_matrix(basis_right), atol = atol))
            @test all(isapprox.(D_D, derivative_matrix(D_coupled), atol = atol))

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
