@testsnippet SubCell begin
    import Optim
end

@testitem "Sub-cell operators" setup=[SubCell] begin
    nodes = collect(-1.1:0.1:0.4)
    N = length(nodes)
    x_L = first(nodes)
    x_M = -0.52
    x_R = last(nodes)
    basis_functions = [one, identity, exp]
    source = GlaubitzLampertNordströmWinters2025()
    for compact in (true, false)
        show(IOContext(devnull, :compact => compact), source)
    end
    Dop = @test_nowarn subcell_operator(basis_functions, nodes, x_M, source)
    for compact in (true, false)
        show(IOContext(devnull, :compact => compact), Dop)
    end

    # grids
    @test grid(Dop) ≈ nodes
    @test all(grid_left(Dop) .<= x_M)
    @test all(grid_right(Dop) .>= x_M)
    @test length(grid_left(Dop)) + length(grid_right(Dop)) == N # x_M is not in the grid

    # mass matrices
    M = mass_matrix(Dop)
    M_L = mass_matrix_left(Dop)
    M_R = mass_matrix_right(Dop)
    @test M ≈ M_L + M_R

    # boundary mass matrices
    B = mass_matrix_boundary(Dop)
    B_L = mass_matrix_boundary_left(Dop)
    B_R = mass_matrix_boundary_right(Dop)
    @test B ≈ B_L + B_R

    for f in basis_functions
        ff = f.(nodes)
        for g in basis_functions
            gg = g.(nodes)
            @test isapprox(ff' * B_L * gg, f(x_M) * g(x_M) - f(x_L) * g(x_L), atol = 1e-14)
            @test isapprox(ff' * B_R * gg, f(x_R) * g(x_R) - f(x_M) * g(x_M), atol = 1e-14)
            @test isapprox(ff' * B * gg, f(x_R) * g(x_R) - f(x_L) * g(x_L), atol = 1e-14)
        end
    end

    # exactness of derivative operator
    # @test all(isapprox.(fs[1], zeros(N); atol = 1e-13))
    # @test fs[2] ≈ ones(N)
    # @test fs[3] ≈ exp.(nodes)

    # SBP properties
    Q_L = Dop.Q_left
    Q_R = Dop.Q_right
    # @test Q_L + Q_L' ≈ B_L
    # @test Q_R + Q_R' ≈ B_R
    D = derivative_matrix(Dop)
    Q = M * D
    # @test Q + Q' ≈ B

    # consistency between matrices
    # @test Q ≈ Q_L + Q_R
    # @test M_L * D ≈ Q_L
    # @test M_R * D ≈ Q_R
end
