@testsnippet SubCell begin
    import Optim
    using LinearAlgebra: issymmetric, dot
end

@testitem "Sub-cell operators" setup=[SubCell] begin
    nodes = collect(-1.1:0.1:0.4)
    N = length(nodes)
    x_L = first(nodes)
    x_M = -0.42
    x_R = last(nodes)
    basis_functions = [one, identity, x -> 0.5 * x^2]
    source = GlaubitzLampertWintersNordström2025()
    for compact in (true, false)
        show(IOContext(devnull, :compact => compact), source)
    end
    K = length(basis_functions)
    for M_local_approximation in ([0, 0], [K, K])
        @testset "M_local_approximation = $M_local_approximation" begin
            Dop = @test_nowarn subcell_operator(basis_functions, nodes, x_M, source;
                                                M_local_approximation,
                                                options = Optim.Options(;
                                                                        iterations = 10000,
                                                                        g_tol = 1e-16))
            for compact in (true, false)
                show(IOContext(devnull, :compact => compact), Dop)
            end

            @test_throws ArgumentError subcell_operator(basis_functions, nodes, x_M, source;
                                                        derivative_order = 2)
            @test_throws ArgumentError subcell_operator(basis_functions, nodes, x_L - 1.0,
                                                        source)
            @test_throws ArgumentError subcell_operator(basis_functions, nodes, x_M, source;
                                                        x0 = zeros(3))

            @test !issymmetric(Dop)
            @test source_of_coefficients(Dop) == source
            @test SummationByPartsOperators.lower_bandwidth(Dop) == length(nodes) - 1
            @test SummationByPartsOperators.upper_bandwidth(Dop) == length(nodes) - 1
            @test derivative_order(Dop) == 1
            @test accuracy_order(Dop) == 0

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
            @test left_boundary_weight(Dop) == M[1, 1] == M_L[1, 1]
            @test right_boundary_weight(Dop) == M[end, end] == M_R[end, end]

            # integration
            u = sin.(nodes)
            u_copy = copy(u)
            @test integrate(cos, u, Dop) == sum(M * cos.(u))
            SummationByPartsOperators.scale_by_mass_matrix!(u, Dop)
            @test_throws DimensionMismatch SummationByPartsOperators.scale_by_mass_matrix!(@view(u[(begin + 1):(end - 1)]),
                                                                                           Dop)

            @test u ≈ M * u_copy
            SummationByPartsOperators.scale_by_inverse_mass_matrix!(u, Dop)
            @test_throws DimensionMismatch SummationByPartsOperators.scale_by_inverse_mass_matrix!(@view(u[(begin + 1):(end - 1)]),
                                                                                                   Dop)
            @test u ≈ u_copy

            # boundary mass matrices
            B = mass_matrix_boundary(Dop)
            B_L = mass_matrix_boundary_left(Dop)
            B_R = mass_matrix_boundary_right(Dop)
            @test B ≈ B_L + B_R

            for f in basis_functions
                ff = f.(nodes)
                for g in basis_functions
                    gg = g.(nodes)
                    @test isapprox(ff' * B_L * gg, f(x_M) * g(x_M) - f(x_L) * g(x_L),
                                   atol = 1e-14)
                    @test isapprox(ff' * B_R * gg, f(x_R) * g(x_R) - f(x_M) * g(x_M),
                                   atol = 1e-14)
                    @test isapprox(ff' * B * gg, f(x_R) * g(x_R) - f(x_L) * g(x_L),
                                   atol = 1e-14)
                end
            end

            # projections
            e_L = left_projection_left(Dop)
            e_M_L = left_projection_right(Dop)
            e_M_R = right_projection_left(Dop)
            e_R = right_projection_right(Dop)
            @test isapprox(dot(e_L, u), u[begin])
            @test isapprox(dot(e_R, u), u[end])
            @test B_L ≈ e_M_L * e_M_L' - e_L * e_L'
            @test B_R ≈ e_R * e_R' - e_M_R * e_M_R'

            # exactness of derivative operator
            fs = [f.(nodes) for f in basis_functions]
            basis_functions_derivatives = [zero, one, identity]
            fs_derivatives = [f.(nodes) for f in basis_functions_derivatives]
            for (f, f_derivative) in zip(fs, fs_derivatives)
                @test all(isapprox.(Dop * f, f_derivative; atol = 1e-12))
            end

            # SBP properties
            Q_L = Dop.Q_left
            Q_R = Dop.Q_right
            @test Q_L + Q_L' ≈ B_L
            @test Q_R + Q_R' ≈ B_R
            D = Matrix(Dop)
            @test derivative_matrix(Dop) == D
            Q = M * D
            @test Q + Q' ≈ B
            @test Matrix(Dop) == D

            # consistency between matrices
            @test Q ≈ Q_L + Q_R
            @test M_L * D ≈ Q_L
            @test M_R * D ≈ Q_R
        end
    end
end

@testitem "Couple SBP operators to sub-cell operators" setup=[SubCell] begin
    basis_functions = [one, identity, x -> 0.5 * x^2]
    x_L = -2.3
    x_M = -1.0
    x_R = 0.4
    D1 = legendre_derivative_operator(x_L, x_M, 3)
    D2 = legendre_derivative_operator(x_M, x_R, 3)
    Dop = @test_nowarn couple_subcell(D1, D2, x_M)
    @test_throws ArgumentError couple_subcell(D1, D2, -2.0)
    @test_throws ArgumentError couple_subcell(D1, D2, 0.0)
    nodes = grid(Dop)

    @test !issymmetric(Dop)
    @test SummationByPartsOperators.lower_bandwidth(Dop) == length(nodes) - 1
    @test SummationByPartsOperators.upper_bandwidth(Dop) == length(nodes) - 1
    @test derivative_order(Dop) == 1
    @test accuracy_order(Dop) == 2

    # grids
    @test grid(Dop) ≈ nodes
    @test all(grid_left(Dop) .<= x_M)
    @test all(grid_right(Dop) .>= x_M)

    # mass matrices
    M = mass_matrix(Dop)
    M_L = mass_matrix_left(Dop)
    M_R = mass_matrix_right(Dop)
    @test M ≈ M_L + M_R
    @test left_boundary_weight(Dop) == M[1, 1] == M_L[1, 1]
    @test right_boundary_weight(Dop) == M[end, end] == M_R[end, end]

    # integration
    u = sin.(nodes)
    u_copy = copy(u)
    @test integrate(cos, u, Dop) == sum(M * cos.(u))
    SummationByPartsOperators.scale_by_mass_matrix!(u, Dop)
    @test_throws DimensionMismatch SummationByPartsOperators.scale_by_mass_matrix!(@view(u[(begin + 1):(end - 1)]),
                                                                                   Dop)

    @test u ≈ M * u_copy
    SummationByPartsOperators.scale_by_inverse_mass_matrix!(u, Dop)
    @test_throws DimensionMismatch SummationByPartsOperators.scale_by_inverse_mass_matrix!(@view(u[(begin + 1):(end - 1)]),
                                                                                           Dop)
    @test u ≈ u_copy

    # boundary mass matrices
    B = mass_matrix_boundary(Dop)
    B_L = mass_matrix_boundary_left(Dop)
    B_R = mass_matrix_boundary_right(Dop)
    @test B ≈ B_L + B_R

    for f in basis_functions
        ff = f.(nodes)
        for g in basis_functions
            gg = g.(nodes)
            @test isapprox(ff' * B_L * gg, f(x_M) * g(x_M) - f(x_L) * g(x_L),
                           atol = 1e-14)
            @test isapprox(ff' * B_R * gg, f(x_R) * g(x_R) - f(x_M) * g(x_M),
                           atol = 1e-14)
            @test isapprox(ff' * B * gg, f(x_R) * g(x_R) - f(x_L) * g(x_L),
                           atol = 1e-14)
        end
    end

    # projections
    e_L = left_projection_left(Dop)
    e_M_L = left_projection_right(Dop)
    e_M_R = right_projection_left(Dop)
    e_R = right_projection_right(Dop)
    @test isapprox(dot(e_L, u), u[begin])
    @test isapprox(dot(e_R, u), u[end])
    @test B_L ≈ e_M_L * e_M_L' - e_L * e_L'
    @test B_R ≈ e_R * e_R' - e_M_R * e_M_R'

    # exactness of derivative operator
    fs = [f.(nodes) for f in basis_functions]
    basis_functions_derivatives = [zero, one, identity]
    fs_derivatives = [f.(nodes) for f in basis_functions_derivatives]
    for (f, f_derivative) in zip(fs, fs_derivatives)
        @test all(isapprox.(Dop * f, f_derivative; atol = 1e-12))
    end

    # SBP properties
    Q_L = Dop.Q_left
    Q_R = Dop.Q_right
    @test Q_L + Q_L' ≈ B_L
    @test Q_R + Q_R' ≈ B_R
    D = Matrix(Dop)
    @test derivative_matrix(Dop) == D
    Q = M * D
    @test Q + Q' ≈ B
    @test Matrix(Dop) == D

    # consistency between matrices
    @test Q ≈ Q_L + Q_R
    @test M_L * D ≈ Q_L
    @test M_R * D ≈ Q_R
end

@testitem "Couple polynomial bases operators to sub-cell operators" setup=[SubCell] begin
    p = 4
    basis_functions = [x -> x^(i - 1) for i in 1:(p + 1)]
    x_L = -2.3
    x_M = -1.1
    x_R = 0.4
    D1 = polynomialbases_derivative_operator(LobattoLegendre, x_L, x_M, p + 1)
    D2 = polynomialbases_derivative_operator(GaussRadauRight, x_M, x_R, p + 1)
    Dop = @test_nowarn couple_subcell(D1, D2, x_M)
    nodes = grid(Dop)

    @test !issymmetric(Dop)
    @test SummationByPartsOperators.lower_bandwidth(Dop) == length(nodes) - 1
    @test SummationByPartsOperators.upper_bandwidth(Dop) == length(nodes) - 1
    @test derivative_order(Dop) == 1
    @test accuracy_order(Dop) == 4

    # grids
    @test grid(Dop) ≈ nodes
    @test all(grid_left(Dop) .<= x_M)
    @test all(grid_right(Dop) .>= x_M)

    # mass matrices
    M = mass_matrix(Dop)
    M_L = mass_matrix_left(Dop)
    M_R = mass_matrix_right(Dop)
    @test M ≈ M_L + M_R
    @test left_boundary_weight(Dop) == M[1, 1] == M_L[1, 1]
    @test right_boundary_weight(Dop) == M[end, end] == M_R[end, end]

    # integration
    u = sin.(nodes)
    u_copy = copy(u)
    @test integrate(cos, u, Dop) == sum(M * cos.(u))
    SummationByPartsOperators.scale_by_mass_matrix!(u, Dop)
    @test_throws DimensionMismatch SummationByPartsOperators.scale_by_mass_matrix!(@view(u[(begin + 1):(end - 1)]),
                                                                                   Dop)

    @test u ≈ M * u_copy
    SummationByPartsOperators.scale_by_inverse_mass_matrix!(u, Dop)
    @test_throws DimensionMismatch SummationByPartsOperators.scale_by_inverse_mass_matrix!(@view(u[(begin + 1):(end - 1)]),
                                                                                           Dop)
    @test u ≈ u_copy

    # boundary mass matrices
    B = mass_matrix_boundary(Dop)
    B_L = mass_matrix_boundary_left(Dop)
    B_R = mass_matrix_boundary_right(Dop)
    @test B ≈ B_L + B_R

    for f in basis_functions
        ff = f.(nodes)
        for g in basis_functions
            gg = g.(nodes)
            @test isapprox(ff' * B_L * gg, f(x_M) * g(x_M) - f(x_L) * g(x_L),
                           atol = 1e-14)
            @test isapprox(ff' * B_R * gg, f(x_R) * g(x_R) - f(x_M) * g(x_M),
                           atol = 1e-14)
            @test isapprox(ff' * B * gg, f(x_R) * g(x_R) - f(x_L) * g(x_L),
                           atol = 1e-12)
        end
    end

    # projections
    e_L = left_projection_left(Dop)
    e_M_L = left_projection_right(Dop)
    e_M_R = right_projection_left(Dop)
    e_R = right_projection_right(Dop)
    @test isapprox(dot(e_L, u), u[begin])
    @test isapprox(dot(e_R, u), u[end])
    @test B_L ≈ e_M_L * e_M_L' - e_L * e_L'
    @test B_R ≈ e_R * e_R' - e_M_R * e_M_R'

    # exactness of derivative operator
    fs = [f.(nodes) for f in basis_functions]
    basis_functions_derivatives = [zero; [x -> i * x^(i - 1) for i in 1:p]...]
    fs_derivatives = [f.(nodes) for f in basis_functions_derivatives]
    for (f, f_derivative) in zip(fs, fs_derivatives)
        @test all(isapprox.(Dop * f, f_derivative; atol = 1e-12))
    end

    # SBP properties
    Q_L = Dop.Q_left
    Q_R = Dop.Q_right
    @test Q_L + Q_L' ≈ B_L
    @test Q_R + Q_R' ≈ B_R
    D = Matrix(Dop)
    @test derivative_matrix(Dop) == D
    Q = M * D
    @test Q + Q' ≈ B
    @test Matrix(Dop) == D

    # consistency between matrices
    @test Q ≈ Q_L + Q_R
    @test M_L * D ≈ Q_L
    @test M_R * D ≈ Q_R
end
