@testitem "Upwind operators with given sigma" begin
    using LinearAlgebra: issymmetric, eigvals

    x_L = -0.2
    x_R = 1.42
    source = GlaubitzRanochaWintersSchlottkeLakemperÖffnerGassner2025()
    for compact in (true, false)
        show(IOContext(devnull, :compact => compact), source)
    end
    for N in 2:6
        D = legendre_derivative_operator(x_L, x_R, N)
        sigma = [-1.0]
        D_upw = upwind_operators(D, sigma, source)
        x = grid(D_upw)
        Dp = Matrix(D_upw.plus)
        Dm = Matrix(D_upw.minus)

        M = mass_matrix(D_upw)
        @test M == mass_matrix(D)
        B = mass_matrix_boundary(D_upw)
        @test B == mass_matrix_boundary(D)
        @test isapprox(Dm' * M + M * Dp, B, atol = 1e-13)

        Qp = M * Dp - B / 2
        Qm = M * Dm - B / 2
        S = Qp + Qp'
        @test Qp - Qm ≈ S
        @test Dp ≈ Dm + inv(M) * S
        @test issymmetric(S)
        lambda = eigvals(S)
        @test count(<(-1e-12), lambda) == length(sigma)
        @test count(x -> abs(x) < 1e-12, lambda) == N - length(sigma)
        for i in 1:(N - length(sigma) - 1)
            @test S * x .^ i≈zeros(N) atol=1e-12
        end
        @test !isapprox(S * x .^ (N - length(sigma)), zeros(N), atol = 1e-12)

        # Accuracy order tests
        Km1 = accuracy_order(D_upw.minus)
        for i in 1:Km1
            @test D_upw.minus * x .^ i ≈ i * x .^ (i - 1)
        end
        K = Km1 + 1
        @test !(D_upw.minus * x .^ K ≈ K * x .^ (K - 1))

        Km1 = accuracy_order(D_upw.plus)
        for i in 1:Km1
            @test D_upw.plus * x .^ i ≈ i * x .^ (i - 1)
        end
        K = Km1 + 1
        @test !(D_upw.plus * x .^ K ≈ K * x .^ (K - 1))

        Km1 = accuracy_order(D_upw.central)
        for i in 1:Km1
            @test D_upw.central * x .^ i ≈ i * x .^ (i - 1)
        end
        K = Km1 + 1
        @test !(D_upw.central * x .^ K ≈ K * x .^ (K - 1))
    end
end

@testitem "Upwind operators (accuracy-optimized)" begin
    using LinearAlgebra: issymmetric, eigvals
    import Optim, ForwardDiff

    x_L = -1.0
    x_R = 1.0
    source = GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized()
    for compact in (true, false)
        show(IOContext(devnull, :compact => compact), source)
    end

    basis_functions = [one, identity, exp]
    basis_functions_derivatives = [zero, one, exp]
    K = length(basis_functions)
    test_functions = [x -> x^2]
    test_functions_derivatives = [x -> 2 * x]
    for N in 5:6
        nodes = collect(range(x_L, x_R, length = N))
        D = function_space_operator(basis_functions, nodes,
                                    GlaubitzNordströmÖffner2023())
        additional_functions = [x -> x^j for j in 2:(N - 2)]
        additional_functions_derivatives = [x -> j * x^(j - 1) for j in 2:(N - 2)]
        @test length(additional_functions) == N - K

        D_upw = upwind_operators(D, basis_functions, additional_functions,
                                 test_functions, source)
        x = grid(D_upw)
        Dp = Matrix(D_upw.plus)
        Dm = Matrix(D_upw.minus)

        M = mass_matrix(D_upw)
        @test M == mass_matrix(D)
        B = mass_matrix_boundary(D_upw)
        @test B == mass_matrix_boundary(D)
        @test isapprox(Dm' * M + M * Dp, B, atol = 1e-12)

        Qp = M * Dp - B / 2
        Qm = M * Dm - B / 2
        S = Qp + Qp'
        @test Qp - Qm ≈ S
        @test Dp ≈ Dm + inv(M) * S
        @test issymmetric(S)
        lambda = eigvals(S)
        @test count(<(-1e-10), lambda) >= 1
        @test count(x -> abs(x) < 1e-10, lambda) >= N - K

        # The upwind operators are exact for the basis functions, but not for `additional_functions` and `test_functions`
        for (basis_function, basis_function_derivative) in zip(basis_functions,
                                                               basis_functions_derivatives)
            for D_op in (D_upw.minus, D_upw.plus, D_upw.central)
                @test D_op * basis_function.(x)≈basis_function_derivative.(x) atol=1e-11
            end
            @test S * basis_function.(x)≈zeros(N) atol=1e-11
        end
        for (additional_functions, additional_functions_derivatives) in zip(additional_functions,
                                                                            additional_functions_derivatives)
            for D_op in (D_upw.minus, D_upw.plus, D_upw.central)
                @test !(D_op * additional_functions.(x) ≈
                        additional_functions_derivatives.(x))
            end
        end
        for (test_function, test_function_derivative) in zip(test_functions,
                                                             test_functions_derivatives)
            for D_op in (D_upw.minus, D_upw.plus, D_upw.central)
                @test !(D_op * test_function.(x) ≈ test_function_derivative.(x))
            end
        end
    end

    # The number of `additional_functions` must be equal to `N - K`
    let N = 5
        nodes = collect(range(x_L, x_R, length = N))
        D = function_space_operator(basis_functions, nodes,
                                    GlaubitzNordströmÖffner2023())
        @test_throws ArgumentError upwind_operators(D, basis_functions, [x -> x^2],
                                                    test_functions, source)
    end

    # The initial guess for the optimization can be set with the `sigma0` keyword
    let N = 5
        nodes = collect(range(x_L, x_R, length = N))
        D = function_space_operator(basis_functions, nodes,
                                    GlaubitzNordströmÖffner2023())
        additional_functions = [x -> x^2, x -> x^3]
        D_upw = upwind_operators(D, basis_functions, additional_functions,
                                 test_functions, source; sigma0 = [-2.0, -1.0])
        @test eltype(D_upw) == Float64
    end
end

@testitem "Upwind operators (accuracy-optimized) are degenerate" begin
    import Optim, ForwardDiff

    # This reproduces the negative result documented for
    # `GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized`: the accuracy objective
    # cannot select a dissipation matrix. It drives the eigenvalues of all enrichment modes
    # that overlap a test function to zero and leaves the remaining ones at their initial
    # guess, since those are invisible to the objective.
    source = GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized()
    basis_functions = [one, identity, exp]
    K = length(basis_functions)
    N = 6
    nodes = collect(range(-1.0, 1.0, length = N))
    D = function_space_operator(basis_functions, nodes, GlaubitzNordströmÖffner2023())

    additional_functions = [x -> x^2, x -> x^3, x -> x^4]
    @test length(additional_functions) == N - K
    # `x^2` and `x^3` are contained in the span of the enrichment modes 1 and 2, but are
    # orthogonal to the enrichment mode 3
    test_functions = [x -> x^2, x -> x^3]
    sigma0 = [-1.0, -1.0, -1.0]

    V = SummationByPartsOperatorsExtra.orthonormal_vandermonde([basis_functions;
                                                                additional_functions],
                                                               nodes)
    sigma = SummationByPartsOperatorsExtra.compute_dissipation_eigenvalues(D, V,
                                                                           test_functions,
                                                                           K,
                                                                           source; sigma0)
    # The modes seen by the objective are driven to zero ...
    @test isapprox(sigma[1], 0.0, atol = 1e-8)
    @test isapprox(sigma[2], 0.0, atol = 1e-8)
    # ... while the mode invisible to the objective simply keeps its initial guess
    @test isapprox(sigma[3], sigma0[3], atol = 1e-9)

    # The objective is driven to (almost) zero, i.e. this is a global minimizer and not a
    # failure of the optimizer
    D_upw = upwind_operators(D, basis_functions, additional_functions, test_functions,
                             source; sigma0)
    x = grid(D_upw)
    objective = sum(sum(abs2, D_op * f.(x) - f_x.(x))
                    for D_op in (D_upw.minus, D_upw.plus),
                        (f, f_x) in zip(test_functions, [x -> 2 * x, x -> 3 * x^2]))
    objective_central = 2 * sum(sum(abs2, Matrix(D) * f.(x) - f_x.(x))
                            for (f, f_x) in zip(test_functions,
                                                [x -> 2 * x, x -> 3 * x^2]))
    @test isapprox(objective, objective_central, rtol = 1e-8)

    # Changing the initial guess changes the resulting operators, which is exactly the
    # artifact described in the docstring
    D_upw2 = upwind_operators(D, basis_functions, additional_functions, test_functions,
                              source; sigma0 = [-1.0, -1.0, -2.0])
    @test !isapprox(Matrix(D_upw.plus), Matrix(D_upw2.plus))
end
