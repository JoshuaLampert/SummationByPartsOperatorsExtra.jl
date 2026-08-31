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

        # `dissipation_matrix` gives the same `S` on its own
        @test isapprox(Matrix(dissipation_matrix(sigma, D, source)), S, atol = 1e-13)

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
        enrichment_functions = [x -> x^j for j in 2:(N - 2)]
        enrichment_functions_derivatives = [x -> j * x^(j - 1) for j in 2:(N - 2)]
        @test length(enrichment_functions) == N - K

        D_upw = upwind_operators(D, basis_functions, source;
                                 test_functions, enrichment_functions)
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

        # The upwind operators are exact for the basis functions, but not for the
        # `enrichment_functions` and `test_functions`
        for (basis_function, basis_function_derivative) in zip(basis_functions,
                                                               basis_functions_derivatives)
            for D_op in (D_upw.minus, D_upw.plus, D_upw.central)
                @test D_op * basis_function.(x)≈basis_function_derivative.(x) atol=1e-11
            end
            @test S * basis_function.(x)≈zeros(N) atol=1e-11
        end
        for (enrichment_function, enrichment_function_derivative) in zip(enrichment_functions,
                                                                         enrichment_functions_derivatives)
            for D_op in (D_upw.minus, D_upw.plus, D_upw.central)
                @test !(D_op * enrichment_function.(x) ≈
                        enrichment_function_derivative.(x))
            end
        end
        for (test_function, test_function_derivative) in zip(test_functions,
                                                             test_functions_derivatives)
            for D_op in (D_upw.minus, D_upw.plus, D_upw.central)
                @test !(D_op * test_function.(x) ≈ test_function_derivative.(x))
            end
        end
    end

    # The number of `enrichment_functions` must be equal to `N - K`
    let N = 5
        nodes = collect(range(x_L, x_R, length = N))
        D = function_space_operator(basis_functions, nodes,
                                    GlaubitzNordströmÖffner2023())
        @test_throws DimensionMismatch upwind_operators(D, basis_functions, source,
                                                        test_functions = test_functions,
                                                        enrichment_functions = [x -> x^2])
    end

    # The `enrichment_functions` default to the lowest-degree Legendre polynomials not already
    # contained in the function space, here the degrees 2 and 3
    let N = 5
        nodes = collect(range(x_L, x_R, length = N))
        D = function_space_operator(basis_functions, nodes,
                                    GlaubitzNordströmÖffner2023())
        D_upw = upwind_operators(D, basis_functions, source; test_functions)
        D_upw_explicit = upwind_operators(D, basis_functions, source; test_functions,
                                          enrichment_functions = [x -> x^2, x -> x^3])
        @test isapprox(Matrix(D_upw.plus), Matrix(D_upw_explicit.plus), atol = 1e-10)
    end

    # The initial guess for the optimization can be set with the `sigma0` keyword
    let N = 5
        nodes = collect(range(x_L, x_R, length = N))
        D = function_space_operator(basis_functions, nodes,
                                    GlaubitzNordströmÖffner2023())
        enrichment_functions = [x -> x^2, x -> x^3]
        D_upw = upwind_operators(D, basis_functions, source; test_functions,
                                 enrichment_functions, sigma0 = [-2.0, -1.0])
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

    enrichment_functions = [x -> x^2, x -> x^3, x -> x^4]
    @test length(enrichment_functions) == N - K
    # `x^2` and `x^3` are contained in the span of the enrichment modes 1 and 2, but are
    # orthogonal to the enrichment mode 3
    test_functions = [x -> x^2, x -> x^3]
    sigma0 = [-1.0, -1.0, -1.0]

    V = SummationByPartsOperatorsExtra.enriched_orthonormal_vandermonde(basis_functions,
                                                                        nodes,
                                                                        enrichment_functions,
                                                                        sqrt(eps()))
    sigma = SummationByPartsOperatorsExtra.dissipation_eigenvalues(test_functions, D, V, K,
                                                                   source; sigma0)
    # The modes seen by the objective are driven to zero ...
    @test isapprox(sigma[1], 0.0, atol = 1e-8)
    @test isapprox(sigma[2], 0.0, atol = 1e-8)
    # ... while the mode invisible to the objective simply keeps its initial guess
    @test isapprox(sigma[3], sigma0[3], atol = 1e-9)

    # The objective is driven to (almost) zero, i.e. this is a global minimizer and not a
    # failure of the optimizer
    D_upw = upwind_operators(D, basis_functions, source; test_functions,
                             enrichment_functions, sigma0)
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
    D_upw2 = upwind_operators(D, basis_functions, source; test_functions,
                              enrichment_functions, sigma0 = [-1.0, -1.0, -2.0])
    @test !isapprox(Matrix(D_upw.plus), Matrix(D_upw2.plus))
end

@testitem "Upwind operators (DG construction)" begin
    using LinearAlgebra: I, dot, issymmetric, eigvals
    import Optim, ForwardDiff

    source = GlaubitzLampertMattssonNiemeläWinters2026DG()
    for compact in (true, false)
        show(IOContext(devnull, :compact => compact), source)
    end

    basis_functions = [one, identity, exp]
    basis_functions_derivatives = [zero, one, exp]
    K = length(basis_functions)
    x_L = -1.0
    x_R = 1.0

    # `N = 4` is excluded: the optimization does not find an `F`-exact central operator for
    # this space and grid (the exactness conditions are overdetermined there), so the exactness
    # tests below would fail for the central operator already
    for N in 5:7
        nodes = collect(range(x_L, x_R, length = N))
        D = function_space_operator(basis_functions, nodes, GlaubitzNordströmÖffner2023())
        D_upw = upwind_operators(D, basis_functions, source; lambda = -1.0)
        x = grid(D_upw)
        Dm = Matrix(D_upw.minus)
        Dp = Matrix(D_upw.plus)

        M = mass_matrix(D_upw)
        @test M == mass_matrix(D)
        B = mass_matrix_boundary(D_upw)
        # Definition of upwind FSBP operators: Q_+ + Q_-' == 0 and Q_+ + Q_+' == S with S
        # symmetric and negative semi-definite
        @test isapprox(Dm' * M + M * Dp, B, atol = 1e-11)
        Qp = M * Dp - B / 2
        Qm = M * Dm - B / 2
        S = Qp + Qp'
        @test Qp - Qm ≈ S
        @test issymmetric(S)
        @test maximum(eigvals(S)) < 1e-10
        # The central operator is recovered as the average
        @test (Dm + Dp) / 2 ≈ Matrix(D)

        # `S` annihilates the function space and is strictly dissipative outside of it
        for (basis_function, basis_function_derivative) in zip(basis_functions,
                                                               basis_functions_derivatives)
            @test isapprox(S * basis_function.(x), zeros(N), atol = 1e-10)
            for D_op in (D_upw.minus, D_upw.plus, D_upw.central)
                @test isapprox(D_op * basis_function.(x), basis_function_derivative.(x),
                               atol = 1e-10)
            end
        end
        for f in (x -> x^2, x -> x^3, sin)
            @test dot(f.(x), S * f.(x)) < -1e-10
        end
    end

    # `S == 0` if `N == K`, i.e. there are no unresolved modes
    let D = legendre_derivative_operator(x_L, x_R, K)
        S = dissipation_matrix(basis_functions, D, source; lambda = -1.0)
        @test isapprox(Matrix(S), zeros(K, K), atol = 1e-12)
    end

    # For a flat `lambda`, `S == -c * (I - P)` with `P` the orthogonal projection onto the
    # nodal values of the function space, independently of the enrichment functions
    let N = 6, c = 0.8
        # only the grid of `D` is used for a fixed `lambda`
        D = legendre_derivative_operator(x_L, x_R, N)
        V_basis = SummationByPartsOperatorsExtra.vandermonde_matrix(basis_functions,
                                                                    grid(D))
        projection = V_basis * ((V_basis' * V_basis) \ V_basis')
        S = dissipation_matrix(basis_functions, D, source; lambda = -c)
        @test isapprox(Matrix(S), -c * (I - projection), atol = 1e-12)

        # A different (valid) enrichment gives the same `S` for a flat `lambda` ...
        enrichment_functions = [x -> x^2, x -> x^5, cos]
        S2 = dissipation_matrix(basis_functions, D, source; lambda = -c,
                                enrichment_functions)
        @test isapprox(Matrix(S), Matrix(S2), atol = 1e-12)
        # ... but not for a non-constant one
        lambda = [-1.0, -0.5, -0.25]
        S3 = dissipation_matrix(basis_functions, D, source; lambda)
        S4 = dissipation_matrix(basis_functions, D, source; lambda,
                                enrichment_functions)
        @test !isapprox(Matrix(S3), Matrix(S4), atol = 1e-8)
    end

    # The default enrichment picks the lowest-degree Legendre polynomials not already contained
    # in the function space. For `F` spanned by the monomials of degree at most `K - 1`, these
    # are the degrees `K, ..., N - 1`, so the construction agrees with the one of
    # `GlaubitzRanochaWintersSchlottkeLakemperÖffnerGassner2025` (both orthonormalize a basis
    # with the same nested sequence of spans, which fixes the columns of `V` up to their signs)
    let N = 6, sigma = [-1.0, -0.7, -0.3]
        source_2025 = GlaubitzRanochaWintersSchlottkeLakemperÖffnerGassner2025()
        D = legendre_derivative_operator(x_L, x_R, N)
        monomials = [x -> x^k for k in 0:(N - length(sigma) - 1)]
        D_upw_2025 = upwind_operators(D, sigma, source_2025)
        D_upw_dg = upwind_operators(D, monomials, source; lambda = sigma)
        @test isapprox(Matrix(D_upw_dg.minus), Matrix(D_upw_2025.minus), atol = 1e-11)
        @test isapprox(Matrix(D_upw_dg.plus), Matrix(D_upw_2025.plus), atol = 1e-11)
    end

    # Error handling
    let N = 6
        D = legendre_derivative_operator(x_L, x_R, N)
        @test_throws ArgumentError dissipation_matrix(basis_functions, D, source;
                                                      lambda = 1.0)
        @test_throws DimensionMismatch dissipation_matrix(basis_functions, D, source;
                                                          lambda = [-1.0, -1.0])
        @test_throws DimensionMismatch dissipation_matrix(basis_functions, D, source;
                                                          lambda = -1.0,
                                                          enrichment_functions = [x -> x^2])
        # Linearly dependent enrichment: `x^2` is used twice
        @test_throws ArgumentError dissipation_matrix(basis_functions, D, source;
                                                      lambda = -1.0,
                                                      enrichment_functions = [x -> x^2,
                                                          x -> x^2,
                                                          x -> x^3])
    end
end

@testitem "Upwind operators (stiffness budget)" begin
    using LinearAlgebra: eigvals, issymmetric
    import Optim, ForwardDiff

    SBPE = SummationByPartsOperatorsExtra
    source = GlaubitzLampertMattssonNiemeläWinters2026DG()
    basis_functions = [one, identity, exp]
    K = length(basis_functions)
    N = 6
    nodes = collect(range(-1.0, 1.0, length = N))
    D = function_space_operator(basis_functions, nodes, GlaubitzNordströmÖffner2023())

    V = SBPE.enriched_orthonormal_vandermonde(basis_functions, nodes, nothing, sqrt(eps()))
    S_shape = SBPE.eigen_dissipation_matrix(V, -ones(N - K))

    # Reference values of the reference given in `GlaubitzLampertMattssonNiemeläWinters2026DG`:
    # the budget-calibrated flat rate is `lambda* = 0.80` at a spectral radius ratio of `1.20`
    budget = StiffnessBudget()
    scale = SBPE.calibrate_stiffness_budget(D, S_shape, source, budget)
    @test isapprox(scale, 0.7989, atol = 1e-4)
    rho_0 = SBPE.semidiscretization_spectral_radius(D, 0 * S_shape, source,
                                                    budget.num_elements)
    rho = SBPE.semidiscretization_spectral_radius(D, scale * S_shape, source,
                                                  budget.num_elements)
    @test isapprox(rho / rho_0, 1 + budget.tol, rtol = 1e-8)

    # The ratio of the spectral radii, and hence the calibrated scale, does not depend on the
    # number of elements used for the calibration
    for num_elements in (3, 4, 8)
        scale_j = SBPE.calibrate_stiffness_budget(D, S_shape, source,
                                                  StiffnessBudget(; num_elements))
        @test isapprox(scale_j, scale, rtol = 1e-6)
    end

    # A tighter budget gives less dissipation, a looser one more
    @test SBPE.calibrate_stiffness_budget(D, S_shape, source,
                                          StiffnessBudget(tol = 1 // 10)) < scale
    @test SBPE.calibrate_stiffness_budget(D, S_shape, source,
                                          StiffnessBudget(tol = 1 // 2)) > scale

    # The budget is the default and yields a valid pair of upwind operators
    D_upw = upwind_operators(D, basis_functions, source)
    x = grid(D_upw)
    M = mass_matrix(D_upw)
    B = mass_matrix_boundary(D_upw)
    @test isapprox(Matrix(D_upw.minus)' * M + M * Matrix(D_upw.plus), B, atol = 1e-11)
    S = M * Matrix(D_upw.plus) - B / 2 + (M * Matrix(D_upw.plus) - B / 2)'
    @test issymmetric(S)
    @test maximum(eigvals(S)) < 1e-10
    for basis_function in basis_functions
        @test isapprox(S * basis_function.(x), zeros(N), atol = 1e-10)
    end
    # ... and it is the same as passing the calibrated value directly
    D_upw_explicit = upwind_operators(D, basis_functions, source; lambda = -scale)
    @test isapprox(Matrix(D_upw.plus), Matrix(D_upw_explicit.plus), atol = 1e-12)

    # A budget that is never binding cannot be calibrated
    @test_throws ArgumentError SBPE.calibrate_stiffness_budget(D, S_shape, source,
                                                               StiffnessBudget(tol = 1e6))
end
