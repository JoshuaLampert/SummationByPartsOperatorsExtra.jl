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

    enriched_basis = SummationByPartsOperatorsExtra.enriched_orthonormal_vandermonde(basis_functions,
                                                                                     nodes,
                                                                                     enrichment_functions,
                                                                                     sqrt(eps()))
    sigma = SummationByPartsOperatorsExtra.dissipation_eigenvalues(test_functions, D,
                                                                   enriched_basis, K,
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

    enriched_basis = SBPE.enriched_orthonormal_vandermonde(basis_functions, nodes, nothing,
                                                           sqrt(eps()))
    S_shape = SBPE.eigen_dissipation_matrix(enriched_basis.V, -ones(N - K))

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
    Qp = M * Matrix(D_upw.plus) - B / 2
    S = Qp + Qp'
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

@testitem "Upwind operators (error-weighted damping rates)" begin
    using LinearAlgebra: Diagonal, diag, dot, eigvals, issymmetric
    import Optim, ForwardDiff

    SBPE = SummationByPartsOperatorsExtra
    source = GlaubitzLampertMattssonNiemeläWinters2026DG()
    basis_functions = [one, identity, exp]
    basis_functions_derivatives = [zero, one, exp]
    K = length(basis_functions)
    N = 6
    nodes = collect(range(-1.0, 1.0, length = N))
    D = function_space_operator(basis_functions, nodes, GlaubitzNordströmÖffner2023())
    P = mass_matrix(D)

    enriched_basis = SBPE.enriched_orthonormal_vandermonde(basis_functions, nodes, nothing,
                                                           sqrt(eps()))
    V = enriched_basis.V
    V_x = SBPE.enriched_derivative_vandermonde(basis_functions_derivatives,
                                               enriched_basis.enrichment_functions_derivatives,
                                               nodes, enriched_basis)
    errors = [sqrt(dot(r, P, r)) for r in eachcol(Matrix(D) * V - V_x)]

    # `V_x` are the nodal derivative values of the orthonormalized enriched basis, so the central
    # operator is exact on the first `K` modes - they span the function space it is built for
    @test all(<(1e-10), errors[1:K])
    @test all(>(1e-2), errors[(K + 1):N])

    weights = ErrorWeights(; basis_functions_derivatives)
    w = SBPE.dissipation_weights(weights, D, enriched_basis, K, basis_functions)
    @test w ≈ errors[(K + 1):N] / sum(errors[(K + 1):N])
    @test sum(w) ≈ 1

    # Reference values of the reference given in `GlaubitzLampertMattssonNiemeläWinters2026DG`:
    # the per-mode ratio of the dissipation-induced derivative error to the central error is
    # `(0.90, 0.16, 0.06)` for the flat choice, but `(0.39, 0.34, 0.35)` for the weighted one
    function error_ratios(lambda)
        S = dissipation_matrix(basis_functions, D, source; lambda)
        dissipation = (P \ Matrix(S)) * V[:, (K + 1):N] / 2
        return [sqrt(dot(r, P, r)) for r in eachcol(dissipation)] ./ errors[(K + 1):N]
    end
    @test isapprox(error_ratios(StiffnessBudget()), [0.90, 0.16, 0.06], atol = 0.02)
    @test isapprox(error_ratios(StiffnessBudget(weights)), [0.39, 0.34, 0.35], atol = 0.02)

    # The equilibration of the reference holds only up to the factor `norm(v_j, inv(P))`: since
    # `v_j` is an eigenvector of `S`, the dissipation-induced derivative error of mode `j` is
    #     norm(inv(P) * S * v_j, P) == abs(lambda_j) * norm(v_j, inv(P)),
    # where the extra factor appears because the modes are normalized in the Euclidean inner
    # product while the error is measured in the `P` norm. This is structural: it holds for every
    # choice of the eigenvalues, not just for the weighted one.
    for lambda in (-1.0, StiffnessBudget(), StiffnessBudget(weights))
        S = Matrix(dissipation_matrix(basis_functions, D, source; lambda))
        for j in (K + 1):N
            v = V[:, j]
            # `S * v == lambda_j * v` and `norm(v) == 1`, so this is the eigenvalue of that mode
            lambda_j = dot(v, S * v)
            @test isapprox(S * v, lambda_j * v, atol = 1e-12)

            dissipation = P \ (S * v)
            @test isapprox(sqrt(dot(dissipation, P, dissipation)),
                           abs(lambda_j) * sqrt(dot(v, inv(P), v)), rtol = 1e-10)
            # ... and the factor may not be dropped, it is far from one here
            @test !isapprox(sqrt(dot(dissipation, P, dissipation)), abs(lambda_j),
                            rtol = 0.1)
        end
    end

    # The weighted choice yields a valid pair of upwind operators, but a different one
    D_upwind = upwind_operators(D, basis_functions, source;
                                lambda = StiffnessBudget(weights))
    x = grid(D_upwind)
    M = mass_matrix(D_upwind)
    B = mass_matrix_boundary(D_upwind)
    @test isapprox(Matrix(D_upwind.minus)' * M + M * Matrix(D_upwind.plus), B, atol = 1e-11)
    Qp = M * Matrix(D_upwind.plus) - B / 2
    S = Qp + Qp'
    @test issymmetric(S)
    @test maximum(eigvals(S)) < 1e-10
    for basis_function in basis_functions
        @test isapprox(S * basis_function.(x), zeros(N), atol = 1e-10)
    end
    @test !isapprox(Matrix(D_upwind.plus),
                    Matrix(upwind_operators(D, basis_functions, source).plus))

    # Error handling
    let enrichment_functions = [x -> x^2, x -> x^5, cos]
        # custom enrichment functions need their derivatives
        @test_throws ArgumentError dissipation_matrix(basis_functions, D, source;
                                                      enrichment_functions,
                                                      lambda = StiffnessBudget(weights))
        weights_full = ErrorWeights(; basis_functions_derivatives,
                                    enrichment_functions_derivatives = [x -> 2 * x,
                                        x -> 5 * x^4, x -> -sin(x)])
        @test dissipation_matrix(basis_functions, D, source; enrichment_functions,
                                 lambda = StiffnessBudget(weights_full)) isa AbstractMatrix
    end
    @test_throws DimensionMismatch dissipation_matrix(basis_functions, D, source;
                                                      lambda = StiffnessBudget(ErrorWeights(;
                                                                                            basis_functions_derivatives = [zero,
                                                                                                one])))
end

@testitem "Upwind operators (FD construction)" begin
    using LinearAlgebra: Diagonal, det, dot, eigvals, issymmetric, norm, rank
    import Optim, ForwardDiff

    SBPE = SummationByPartsOperatorsExtra
    source = GlaubitzLampertMattssonNiemeläWinters2026FD()
    for compact in (true, false)
        show(IOContext(devnull, :compact => compact), source)
    end

    x_L = 0.0
    x_R = 1.0

    # Reduction to the polynomial case: for `F` the polynomials of degree at most `K - 1` on
    # equidistant nodes, the generalized divided differences are the classical undivided
    # differences of order `K`, and the annihilation matrix is Toeplitz
    let N = 12
        monomials = [x -> x^k for k in 0:2]
        nodes = collect(range(x_L, x_R, length = N))
        D_tilde = annihilation_matrix(monomials, nodes; normalization = :undivided)
        @test size(D_tilde) == (N - 3, N)
        for i in 1:(N - 3)
            @test isapprox(D_tilde[i, i:(i + 3)], [-1.0, 3.0, -3.0, 1.0], atol = 1e-12)
            @test all(iszero, D_tilde[i, 1:(i - 1)])
            @test all(iszero, D_tilde[i, (i + 4):N])
        end
        # The two normalizations differ by a single global factor on an equidistant grid
        D_tilde_2 = annihilation_matrix(monomials, nodes)
        @test isapprox(D_tilde, sqrt(20) * D_tilde_2, atol = 1e-12)
        @test all(isapprox(norm(D_tilde_2[i, :]), 1) for i in 1:(N - 3))
        # The rows are linearly independent, so the kernel is exactly the function space
        @test rank(D_tilde) == N - 3
    end

    # Determinant representation of the generalized divided differences: the coefficients are
    # the cofactors of the local Vandermonde matrix, and the generalized divided difference of a
    # function is the determinant of that matrix with the sampled function appended as a column.
    # This is an independent route to the same vectors, checked on a non-uniform grid and a
    # non-polynomial function space, where neither the polynomial reduction above nor a Toeplitz
    # structure applies.
    let basis_functions = [one, identity, exp]
        K = length(basis_functions)
        nodes = [0.0, 0.13, 0.31, 0.52, 0.6, 0.87, 1.0]
        N = length(nodes)
        D_tilde = annihilation_matrix(basis_functions, nodes)
        V = SBPE.vandermonde_matrix(basis_functions, nodes)
        for i in 1:(N - K)
            window = i:(i + K)
            V_window = V[window, :]
            # `(-1)^(K + j)` times the determinant of `V_window` with the row of the node
            # `x_{i + j}` deleted
            cofactors = [(-1)^(K + j) * det(V_window[setdiff(1:(K + 1), j + 1), :])
                         for j in 0:K]
            # Appending the nodal values of a function to the local Vandermonde matrix and
            # expanding the determinant along that column gives its generalized divided
            # difference
            for f in (sin, exp, x -> x^4)
                values = f.(nodes[window])
                @test isapprox(det(hcat(V_window, values)), dot(cofactors, values),
                               atol = 1e-14)
            end
            # The computed coefficients agree with the cofactors up to the normalization
            @test isapprox(D_tilde[i, window],
                           sign(cofactors[end]) * cofactors / norm(cofactors), atol = 1e-12)
            # The span of `1`, `x`, and `exp` is an extended complete Chebyshev system, so the
            # cofactors share their sign and the coefficients alternate like the classical
            # difference stencils
            @test all(sign(D_tilde[i, i + j]) == (-1)^(K + j) for j in 0:K)
        end
    end

    # The classical upwind SBP operators of Mattsson (2017) with interior order 5 are members of
    # the family: their central operator has the same half bandwidth `K` as the dissipation
    # matrix, and with the scaling `epsilon = 2 / 3` the interior agrees. The boundary closure
    # needs the weighting matrix `C`, which gives the outermost window half the weight - the
    # trapezoidal rule for the dissipation functional, where `C = I` is the rectangle rule.
    let N = 24
        monomials = [x -> x^k for k in 0:2]
        D_classical = upwind_operators(Mattsson2017, derivative_order = 1,
                                       accuracy_order = 5, xmin = x_L, xmax = x_R, N = N)
        K = length(monomials)
        D_central = D_classical.central
        M = mass_matrix(D_classical)
        B = mass_matrix_boundary(D_classical)
        Qp = M * Matrix(D_classical.plus) - B / 2
        S_classical = Qp + Qp'
        D_tilde = annihilation_matrix(monomials, collect(grid(D_classical)))
        S = -D_tilde' * D_tilde
        i = N ÷ 2
        epsilon = S_classical[i, i] / S[i, i]
        @test isapprox(epsilon, 2 / 3, atol = 1e-12)
        @test isapprox(S_classical[i, :], epsilon * S[i, :], atol = 1e-12)

        # With `C = I` the interior of the operators agrees, but the boundary closure does not
        D_upwind = upwind_operators(D_central, monomials, source; epsilon)
        @test isapprox(mass_matrix(D_upwind), M)
        difference = Matrix(D_upwind.minus) - Matrix(D_classical.minus)
        @test isapprox(difference[(2 * K - 1):(N - 2 * K + 2), :],
                       zeros(N - 4 * K + 4, N), atol = 1e-11)
        @test all(norm(difference[r, :]) > 0.1 for r in (1:(2 * K - 2)))

        # The trapezoidal weighting reproduces the classical operators completely
        C = Diagonal([0.5; ones(N - K - 2); 0.5])
        D_trapezoidal = upwind_operators(D_central, monomials, source; epsilon, C)
        @test isapprox(Matrix(dissipation_matrix(monomials, D_central, source; epsilon, C)),
                       S_classical, atol = 1e-11)
        @test isapprox(Matrix(D_trapezoidal.minus), Matrix(D_classical.minus), atol = 1e-11)
        @test isapprox(Matrix(D_trapezoidal.plus), Matrix(D_classical.plus), atol = 1e-11)
    end

    # Properties of the dissipation matrix for a non-polynomial function space
    let N = 12
        basis_functions = [one, identity, exp]
        basis_functions_derivatives = [zero, one, exp]
        K = length(basis_functions)
        nodes = collect(range(-1.0, 1.0, length = N))
        D = function_space_operator(basis_functions, nodes, GlaubitzNordströmÖffner2023())
        D_upw = upwind_operators(D, basis_functions, source)
        x = grid(D_upw)
        M = mass_matrix(D_upw)
        B = mass_matrix_boundary(D_upw)
        @test isapprox(Matrix(D_upw.minus)' * M + M * Matrix(D_upw.plus), B, atol = 1e-11)
        Qp = M * Matrix(D_upw.plus) - B / 2
        S = Qp + Qp'
        @test issymmetric(S)
        @test maximum(eigvals(S)) < 1e-12
        # `S` is banded with bandwidth at most `2K + 1`
        for i in 1:N, j in 1:N
            abs(i - j) > K && @test isapprox(S[i, j], 0, atol = 1e-14)
        end
        # `S` annihilates the function space, the upwind operators are exact for it, and `S` is
        # strictly dissipative outside of it
        for (basis_function, basis_function_derivative) in zip(basis_functions,
                                                               basis_functions_derivatives)
            @test isapprox(S * basis_function.(x), zeros(N), atol = 1e-12)
            for D_op in (D_upw.minus, D_upw.plus, D_upw.central)
                @test isapprox(D_op * basis_function.(x), basis_function_derivative.(x),
                               atol = 1e-10)
            end
        end
        for f in (x -> x^2, x -> x^3, sin)
            @test dot(f.(x), S * f.(x)) < -1e-12
        end
        @test isapprox((Matrix(D_upw.minus) + Matrix(D_upw.plus)) / 2, Matrix(D))

        # A non-constant weighting matrix `C` gives a different, but still admissible, `S`
        C = Diagonal(range(0.5, 2.0, length = N - K))
        S_weighted = Matrix(dissipation_matrix(basis_functions, D, source; epsilon = 1.0,
                                               C))
        @test !isapprox(S_weighted,
                        Matrix(dissipation_matrix(basis_functions, D, source;
                                                  epsilon = 1.0)))
        @test issymmetric(S_weighted)
        @test maximum(eigvals(S_weighted)) < 1e-12
        for basis_function in basis_functions
            @test isapprox(S_weighted * basis_function.(x), zeros(N), atol = 1e-12)
        end
    end

    # Locality: in contrast to the DG-type construction, the dissipation vanishes under
    # refinement on smooth functions outside of the function space, so that the upwind operators
    # stay consistent. In the interior it does so at the expected rate `2K - 1 = 5`; the overall
    # rate is limited by the one-sided windows at the boundary.
    let monomials = [x -> x^k for k in 0:2]
        source_dg = GlaubitzLampertMattssonNiemeläWinters2026DG()
        residuals_fd = Float64[]
        residuals_fd_interior = Float64[]
        residuals_dg = Float64[]
        for N in (24, 48)
            D = derivative_operator(MattssonNordström2004(), 1, 4, x_L, x_R, N)
            x = collect(grid(D))
            P = mass_matrix(D)
            S_fd = dissipation_matrix(monomials, D, source; epsilon = 1.0)
            S_dg = dissipation_matrix(monomials, D, source_dg; lambda = -1.0)
            push!(residuals_fd, maximum(abs, P \ (S_fd * sin.(x))))
            push!(residuals_fd_interior, maximum(abs, (P \ (S_fd * sin.(x)))[9:(N - 8)]))
            push!(residuals_dg, maximum(abs, P \ (S_dg * sin.(x))))
        end
        @test residuals_fd[1] / residuals_fd[2] > 2
        @test residuals_fd_interior[1] / residuals_fd_interior[2] > 16
        @test residuals_dg[2] >= residuals_dg[1]
    end

    # Stiffness budget
    let monomials = [x -> x^k for k in 0:2]
        budget = StiffnessBudget()
        epsilons = Float64[]
        for N in (24, 48, 96)
            D = derivative_operator(MattssonNordström2004(), 1, 4, x_L, x_R, N)
            D_tilde = annihilation_matrix(monomials, collect(grid(D)))
            S_shape = -D_tilde' * D_tilde
            epsilon = SBPE.calibrate_stiffness_budget(D, S_shape, source, budget)
            push!(epsilons, epsilon)
            rho_0 = SBPE.semidiscretization_spectral_radius(D, 0 * S_shape, source,
                                                            budget.num_elements)
            rho = SBPE.semidiscretization_spectral_radius(D, epsilon * S_shape, source,
                                                          budget.num_elements)
            @test isapprox(rho / rho_0, 1 + budget.tol, rtol = 1e-8)
            # The budget is the default, so it gives the same dissipation matrix
            @test isapprox(Matrix(dissipation_matrix(monomials, D, source)),
                           epsilon * S_shape, atol = 1e-12)
            # A tighter budget gives less dissipation
            @test SBPE.calibrate_stiffness_budget(D, S_shape, source,
                                                  StiffnessBudget(tol = 1 // 10)) < epsilon
        end
        # The calibrated scaling is essentially resolution independent
        @test all(isapprox(epsilon, 1.0, rtol = 0.1) for epsilon in epsilons)
    end

    # Error handling
    let N = 10
        monomials = [x -> x^k for k in 0:2]
        nodes = collect(range(x_L, x_R, length = N))
        D = derivative_operator(MattssonNordström2004(), 1, 4, x_L, x_R, N)
        @test_throws ArgumentError dissipation_matrix(monomials, D, source;
                                                      epsilon = -1.0)
        @test_throws ArgumentError dissipation_matrix(monomials, D, source;
                                                      normalization = :unknown)
        # The FD construction has no per-mode weights
        weights = ErrorWeights(basis_functions_derivatives = [zero, one, x -> 2 * x])
        @test_throws ArgumentError dissipation_matrix(monomials, D, source;
                                                      epsilon = StiffnessBudget(weights))
        # There must be at least one window of `K + 1` nodes
        @test_throws ArgumentError annihilation_matrix([x -> x^k for k in 0:(N - 1)], nodes)
        # Linearly dependent basis functions
        @test_throws ArgumentError annihilation_matrix([one, one], nodes)
        # A window that is not unisolvent after removing one of its nodes: for `F` spanned by
        # `1` and `x^2` on nodes symmetric about the origin, removing the node at the origin
        # leaves two nodes with the same value of `x^2`
        @test_throws ArgumentError annihilation_matrix([one, x -> x^2], [-1.0, 0.0, 1.0])
    end
end
