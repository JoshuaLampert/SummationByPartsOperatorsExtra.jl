@testitem "Upwind operators" begin
    using LinearAlgebra: issymmetric, eigvals

    x_L = -0.2
    x_R = 1.42
    source = GlaubitzEtAl2025()
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
        @test issymmetric(S)
        lambda = eigvals(S)
        @test count(x -> x < -1e-12, lambda) == length(sigma)
        @test count(x -> abs(x) < 1e-12, lambda) == N - length(sigma)
        @test Dp ≈ Dm + inv(M) * S
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
