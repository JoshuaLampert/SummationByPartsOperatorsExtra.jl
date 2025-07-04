@testsnippet PolynomialBasesDerivativeOperator begin
    using LinearAlgebra: issymmetric
end

@testitem "PolynomialBases operators" setup=[PolynomialBasesDerivativeOperator] begin
    for basis_type in (LobattoLegendre, GaussLegendre,
                       GaussRadauLeft, GaussRadauRight)
        for p in (2, 4, 6)
            xmin = -2.0
            xmax = 1.4
            N = p + 1
            D = polynomialbases_derivative_operator(basis_type; xmin, xmax, N)

            for compact in (true, false)
                show(IOContext(devnull, :compact => compact), D)
            end
            @test isa(D, PolynomialBasesDerivativeOperator)
            @test derivative_order(D) == 1
            @test issymmetric(D) == false
            @test eltype(D) == Float64
            @test SummationByPartsOperators.lower_bandwidth(D) == N - 1
            @test SummationByPartsOperators.upper_bandwidth(D) == N - 1
            nodes = grid(D)
            @test accuracy_order(D) == p
            @test length(nodes) == N
            @test D.xmin == xmin
            @test D.xmax == xmax

            D_mat = Matrix(D)
            P = mass_matrix(D)
            B = mass_matrix_boundary(D)
            @test all(isapprox(P * D_mat + D_mat' * P, B, atol = 1e-13))

            basis_functions = [x -> x^(i - 1) for i in 1:N]
            basis_functions_derivatives = [zero, (x -> (i - 1) * x^(i - 2) for i in 2:N)...]
            for (f, f_deriv) in zip(basis_functions, basis_functions_derivatives)
                @test all(isapprox(D * f.(nodes), f_deriv.(nodes), atol = 1e-12))
            end

            for f in basis_functions
                ff = f.(nodes)
                for g in basis_functions
                    gg = g.(nodes)
                    @test isapprox(ff' * B * gg, f(xmax) * g(xmax) - f(xmin) * g(xmin),
                                   atol = 1e-11)
                end
            end

            u = sinpi.(nodes)
            u_copy = copy(u)
            scale_by_mass_matrix!(u, D)
            @test_throws DimensionMismatch scale_by_mass_matrix!(ones(N + 1), D)
            @test isapprox(integrate(u_copy, D), sum(u), atol = 1e-13)
            scale_by_inverse_mass_matrix!(u, D)
            @test_throws DimensionMismatch scale_by_mass_matrix!(ones(N + 1), D)
            @test all(isapprox(u, u_copy, atol = 1e-12))

            @test SummationByPartsOperators.get_weight(D, 1) == left_boundary_weight(D)
            @test SummationByPartsOperators.get_weight(D, N) == right_boundary_weight(D)
        end
    end
end
