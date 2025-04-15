@testsnippet Reproducing1D begin
    using Optim: Optim, BFGS
    xmin = -2.0
    xmax = 4.0
    vol = xmax - xmin
    N = 20
    verbose = false
    opt_kwargs = (;
                  options = Optim.Options(f_abstol = 1e-25, g_tol = 1e-16,
                                          iterations = 50000),
                  opt_alg = BFGS())
    atol = 1e-13

    function block_banded_sparsity_pattern(N, bandwidth, size_boundary)
        different_values = true
        sigma = ones(get_nsigma(N; bandwidth, size_boundary,
                                different_values))
        S = SummationByPartsOperatorsExtra.create_S(sigma, N, bandwidth, size_boundary,
                                                    different_values, nothing)
        return get_sparsity_pattern(S)
    end
end

@testitem "Reproducing legendre_derivative_operator" setup=[Reproducing1D] begin
    for n in 2:6
        @testset "n = $n" begin
            D_legendre = legendre_derivative_operator(xmin, xmax, n)
            basis = [x -> x^i for i in 0:(n - 1)]
            nodes = collect(grid(D_legendre))

            @testset "function_space_operator" begin
                D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
                                            verbose, opt_kwargs...)
                @test isapprox(Matrix(D), Matrix(D_legendre); atol) # equal
                @test isapprox(mass_matrix(D), mass_matrix(D_legendre); atol) # equal

                sparsity_pattern = get_sparsity_pattern(D)
                D_sparsity_pattern = function_space_operator(basis, nodes,
                                                             GlaubitzNordströmÖffner2023();
                                                             sparsity_pattern,
                                                             verbose, opt_kwargs...)
                @test isapprox(Matrix(D_sparsity_pattern), Matrix(D_legendre); atol) # equal
                @test isapprox(mass_matrix(D_sparsity_pattern), mass_matrix(D_legendre);
                               atol) # equal
            end
        end
    end
end

@testitem "Reproducing MattssonNordström2004" setup=[Reproducing1D] begin
    source = MattssonNordström2004()
    for p in (2, 4)
        D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
        basis = [x -> x^i for i in 0:div(p, 2)]
        nodes = collect(grid(D_poly))
        bandwidth = div(p, 2)
        size_boundary = p
        @testset "p = $p (equal)" begin
            different_values = false

            @testset "function_space_operator" begin
                D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
                                            bandwidth, size_boundary, different_values,
                                            verbose, opt_kwargs...)
                @test isapprox(Matrix(D), Matrix(D_poly); 1e-12) # equal
                @test isapprox(mass_matrix(D), mass_matrix(D_poly); 1e-12) # equal
            end
        end

        @testset "p = $p with different_values = true (almost equal)" begin
            different_values = true

            @testset "function_space_operator" verbose=true begin
                D = function_space_operator(basis, nodes, GlaubitzNordströmÖffner2023();
                                            bandwidth, size_boundary, different_values,
                                            verbose, opt_kwargs...)
                @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-3) # almost equal
                @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol = 1e-4) # almost equal

                x_poly = get_optimization_entries(D_poly;
                                                  bandwidth, size_boundary,
                                                  different_values)
                D_x0 = function_space_operator(basis, nodes,
                                               GlaubitzNordströmÖffner2023();
                                               bandwidth, size_boundary, different_values,
                                               verbose, opt_kwargs...,
                                               x0 = x_poly)
                @test isapprox(Matrix(D_x0), Matrix(D_poly); atol) # equal
                @test isapprox(mass_matrix(D_x0), mass_matrix(D_poly); atol) # equal
                x = get_optimization_entries(D_x0; bandwidth, size_boundary,
                                             different_values)
                @test isapprox(x, x_poly; atol)

                sparsity_pattern = block_banded_sparsity_pattern(N, bandwidth,
                                                                 size_boundary)
                @test all(get_sparsity_pattern(D) .== sparsity_pattern)

                D_sparsity_pattern = function_space_operator(basis, nodes,
                                                             GlaubitzNordströmÖffner2023();
                                                             sparsity_pattern,
                                                             verbose, opt_kwargs...)
                @test isapprox(Matrix(D_sparsity_pattern), Matrix(D_poly); atol = 1e-3) # almost equal
                @test isapprox(mass_matrix(D_sparsity_pattern), mass_matrix(D_poly);
                               atol = 1e-4) # almost equal

                # This neeeds only 1 iteration
                x_poly_sparsity_pattern = get_optimization_entries(D_poly;
                                                                   sparsity_pattern)
                D_sparsity_pattern_x0 = function_space_operator(basis, nodes,
                                                                GlaubitzNordströmÖffner2023();
                                                                sparsity_pattern,
                                                                verbose, opt_kwargs...,
                                                                x0 = x_poly_sparsity_pattern)
                @test isapprox(Matrix(D_sparsity_pattern_x0), Matrix(D_poly); atol) # equal
                @test isapprox(mass_matrix(D_sparsity_pattern_x0), mass_matrix(D_poly);
                               atol) # equal
                x = get_optimization_entries(D_sparsity_pattern_x0;
                                             sparsity_pattern)
                @test isapprox(x, x_poly_sparsity_pattern; atol)
            end
        end
    end
end
