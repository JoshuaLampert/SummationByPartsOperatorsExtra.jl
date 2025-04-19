@testsnippet Reproducing1D begin
    using Optim: Optim, BFGS
    using StaticArrays: SVector

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

            @testset "function_space_operator" begin
                basis = [x -> x^i for i in 0:(n - 1)]
                nodes = collect(grid(D_legendre))
                source = GlaubitzNordströmÖffner2023()
                D = function_space_operator(basis, nodes, source;
                                            verbose, opt_kwargs...)
                @test isapprox(Matrix(D), Matrix(D_legendre); atol) # equal
                @test isapprox(mass_matrix(D), mass_matrix(D_legendre); atol) # equal

                sparsity_pattern = get_sparsity_pattern(D)
                D_sparsity_pattern = function_space_operator(basis, nodes,
                                                             source;
                                                             sparsity_pattern,
                                                             verbose, opt_kwargs...)
                @test isapprox(Matrix(D_sparsity_pattern), Matrix(D_legendre); atol) # equal
                @test isapprox(mass_matrix(D_sparsity_pattern), mass_matrix(D_legendre);
                               atol) # equal
                @test isapprox(mass_matrix_boundary(D), mass_matrix_boundary(D_legendre);
                               atol) # equal
            end

            @testset "multidimensional_function_space_operator" begin
                basis = [x -> x[1]^i for i in 0:(n - 1)]
                nodes = SVector.(grid(D_legendre))
                boundary_indices_ = [1, n]
                normals_ = [SVector(-1.0), SVector(1.0)]
                moments = compute_moments_boundary(basis, nodes, normals_)
                source = GlaubitzIskeLampertÖffner2025()
                D = multidimensional_function_space_operator(basis, nodes,
                                                             boundary_indices_,
                                                             normals_,
                                                             moments, vol,
                                                             source;
                                                             verbose, opt_kwargs...)
                @test isapprox(Matrix(D[1]), Matrix(D_legendre); atol) # equal
                @test isapprox(mass_matrix(D), mass_matrix(D_legendre); atol) # equal
                @test isapprox(mass_matrix_boundary(D, 1), mass_matrix_boundary(D_legendre);
                               atol) # equal

                sparsity_pattern = get_sparsity_pattern(D)
                sparsity_patterns = (sparsity_pattern,)
                D_sparsity_pattern = multidimensional_function_space_operator(basis,
                                                                              nodes,
                                                                              boundary_indices_,
                                                                              normals_,
                                                                              moments,
                                                                              vol,
                                                                              source;
                                                                              sparsity_patterns,
                                                                              verbose,
                                                                              opt_kwargs...)
                @test isapprox(Matrix(D_sparsity_pattern[1]), Matrix(D_legendre); atol) # equal
                @test isapprox(mass_matrix(D_sparsity_pattern), mass_matrix(D_legendre);
                               atol) # equal
                @test isapprox(mass_matrix_boundary(D_sparsity_pattern, 1),
                               mass_matrix_boundary(D_legendre); atol) # equal
            end
        end
    end
end

@testitem "Reproducing MattssonNordström2004" setup=[Reproducing1D] begin
    source = MattssonNordström2004()
    for p in (2, 4)
        D_poly = derivative_operator(source, 1, p, xmin, xmax, N)
        bandwidth = div(p, 2)
        size_boundary = p
        @testset "p = $p (equal)" begin
            different_values = false

            @testset "function_space_operator" begin
                basis = [x -> x^i for i in 0:div(p, 2)]
                nodes = collect(grid(D_poly))
                source = GlaubitzNordströmÖffner2023()
                D = function_space_operator(basis, nodes, source;
                                            bandwidth, size_boundary, different_values,
                                            verbose, opt_kwargs...)
                @test isapprox(Matrix(D), Matrix(D_poly); 1e-12) # equal
                @test isapprox(mass_matrix(D), mass_matrix(D_poly); 1e-12) # equal
            end

            @testset "multidimensional_function_space_operator" begin
                basis = [x -> x[1]^i for i in 0:div(p, 2)]
                nodes = SVector.(grid(D_poly))
                boundary_indices_ = [1, N]
                normals_ = [SVector(-1.0), SVector(1.0)]
                moments = compute_moments_boundary(basis, nodes, normals_)
                source = GlaubitzIskeLampertÖffner2025()
                D = multidimensional_function_space_operator(basis, nodes,
                                                             boundary_indices_,
                                                             normals_,
                                                             moments, vol,
                                                             source;
                                                             bandwidth, size_boundary,
                                                             different_values,
                                                             verbose, opt_kwargs...)
                @test isapprox(Matrix(D[1]), Matrix(D_poly); 1e-12) # equal
                @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
                @test isapprox(mass_matrix_boundary(D, 1), mass_matrix_boundary(D_poly);
                               atol) # equal
            end
        end

        @testset "p = $p with different_values = true (almost equal)" begin
            different_values = true

            @testset "function_space_operator" begin
                source = GlaubitzNordströmÖffner2023()
                basis = [x -> x^i for i in 0:div(p, 2)]
                nodes = collect(grid(D_poly))
                D = function_space_operator(basis, nodes, source;
                                            bandwidth, size_boundary, different_values,
                                            verbose, opt_kwargs...)
                @test isapprox(Matrix(D), Matrix(D_poly); atol = 1e-3) # almost equal
                @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol = 1e-4) # almost equal

                x_poly = get_optimization_entries(D_poly;
                                                  bandwidth, size_boundary,
                                                  different_values)
                D_x0 = function_space_operator(basis, nodes,
                                               source;
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
                                                             source;
                                                             sparsity_pattern,
                                                             verbose, opt_kwargs...)
                @test isapprox(Matrix(D_sparsity_pattern), Matrix(D_poly); atol = 1e-3) # almost equal
                @test isapprox(mass_matrix(D_sparsity_pattern), mass_matrix(D_poly);
                               atol = 1e-4) # almost equal

                # This needs only 1 iteration
                x_poly_sparsity_pattern = get_optimization_entries(D_poly;
                                                                   sparsity_pattern)
                D_sparsity_pattern_x0 = function_space_operator(basis, nodes,
                                                                source;
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

            @testset "multidimensional_function_space_operator" begin
                basis = [x -> x[1]^i for i in 0:div(p, 2)]
                nodes = SVector.(grid(D_poly))
                boundary_indices_ = [1, N]
                normals_ = [SVector(-1.0), SVector(1.0)]
                moments = compute_moments_boundary(basis, nodes, normals_)
                source = GlaubitzIskeLampertÖffner2025()
                D = multidimensional_function_space_operator(basis, nodes,
                                                             boundary_indices_,
                                                             normals_,
                                                             moments, vol,
                                                             source;
                                                             bandwidth, size_boundary,
                                                             different_values,
                                                             verbose, opt_kwargs...)
                @test isapprox(Matrix(D[1]), Matrix(D_poly); atol = 1e-1) # almost equal
                @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol = 1e-2) # almost equal
                @test isapprox(mass_matrix_boundary(D, 1), mass_matrix_boundary(D_poly);
                               atol) # equal

                x_poly = get_multidimensional_optimization_entries(D_poly; bandwidth,
                                                                   size_boundary,
                                                                   different_values)
                D = multidimensional_function_space_operator(basis, nodes,
                                                             boundary_indices_,
                                                             normals_,
                                                             moments, vol,
                                                             source;
                                                             bandwidth, size_boundary,
                                                             different_values,
                                                             verbose, opt_kwargs...,
                                                             x0 = x_poly)
                @test isapprox(Matrix(D[1]), Matrix(D_poly); atol) # equal
                @test isapprox(mass_matrix(D), mass_matrix(D_poly); atol) # equal
                @test isapprox(mass_matrix_boundary(D, 1), mass_matrix_boundary(D_poly);
                               atol) # equal
                x = get_multidimensional_optimization_entries(D; bandwidth, size_boundary,
                                                              different_values)
                @test isapprox(x, x_poly; atol)

                sparsity_pattern = block_banded_sparsity_pattern(N, bandwidth,
                                                                 size_boundary)
                @test all(get_sparsity_pattern(D) .== sparsity_pattern)
                sparsity_patterns = (sparsity_pattern,)
                D_sparsity_pattern = multidimensional_function_space_operator(basis,
                                                                              nodes,
                                                                              boundary_indices_,
                                                                              normals_,
                                                                              moments,
                                                                              vol,
                                                                              source;
                                                                              sparsity_patterns,
                                                                              verbose,
                                                                              opt_kwargs...)
                @test isapprox(Matrix(D_sparsity_pattern[1]), Matrix(D_poly);
                               atol = 1e-1) # almost equal
                @test isapprox(mass_matrix(D_sparsity_pattern), mass_matrix(D_poly);
                               atol = 1e-2) # almost equal
                @test isapprox(mass_matrix_boundary(D_sparsity_pattern, 1),
                               mass_matrix_boundary(D_poly); atol) # equal

                # This needs only 1 iteration
                x_poly_sparsity_pattern = get_multidimensional_optimization_entries(D_poly;
                                                                                    sparsity_patterns)
                D_sparsity_pattern_x0 = multidimensional_function_space_operator(basis,
                                                                                 nodes,
                                                                                 boundary_indices_,
                                                                                 normals_,
                                                                                 moments,
                                                                                 vol,
                                                                                 source;
                                                                                 sparsity_patterns,
                                                                                 verbose,
                                                                                 opt_kwargs...,
                                                                                 x0 = x_poly_sparsity_pattern)
                @test isapprox(Matrix(D_sparsity_pattern_x0[1]), Matrix(D_poly); atol) # equal
                @test isapprox(mass_matrix(D_sparsity_pattern_x0), mass_matrix(D_poly);
                               atol) # equal
                @test isapprox(mass_matrix_boundary(D_sparsity_pattern_x0, 1),
                               mass_matrix_boundary(D_poly); atol) # equal
                x = get_multidimensional_optimization_entries(D_sparsity_pattern_x0;
                                                              sparsity_patterns)
                @test isapprox(x, x_poly_sparsity_pattern; atol)
            end
        end
    end
end

@testitem "Reproducing polynomial tensor product SBP operators with MFSBP operators (2D)" setup=[Reproducing1D] begin
    ymin = -2.0
    ymax = 1.0
    N_x = 10
    N_y = 9
    for p in (2, 4)
        @testset "p = $p (equal)" begin
            D_1 = derivative_operator(MattssonNordström2004(), derivative_order = 1,
                                      accuracy_order = p,
                                      xmin = xmin, xmax = xmax, N = N_x)
            D_2 = derivative_operator(MattssonNordström2004(), derivative_order = 1,
                                      accuracy_order = p,
                                      xmin = ymin, xmax = ymax, N = N_y)

            D_t = tensor_product_operator_2D(D_1, D_2)

            basis = Function[x -> x[1]^i * x[2]^j for i in 0:div(p, 2), j in 0:div(p, 2)]
            nodes = grid(D_t)
            boundary_indices_ = boundary_indices(D_t)
            normals_ = normals(D_t)
            moments = compute_moments_boundary(basis, D_t)
            vol = (xmax - xmin) * (ymax - ymin)
            source = GlaubitzIskeLampertÖffner2025()
            sparsity_patterns = get_sparsity_pattern(D_t)

            corners_x_dir = [N_y + 1, # lower left corner
                N_x + N_y, # lower right corner
                N_x + N_y + 1, # upper left corner
                2 * N_x + N_y] # upper right corner
            corners_y_dir = [1, # lower left corner
                N_y,  # upper left corner
                2 * N_x + N_y + 1, # lower right corner
                2 * (N_x + N_y)] # upper right corner
            corners = (corners_x_dir, corners_y_dir)

            D = multidimensional_function_space_operator(basis,
                                                         nodes,
                                                         boundary_indices_,
                                                         normals_,
                                                         moments,
                                                         vol,
                                                         source;
                                                         sparsity_patterns,
                                                         corners,
                                                         verbose,
                                                         opt_kwargs...)

            D_t_MFSBP = TensorProductOperator(D, N_x, N_y)
            atol = p == 2 ? 1e-11 : 1e1 # almost equal (some tests also pass with lower `atol`)
            @test isapprox(Matrix(D_t_MFSBP[1]), Matrix(D_t[1]); atol) # equal
            @test isapprox(Matrix(D_t_MFSBP[2]), Matrix(D_t[2]); atol) # equal
            @test isapprox(mass_matrix(D_t_MFSBP), mass_matrix(D_t); atol) # equal
            @test isapprox(mass_matrix_boundary(D_t_MFSBP, 1),
                           mass_matrix_boundary(D_t, 1); atol) # equal
            @test isapprox(mass_matrix_boundary(D_t_MFSBP, 2),
                           mass_matrix_boundary(D_t, 2); atol) # equal

            # This needs only 1 (or 2) iteration(s)
            x_tensor = get_multidimensional_optimization_entries(D_t; sparsity_patterns)

            D_x0 = multidimensional_function_space_operator(basis,
                                                            nodes,
                                                            boundary_indices_,
                                                            normals_,
                                                            moments,
                                                            vol,
                                                            source;
                                                            sparsity_patterns,
                                                            corners,
                                                            verbose,
                                                            opt_kwargs...,
                                                            x0 = x_tensor)

            D_t_x0 = TensorProductOperator(D_x0, N_x, N_y)
            atol = 1e-13
            @test isapprox(Matrix(D_t_x0[1]), Matrix(D_t[1]); atol) # equal
            @test isapprox(Matrix(D_t_x0[2]), Matrix(D_t[2]); atol) # equal
            @test isapprox(mass_matrix(D_t_x0), mass_matrix(D_t); atol) # equal
            @test isapprox(mass_matrix_boundary(D_t_x0, 1),
                           mass_matrix_boundary(D_t, 1); atol) # equal
            @test isapprox(mass_matrix_boundary(D_t_x0, 2),
                           mass_matrix_boundary(D_t, 2); atol) # equal
            x = get_multidimensional_optimization_entries(D_t_x0; sparsity_patterns)
            @test isapprox(x, x_tensor; atol)
        end
    end
end
