@testitem "moments" begin
    using StaticArrays: SVector
    D = derivative_operator(MattssonNordström2004(), 1, 4, -1.0, 1.0, 10)
    nodes = grid(D)
    functions = [one, identity, x -> x^2, exp, sin]
    normals = [SVector(-1.0), SVector(1.0)]
    @test compute_moments_boundary(functions, nodes, normals) ==
          compute_moments_boundary(functions, D)
    @test compute_moments_boundary(functions, D) ==
          ([0.0 2.0 0.0 2.3504023872876028 1.682941969615793
            2.0 0.0 2.0 3.0861612696304874 0.0
            0.0 2.0 0.0 2.3504023872876028 1.682941969615793
            2.3504023872876028 3.0861612696304874 2.3504023872876028 7.253720815694037 2.5969151628319547
            1.682941969615793 0.0 1.682941969615793 2.5969151628319547 0.0],)

    D_2 = tensor_product_operator_2D(D)
    functions_2 = [x -> 1.0, x -> x[1], x -> x[2], x -> sin(x[1] * x[2])]
    @test all(compute_moments_boundary(functions_2, D_2) .≈
              ([0.0 4.0 0.0 0.0
                4.0 0.0 0.0 0.0
                0.0 0.0 0.0 1.2044456338738128
                0.0 0.0 1.2044456338738128 0.0],
               [0.0 0.0 4.0 0.0
                0.0 0.0 0.0 1.2044456338738128
                4.0 0.0 0.0 0.0
                0.0 1.2044456338738128 0.0 0.0]))
end

@testitem "corners" begin
    D = derivative_operator(MattssonNordström2004(), 1, 2, -1.0, 1.0, 4)
    D_2 = tensor_product_operator_2D(D)
    # The `TensorProductOperator` has the information about the corners in SummationByPartsOperators.jl
    # and therefore defines a `mass_matrix_boundary` function. Here, we also define a `mass_matrix_boundary`
    # function for the `MultidimensionalMatrixDerivativeOperator` by finding the corner indices based on
    # the `boundary_indices`. With this, we can define a general `mass_matrix_boundary` function.
    # Here, we test that the `mass_matrix_boundary` function for the `MultidimensionalMatrixDerivativeOperator`
    # defined in SummationByPartsOperatorsExtra.jl is the same as the one for the `TensorProductOperator`
    # defined in SummationByPartsOperators.jl.
    # We need to bring the normals in the correct order (x indices first, y indices second). We know that
    # in the `TensorProductOperator` for the first two corners, the y indices are first. So we need to
    # swap the first two corners.
    ns = copy(normals(D_2))
    (corners_x, corners_y) = SummationByPartsOperatorsExtra.find_corners(D_2.boundary_indices)
    ns[corners_x[1]], ns[corners_y[1]] = ns[corners_y[1]], ns[corners_x[1]]
    ns[corners_x[2]], ns[corners_y[2]] = ns[corners_y[2]], ns[corners_x[2]]
    D_2_multi = MultidimensionalMatrixDerivativeOperator(grid(D_2), boundary_indices(D_2),
                                                         ns, D_2.weights,
                                                         D_2.weights_boundary, D_2.Ds,
                                                         D_2.accuracy_order, D_2.source)
    @test mass_matrix_boundary(D_2, 1) == mass_matrix_boundary(D_2_multi, 1)
    @test mass_matrix_boundary(D_2, 2) == mass_matrix_boundary(D_2_multi, 2)
end

@testitem "visualization" begin
    import Meshes, CairoMakie
    N_x = N_y = 4
    D = derivative_operator(MattssonNordström2004(), 1, 2, -1.0, 1.0, N_x)
    D_2 = tensor_product_operator_2D(D)

    corner_indices = [N_y + 1, # lower left corner
        N_x + N_y, # lower right corner
        N_x + N_y + 1, # upper left corner
        2 * N_x + N_y] # upper right corner
    @test_nowarn plot_nodes(D_2; pointsize = 10)
    @test_nowarn plot_nodes(D_2; corner_indices = corner_indices, pointsize = 10)
    @test_nowarn plot_normals(D_2; pointsize = 10)
    @test_nowarn plot_sparsity_pattern(get_sparsity_pattern(D_2)[1], grid(D_2), 6)
end

@testitem "AnalysisCallback" begin
    N_x = N_y = 4
    D = derivative_operator(MattssonNordström2004(), 1, 2, -1.0, 1.0, N_x)
    D_2 = tensor_product_operator_2D(D)
    a = (1.0, 1.0)
    g(x, t) = 0.0
    semi = MultidimensionalLinearAdvectionNonperiodicSemidiscretization(D_2, a, g)
    analysis_callback = AnalysisCallback(semi; dt = 0.1)
    for compact in (true, false)
        show(IOContext(devnull, :compact => compact), analysis_callback)
    end
    @test length(tstops(analysis_callback)) == 0
    @test length(quantities(analysis_callback)) == 0
    analysis_callback = AnalysisCallback(semi; interval = 10)
    for compact in (true, false)
        show(IOContext(devnull, :compact => compact), analysis_callback)
    end
    @test length(tstops(analysis_callback)) == 0
    @test length(quantities(analysis_callback)) == 0

    @test_throws ArgumentError AnalysisCallback(semi; interval = 10, dt = 0.1)
end

@testitem "MultidimensionalLinearAdvectionNonperiodicSemidiscretization" begin
    N_x = N_y = 4
    D = derivative_operator(MattssonNordström2004(), 1, 2, -1.0, 1.0, N_x)
    D_2 = tensor_product_operator_2D(D)
    a = (1.0, 1.0)
    g(x, t) = 0.0
    semi = MultidimensionalLinearAdvectionNonperiodicSemidiscretization(D_2, a, g)
    for compact in (true, false)
        show(IOContext(devnull, :compact => compact), semi)
    end
end
