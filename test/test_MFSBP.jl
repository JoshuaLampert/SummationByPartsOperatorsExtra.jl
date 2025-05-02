@testsnippet MFSBP begin
    import Optim
    using StaticArrays: SVector
end

@testitem "MFSBP 1D" setup=[MFSBP] begin
    N = 5
    xmin = -1.0
    xmax = 1.0
    nodes = SVector.(collect(range(xmin, xmax, length = N)))
    boundary_indices_ = [1, N]
    normals_ = [SVector(-1.0), SVector(1.0)]
    source = GlaubitzIskeLampertÖffner2025()
    for compact in (true, false)
        show(IOContext(devnull, :compact => compact), source)
    end
    basis_functions = [x -> one(x[1]), x -> x[1], x -> exp(x[1])]
    moments = compute_moments_boundary(basis_functions, nodes, normals_)
    vol = xmax - xmin

    # Test errors
    @test_throws ArgumentError multidimensional_function_space_operator(basis_functions,
                                                                        nodes,
                                                                        boundary_indices_,
                                                                        normals_,
                                                                        moments, vol,
                                                                        source;
                                                                        derivative_order = 2)
    @test_throws ArgumentError multidimensional_function_space_operator(basis_functions,
                                                                        nodes,
                                                                        boundary_indices_,
                                                                        normals_,
                                                                        moments, vol,
                                                                        source;
                                                                        sparsity_patterns = (ones(Bool,
                                                                                                  N,
                                                                                                  N),))
    @test_throws ArgumentError multidimensional_function_space_operator(basis_functions,
                                                                        nodes,
                                                                        boundary_indices_,
                                                                        normals_,
                                                                        moments, vol,
                                                                        source;
                                                                        bandwidth = 2)
    @test_throws ArgumentError multidimensional_function_space_operator(basis_functions,
                                                                        nodes,
                                                                        boundary_indices_,
                                                                        normals_,
                                                                        moments, vol,
                                                                        source;
                                                                        x0 = zeros(3))

    D = multidimensional_function_space_operator(basis_functions, nodes,
                                                 boundary_indices_, normals_,
                                                 moments, vol, source)
    for compact in (true, false)
        show(IOContext(devnull, :compact => compact), D)
    end

    @test grid(D) ≈ nodes
    x = first.(nodes)
    @test all(isapprox.(D[1] * ones(N), zeros(N); atol = 1e-13))
    @test D[1] * x ≈ ones(N)
    @test D[1] * exp.(x) ≈ exp.(x)
    M = mass_matrix(D)
    @test M * D[1] + D[1]' * M ≈ mass_matrix_boundary(D)
end

@testitem "MFSBP 2D" setup=[MFSBP] begin
    N_x, N_y = 5, 5
    N = N_x * N_y
    xmin = -1.0
    xmax = 1.0
    ymin = -1.0
    ymax = 1.0
    p = 2
    D_1 = derivative_operator(MattssonNordström2004(), derivative_order = 1,
                              accuracy_order = p,
                              xmin = xmin, xmax = xmax, N = N_x)
    D_2 = derivative_operator(MattssonNordström2004(), derivative_order = 1,
                              accuracy_order = p,
                              xmin = ymin, xmax = ymax, N = N_y)

    D_t = tensor_product_operator_2D(D_1, D_2)

    basis_functions = Function[x -> x[1]^i * x[2]^j for i in 0:div(p, 2), j in 0:div(p, 2)]
    nodes = grid(D_t)
    boundary_indices_ = boundary_indices(D_t)
    normals_ = normals(D_t)
    moments = compute_moments_boundary(basis_functions, D_t)
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

    D = multidimensional_function_space_operator(basis_functions, nodes,
                                                 boundary_indices_, normals_,
                                                 moments, vol, source;
                                                 sparsity_patterns,
                                                 corners,)
    for compact in (true, false)
        show(IOContext(devnull, :compact => compact), D)
    end

    @test grid(D) ≈ nodes
    x = first.(nodes)
    y = last.(nodes)
    @test all(isapprox.(D[1] * ones(N), zeros(N); atol = 1e-13))
    @test all(isapprox.(D[2] * ones(N), zeros(N); atol = 1e-13))
    @test D[1] * x ≈ ones(N)
    @test all(isapprox.(D[2] * x, zeros(N); atol = 1e-13))
    @test all(isapprox.(D[1] * y, zeros(N); atol = 1e-13))
    @test D[2] * y ≈ ones(N)
    M = mass_matrix(D)
    # Wrap operator in TensorProductOperator, such that we can call `mass_matrix_boundary`
    # because we know the corners
    D_t_FSBP = TensorProductOperator(D, N_x, N_y)
    @test M * D[1] + D[1]' * M ≈ mass_matrix_boundary(D_t_FSBP, 1)
    @test M * D[2] + D[2]' * M ≈ mass_matrix_boundary(D_t_FSBP, 2)
end
