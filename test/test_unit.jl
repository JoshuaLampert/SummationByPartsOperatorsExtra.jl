@testitem "interpolation" begin
    xmin = -1.0
    xmax = 1.0
    N = 4
    D = derivative_operator(MattssonNordström2004(), 1, 2, xmin, xmax, N)
    A = interpolation_matrix([xmin, xmax], D)
    A_ref = zeros(2, N)
    A_ref[1, 1] = 1.0
    A_ref[end, end] = 1.0
    @test isapprox(A, A_ref, atol = 1e-14)
    A = zeros(2, N)
    interpolation_matrix!(A, [xmin, xmax], D)
    @test isapprox(A, A_ref, atol = 1e-14)
end

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

@testitem "orthonormal_vandermonde_matrices" begin
    using LinearAlgebra: I, cond, norm
    import Optim, ForwardDiff

    ext = Base.get_extension(SummationByPartsOperatorsExtra, :OptimForwardDiffExt)
    nodes = collect(range(-1.0, 1.0, length = 12))
    N = length(nodes)

    function orthonormalize(basis_functions)
        basis_functions_derivatives = [x -> ForwardDiff.derivative(f, x)
                                       for f in basis_functions]
        return ext.orthonormal_vandermonde_matrices(basis_functions,
                                                    basis_functions_derivatives, nodes)
    end
    stacked(basis_functions) = [reduce(hcat, [f.(nodes) for f in basis_functions]);
                                reduce(hcat,
                                       [x -> ForwardDiff.derivative(f, x)
                                        for f in basis_functions] .|>
                                       (fd -> fd.(nodes)))]

    for basis_functions in ([one, identity, exp],
                            [x -> x^k for k in 0:5],
                            [x -> x^k for k in 0:(N - 1)],
                            # Badly conditioned: cond(W) is of the order of 1e15 here
                            [x -> (x + 5.0)^k for k in 0:9])
        V, V_x = orthonormalize(basis_functions)
        K = length(basis_functions)
        @test size(V) == size(V_x) == (N, K)
        # The basis is orthonormal with respect to the discrete H^1 inner product. Note that the
        # classical Gram-Schmidt process used previously loses orthogonality proportionally to
        # `cond(W)^2` and fails completely for the badly conditioned basis above, while the
        # Householder QR decomposition is orthogonal to machine precision for all of them.
        @test isapprox(V' * V + V_x' * V_x, I, atol = 1e-13)
    end

    # The orthonormalized basis is obtained from the original one by a *lower triangular*
    # transformation, i.e. the `k`-th orthonormalized function is a combination of the first `k`
    # original ones. We only check this for reasonably conditioned bases, since the spans
    # themselves are not numerically well-defined otherwise.
    for basis_functions in ([one, identity, exp], [x -> x^k for k in 0:5])
        V, V_x = orthonormalize(basis_functions)
        W = stacked(basis_functions)
        W_orthonormalized = [V; V_x]
        for k in axes(W, 2)
            W_k = W[:, 1:k]
            residual = W_orthonormalized[:, k] - W_k * (W_k \ W_orthonormalized[:, k])
            @test isapprox(norm(residual), 0.0, atol = 1e-12)
        end
    end

    # The signs are normalized as for the Gram-Schmidt process, i.e. the diagonal of the
    # transformation matrix is positive. Together with the lower triangular structure this makes
    # the orthonormalized basis unique.
    let basis_functions = [one, identity, exp]
        V, _ = orthonormalize(basis_functions)
        V_original = reduce(hcat, [f.(nodes) for f in basis_functions])
        A_transposed = V_original \ V
        for k in axes(A_transposed, 2)
            @test A_transposed[k, k] > 0
        end
    end
end
