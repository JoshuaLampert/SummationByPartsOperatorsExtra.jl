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
