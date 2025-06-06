@testsnippet ReproducingSubcell1D begin
    using Optim: Optim
    using LinearAlgebra: diag
    using FastGaussQuadrature: gaussradau

    # Map interval [a, b] (reference interval) to [c, d] (physical interval)
    linear_map(x, a, b, c, d) = c + (x - a) / (b - a) * (d - c)
end

@testitem "Reproducing Gauss-Radau" setup=[ReproducingSubcell1D] begin
    # Gauss-Radau nodes are defined on [-1, 1]
    a = -1.0
    b = 1.0
    # Construct `D` on [-1, 1] with subcells [-1, x_M] and [x_M, 1]
    c = -1.0
    d = 1.0
    x_M = 0.1
    source = GlaubitzLampertNordströmWinters2025()
    for n in 2:6
        nodes_ref, weights = gaussradau(n)
        basis = [x -> x^(i - 1) for i in 1:n]
        # Map the reference nodes to the left part and use symmetry
        # to construct the right part of the interval
        nodes = [
            linear_map.(nodes_ref, a, b, c, x_M)...,
            linear_map.(-reverse(nodes_ref), a, b, x_M, d)...
        ]
        D = subcell_operator(basis, nodes, x_M, source;
                             options = Optim.Options(; iterations = 10000,
                                                     g_tol = 1e-16))

        weights_D = diag(mass_matrix(D))
        @test all((b - a) / (x_M - c) * weights_D[1:n] .≈ weights)
        @test all((b - a) / (d - x_M) * weights_D[(n + 1):end] .≈ reverse(weights))
    end
end
