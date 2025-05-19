function assert_first_derivative_order(derivative_order)
    if derivative_order != 1
        throw(ArgumentError("Derivative order $derivative_order not implemented."))
    end
end

function assert_correct_bandwidth(N, bandwidth, size_boundary)
    if (N < 2 * size_boundary + bandwidth || bandwidth < 1) &&
       (bandwidth != N - 1)
        throw(ArgumentError("2 * size_boundary + bandwidth = $(2 * size_boundary + bandwidth) needs to be smaller than or equal to N = $N and bandwidth = $bandwidth needs to be at least 1."))
    end
end

function assert_correct_sparsity_pattern(sparsity_pattern)
    if !(sparsity_pattern isa UpperTriangular || issymmetric(sparsity_pattern)) ||
       !all(diag(sparsity_pattern) .== 0)
        throw(ArgumentError("Sparsity pattern has to be symmetric with all diagonal entries being false or `UpperTriangular`."))
    end
end
