function SummationByPartsOperatorsExtra.function_space_operator(basis_functions,
                                                                nodes::Vector{T},
                                                                source::SourceOfCoefficients;
                                                                derivative_order = 1,
                                                                accuracy_order = 0,
                                                                bandwidth = length(nodes) -
                                                                            1,
                                                                size_boundary = 2 *
                                                                                bandwidth,
                                                                different_values = true,
                                                                sparsity_pattern = nothing,
                                                                opt_alg = default_opt_alg(source),
                                                                options = default_options(source),
                                                                autodiff = :forward,
                                                                x0 = nothing,
                                                                verbose = false) where {T,
                                                                                        SourceOfCoefficients
                                                                                        }
    assert_first_derivative_order(derivative_order)
    sort!(nodes)
    weights, D = construct_function_space_operator(basis_functions, nodes, source;
                                                   bandwidth, size_boundary,
                                                   different_values, sparsity_pattern,
                                                   opt_alg, options, autodiff, x0, verbose)
    return MatrixDerivativeOperator(first(nodes), last(nodes), nodes, weights, D,
                                    accuracy_order, source)
end
