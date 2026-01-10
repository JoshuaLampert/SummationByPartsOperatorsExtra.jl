function SummationByPartsOperatorsExtra.function_space_operator(basis_functions,
                                                                nodes::Vector{T},
                                                                source::SourceOfCoefficients;
                                                                derivative_order = 1,
                                                                accuracy_order = 0,
                                                                kwargs...) where {T,
                                                                                  SourceOfCoefficients
                                                                                  }
    assert_first_derivative_order(derivative_order)
    sort!(nodes)
    weights, D = construct_function_space_operator(basis_functions, nodes, source;
                                                   kwargs...)
    return MatrixDerivativeOperator(first(nodes), last(nodes), nodes, weights, D,
                                    accuracy_order, source)
end
