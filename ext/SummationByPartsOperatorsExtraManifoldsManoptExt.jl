module SummationByPartsOperatorsExtraManifoldsManoptExt

using Manopt: quasi_Newton
using SummationByPartsOperatorsExtra: SummationByPartsOperatorsExtra,
                                      GlaubitzIskeLampertÖffner2025,
                                      MatrixDerivativeOperator

include("utils.jl")
include("function_space_operators.jl")

default_opt_alg(::GlaubitzIskeLampertÖffner2025) = quasi_Newton
default_options(::GlaubitzIskeLampertÖffner2025) = NamedTuple()

function construct_function_space_operator(basis_functions, nodes,
                                           source::GlaubitzIskeLampertÖffner2025;
                                           bandwidth = length(nodes) - 1,
                                           size_boundary = 2 * bandwidth,
                                           different_values = true,
                                           sparsity_pattern = nothing,
                                           opt_alg = default_opt_alg(source),
                                           options = default_options(source),
                                           autodiff = :forward,
                                           x0 = nothing, verbose = false)
    T = eltype(nodes)
    N = length(nodes)
    weights = zeros(T, N)
    D = zeros(T, N, N)
    return weights, D
end
end
