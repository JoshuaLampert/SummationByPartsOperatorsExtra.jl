module SummationByPartsOperatorsExtra

using LinearAlgebra: UpperTriangular, diag
using Reexport: @reexport
@reexport using SummationByPartsOperators
using SummationByPartsOperators: AbstractNonperiodicDerivativeOperator,
                                 AbstractMultidimensionalMatrixDerivativeOperator

include("function_space_operators.jl")
include("multidimensional_function_space_operators.jl")
include("utils/utils.jl")
export get_sparsity_pattern
export get_nsigma, get_optimization_entries, get_multidimensional_optimization_entries
export compute_moments_boundary
export GlaubitzIskeLampertÖffner2025, multidimensional_function_space_operator
end
