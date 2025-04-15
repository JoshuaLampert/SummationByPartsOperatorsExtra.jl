module SummationByPartsOperatorsExtra

using LinearAlgebra: UpperTriangular, diag
using Reexport: @reexport
@reexport using SummationByPartsOperators

include("function_space_operators.jl")
include("utils/utils.jl")
export get_sparsity_pattern
export get_nsigma, get_optimization_entries
end
