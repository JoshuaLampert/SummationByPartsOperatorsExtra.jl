module SummationByPartsOperatorsExtra

using Reexport: @reexport
@reexport using SummationByPartsOperators
using SummationByPartsOperators: AbstractMultidimensionalMatrixDerivativeOperator

include("function_space_operators.jl")
end
