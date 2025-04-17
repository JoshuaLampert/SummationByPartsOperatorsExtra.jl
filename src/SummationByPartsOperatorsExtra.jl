module SummationByPartsOperatorsExtra

using AutoHashEquals: @auto_hash_equals
using ArgCheck: @argcheck
using LinearAlgebra: LinearAlgebra, Diagonal, UpperTriangular, diag, issymmetric
import LinearAlgebra: mul!
using Reexport: @reexport
@reexport using SummationByPartsOperators
using SimpleUnPack: @unpack

include("utils/utils.jl")
include("function_space_operators.jl")
include("subcell_operators.jl")
export get_sparsity_pattern
export get_nsigma, get_optimization_entries
export SubcellOperator, GlaubitzLampertNordströmWinters2025, subcell_operator,
       derivative_matrix, grid_left, grid_right, mass_matrix_left, mass_matrix_right,
       mass_matrix_boundary_left, mass_matrix_boundary_right
end
