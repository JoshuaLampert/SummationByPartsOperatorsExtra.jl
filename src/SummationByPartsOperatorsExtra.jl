module SummationByPartsOperatorsExtra

using AutoHashEquals: @auto_hash_equals
using ArgCheck: @argcheck
using LinearAlgebra: LinearAlgebra, Diagonal, UpperTriangular, diag
import LinearAlgebra: mul!, diagind, norm
using Reexport: @reexport
@reexport using SummationByPartsOperators
using SimpleUnPack: @unpack
@reexport using StaticArrays: SVector
using StatsBase: countmap
using SummationByPartsOperators: AbstractDerivativeOperator,
                                 AbstractNonperiodicDerivativeOperator,
                                 AbstractMultidimensionalMatrixDerivativeOperator,
                                 scale_by_mass_matrix!,
                                 scale_by_inverse_mass_matrix!

include("utils/utils.jl")
include("function_space_operators.jl")
include("multidimensional_function_space_operators.jl")
include("subcell_operators.jl")
export get_nsigma, get_optimization_entries, get_multidimensional_optimization_entries
export get_sparsity_pattern, neighborhood_sparsity_pattern
export compute_moments_boundary
export plot_nodes, plot_normals
export GlaubitzIskeLampertÖffner2025, multidimensional_function_space_operator
export SubcellOperator, GlaubitzLampertNordströmWinters2025, subcell_operator, couple_subcell,
       derivative_matrix, grid_left, grid_right, mass_matrix_left, mass_matrix_right,
       mass_matrix_boundary_left, mass_matrix_boundary_right
end
