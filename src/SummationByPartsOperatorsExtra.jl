"""
    SummationByPartsOperatorsExtra

**SummationByPartsOperatorsExtra.jl** is a [Julia](https://julialang.org/) package that
implements some extra functionality for the package [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl).
SummationByPartsOperatorsExtra.jl is still in an early stage of development and is meant to be used
for research purposes. Maybe some parts of the package will be moved to SummationByPartsOperators.jl in the future.
Until now, the package focuses on the implementation of function space summation-by-parts operators in one and multiple dimensions
and on subcell summation-by-parts operators.

See also: [SummationByPartsOperatorsExtra.jl](https://github.com/JoshuaLampert/SummationByPartsOperatorsExtra.jl)
"""
module SummationByPartsOperatorsExtra

using AutoHashEquals: @auto_hash_equals
using ArgCheck: @argcheck
using DiffEqCallbacks: PeriodicCallback, PeriodicCallbackAffect
using LinearAlgebra: LinearAlgebra, Diagonal, UpperTriangular, diag, dot
import LinearAlgebra: mul!, diagind, norm
using Reexport: @reexport
import SciMLBase: u_modified!, get_tmp_cache
@reexport using SummationByPartsOperators
using SimpleUnPack: @unpack
@reexport using StaticArrays: SVector
using StatsBase: countmap
using SummationByPartsOperators: AbstractDerivativeOperator,
                                 AbstractNonperiodicDerivativeOperator,
                                 AbstractMultidimensionalMatrixDerivativeOperator,
                                 AbstractSemidiscretization,
                                 scale_by_mass_matrix!,
                                 scale_by_inverse_mass_matrix!

include("utils/utils.jl")
include("function_space_operators.jl")
include("multidimensional_function_space_operators.jl")
include("subcell_operators.jl")
include("conservation_laws/analysis_callback.jl")
include("conservation_laws/multidimensional_linear_advection.jl")
export get_nsigma, get_optimization_entries, get_multidimensional_optimization_entries
export get_sparsity_pattern, neighborhood_sparsity_pattern
export compute_moments_boundary
export plot_nodes, plot_normals
export GlaubitzIskeLampertÖffner2025, multidimensional_function_space_operator
export SubcellOperator, GlaubitzLampertNordströmWinters2025, subcell_operator,
       derivative_matrix, grid_left, grid_right, mass_matrix_left, mass_matrix_right,
       mass_matrix_boundary_left, mass_matrix_boundary_right,
       left_projection_left, left_projection_right,
       right_projection_left, right_projection_right
export AnalysisCallback, tstops, quantities
export MultidimensionalLinearAdvectionNonperiodicSemidiscretization
end
