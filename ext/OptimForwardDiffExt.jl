module OptimForwardDiffExt

using Optim: Optim, Options, BFGS, optimize, minimizer
using LinearAlgebra: mul!
import ForwardDiff

import SummationByPartsOperatorsExtra: construct_function_space_operator,
                                       default_opt_alg,
                                       default_options
using SummationByPartsOperatorsExtra: SummationByPartsOperatorsExtra,
                                      GlaubitzNordströmÖffner2023,
                                      GlaubitzLampertWintersNordström2025,
                                      GlaubitzIskeLampertÖffner2026Multidimensional,
                                      MatrixDerivativeOperator,
                                      MultidimensionalMatrixDerivativeOperator,
                                      SubcellOperator

using PreallocationTools: DiffCache, get_tmp

include("utils.jl")
include("function_space_operators_optim.jl")
include("multidimensional_function_space_operators.jl")
include("subcell_operators.jl")
end
