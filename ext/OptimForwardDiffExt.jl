module OptimForwardDiffExt

using Optim: Optim, Options, BFGS, LBFGSB, optimize, minimizer
import ForwardDiff

import SummationByPartsOperatorsExtra: construct_function_space_operator,
                                       default_opt_alg,
                                       default_options
using SummationByPartsOperatorsExtra: SummationByPartsOperatorsExtra,
                                      SummationByPartsOperators,
                                      GlaubitzNordströmÖffner2023,
                                      GlaubitzLampertWintersNordström2025,
                                      GlaubitzIskeLampertÖffner2026Multidimensional,
                                      GlaubitzEtAl2026,
                                      MatrixDerivativeOperator,
                                      MultidimensionalMatrixDerivativeOperator,
                                      SubcellOperator,
                                      UpwindOperators

using PreallocationTools: DiffCache, get_tmp

include("utils.jl")
include("function_space_operators_optim.jl")
include("multidimensional_function_space_operators.jl")
include("subcell_operators.jl")
include("upwind_operators.jl")
end
