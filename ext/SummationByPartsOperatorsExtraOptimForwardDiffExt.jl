module SummationByPartsOperatorsExtraOptimForwardDiffExt

using Optim: Optim, Options, LBFGS, optimize, minimizer
import ForwardDiff

import SummationByPartsOperatorsExtra: construct_function_space_operator,
                                       default_opt_alg,
                                       default_options
using SummationByPartsOperatorsExtra: SummationByPartsOperatorsExtra,
                                      GlaubitzNordströmÖffner2023,
                                      GlaubitzIskeLampertÖffner2025,
                                      GlaubitzLampertNordströmWinters2025,
                                      MatrixDerivativeOperator,
                                      MultidimensionalMatrixDerivativeOperator,
                                      SubcellOperator

using PreallocationTools: DiffCache, get_tmp

include("utils.jl")
include("function_space_operators_optim.jl")
include("multidimensional_function_space_operators.jl")
include("subcell_operators.jl")
end
