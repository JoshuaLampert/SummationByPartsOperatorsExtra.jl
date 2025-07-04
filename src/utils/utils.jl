boundary_left(D::AbstractNonperiodicDerivativeOperator) = first(grid(D))
boundary_right(D::AbstractNonperiodicDerivativeOperator) = last(grid(D))

include("interpolation.jl")
include("optimization.jl")
include("sparsity_patterns.jl")
include("moments.jl")
include("corners.jl")
include("visualization.jl")
