module SummationByPartsOperatorsExtraManifoldsManoptForwardDiffExt

using Manifolds: Manifolds, SkewSymmetricMatrices, PositiveVectors, ProductManifold,
                 check_point
using Manopt: quasi_Newton, interior_point_Newton, augmented_Lagrangian_method,
              ApproxHessianBFGS, ManifoldGradientObjective, ConstrainedManifoldObjective,
              DebugFeasibility,
              StopAfterIteration, StopWhenGradientNormLess, StopWhenCostLess
using LinearAlgebra: eigvals, Symmetric
import RecursiveArrayTools: ArrayPartition
import ForwardDiff
import DifferentiableEigen
import SummationByPartsOperatorsExtra: construct_function_space_operator,
                                       default_opt_alg,
                                       default_options
using SummationByPartsOperatorsExtra: SummationByPartsOperatorsExtra,
                                      GlaubitzIskeLampertÖffner2025Basic,
                                      GlaubitzIskeLampertÖffner2025Regularized,
                                      GlaubitzIskeLampertÖffner2025EigenvalueProperty,
                                      MatrixDerivativeOperator

include("utils.jl")
include("function_space_operators.jl")

default_opt_alg(::GlaubitzIskeLampertÖffner2025Basic) = quasi_Newton
default_opt_alg(::GlaubitzIskeLampertÖffner2025Regularized) = augmented_Lagrangian_method
function default_opt_alg(::GlaubitzIskeLampertÖffner2025EigenvalueProperty)
    augmented_Lagrangian_method
end
function default_options(::GlaubitzIskeLampertÖffner2025Basic,
                         verbose)
    if verbose
        debug = [:Iteration,
            :Time,
            " | ",
            (:Cost, "f(x): %.6e"),
            " | ",
            (:GradientNorm, "||∇f(x)||: %.6e"),
            "\n",
            :Stop]
    else
        debug = []
    end
    return (;
            debug = debug,
            stopping_criterion = StopAfterIteration(10000) |
                                 StopWhenGradientNormLess(1e-16) |
                                 StopWhenCostLess(1e-28))
end
function default_options(::GlaubitzIskeLampertÖffner2025Regularized,
                         verbose)
    if verbose
        debug = [:Iteration,
            :Time,
            " | ",
            (:Cost, "f(x): %.6e"),
            " | ",
            (:GradientNorm, "||∇f(x)||: %.6e"),
            " | ",
            DebugFeasibility(["feasible: ", :Feasible, ", total violation: ", :TotalEq]),
            "\n",
            :Stop]
    else
        debug = []
    end
    return (;
            debug = debug,
            stopping_criterion = StopAfterIteration(1000) |
                                 StopWhenGradientNormLess(1e-16) |
                                 StopWhenCostLess(1e-28))
end
function default_options(::GlaubitzIskeLampertÖffner2025EigenvalueProperty,
                         verbose)
    if verbose
        debug = [:Iteration,
            :Time,
            " | ",
            (:Cost, "f(x): %.6e"),
            " | ",
            (:GradientNorm, "||∇f(x)||: %.6e"),
            " | ",
            DebugFeasibility(["feasible: ", :Feasible, ", total violation: ", :TotalInEq]),
            "\n",
            :Stop]
    else
        debug = []
    end
    return (;
            debug = debug,
            stopping_criterion = StopAfterIteration(1000) |
                                 StopWhenGradientNormLess(1e-16) |
                                 StopWhenCostLess(1e-28))
end

function construct_function_space_operator(basis_functions, nodes,
                                           source::Union{GlaubitzIskeLampertÖffner2025Basic,
                                                         GlaubitzIskeLampertÖffner2025Regularized,
                                                         GlaubitzIskeLampertÖffner2025EigenvalueProperty};
                                           basis_functions_weights = ones(eltype(nodes),
                                                                          length(basis_functions)),
                                           regularization_functions = nothing,
                                           bandwidth = length(nodes) - 1,
                                           size_boundary = 2 * bandwidth,
                                           different_values = true,
                                           sparsity_pattern = nothing,
                                           autodiff = :forward,
                                           x0 = nothing, verbose = false,
                                           opt_alg = default_opt_alg(source),
                                           options = default_options(source, verbose))
    T = eltype(nodes)
    K = length(basis_functions)
    N = length(nodes)

    assert_correct_bandwidth(N, bandwidth, size_boundary)
    if !isnothing(sparsity_pattern)
        assert_correct_sparsity_pattern(sparsity_pattern)
        sparsity_pattern = UpperTriangular(sparsity_pattern)
    end
    assert_correct_length_basis_functions_weights(basis_functions_weights, basis_functions)
    if source isa GlaubitzIskeLampertÖffner2025Regularized
        @assert !isnothing(regularization_functions) "regularization_functions must be provided for GlaubitzIskeLampertÖffner2025Regularized"
        regularization_functions_derivatives = [x -> ForwardDiff.derivative(regularization_function,
                                                                            x)
                                                for regularization_function in regularization_functions]
    end
    L = get_nsigma(N; bandwidth, size_boundary, different_values, sparsity_pattern)

    basis_functions_derivatives = [x -> ForwardDiff.derivative(basis_functions[i], x)
                                   for i in 1:K]
    basis_functions_orthonormalized, basis_functions_orthonormalized_derivatives = orthonormalize_gram_schmidt(basis_functions,
                                                                                                               basis_functions_derivatives,
                                                                                                               nodes)
    # This weights column k, i.e. basis function k, with the weight `basis_functions_weights[k]`
    V = vandermonde_matrix(basis_functions_orthonormalized, nodes) *
        Diagonal(basis_functions_weights)
    V_x = vandermonde_matrix(basis_functions_orthonormalized_derivatives, nodes) *
          Diagonal(basis_functions_weights)
    B = spzeros(T, N, N)
    B[1, 1] = -1
    B[N, N] = 1
    R = B * V / 2
    x_length = last(nodes) - first(nodes)

    if isnothing(x0)
        # TODO: This does not satisfy the constraints for the constrained case and will therefore fail.
        # In this case, should we run the unconstrained optimization first to get a good initial guess?
        # Or should we just enforce the user to pass a suitable initial guess, e.g., by first running the
        # unconstrained case with the same nodes and basis, but without the regularization?
        x0 = [zeros(T, L); invsig.(1 / N * ones(T, N))]
    else
        n_total = L + N
        if length(x0) != n_total
            throw(ArgumentError("Initial guess to has be L + N = $n_total long"))
        end
    end

    sigma0, rho0 = split_x_function_space_operator(x0, L)

    M1 = SkewSymmetricMatrices(N)
    M2 = PositiveVectors(N)
    M = ProductManifold(M1, M2)
    S0 = SummationByPartsOperatorsExtra.create_S(sigma0, N, bandwidth, size_boundary,
                                                 different_values, sparsity_pattern)
    p0 = diag(create_P(rho0, x_length))
    x0 = ArrayPartition(S0, p0)

    param = (; V, V_x, R, B)
    if source isa GlaubitzIskeLampertÖffner2025Regularized
        G = vandermonde_matrix(regularization_functions, nodes)
        G_x = vandermonde_matrix(regularization_functions_derivatives, nodes)
        R_G = B * G / 2
        param = (; param..., G, G_x, R_G)
    end
    if autodiff == :forward
        autodiff = Manifolds.ManifoldDiff.AutoForwardDiff()
    end
    objective = get_objective_function(source, param, autodiff)
    x = opt_alg(M, objective, x0; options...)

    S, p = x.x
    weights = p
    Q = S + B / 2
    D = inv(Diagonal(p)) * Q
    return weights, D
end

function optimization_gradient_function_space_operator(M, f, x, autodiff)
    b = Manifolds.ManifoldDiff.TangentDiffBackend(autodiff)
    return Manifolds.ManifoldDiff.gradient(M, x -> f(M, x), x, b)
end
function optimization_jacobian_function_space_operator(M, f, x, autodiff)
    b = Manifolds.ManifoldDiff.TangentDiffBackend(autodiff)
    return Manifolds.ManifoldDiff.gradient(M, x -> f(M, x), x, b)
end

function get_objective_function(::GlaubitzIskeLampertÖffner2025Basic,
                                param, autodiff)
    f(M, x) = optimization_function_function_space_operator(M, x, param)
    grad_f(M, x) = optimization_gradient_function_space_operator(M, f, x, autodiff)
    return ManifoldGradientObjective(f, grad_f)
end
function get_objective_function(::GlaubitzIskeLampertÖffner2025Regularized,
                                param, autodiff)
    f(M, x) = optimization_function_function_space_operator_G(M, x, param)
    grad_f(M, x) = optimization_gradient_function_space_operator(M, f, x, autodiff)
    hess_f(M, p, Xp) = ApproxHessianBFGS(M, p, grad_f)(M, p, Xp)

    h1(M, x) = optimization_function_function_space_operator(M, x, param)
    h(M, x) = [h1(M, x)]
    grad_h(M, x) = [optimization_gradient_function_space_operator(M, h1, x, autodiff)]
    hess_h(M, p, Xp) = ApproxHessianBFGS(M, p, grad_h)(M, p, Xp)
    return ConstrainedManifoldObjective(f, grad_f; hess_f = hess_f,
                                        g = nothing, grad_g = nothing, h = h,
                                        grad_h = grad_h, hess_h = hess_h,
                                        equality_constraints = 1,
                                        atol = 1e-28)
end
function get_objective_function(::GlaubitzIskeLampertÖffner2025EigenvalueProperty,
                                param, autodiff)
    N, N = size(param.V)
    f(M, x) = optimization_function_function_space_operator(M, x, param)
    grad_f(M, x) = optimization_gradient_function_space_operator(M, f, x, autodiff)
    hess_f(M, p, Xp) = ApproxHessianBFGS(M, p, grad_f)(M, p, Xp)

    g1(M, x) = eigenvalue_property(M, x, param)
    g(M, x) = [g1(M, x)]
    grad_g(M, x) = [optimization_gradient_function_space_operator(M, g1, x, autodiff)]
    hess_g(M, p, Xp) = ApproxHessianBFGS(M, p, grad_g)(M, p, Xp)
    return ConstrainedManifoldObjective(f, grad_f; hess_f = hess_f,
                                        g = g, grad_g = grad_g, hess_g = hess_g,
                                        h = nothing, grad_h = nothing,
                                        inequality_constraints = 1,
                                        atol = 1e-28)
end

function optimization_function_function_space_operator(M, x, param)
    (; V, V_x, R) = param
    S, p = x.x

    A = S * V - Diagonal(p) * V_x + R
    return sum(abs2, A)
end

function optimization_function_function_space_operator_G(M, x, param)
    (; G, G_x, R_G) = param
    S, p = x.x

    A = S * G - Diagonal(p) * G_x + R_G
    return sum(abs2, A)
end

# The eigenvalue property is satisifed if each value in the vector of the
# return value of this function is non-positive.
function eigenvalue_property(M, x, param)
    (; B) = param
    S, p = x.x

    Q = S + B / 2
    D = inv(Diagonal(p)) * Q
    sigma = 1.0
    D[1, 1] += sigma / p[1]
    lambdas = DifferentiableEigen.eigen(D)[1]
    return -minimum(real(lambdas))
end

end
