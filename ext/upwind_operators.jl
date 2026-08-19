
function default_opt_alg(::GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized)
    return LBFGSB()
end
function default_options(::GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized,
                         verbose)
    return Options(g_tol = 1e-10, iterations = 10000, show_trace = verbose)
end

function SummationByPartsOperatorsExtra.compute_dissipation_eigenvalues(D, V,
                                                                        test_functions,
                                                                        K,
                                                                        source::GlaubitzLampertMattssonNiemeläWinters2026AccuracyOptimized;
                                                                        autodiff = Optim.ADTypes.AutoForwardDiff(),
                                                                        sigma0 = nothing,
                                                                        verbose = false,
                                                                        opt_alg = default_opt_alg(source),
                                                                        options = default_options(source,
                                                                                                  verbose))
    nodes = grid(D)
    N = length(nodes)
    NmK = N - K
    sigma0 = isnothing(sigma0) ? fill(real(D)(-1.0), NmK) : sigma0

    P = mass_matrix(D)
    P_inv = inv(P)
    D_central = Matrix(D)
    test_functions_derivatives = [x -> ForwardDiff.derivative(test_function, x)
                                  for test_function in test_functions]
    test_functions_values = [test_function.(nodes) for test_function in test_functions]
    test_functions_derivatives_values = [test_function_derivative.(nodes)
                                         for test_function_derivative in test_functions_derivatives]
    p = (; P_inv, D_central, V, test_functions_values, test_functions_derivatives_values)
    f(sigma) = optimization_function_upwind_function_space_operator(sigma, p)
    result = optimize(f, fill(-Inf, NmK), fill(0.0, NmK), sigma0,
                      opt_alg, options; autodiff)
    verbose && show(stdout, "text/plain", result)
    sigma = minimizer(result)
    return sigma
end

function optimization_function_upwind_function_space_operator(sigma, p)
    (; P_inv, D_central, V, test_functions_values, test_functions_derivatives_values) = p

    S = eigen_dissipation_matrix(V, sigma)
    dissipation = (P_inv * S) / 2
    D_minus = D_central - dissipation
    D_plus = D_central + dissipation

    # Compute the residuals for the test functions
    residuals = zeros(eltype(sigma), length(test_functions_values))
    for (i, (test_function_values, test_function_derivative_values)) in enumerate(zip(test_functions_values,
                                                                                      test_functions_derivatives_values))
        residuals[i] = sum(abs2,
                           D_plus * test_function_values -
                           test_function_derivative_values) +
                       sum(abs2,
                           D_minus * test_function_values -
                           test_function_derivative_values)
    end
    return sum(residuals)
end
