# AGENTS.md

This file provides guidance to coding agents (e.g. Claude Code, which reads it via the
`@AGENTS.md` import in `CLAUDE.md`) when working with code in this repository.

## What this package is

SummationByPartsOperatorsExtra.jl is a Julia research package that adds extra functionality on top of
[SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl) (SBP). It `@reexport`s
SummationByPartsOperators.jl and PolynomialBases.jl, so users get all of those symbols by `using SummationByPartsOperatorsExtra`.
The focus is on **function space SBP operators** (one- and multi-dimensional) and **subcell SBP operators**, mostly
constructed by numerical optimization. Requires Julia ≥ 1.11.

## Common commands

Tests use [TestItemRunner](https://github.com/julia-actions/TestItemRunner). Each test is an isolated `@testitem`
with `setup=[...]` snippets (`Setup`, `AdditionalImports`, `PDEExamples`, ...) defined in `test/runtests.jl` and
`test/test_util.jl`.

Run individual test items via the persistent `mcp__julia__julia_eval` session (`env_path` = the package root), not `julia --project=test`. Use `TestEnv.activate()` to build the same merged env
(package + test deps) that `Pkg.test` uses, then filter with TestItemRunner. The session stays warm, so later items
skip recompilation:

```julia
using TestEnv
TestEnv.activate() # merges test/Project.toml deps with the package; activates a temp env
using TestItemRunner
cd("test")
@run_package_tests filter = ti -> occursin("rbf_fd_advection_2d_basic.jl", ti.name)
```

The `filter` matches against `ti.name` (the string in `@testitem "..."`), `ti.tags`, and `ti.filename`. Do `TestEnv.activate()`
once per session; afterwards just re-run the `@run_package_tests` line with a new filter. Do not run the whole suite.

Format code (CI enforces SciML style via `.JuliaFormatter.toml` — `yas_style_nesting`, `align_struct_field`,
`always_use_return`):

```sh
julia -e 'using JuliaFormatter; format(".")'
```

### Running examples or development scripts (the `run/` project)

For interactive experimentation, use the `run/` project (additional packages needed for examples live there) so the main `Project.toml`
stays clean: `julia --project=run`. See `docs/src/development.md`. The `examples/` directory holds runnable example
scripts (RBF function space operators for advection, etc.), which are exercised by `test/test_examples.jl`.

Run examples or development scripts via the MCP Julia tools with `env_path` pointing at `run/` for an interactive and persistent session.

## Architecture: stubs in `src/`, implementations in `ext/`

This is the most important thing to understand before editing. The heavy operator-construction routines depend on
optimization and autodiff packages that are **weak dependencies**. The pattern is:

- `src/` declares empty function stubs, e.g. `function function_space_operator end` and
  `function construct_function_space_operator end` (see `src/function_space_operators.jl`). Calling these without the
  relevant optimization package loaded throws a "function not defined / load Optim" style error.
- `ext/` provides the actual methods, loaded automatically only when the weak deps are present.

The two optimizer backends, mapped by their `SourceOfCoefficients` type:

| Source type | Backend extension | Optimizer |
|---|---|---|
| `GlaubitzIskeLampertÖffner2026Basic` / `Regularized` / `EigenvalueProperty` | `ManifoldsManoptForwardDiffExt` | Manopt.jl on a manifold |
| `GlaubitzLampertWintersNordström2025` (subcell), `GlaubitzIskeLampertÖffner2026Multidimensional`, multidim/standard FSBP | `OptimForwardDiffExt` | Optim.jl (BFGS default) |

Both backends extend the same internal seams: `construct_function_space_operator`, `default_opt_alg(source)`,
`default_options(source)`. When adding a new operator family, add the stub + exported `SourceOfCoefficients` struct in
`src/`, then implement `construct_*` and `default_*` for it in the appropriate `ext/` file. ForwardDiff supplies
gradients/Hessians for the optimization in both backends.

Other extensions: `MeshesExt` (geometry/normals for multidimensional operators), `MeshesMakieExt` and the visualization
helpers (`plot_nodes`, `plot_normals`, `plot_sparsity_pattern`) — plotting only works once Makie is loaded.

### Module layout

- `src/SummationByPartsOperatorsExtra.jl` — module root: imports, `include`s, and the full export list. Start here to
  see the public API surface.
- `src/function_space_operators.jl` / `src/multidimensional_function_space_operators.jl` — FSBP operator stubs + source types.
- `src/subcell_operators.jl` — `SubcellOperator`, `couple_subcell`, and the many left/right accessors
  (`grid_left`, `mass_matrix_right`, `integrate_left`, projections, …).
- `src/polynomialbases_operators.jl` — `PolynomialBasesDerivativeOperator` bridging PolynomialBases.jl.
- `src/conservation_laws/` — `AnalysisCallback` and a multidimensional linear advection semidiscretization (SciML-based).
- `src/utils/` — moments (`compute_moments_boundary`), sparsity patterns, interpolation, corners, optimization helpers, visualization.

## Conventions

- Formatting: SciML style with `always_use_return = true`, `align_struct_field = true`, `yas_style_nesting = true`
  (see `.JuliaFormatter.toml`). The `always_use_return` rule means explicit `return` is expected at the end of functions.
- Operator constructors return SBP-operator types from SummationByPartsOperators.jl (e.g. `MatrixDerivativeOperator`,
  `MultidimensionalMatrixDerivativeOperator`), so the result behaves like a normal SBP operator downstream.
- Source/citation types (the `Glaubitz…` structs) subtype `SourceOfCoefficients` and carry a `Base.show` with the paper
  reference; they are the dispatch key selecting construction algorithm and optimizer.
- `test/test_aqua.jl` (Aqua.jl) and ExplicitImports checks run in CI — keep imports explicit and avoid the issues Aqua flags.
