# Changelog

SummationByPartsOperatorsExtra.jl follows the interpretation of
[semantic versioning (semver)](https://julialang.github.io/Pkg.jl/dev/compatibility/#Version-specifier-format-1)
used in the Julia ecosystem. Notable changes will be documented in this file
for human readability.

## Changes in the v0.1 lifecycle

#### Added

- Allow Julia v1.10 ([#94]).

#### Fixed

- `MultidimensionalLinearAdvectionNonperiodicSemidiscretization` was not energy stable at corners
  adjacent to one inflow and one outflow boundary face, e.g., for tensor product operators and
  operators constructed with `corners`. The boundary data is now imposed at the nodes with
  `τ_j < 0`, where `τ = diag(a[1] * B_1 + a[2] * B_2)`, which coincides with the previous rule
  at boundary nodes with a single normal ([#112]).
- The weights of sub-cell operators constructed with `subcell_operator` were bounded by one, such
  that the operators were not exact (without any warning) if a sub-cell was longer than its number
  of nodes. The weights are now scaled by the length of the sub-cell. Note that this changes the
  meaning of the weight entries of a user-provided initial guess `x0`.
