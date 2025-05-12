# SummationByPartsOperatorsExtra.jl

[![Build Status](https://github.com/JoshuaLampert/SummationByPartsOperatorsExtra.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JoshuaLampert/SummationByPartsOperatorsExtra.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/JoshuaLampert/SummationByPartsOperatorsExtra.jl/graph/badge.svg?token=ZnS5D3tWSK)](https://codecov.io/gh/JoshuaLampert/SummationByPartsOperatorsExtra.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

**SummationByPartsOperatorsExtra.jl** is a [Julia](https://julialang.org/) package that
implements some extra functionality for the package [SummationByPartsOperators.jl](https://github.com/ranocha/SummationByPartsOperators.jl).
SummationByPartsOperatorsExtra.jl is still in an early stage of development and is meant to be used
for research purposes. Maybe some parts of the package will be moved to SummationByPartsOperators.jl in the future.
Until now, the package focuses on the implementation of function space summation-by-parts operators in one and multiple dimensions.

## Installation

If you have not yet installed Julia, then you first need to [download Julia](https://julialang.org/downloads/). Please
[follow the instructions for your operating system](https://julialang.org/downloads/platform/). SummationByPartsOperatorsExtra.jl
works with Julia v1.11 and newer. You can install SummationByPartsOperatorsExtra.jl by
executing the following commands from the Julia REPL

```julia
julia> using Pkg

julia> Pkg.add("https://github.com/JoshuaLampert/SummationByPartsOperatorsExtra.jl")
```

## Usage

In the Julia REPL, first load the package SummationByPartsOperatorsExtra.jl

```julia
julia> using SummationByPartsOperatorsExtra
```

SummationByPartsOperatorsExtra.jl is built on top of the package SummationByPartsOperators.jl and exports all the functions
and types of the package.

## Authors

The package is developed and maintained by Joshua Lampert (University of Hamburg).

## License and contributing

SummationByPartsOperatorsExtra.jl is published under the MIT license (see [License](https://github.com/JoshuaLampert/SummationByPartsOperatorsExtra.jl/blob/main/LICENSE)).
We are pleased to accept contributions from everyone, preferably in the form of a PR.
