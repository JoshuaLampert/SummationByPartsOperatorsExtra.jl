using TrixiTest: @test_trixi_include_base

# Use a macro to avoid world age issues when defining new initial conditions etc.
# inside an example.
"""
    @test_trixi_include(example, args...)

Test by calling `trixi_include(example; args...)`.
By default, only the absence of error output is checked.

This is a thin wrapper around `TrixiTest.@test_trixi_include_base`, analogous to the
one Trixi.jl uses in its `test/test_trixi.jl`.
"""
macro test_trixi_include(example, args...)
    ex = quote
        @test_trixi_include_base($example, $(args...))
    end
    return esc(ex)
end
