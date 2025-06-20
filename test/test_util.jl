using TrixiTest: @trixi_test_nowarn

# Use a macro to avoid world age issues when defining new initial conditions etc.
# inside an example.
"""
    @test_trixi_include(example)

Test by calling `trixi_include(example; parameters...)`.
By default, only the absence of error output is checked.
"""
macro test_trixi_include(example, args...)
    local kwargs = Pair{Symbol, Any}[]
    for arg in args
        if arg.head == :(=)
            push!(kwargs, Pair(arg.args...))
        end
    end

    quote
        println("═"^100)
        println($example)

        # evaluate examples in the scope of the module they're called from
        @trixi_test_nowarn trixi_include(@__MODULE__, $example; $kwargs...)

        println("═"^100)
    end
end
