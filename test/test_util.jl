# Copied from TrixiBase.jl. See https://github.com/trixi-framework/TrixiBase.jl/issues/9.
"""
    @test_nowarn_mod expr

Modified version of `@test_nowarn expr` that prints the content of `stderr` when
it is not empty and ignores some common info statements printed in KernelInterpolation.jl
uses.
"""
macro test_nowarn_mod(expr, additional_ignore_content = String[])
    quote
        let fname = tempname()
            try
                ret = open(fname, "w") do f
                    redirect_stderr(f) do
                        $(esc(expr))
                    end
                end
                stderr_content = read(fname, String)
                if !isempty(stderr_content)
                    println("Content of `stderr`:\n", stderr_content)
                end

                # Patterns matching the following ones will be ignored. Additional patterns
                # passed as arguments can also be regular expressions, so we just use the
                # type `Any` for `ignore_content`.
                ignore_content = Any["[ Info: You just called `trixi_include`. Julia may now compile the code, please be patient.\n"]
                append!(ignore_content, $additional_ignore_content)
                for pattern in ignore_content
                    stderr_content = replace(stderr_content, pattern => "")
                end

                # We also ignore simple module redefinitions for convenience. Thus, we
                # check whether every line of `stderr_content` is of the form of a
                # module replacement warning.
                @test occursin(r"^(WARNING: replacing module .+\.\n)*$", stderr_content)
                ret
            finally
                rm(fname, force = true)
            end
        end
    end
end