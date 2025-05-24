using Documenter
using SummationByPartsOperatorsExtra

# Define module-wide setups such that the respective modules are available in doctests
DocMeta.setdocmeta!(SummationByPartsOperatorsExtra, :DocTestSetup,
                    :(using SummationByPartsOperatorsExtra);
                    recursive = true)

makedocs(;
         modules = [SummationByPartsOperatorsExtra],
         authors = "Joshua Lampert <joshua.lampert@uni-hamburg.de>",
         repo = Remotes.GitHub("JoshuaLampert", "SummationByPartsOperatorsExtra.jl"),
         sitename = "SummationByPartsOperatorsExtra.jl",
         format = Documenter.HTML(;
                                  prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://JoshuaLampert.github.io/SummationByPartsOperatorsExtra.jl/stable",
                                  edit_link = "main"),
         pages = ["Home" => "index.md",
             "Development" => "development.md",
             "Reference" => "ref.md",
             "License" => "license.md"])

deploydocs(;
           repo = "github.com/JoshuaLampert/SummationByPartsOperatorsExtra.jl",
           devbranch = "main",
           push_preview = true)
