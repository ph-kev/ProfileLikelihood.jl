using Documenter, ProfileLikelihood, DifferentialEquations, Distributions, Plots, LaTeXStrings, Measures, Interpolations

makedocs(sitename = "ProfileLikelihood.jl",
         pages = ["index.md",
         "optimizer.md",
         "tutorial.md",
         "api.md",
         ])

deploydocs(
    repo = "github.com/ph-kev/ProfileLikelihood.jl",
)