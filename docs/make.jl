using Documenter, ProfileLikelihood

makedocs(sitename = "ProfileLikelihood.jl",
         pages = ["index.md",
         "optimizer.md",
         "tutorial.md",
         "api.md",
         ])

deploydocs(
    repo = "github.com/ph-kev/ProfileLikelihood.jl",
)