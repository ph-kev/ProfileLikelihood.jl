# ProfileLikelihood.jl: Estimating Parameters and Finding Profile Likelihood

```@contents
Pages = ["index.md",
         "optimizer.md",
         "tutorial.md",
         "api.md",]
Depth = 1
```

## Installation 
To install, use the following commands inside Julia REPL: 
```julia 
julia> ]

(@v1.7) pkg> add https://github.com/ph-kev/ProfileLikelihood.jl
```

To use the package, add the command: 
```julia 
using ProfileLikelihood
```

## Introduction 
The package `ProfileLikelihood.jl` provide methods for generating perfect and noisy data, 
estimating parameters, generating profile likelihood, and finding confidence intervals of the estimated parameters. This package is written in mind for use in epidemiology, but it should work well for other fields such as systems biology. More information can be found at Raul et. al.'s "... exploiting the profile likelihood" which introduces the method of profile likelihood in the field of systems biology [1].

## Features  
- Generate perfect and noisy data from the state variables of a system of differential equations 
- Generate incidence data from cumulative data 
- Built-in objective functions derived from maximum likelihood estimation when the error follows relative error or constant variance error or when the data follow a Poisson distribution
- Utilize the `Optimization.jl` package which allow for optimization methods from the packages `BlackBoxOptim.jl`, `Evolutionary.jl`, `Metaheuristics.jl`, `MultistartOptimization.jl`, `NLopt.jl`, `NOMAD.jl`, and `Optim.jl`
- Implemented a simple fixed step-size algorithm to find the profile likelihood 
- Find the confidence intervals of the estimated parameters using likelihood-based approach 

## Limitations 
- No support for different types of data taken at different time points 
- No adaptive step size algorithm to find the profile likelihood 
- No support for finding a two-dimensional profile likelihood where two parameters are fixed as opposed to one parameter being fixed  
- No support for automatic differentiation of the likelihood function 

## References 
[1] A. Raue, C. Kreutz, T. Maiwald, J. Bachmann, M. Schilling, U. Klingmüller, J. Timmer, Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood, Bioinformatics, Volume 25, Issue 15, 1 August 2009, Pages 1923–1929, [https://doi.org/10.1093/bioinformatics/btp358](https://doi.org/10.1093/bioinformatics/btp358)
