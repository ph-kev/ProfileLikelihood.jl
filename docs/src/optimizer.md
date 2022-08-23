# Optimizers 

The package `ProfileLikelihood.jl` utilizes `Optimization.jl` to exports a variety of
optimization methods. `ProfileLikelihood.jl` currently
supports optimization methods from 
- `BlackBoxOptim.jl`,
- `Evolutionary.jl`, 
- `Metaheuristics.jl`, 
- `MultistartOptimization.jl`, 
- `NLopt.jl`,
- `NOMAD.jl`,
- `Optim.jl`. 
See `Optimization.jl` and `ProfileLikelihood.jl` documentation to further understand 
how to use the optimization methods. Another resource is the tutorial and the `examples` folder on the
[Github](https://github.com/ph-kev/ProfileLikelihood.jl) of this package.

## Note on Optimization 
For generating profile likelihood plots, we must find the global minimum for each point of the
profile likelihood plots. Finding the global minimum is much more difficult than finding the 
local minimum. Thus, it is recommended that after finding the global minimum to use another optimization method to ensure that the global minimum is actually found.

Note that this is based on personal experience on using these optimization methods. For generating profile likelihood plots, I recommend the following optimization methods: 
- `NOMADOpt()` from `NOMAD.jl`,
- `BBO_generating_set_search()` from `BlackBoxOptim.jl`,
- `OptimizationMetaheuristics.DE()` from `Metaheuristics.jl`,
- `MultistartOptimization.TikTak(n)` with `NLopt.LN_NELDERMEAD` from `MultistartOptimization.jl` and `NLopt.jl`
from most recommended to least recommended. `NOMADOpt()` is reasonably accurate and fast at 
finding the global minimum compared to the rest of the methods. However, `BBO_generating_set_search()` is the most accurate at finding the global minimum, but 
is typically the slowest. Other optimization methods that are worth using are `OptimizationMetaheuristics.DE()` and `MultistartOptimization.TikTak(n)` with `NLopt.LN_NELDERMEAD` which could be used to check if the optimization method was successful at finding the global minimum.