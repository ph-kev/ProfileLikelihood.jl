module ProfileLikelihood
using Reexport

@reexport using DifferentialEquations
@reexport using Random
@reexport using Distributions
@reexport using Optimization
@reexport using OptimizationBBO
@reexport using OptimizationEvolutionary
@reexport using OptimizationMetaheuristics
@reexport using OptimizationMultistartOptimization
@reexport using OptimizationNLopt
@reexport using OptimizationNOMAD
@reexport using OptimizationOptimJL
@reexport using Roots

export generate_data, generate_incidence_data, 
relative_error, poisson_error, const_variance_error, likelihood_const, likelihood, 
estimate_params, estimate_params_multistart,
find_threshold, 
find_profile_likelihood, 
find_roots

include("generating_data.jl")
include("error_functions.jl")
include("likelihood_function.jl")
include("estimating_parameters.jl")
include("threshold.jl")
include("finding_PL.jl")
include("processing_data.jl")

end
