# Documentation

## Public API

Documentation for `ProfileLikelihood.jl`'s public functions. 

### Error/objective functions 
```@docs
relative_error(data::AbstractVector{<:Real}, sol::AbstractVector{<:Real}, 
                        noise_level::Real) 
```

```@docs
poisson_error(data::AbstractVector{<:Real}, sol::AbstractVector{<:Real}) 
```

```@docs
const_variance_error(data::AbstractVector{<:Real}, sol::AbstractVector{<:Real}, sigma::Real)
```

```@docs
likelihood_const(obj::String; noise_level::Real=0.0, times::AbstractVector{<:Real}=Vector{Real}(), data::AbstractVector{<:Real}=Vector{Real}(), sigma::Real = -1.0) 
```
### Generate data 
```@docs 
generate_data(index::Integer, seed::Integer, dist::Function, 
                       prob::SciMLBase.AbstractDEProblem, 
                       alg::SciMLBase.AbstractDEAlgorithm, 
                       times::AbstractVector{<:Real}; 
                       incidence_obs_status::Bool = false, 
                       kwargs...)
```

```@docs
generate_incidence_data(index::Integer, 
                                 prob::SciMLBase.AbstractDEProblem, 
                                 alg::SciMLBase.AbstractDEAlgorithm, 
                                 times::AbstractVector{<:Real}; 
                                 kwargs...)
```

### Estimating parameters 
```@docs
estimate_params
```

```@docs 
estimate_params_multistart
```
### Likelihood function 
```@docs 
likelihood
```

### Threshold 
```@docs 
find_threshold(confidence::Real, nums_params::Integer, loss::Real)
```

### Finding profile likelihood 
```@docs
find_profile_likelihood
```

```@docs
find_profile_likelihood_multistart
```
### Processing data 
```@docs
find_roots
```

## Internals 
Documentation for `ProfileLikelihood.jl`'s private functions which are not intended for use. 
```@docs
generate_incidence_data(sol::AbstractVector{<:Real}) 
```

```@docs
ProfileLikelihood.min_point
```

```@docs
ProfileLikelihood.go_right_PL
```

```@docs
ProfileLikelihood.go_left_PL
```

```@docs
ProfileLikelihood.go_right_PL_multistart
```

```@docs
ProfileLikelihood.go_left_PL_multistart
```