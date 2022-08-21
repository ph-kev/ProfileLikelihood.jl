"""
    likelihood(params::AbstractVector{<:Real}, 
               data::AbstractVector{<:AbstractVector{<:Real}},
               sol_obs::AbstractVector{<:Integer}, 
               prob::SciMLBase.AbstractDEProblem, 
               alg::SciMLBase.AbstractDEAlgorithm, 
               times::AbstractVector{<:Real}, 
               obj_arr::AbstractVector{<:Function}; 
               incidence_obs::AbstractVector{<:Integer} = Int64[], 
               param_index::Integer=0, 
               param_eval::Real=0.0, 
               solver_diff_opts::Dict = Dict())::Float64

This computes the likelihood based on the given parameters `params`, `data`, and likelihood functions in `obj_arr`. 

Each objective function in `obj_arr` corresponds to data described by `sol_obs` and `incidence_obs`. For instance, if `obj_arr = [obj1 obj2]`,`sol_obs = [2]`, `incidence_obs = [3]`, then `obj1` use data corresponding to the second state variable of the DEs and `obj2` use incidence data of the third state variable of the DEs. Noisy data or any other type of data should be passed to `data` and in the same order as in the arrays `sol_obs` and `incidence_obs`.
Fixing a parameter can be done using `param_index`. Instead, this return the fitted parameters with `param_eval` for the fixed parameter and accordingly, the loss is the objective function evaluated at the fitted parameters with `param_eval` for the fixed parameter. 

# Arguments
- `params::AbstractVector{<:Real}`: Parameters to evaluate at. If `param_index` and `param_eval` are used, then `params` should not contain the fixed parameter.
- `data::AbstractVector{<:AbstractVector{<:Real}}`: Vector of data used for computing likelihood. `data` must be in the same order as in `sol_obs` and `incidence_obs`.
- `sol_obs::AbstractVector{<:Integer}`: Indices of the state variables of the DEs to be used for sampling data points.
- `prob::SciMLBase.AbstractDEProblem`: DE Problem (see `DifferentialEquations.jl` for more information).
- `alg::SciMLBase.AbstractDEAlgorithm`: DE Solver Algorithm (see `DifferentialEquations.jl` for more information).
- `times::AbstractVector{<:Real}`: Times that the data points will be sampled at. 
- `obj_arr::AbstractVector`: Vector of objective functions. 

# Keywords
- `incidence_obs::AbstractVector{<:Integer} = []`: Indices of the state variables of the DEs to find incidence data of. The state variables must be cumulative data.   
- `param_index::Integer=0`: Index of the parameter vector that is fixed. 
- `param_eval::Real=0.0`: Value for the fixed parameter.  
- `solver_diff_opts::Dict = Dict()`: Keyword arguments to be passed into the DE solver. See `DifferentialEquations.jl`'s Common Solver Options.

# Returns
- `loss`: Likelihood of observing `data` given `params`.
"""
function likelihood(params::AbstractVector{<:Real}, 
                    data::AbstractVector{<:AbstractVector{<:Real}},
                    sol_obs::AbstractVector{<:Integer}, 
                    prob::SciMLBase.AbstractDEProblem, 
                    alg::SciMLBase.AbstractDEAlgorithm, 
                    times::AbstractVector{<:Real}, 
                    obj_arr::AbstractVector{<:Function}; 
                    incidence_obs::AbstractVector{<:Integer} = Int64[], 
                    param_index::Integer=0, 
                    param_eval::Real=0.0, 
                    solver_diff_opts::Dict = Dict())::Float64
    sol_obs_copy = vcat(sol_obs, incidence_obs)
    if param_index != 0 # check if the parameter is fixed or not
        params_copy = copy(params)
        insert!(params_copy, param_index, param_eval)
        prob_cur = remake(prob, p=params_copy)
    else
        prob_cur = remake(prob, p=params)
    end
    # solve ODEs
    sol = solve(
        prob_cur,
        alg,
        saveat=times,
        save_idxs=sol_obs_copy;
        solver_diff_opts...
    )
    loss = 0.0
    index = 0 
    for ind in eachindex(sol_obs)
        loss += obj_arr[ind](data[ind], sol[ind,:])
        index += 1
    end
    for ind in eachindex(incidence_obs)
        incidence_sol = generate_incidence_data(sol[ind + index,:])
        # remove the first data point for cumulative data since it is always 0  
        loss += obj_arr[ind + index](data[ind + index][2:end], incidence_sol[2:end]) 
    end
    return loss
end